from __future__ import annotations

import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------
# Path bootstrap
# This lets new comparison scripts work even if older files still use
# flat imports like `from zip_strategy import ...`
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

for p in [
    PROJECT_ROOT,
    PROJECT_ROOT / "core",
    PROJECT_ROOT / "markets",
    PROJECT_ROOT / "strategies",
    PROJECT_ROOT / "baseline",
    PROJECT_ROOT / "pricing",
]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from household import get_household_df
from order_book import Order, OrderBook
from tariff import load_fit_profile, load_tou_profile

from aa_strategy import AAStrategy
from static_strategy import StaticStrategy
from zip_strategy import ZIPStrategy

from cda import CDA_mechanism
from call_market import CallMarketMechanism
from uniform_price_market import UniformPriceDoubleAuction

from pricing_rules import get_pricing_rule


TARIFF_TARGET_YEAR = int(os.getenv("TARIFF_TARGET_YEAR", "2025"))
SIM_SEASON = os.getenv("SIM_SEASON", "").strip().lower() or None
TARIFF_AGG = os.getenv("TARIFF_AGG", "median").strip().lower()

HOUSEHOLD_SEASON = os.getenv("HOUSEHOLD_SEASON", SIM_SEASON or "summer").strip().lower()
HOUSEHOLD_DAYS = os.getenv("HOUSEHOLD_DAYS", "").strip()
HOUSEHOLD_IDS = os.getenv("HOUSEHOLD_IDS", "").strip()

MARKET_NAME = os.getenv("MARKET_NAME", "cda").strip().lower()
STRATEGY_NAME = os.getenv("STRATEGY_NAME", "static").strip().lower()
PRICING_NAME = os.getenv("PRICING_NAME", "midpoint").strip().lower()
PRICING_K = float(os.getenv("PRICING_K", "0.5"))


def parse_household_days(value: str):
    return None if value == "" else int(value)


def parse_selected_households(value: str):
    if not value:
        return None
    ids = [x.strip() for x in value.split(",") if x.strip()]
    return ids or None


def validate_household_df(household_df: pd.DataFrame):
    required_cols = {"h_id", "DateTime", "import_energy", "export_energy"}
    missing = required_cols - set(household_df.columns)
    if missing:
        raise ValueError(f"household_df missing required columns: {sorted(missing)}")
    if household_df.empty:
        raise ValueError("household_df is empty")


def build_strategy(h_id: str, side: str, strategy_name: str):
    key = strategy_name.strip().lower()

    if key == "zip":
        return ZIPStrategy(h_id=h_id, side=side)

    if key == "static":
        return StaticStrategy(h_id=h_id, side=side)

    if key == "aa":
        return AAStrategy(h_id=h_id, side=side)

    raise ValueError("strategy_name must be one of: 'static', 'zip', 'aa'")


class CDAWithPricing(CDA_mechanism):
    """
    Wrapper around your existing CDA implementation so we can compare
    different pricing rules without rewriting cda.py.
    """

    def __init__(
        self,
        *args,
        pricing_name: str = "midpoint",
        pricing_kwargs: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pricing_name = pricing_name
        self.pricing_kwargs = pricing_kwargs or {}
        self._pricing_fn = get_pricing_rule(pricing_name, **self.pricing_kwargs)

    def trade_price(self, best_bid, best_ask):
        return float(
            self._pricing_fn(
                float(best_bid.submitted_price),
                float(best_ask.submitted_price),
            )
        )


def build_market(
    market_name: str,
    order_book,
    trader_registry,
    verbose: bool = True,
    pricing_name: str | None = None,
    pricing_kwargs: dict | None = None,
):
    key = market_name.strip().lower()

    if key == "cda":
        return CDAWithPricing(
            order_book=order_book,
            trader_registry=trader_registry,
            max_trade_rounds=150,
            max_no_trade_rounds=50,
            verbose=verbose,
            pricing_name=pricing_name or "midpoint",
            pricing_kwargs=pricing_kwargs or {},
        )

    if key in {"call", "periodic_call", "periodic_double_auction"}:
        return CallMarketMechanism(
            order_book=order_book,
            trader_registry=trader_registry,
            verbose=verbose,
        )

    if key in {"uniform", "uniform_price", "uniform_price_double_auction"}:
        return UniformPriceDoubleAuction(
            order_book=order_book,
            trader_registry=trader_registry,
            verbose=verbose,
        )

    raise ValueError(
        f"Unsupported market_name={market_name!r}. "
        "Choose from: 'cda', 'call', 'uniform_price'."
    )


def gen_orders_and_slot(
    slot_df,
    tou_profile,
    fit_profile,
    trader_registry,
    order_id_start=0,
    strategy_name="static",
):
    order_book = OrderBook()
    order_counter = order_id_start

    slot_df = slot_df.sort_values(["DateTime", "h_id"]).reset_index(drop=True)

    for _, row in slot_df.iterrows():
        h_id = str(row["h_id"])
        dt = pd.to_datetime(row["DateTime"])
        hour = int(dt.hour)

        import_energy = float(row["import_energy"])
        export_energy = float(row["export_energy"])

        fit_price = fit_profile.get_price(hour)
        tou_price = tou_profile.get_price(hour)

        if import_energy > 0:
            side = "buy"
            quantity = import_energy
            limit_price = tou_price
        elif export_energy > 0:
            side = "sell"
            quantity = export_energy
            limit_price = fit_price
        else:
            continue

        trader_key = (h_id, side)

        if trader_key not in trader_registry:
            trader_registry[trader_key] = build_strategy(
                h_id=h_id,
                side=side,
                strategy_name=strategy_name,
            )

        strategy = trader_registry[trader_key]

        if hasattr(strategy, "set_slot_context"):
            strategy.set_slot_context(dt)

        submitted_price = float(
            strategy.generate_shout(
                fit_price=fit_price,
                tou_price=tou_price,
            )
        )

        order_counter += 1
        order = Order(
            order_id=f"O{order_counter}",
            h_id=h_id,
            trader_key=trader_key,
            DateTime=dt,
            hour=hour,
            side=side,
            quantity=quantity,
            remaining_quantity=quantity,
            limit_price=float(limit_price),
            submitted_price=float(submitted_price),
        )
        order_book.add_order(order)

    return order_book, trader_registry, order_counter


def run_one_slot(
    slot_df,
    tou_profile,
    fit_profile,
    trader_registry,
    order_id_start,
    market_name="cda",
    strategy_name="static",
    pricing_name="midpoint",
    pricing_kwargs=None,
    verbose=True,
):
    order_book, trader_registry, order_id_end = gen_orders_and_slot(
        slot_df=slot_df,
        tou_profile=tou_profile,
        fit_profile=fit_profile,
        trader_registry=trader_registry,
        order_id_start=order_id_start,
        strategy_name=strategy_name,
    )

    mechanism = build_market(
        market_name=market_name,
        order_book=order_book,
        trader_registry=trader_registry,
        verbose=verbose,
        pricing_name=pricing_name,
        pricing_kwargs=pricing_kwargs,
    )

    result = mechanism.run_market() if hasattr(mechanism, "run_market") else mechanism.run_cda()

    return result, trader_registry, order_id_end


def run_market_sessions(
    household_df,
    market_name="cda",
    strategy_name="static",
    pricing_name="midpoint",
    pricing_kwargs=None,
    verbose=True,
):
    validate_household_df(household_df)

    household_df = household_df.copy()
    household_df["DateTime"] = pd.to_datetime(household_df["DateTime"], errors="coerce")
    household_df = household_df.dropna(subset=["DateTime"]).copy()
    household_df = household_df.sort_values(["DateTime", "h_id"]).reset_index(drop=True)

    tariff_season = SIM_SEASON or HOUSEHOLD_SEASON

    tou_profile = load_tou_profile(
        target_year=TARIFF_TARGET_YEAR,
        season=tariff_season,
        agg=TARIFF_AGG,
    )
    fit_profile = load_fit_profile(
        target_year=TARIFF_TARGET_YEAR,
        season=tariff_season,
        agg=TARIFF_AGG,
    )

    trader_registry = {}
    order_id_counter = 0
    slot_results = []

    grouped = household_df.groupby("DateTime", sort=True)

    for dt, slot_df in grouped:
        result, trader_registry, order_id_counter = run_one_slot(
            slot_df=slot_df.reset_index(drop=True),
            tou_profile=tou_profile,
            fit_profile=fit_profile,
            trader_registry=trader_registry,
            order_id_start=order_id_counter,
            market_name=market_name,
            strategy_name=strategy_name,
            pricing_name=pricing_name,
            pricing_kwargs=pricing_kwargs,
            verbose=verbose,
        )
        slot_results.append({"DateTime": dt, "result": result})

    total_num_trades = sum(int(item["result"].get("num_trades", 0)) for item in slot_results)
    total_num_unmatched_orders = sum(
        int(item["result"].get("num_unmatched_orders", 0)) for item in slot_results
    )

    return {
    "market_name": market_name,
    "strategy_name": strategy_name,
    "pricing_name": pricing_name,
    "slot_results": slot_results,
    "total_num_trades": total_num_trades,
    "total_num_unmatched_orders": total_num_unmatched_orders,
}


def summarise_market_result(market_result: dict[str, Any]) -> dict[str, Any]:
    slot_results = market_result.get("slot_results", [])

    total_matched_volume = 0.0
    total_unmatched_buy = 0.0
    total_unmatched_sell = 0.0
    trade_prices = []
    clearing_prices = []

    for item in slot_results:
        result = item["result"]
        committed_trades = result.get("committed_trades", [])
        unmatched_orders = result.get("unmatched_orders", [])

        total_matched_volume += sum(float(t.quantity) for t in committed_trades)
        total_unmatched_buy += sum(
            float(o.remaining_quantity) for o in unmatched_orders if o.side == "buy"
        )
        total_unmatched_sell += sum(
            float(o.remaining_quantity) for o in unmatched_orders if o.side == "sell"
        )

        trade_prices.extend(float(t.trade_price) for t in committed_trades)

        cp = result.get("clearing_price")
        if cp is not None:
            clearing_prices.append(float(cp))

    mean_trade_price = float(sum(trade_prices) / len(trade_prices)) if trade_prices else None
    mean_clearing_price = (
        float(sum(clearing_prices) / len(clearing_prices)) if clearing_prices else None
    )

    return {
        "market_name": market_result.get("market_name"),
        "strategy_name": market_result.get("strategy_name"),
        "pricing_name": market_result.get("pricing_name"),
        "num_slots": len(slot_results),
        "total_num_trades": int(market_result.get("total_num_trades", 0)),
        "total_num_unmatched_orders": int(market_result.get("total_num_unmatched_orders", 0)),
        "total_matched_volume": float(total_matched_volume),
        "total_unmatched_buy": float(total_unmatched_buy),
        "total_unmatched_sell": float(total_unmatched_sell),
        "mean_trade_price": mean_trade_price,
        "mean_clearing_price": mean_clearing_price,
    }



def result_to_serialisable(obj: Any):
    # 1) dataclass -> 先转 dict，再继续递归
    if is_dataclass(obj):
        return result_to_serialisable(asdict(obj))

    # 2) dict -> 递归处理 value
    if isinstance(obj, dict):
        return {k: result_to_serialisable(v) for k, v in obj.items()}

    # 3) list / tuple -> 递归处理每个元素
    if isinstance(obj, list):
        return [result_to_serialisable(v) for v in obj]

    if isinstance(obj, tuple):
        return [result_to_serialisable(v) for v in obj]

    # 4) pandas 时间类型
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    # 5) 原生 datetime/date
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # 6) pandas 缺失值
    if pd.isna(obj):
        return None

    return obj


def load_default_households(season=None, days=None, selected_households=None):
    return get_household_df(
        season=season or HOUSEHOLD_SEASON,
        days=days,
        selected_households=selected_households,
    )