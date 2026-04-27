import os
import pandas as pd

from tariff import load_tou_profile, load_fit_profile
from zip_strategy import ZIPStrategy
from static_strategy import StaticStrategy
from aa_strategy import AAStrategy
from order_book import Order, OrderBook
from cda import CDA_mechanism
from household import get_household_df


TARIFF_TARGET_YEAR = int(os.getenv("TARIFF_TARGET_YEAR", "2025"))
SIM_SEASON = os.getenv("SIM_SEASON", "").strip().lower() or None
TARIFF_AGG = os.getenv("TARIFF_AGG", "median").strip().lower()

HOUSEHOLD_SEASON = os.getenv("HOUSEHOLD_SEASON", SIM_SEASON or "summer").strip().lower()
HOUSEHOLD_DAYS = os.getenv("HOUSEHOLD_DAYS", "").strip()
HOUSEHOLD_IDS = os.getenv("HOUSEHOLD_IDS", "").strip()

STRATEGY_NAME = os.getenv("STRATEGY_NAME", "zip").strip().lower()


def parse_household_days(value: str):
    if value == "":
        return None
    return int(value)


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


def build_strategy(h_id, side, strategy_name):
    if strategy_name == "zip":
        return ZIPStrategy(h_id=h_id, side=side)

    if strategy_name == "static":
        return StaticStrategy(h_id=h_id, side=side)

    if strategy_name == "aa":
        return AAStrategy(h_id=h_id, side=side)

    raise ValueError(
        f"Unsupported strategy_name={strategy_name!r}. "
        f"Choose from: 'zip', 'static', 'aa'."
    )


def gen_orders_and_slot(
    slot_df,
    tou_profile,
    fit_profile,
    trader_registry,
    order_id_start=0,
    strategy_name="zip",
):
    order_book = OrderBook()
    order_counter = order_id_start

    slot_df = slot_df.sort_values(["DateTime", "h_id"]).reset_index(drop=True)

    for _, row in slot_df.iterrows():
        h_id = str(row["h_id"])
        DateTime = pd.to_datetime(row["DateTime"])
        hour = int(DateTime.hour)

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
            strategy.set_slot_context(DateTime)

        submitted_price = strategy.generate_shout(
            fit_price=fit_price,
            tou_price=tou_price,
        )

        order_counter += 1
        order_id = f"O{order_counter}"

        order = Order(
            order_id=order_id,
            h_id=h_id,
            trader_key=trader_key,
            DateTime=DateTime,
            hour=hour,
            side=side,
            quantity=quantity,
            remaining_quantity=quantity,
            limit_price=limit_price,
            submitted_price=submitted_price,
        )
        order_book.add_order(order)

    return order_book, trader_registry, order_counter


def print_household(household_df, max_rows=10):
    print("=== HOUSEHOLD INPUT PREVIEW ===")
    preview_cols = ["DateTime", "h_id", "import_energy", "export_energy"]
    print(household_df[preview_cols].head(max_rows))


def print_slot_summary(slot_df):
    num_buyers = int((slot_df["import_energy"] > 0).sum())
    num_sellers = int((slot_df["export_energy"] > 0).sum())

    print("\n=== SLOT SUMMARY ===")
    print(f"DateTime = {slot_df['DateTime'].iloc[0]}")
    print(f"num_buyers  = {num_buyers}")
    print(f"num_sellers = {num_sellers}")
    print(f"total_import_energy = {slot_df['import_energy'].sum():.3f} kWh")
    print(f"total_export_energy = {slot_df['export_energy'].sum():.3f} kWh")


def print_order_book(order_book):
    print("\n=== INITIAL ORDER BOOK ===")
    order_book.print_book()


def print_results(result):
    print("\n=== COMMITTED TRADES ===")
    if not result["committed_trades"]:
        print("[empty]")
    for trade in result["committed_trades"]:
        print(
            f"trade_id={trade.trade_id}, "
            f"buyer_h_id={trade.buyer_h_id}, seller_h_id={trade.seller_h_id}, "
            f"buyer_order_id={trade.buyer_order_id}, seller_order_id={trade.seller_order_id}, "
            f"DateTime={trade.DateTime}, hour={trade.hour}, "
            f"quantity={trade.quantity:.3f} kWh, "
            f"trade_price={trade.trade_price:.4f} GBP/kWh, "
            f"trade_value={trade.trade_value:.4f}, "
            f"trade_round={trade.trade_round}"
        )

    print("\n=== UNMATCHED ORDERS ===")
    if not result["unmatched_orders"]:
        print("[empty]")
    for order in result["unmatched_orders"]:
        print(
            f"unmatched_order_id={order.unmatched_order_id}, "
            f"order_id={order.order_id}, h_id={order.h_id}, "
            f"DateTime={order.DateTime}, hour={order.hour}, side={order.side}, "
            f"original_quantity={order.original_quantity:.3f} kwh, "
            f"remaining_quantity={order.remaining_quantity:.3f} kwh, "
            f"limit_price={order.limit_price:.4f} GBP/kWh, "
            f"submitted_price={order.submitted_price:.4f} GBP/kWh"
        )

    print("\n=== SUMMARY ===")
    print(f"num_trades = {result['num_trades']}")
    print(f"num_unmatched_orders = {result['num_unmatched_orders']}")


def run_one_slot(
    slot_df,
    tou_profile,
    fit_profile,
    trader_registry,
    order_id_start,
    strategy_name="zip",
    verbose=True,
):
    if verbose:
        print_slot_summary(slot_df)

    order_book, trader_registry, order_id_end = gen_orders_and_slot(
        slot_df=slot_df,
        tou_profile=tou_profile,
        fit_profile=fit_profile,
        trader_registry=trader_registry,
        order_id_start=order_id_start,
        strategy_name=strategy_name,
    )

    if verbose:
        print_order_book(order_book)

    mechanism = CDA_mechanism(
        order_book=order_book,
        trader_registry=trader_registry,
        max_trade_rounds=150,
        max_no_trade_rounds=50,
        verbose=verbose,
    )

    result = mechanism.run_cda()

    if verbose:
        print_results(result)

    return result, trader_registry, order_id_end


def run_market_sessions(household_df, strategy_name="zip", verbose=True):
    validate_household_df(household_df)

    household_df = household_df.copy()
    household_df["DateTime"] = pd.to_datetime(household_df["DateTime"], errors="coerce")
    household_df = household_df.dropna(subset=["DateTime"]).copy()
    household_df = household_df.sort_values(["DateTime", "h_id"]).reset_index(drop=True)

    if household_df.empty:
        raise ValueError("household_df is empty")

    if verbose:
        print("\n===== RUN CONFIG CHECK =====")
        print("STRATEGY_NAME      =", strategy_name)
        print("HOUSEHOLD_SEASON   =", HOUSEHOLD_SEASON)
        print("SIM_SEASON         =", SIM_SEASON)
        print("TARIFF_TARGET_YEAR =", TARIFF_TARGET_YEAR)

        print("\n===== HOUSEHOLD DATA CHECK =====")
        print("min DateTime:", household_df["DateTime"].min())
        print("max DateTime:", household_df["DateTime"].max())
        print("unique months:", sorted(household_df["DateTime"].dt.month.unique().tolist()))
        print("num households:", household_df["h_id"].nunique())
        print(household_df[["DateTime", "h_id", "import_energy", "export_energy"]].head(10))

        print_household(household_df)

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

    if verbose:
        print("\n===== TARIFF CHECK =====")
        print("ToU season:", tou_profile.season)
        print("FiT season:", fit_profile.season)
        print("ToU target_year:", tou_profile.target_year)
        print("FiT target_year:", fit_profile.target_year)
        print(tou_profile.to_dataframe().head())
        print(fit_profile.to_dataframe().head())

    trader_registry = {}
    order_id_counter = 0
    slot_results = []

    grouped = household_df.groupby("DateTime", sort=True)

    for dt, slot_df in grouped:
        if verbose:
            print("\n" + "#" * 90)
            print(f"RUNNING MARKET SESSION FOR SLOT: {dt}")
            print("#" * 90)

        result, trader_registry, order_id_counter = run_one_slot(
            slot_df=slot_df.reset_index(drop=True),
            tou_profile=tou_profile,
            fit_profile=fit_profile,
            trader_registry=trader_registry,
            order_id_start=order_id_counter,
            strategy_name=strategy_name,
            verbose=verbose,
        )

        slot_results.append({"DateTime": dt, "result": result})

    total_num_trades = sum(item["result"]["num_trades"] for item in slot_results)
    total_num_unmatched_orders = sum(
        item["result"]["num_unmatched_orders"] for item in slot_results
    )

    if verbose:
        print("\n" + "=" * 90)
        print("OVERALL SUMMARY ACROSS ALL SLOTS")
        print("=" * 90)
        print(f"strategy_name = {strategy_name}")
        print(f"num_slots = {len(slot_results)}")
        print(f"total_num_trades = {total_num_trades}")
        print(f"total_num_unmatched_orders = {total_num_unmatched_orders}")

    return {
        "strategy_name": strategy_name,
        "slot_results": slot_results,
        "total_num_trades": total_num_trades,
        "total_num_unmatched_orders": total_num_unmatched_orders,
        "trader_registry": trader_registry,
    }


def main(
    household_df=None,
    season=None,
    days=None,
    selected_households=None,
    strategy_name=None,
    verbose=True,
):
    use_strategy = (strategy_name or STRATEGY_NAME).strip().lower()

    if household_df is None:
        use_season = season or HOUSEHOLD_SEASON
        household_df = get_household_df(
            season=use_season,
            days=days,
            selected_households=selected_households,
        )

    return run_market_sessions(
        household_df=household_df,
        strategy_name=use_strategy,
        verbose=verbose,
    )


if __name__ == "__main__":
    household_df = get_household_df(
        season=HOUSEHOLD_SEASON,
        days=parse_household_days(HOUSEHOLD_DAYS),
        selected_households=parse_selected_households(HOUSEHOLD_IDS),
    )
    main(
        household_df=household_df,
        strategy_name=STRATEGY_NAME,
        verbose=False,
    )