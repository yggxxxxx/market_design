from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for p in [
    PROJECT_ROOT,
    PROJECT_ROOT / "core",
    PROJECT_ROOT / "baseline",
    PROJECT_ROOT / "markets",
    PROJECT_ROOT / "strategies",
    PROJECT_ROOT / "pricing",
]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from household import get_household_df
from main import main as run_market_main
from grid_only import compare_strategy_to_grid_only


SEASON_ORDER = ["spring", "summer", "autumn", "winter"]
STRATEGY_ORDER = ["static", "zip", "aa"]

TARIFF_TARGET_YEAR = int(os.getenv("TARIFF_TARGET_YEAR", "2025"))
TARIFF_AGG = os.getenv("TARIFF_AGG", "median").strip().lower()
HOUSEHOLD_DAYS = os.getenv("HOUSEHOLD_DAYS", "").strip()
HOUSEHOLD_IDS = os.getenv("HOUSEHOLD_IDS", "").strip()

OUTPUT_DIR = Path(os.getenv("COMPARISON_OUTPUT_DIR", "results/bidding_strategy"))


def parse_household_days(value: str):
    if value == "":
        return None
    return int(value)


def parse_selected_households(value: str):
    if not value:
        return None
    ids = [x.strip() for x in value.split(",") if x.strip()]
    return ids or None


def ensure_output_dir(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def extract_trade_records(market_result: dict, season: str, strategy: str):
    rows = []
    for item in market_result.get("slot_results", []):
        dt = pd.to_datetime(item["DateTime"])
        result = item["result"]
        for trade in result.get("committed_trades", []):
            rows.append(
                {
                    "season": season,
                    "strategy": strategy,
                    "DateTime": dt,
                    "date": dt.date(),
                    "hour": int(dt.hour),
                    "buyer_h_id": trade.buyer_h_id,
                    "seller_h_id": trade.seller_h_id,
                    "quantity": float(trade.quantity),
                    "trade_price": float(trade.trade_price),
                    "trade_value": float(trade.trade_value),
                    "trade_round": int(trade.trade_round),
                }
            )
    return pd.DataFrame(rows)


def weighted_average(values: pd.Series, weights: pd.Series):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def aggregate_summary(compare_df: pd.DataFrame, trades_df: pd.DataFrame, season: str, strategy: str):
    num_days = int(compare_df["date"].nunique()) if not compare_df.empty else 0
    total_feasible = float(compare_df["feasible_p2p_volume"].sum()) if not compare_df.empty else 0.0
    total_matched = float(compare_df["matched_volume"].sum()) if not compare_df.empty else 0.0

    if trades_df.empty:
        avg_trade_price = np.nan
        median_trade_price = np.nan
        weighted_trade_price = np.nan
    else:
        avg_trade_price = float(trades_df["trade_price"].mean())
        median_trade_price = float(trades_df["trade_price"].median())
        weighted_trade_price = weighted_average(trades_df["trade_price"], trades_df["quantity"])

    return {
        "season": season,
        "strategy": strategy,
        "num_days": num_days,
        "total_grid_only_net_cost": float(compare_df["grid_only_net_cost"].sum()),
        "total_strategy_external_net_cost": float(compare_df["strategy_external_net_cost"].sum()),
        "total_community_saving": float(compare_df["community_saving"].sum()),
        "total_buyer_savings": float(compare_df["buyer_savings"].sum()),
        "total_seller_gains": float(compare_df["seller_gains"].sum()),
        "total_social_benefit": float(compare_df["total_social_benefit"].sum()),
        "max_abs_saving_gap_check": float(compare_df["saving_gap_check"].abs().max()),
        "total_grid_only_import": float(compare_df["total_import"].sum()),
        "total_grid_only_export": float(compare_df["total_export"].sum()),
        "total_strategy_external_import": float(compare_df["unmatched_buy"].sum()),
        "total_strategy_external_export": float(compare_df["unmatched_sell"].sum()),
        "total_external_import_reduction": float(compare_df["external_import_reduction"].sum()),
        "total_external_export_reduction": float(compare_df["external_export_reduction"].sum()),
        "total_p2p_matched_volume": total_matched,
        "total_num_trades": int(compare_df["num_trades"].sum()),
        "total_num_unmatched_orders": int(compare_df["num_unmatched_orders"].sum()),
        "total_feasible_p2p_volume": total_feasible,
        "overall_match_rate": float(total_matched / total_feasible) if total_feasible > 0 else np.nan,
        "avg_slot_saving": float(compare_df["community_saving"].mean()),
        "avg_daily_matched_volume": float(total_matched / num_days) if num_days > 0 else np.nan,
        "avg_daily_num_trades": float(compare_df["num_trades"].sum() / num_days) if num_days > 0 else np.nan,
        "avg_daily_saving": float(compare_df["community_saving"].sum() / num_days) if num_days > 0 else np.nan,
        "avg_trade_price": avg_trade_price,
        "median_trade_price": median_trade_price,
        "weighted_avg_trade_price": weighted_trade_price,
    }


def build_annual_summary(season_summary_df: pd.DataFrame):
    annual = (
        season_summary_df.groupby("strategy", as_index=False)
        .agg(
            annual_num_days=("num_days", "sum"),
            annual_total_grid_only_net_cost=("total_grid_only_net_cost", "sum"),
            annual_total_strategy_external_net_cost=("total_strategy_external_net_cost", "sum"),
            annual_total_community_saving=("total_community_saving", "sum"),
            annual_total_buyer_savings=("total_buyer_savings", "sum"),
            annual_total_seller_gains=("total_seller_gains", "sum"),
            annual_total_social_benefit=("total_social_benefit", "sum"),
            annual_total_grid_only_import=("total_grid_only_import", "sum"),
            annual_total_grid_only_export=("total_grid_only_export", "sum"),
            annual_total_strategy_external_import=("total_strategy_external_import", "sum"),
            annual_total_strategy_external_export=("total_strategy_external_export", "sum"),
            annual_total_external_import_reduction=("total_external_import_reduction", "sum"),
            annual_total_external_export_reduction=("total_external_export_reduction", "sum"),
            annual_total_p2p_matched_volume=("total_p2p_matched_volume", "sum"),
            annual_total_num_trades=("total_num_trades", "sum"),
            annual_total_num_unmatched_orders=("total_num_unmatched_orders", "sum"),
            annual_total_feasible_p2p_volume=("total_feasible_p2p_volume", "sum"),
        )
        .sort_values("strategy")
        .reset_index(drop=True)
    )

    annual["annual_overall_match_rate"] = np.where(
        annual["annual_total_feasible_p2p_volume"] > 0,
        annual["annual_total_p2p_matched_volume"] / annual["annual_total_feasible_p2p_volume"],
        np.nan,
    )
    annual["annual_avg_daily_matched_volume"] = np.where(
        annual["annual_num_days"] > 0,
        annual["annual_total_p2p_matched_volume"] / annual["annual_num_days"],
        np.nan,
    )
    annual["annual_avg_daily_saving"] = np.where(
        annual["annual_num_days"] > 0,
        annual["annual_total_community_saving"] / annual["annual_num_days"],
        np.nan,
    )
    annual["annual_avg_daily_num_trades"] = np.where(
        annual["annual_num_days"] > 0,
        annual["annual_total_num_trades"] / annual["annual_num_days"],
        np.nan,
    )
    return annual


def build_relative_improvement_table(season_summary_df: pd.DataFrame):
    rows = []
    metrics = [
        "total_community_saving",
        "total_p2p_matched_volume",
        "total_num_trades",
        "overall_match_rate",
        "total_strategy_external_net_cost",
    ]

    for season in SEASON_ORDER:
        sub = season_summary_df[season_summary_df["season"] == season].set_index("strategy")
        if sub.empty:
            continue

        row = {"season": season}
        pairs = [
            ("zip", "static", "zip_vs_static"),
            ("aa", "static", "aa_vs_static"),
            ("aa", "zip", "aa_vs_zip"),
        ]

        for metric in metrics:
            for a, b, label in pairs:
                if a not in sub.index or b not in sub.index:
                    row[f"{metric}_{label}_pct"] = np.nan
                    continue
                base = float(sub.loc[b, metric])
                new = float(sub.loc[a, metric])
                if abs(base) < 1e-12:
                    row[f"{metric}_{label}_pct"] = np.nan
                else:
                    row[f"{metric}_{label}_pct"] = 100.0 * (new - base) / abs(base)
        rows.append(row)

    return pd.DataFrame(rows)


def run_one_experiment(season: str, strategy: str, days=None, selected_households=None, verbose=False):
    household_df = get_household_df(
        season=season,
        days=days,
        selected_households=selected_households,
    )

    market_result = run_market_main(
        household_df=household_df,
        strategy_name=strategy,
        verbose=verbose,
    )

    compare_df = compare_strategy_to_grid_only(
        household_df=household_df,
        market_result=market_result,
        target_year=TARIFF_TARGET_YEAR,
        season=season,
        agg=TARIFF_AGG,
    )
    compare_df["season"] = season
    compare_df["strategy"] = strategy
    compare_df["feasible_p2p_volume"] = compare_df[["total_import", "total_export"]].min(axis=1)
    compare_df["slot_match_rate"] = np.where(
        compare_df["feasible_p2p_volume"] > 0,
        compare_df["matched_volume"] / compare_df["feasible_p2p_volume"],
        np.nan,
    )

    trades_df = extract_trade_records(market_result, season=season, strategy=strategy)
    summary = aggregate_summary(compare_df, trades_df, season, strategy)

    return compare_df, trades_df, summary


def main():
    days = parse_household_days(HOUSEHOLD_DAYS)
    selected_households = parse_selected_households(HOUSEHOLD_IDS)
    output_dir = ensure_output_dir(OUTPUT_DIR)

    all_compare = []
    all_trades = []
    summary_rows = []

    for season in SEASON_ORDER:
        for strategy in STRATEGY_ORDER:
            print(f"Running season={season}, strategy={strategy} ...")
            compare_df, trades_df, summary = run_one_experiment(
                season=season,
                strategy=strategy,
                days=days,
                selected_households=selected_households,
                verbose=False,
            )
            all_compare.append(compare_df)
            all_trades.append(trades_df)
            summary_rows.append(summary)

    all_compare_df = pd.concat(all_compare, ignore_index=True) if all_compare else pd.DataFrame()
    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    season_summary_df = pd.DataFrame(summary_rows)

    if not season_summary_df.empty:
        season_summary_df["season"] = pd.Categorical(
            season_summary_df["season"], categories=SEASON_ORDER, ordered=True
        )
        season_summary_df["strategy"] = pd.Categorical(
            season_summary_df["strategy"], categories=STRATEGY_ORDER, ordered=True
        )
        season_summary_df = season_summary_df.sort_values(["season", "strategy"]).reset_index(drop=True)

    annual_summary_df = build_annual_summary(season_summary_df)
    relative_improvement_df = build_relative_improvement_table(season_summary_df)

    all_compare_df.to_csv(output_dir / "all_compare_results.csv", index=False)
    all_trades_df.to_csv(output_dir / "all_trade_results.csv", index=False)
    season_summary_df.to_csv(output_dir / "season_strategy_summary.csv", index=False)
    annual_summary_df.to_csv(output_dir / "annual_strategy_summary.csv", index=False)
    relative_improvement_df.to_csv(output_dir / "relative_improvement_summary.csv", index=False)

    print("\\n=== BIDDING STRATEGY COMPARISON COMPLETE ===")
    print(f"Results saved to: {output_dir}")
    print("- season_strategy_summary.csv")
    print("- annual_strategy_summary.csv")
    print("- relative_improvement_summary.csv")
    print("- all_compare_results.csv")
    print("- all_trade_results.csv")


if __name__ == "__main__":
    main()