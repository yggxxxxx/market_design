from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for p in [
    PROJECT_ROOT,
    PROJECT_ROOT / "core",
    PROJECT_ROOT / "baseline",
    PROJECT_ROOT / "strategies",
    PROJECT_ROOT / "markets",
    PROJECT_ROOT / "pricing",
]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from grid_only import aggregate_grid_only_result, compare_strategy_to_grid_only
from household import get_household_df
from market_runner import result_to_serialisable, run_market_sessions, summarise_market_result


RESULT_DIR = PROJECT_ROOT / "results" / "market_mechanism"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def run_one_configuration(
    household_df,
    market_name: str,
    pricing_name: str,
    label: str,
    verbose: bool = False,
):
    market_result = run_market_sessions(
        household_df=household_df,
        market_name=market_name,
        strategy_name="static",
        pricing_name=pricing_name,
        pricing_kwargs=None,
        verbose=verbose,
    )

    compare_df = compare_strategy_to_grid_only(
        household_df=household_df,
        market_result=market_result,
    )

    aggregate = aggregate_grid_only_result(
        compare_df,
        season=None,
        strategy_name=label,
    )

    summary = {
        **summarise_market_result(market_result),
        **aggregate,
    }

    return market_result, compare_df, summary


def main(
    season: str = "all",
    days: int | None = None,
    selected_households: list[str] | None = None,
    verbose: bool = False,
):
    household_df = get_household_df(
        season=season,
        days=days,
        selected_households=selected_households,
    )

    configs = [
        {"market_name": "cda", "pricing_name": "midpoint", "label": "CDA+Midpoint+Static"},
        {"market_name": "call", "pricing_name": "call_price", "label": "CALL+CallPrice+Static"},
        {"market_name": "uniform_price", "pricing_name": "uniform_price", "label": "UniformPrice+Static"},
    ]

    summaries = []

    for cfg in configs:
        market_result, compare_df, summary = run_one_configuration(
            household_df=household_df,
            market_name=cfg["market_name"],
            pricing_name=cfg["pricing_name"],
            label=cfg["label"],
            verbose=verbose,
        )

        summaries.append(summary)

        compare_df.to_csv(RESULT_DIR / f"{cfg['market_name']}_compare.csv", index=False)

        with open(RESULT_DIR / f"{cfg['market_name']}_market_result.json", "w", encoding="utf-8") as f:
            json.dump(result_to_serialisable(market_result), f, ensure_ascii=False, indent=2)

    summary_df = pd.DataFrame(summaries)

    show_cols = [
        "market_name",
        "strategy_name",
        "pricing_name",
        "total_community_saving",
        "avg_slot_saving",
        "total_p2p_matched_volume",
        "matched_volume_ratio",
        "total_num_trades",
        "avg_trade_size",
        "total_buyer_savings",
        "total_seller_gains",
        "total_strategy_external_import",
        "total_strategy_external_export",
    ]

    print(summary_df[show_cols].sort_values("total_community_saving", ascending=False))
    summary_df.to_csv(RESULT_DIR / "market_mechanism_summary.csv", index=False)

if __name__ == "__main__":
    main()