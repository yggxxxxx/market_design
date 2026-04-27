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


RESULT_DIR = PROJECT_ROOT / "results" / "pricing_mechanism"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def run_one_configuration(
    household_df,
    pricing_name: str,
    label: str,
    pricing_kwargs=None,
    verbose: bool = False,
):
    market_result = run_market_sessions(
        household_df=household_df,
        market_name="cda",
        strategy_name="static",
        pricing_name=pricing_name,
        pricing_kwargs=pricing_kwargs,
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

    if pricing_kwargs:
        summary.update({f"pricing_{k}": v for k, v in pricing_kwargs.items()})

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
        {"pricing_name": "midpoint", "pricing_kwargs": None, "label": "CDA+Static+Midpoint"},
        {"pricing_name": "pay_as_bid", "pricing_kwargs": None, "label": "CDA+Static+PayAsBid"},
        {"pricing_name": "pay_as_ask", "pricing_kwargs": None, "label": "CDA+Static+PayAsAsk"},
        {"pricing_name": "k_factor", "pricing_kwargs": {"k": 0.25}, "label": "CDA+Static+KFactor025"},
        {"pricing_name": "k_factor", "pricing_kwargs": {"k": 0.75}, "label": "CDA+Static+KFactor075"},
    ]

    summaries = []

    for idx, cfg in enumerate(configs, start=1):
        market_result, compare_df, summary = run_one_configuration(
            household_df=household_df,
            pricing_name=cfg["pricing_name"],
            label=cfg["label"],
            pricing_kwargs=cfg["pricing_kwargs"],
            verbose=verbose,
        )

        summaries.append(summary)

        compare_df.to_csv(RESULT_DIR / f"pricing_{idx}_compare.csv", index=False)

        with open(RESULT_DIR / f"pricing_{idx}_market_result.json", "w", encoding="utf-8") as f:
            json.dump(result_to_serialisable(market_result), f, ensure_ascii=False, indent=2)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULT_DIR / "pricing_mechanism_summary.csv", index=False)

    show_cols = [
        "pricing_name",
        "pricing_k",
        "total_community_saving",
        "avg_slot_saving",
        "total_strategy_external_net_cost",
        "total_buyer_savings",
        "total_seller_gains",
        "total_social_benefit",
        "total_p2p_matched_volume",
        "total_num_trades",
    ]

    print(summary_df[show_cols].sort_values("total_community_saving", ascending=False))
    return summary_df


if __name__ == "__main__":
    main()