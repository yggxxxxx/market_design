import os
import pandas as pd

from tariff import load_tou_profile, load_fit_profile


TARIFF_TARGET_YEAR = int(os.getenv("TARIFF_TARGET_YEAR", "2025"))
SIM_SEASON = os.getenv("SIM_SEASON", "").strip().lower() or None
TARIFF_AGG = os.getenv("TARIFF_AGG", "median").strip().lower()
HOUSEHOLD_SEASON = os.getenv("HOUSEHOLD_SEASON", SIM_SEASON or "summer").strip().lower()


def _effective_target_year(target_year):
    return TARIFF_TARGET_YEAR if target_year is None else int(target_year)


def _effective_season(season):
    if season is not None:
        season = str(season).strip().lower()
        return season or None
    return SIM_SEASON or HOUSEHOLD_SEASON


def _effective_agg(agg):
    if agg is None:
        return TARIFF_AGG
    return str(agg).strip().lower()


def _validate_household_df(household_df: pd.DataFrame):
    required_cols = {"DateTime", "import_energy", "export_energy"}
    missing = required_cols - set(household_df.columns)
    if missing:
        raise ValueError(f"household_df missing required columns: {sorted(missing)}")
    if household_df is None or household_df.empty:
        raise ValueError("household_df is empty")


def _build_tariff_profiles(
    target_year=None,
    season=None,
    agg=None,
    tou_csv_path=None,
    fit_csv_path=None,
):
    use_target_year = _effective_target_year(target_year)
    use_season = _effective_season(season)
    use_agg = _effective_agg(agg)

    tou_profile = load_tou_profile(
        csv_path=tou_csv_path,
        target_year=use_target_year,
        season=use_season,
        agg=use_agg,
    )
    fit_profile = load_fit_profile(
        csv_path=fit_csv_path,
        target_year=use_target_year,
        season=use_season,
        agg=use_agg,
    )
    return tou_profile, fit_profile


def build_grid_only_slot_df(
    household_df,
    target_year=None,
    season=None,
    agg=None,
    tou_csv_path=None,
    fit_csv_path=None,
):
    """
    Grid-only baseline:
    all import_energy is bought from the grid at ToU price,
    all export_energy is sold to the grid at FiT price.
    """
    _validate_household_df(household_df)

    tou_profile, fit_profile = _build_tariff_profiles(
        target_year=target_year,
        season=season,
        agg=agg,
        tou_csv_path=tou_csv_path,
        fit_csv_path=fit_csv_path,
    )

    df = household_df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df["import_energy"] = pd.to_numeric(df["import_energy"], errors="coerce")
    df["export_energy"] = pd.to_numeric(df["export_energy"], errors="coerce")
    df = df.dropna(subset=["DateTime", "import_energy", "export_energy"]).copy()

    slot_df = (
        df.groupby("DateTime", as_index=False)
        .agg(
            total_import=("import_energy", "sum"),
            total_export=("export_energy", "sum"),
        )
        .sort_values("DateTime")
        .reset_index(drop=True)
    )

    slot_df["hour"] = slot_df["DateTime"].dt.hour.astype(int)
    slot_df["tou_price"] = slot_df["hour"].apply(tou_profile.get_price)
    slot_df["fit_price"] = slot_df["hour"].apply(fit_profile.get_price)

    slot_df["grid_only_import_cost"] = slot_df["total_import"] * slot_df["tou_price"]
    slot_df["grid_only_export_revenue"] = slot_df["total_export"] * slot_df["fit_price"]
    slot_df["grid_only_net_cost"] = (
        slot_df["grid_only_import_cost"] - slot_df["grid_only_export_revenue"]
    )

    slot_df["date"] = slot_df["DateTime"].dt.date

    return slot_df


def build_strategy_slot_df(
    market_result,
    target_year=None,
    season=None,
    agg=None,
    tou_csv_path=None,
    fit_csv_path=None,
):
    """
    Strategy-side slot summary from market_result.
    P2P matched volume is internal.
    Only unmatched buy/sell goes to the external grid.
    """
    if market_result is None or "slot_results" not in market_result:
        raise ValueError("market_result must contain 'slot_results'")

    tou_profile, fit_profile = _build_tariff_profiles(
        target_year=target_year,
        season=season,
        agg=agg,
        tou_csv_path=tou_csv_path,
        fit_csv_path=fit_csv_path,
    )

    rows = []

    for item in market_result["slot_results"]:
        dt = pd.to_datetime(item["DateTime"])
        result = item["result"]

        committed_trades = result.get("committed_trades", [])
        unmatched_orders = result.get("unmatched_orders", [])

        hour = int(dt.hour)
        tou_price = tou_profile.get_price(hour)
        fit_price = fit_profile.get_price(hour)

        matched_volume = float(sum(float(t.quantity) for t in committed_trades))

        unmatched_buy = float(
            sum(float(o.remaining_quantity) for o in unmatched_orders if o.side == "buy")
        )
        unmatched_sell = float(
            sum(float(o.remaining_quantity) for o in unmatched_orders if o.side == "sell")
        )

        strategy_external_import_cost = unmatched_buy * tou_price
        strategy_external_export_revenue = unmatched_sell * fit_price
        strategy_external_net_cost = (
            strategy_external_import_cost - strategy_external_export_revenue
        )

        buyer_savings = float(
            sum(float(t.quantity) * (tou_price - float(t.trade_price)) for t in committed_trades)
        )
        seller_gains = float(
            sum(float(t.quantity) * (float(t.trade_price) - fit_price) for t in committed_trades)
        )
        total_social_benefit = buyer_savings + seller_gains

        rows.append(
            {
                "DateTime": dt,
                "matched_volume": matched_volume,
                "unmatched_buy": unmatched_buy,
                "unmatched_sell": unmatched_sell,
                "strategy_external_import_cost": strategy_external_import_cost,
                "strategy_external_export_revenue": strategy_external_export_revenue,
                "strategy_external_net_cost": strategy_external_net_cost,
                "buyer_savings": buyer_savings,
                "seller_gains": seller_gains,
                "total_social_benefit": total_social_benefit,
                "num_trades": int(result.get("num_trades", len(committed_trades))),
                "num_unmatched_orders": int(
                    result.get("num_unmatched_orders", len(unmatched_orders))
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "DateTime",
                "matched_volume",
                "unmatched_buy",
                "unmatched_sell",
                "strategy_external_import_cost",
                "strategy_external_export_revenue",
                "strategy_external_net_cost",
                "buyer_savings",
                "seller_gains",
                "total_social_benefit",
                "num_trades",
                "num_unmatched_orders",
            ]
        )

    strategy_slot_df = pd.DataFrame(rows).sort_values("DateTime").reset_index(drop=True)
    return strategy_slot_df


def compare_strategy_to_grid_only(
    household_df,
    market_result,
    target_year=None,
    season=None,
    agg=None,
    tou_csv_path=None,
    fit_csv_path=None,
):
    """
    Compare ZIP-CDA strategy with grid-only baseline at community level.

    Robust to missing strategy slots:
    if a DateTime exists in grid_df but is absent in strategy_df,
    it is treated as:
      matched_volume = 0
      unmatched_buy = total_import
      unmatched_sell = total_export
    """
    grid_df = build_grid_only_slot_df(
        household_df=household_df,
        target_year=target_year,
        season=season,
        agg=agg,
        tou_csv_path=tou_csv_path,
        fit_csv_path=fit_csv_path,
    )

    strategy_df = build_strategy_slot_df(
        market_result=market_result,
        target_year=target_year,
        season=season,
        agg=agg,
        tou_csv_path=tou_csv_path,
        fit_csv_path=fit_csv_path,
    )

    compare_df = grid_df.merge(strategy_df, on="DateTime", how="left")

    compare_df["matched_volume"] = compare_df["matched_volume"].fillna(0.0)
    compare_df["buyer_savings"] = compare_df["buyer_savings"].fillna(0.0)
    compare_df["seller_gains"] = compare_df["seller_gains"].fillna(0.0)
    compare_df["total_social_benefit"] = compare_df["total_social_benefit"].fillna(0.0)
    compare_df["num_trades"] = compare_df["num_trades"].fillna(0).astype(int)
    compare_df["num_unmatched_orders"] = compare_df["num_unmatched_orders"].fillna(0).astype(int)

    compare_df["unmatched_buy"] = compare_df["unmatched_buy"].where(
        compare_df["unmatched_buy"].notna(),
        compare_df["total_import"],
    )
    compare_df["unmatched_sell"] = compare_df["unmatched_sell"].where(
        compare_df["unmatched_sell"].notna(),
        compare_df["total_export"],
    )

    compare_df["strategy_external_import_cost"] = compare_df[
        "strategy_external_import_cost"
    ].where(
        compare_df["strategy_external_import_cost"].notna(),
        compare_df["unmatched_buy"] * compare_df["tou_price"],
    )

    compare_df["strategy_external_export_revenue"] = compare_df[
        "strategy_external_export_revenue"
    ].where(
        compare_df["strategy_external_export_revenue"].notna(),
        compare_df["unmatched_sell"] * compare_df["fit_price"],
    )

    compare_df["strategy_external_net_cost"] = compare_df[
        "strategy_external_net_cost"
    ].where(
        compare_df["strategy_external_net_cost"].notna(),
        compare_df["strategy_external_import_cost"]
        - compare_df["strategy_external_export_revenue"],
    )

    compare_df["community_saving"] = (
        compare_df["grid_only_net_cost"] - compare_df["strategy_external_net_cost"]
    )

    compare_df["saving_gap_check"] = (
        compare_df["community_saving"] - compare_df["total_social_benefit"]
    )

    compare_df["external_import_reduction"] = (
        compare_df["total_import"] - compare_df["unmatched_buy"]
    )
    compare_df["external_export_reduction"] = (
        compare_df["total_export"] - compare_df["unmatched_sell"]
    )

    compare_df["hour"] = compare_df["DateTime"].dt.hour
    compare_df["date"] = compare_df["DateTime"].dt.date

    return compare_df.sort_values("DateTime").reset_index(drop=True)


def aggregate_grid_only_result(compare_df, season=None, strategy_name="ZIP-CDA"):
    if compare_df is None or compare_df.empty:
        raise ValueError("compare_df is empty")

    total_grid_only_import = float(compare_df["total_import"].sum())
    total_grid_only_export = float(compare_df["total_export"].sum())
    total_strategy_external_import = float(compare_df["unmatched_buy"].sum())
    total_strategy_external_export = float(compare_df["unmatched_sell"].sum())
    total_external_import_reduction = float(compare_df["external_import_reduction"].sum())
    total_external_export_reduction = float(compare_df["external_export_reduction"].sum())
    total_p2p_matched_volume = float(compare_df["matched_volume"].sum())
    total_num_trades = int(compare_df["num_trades"].sum())

    return {
        "season": season,
        "strategy": strategy_name,

        # economics
        "total_grid_only_net_cost": float(compare_df["grid_only_net_cost"].sum()),
        "total_strategy_external_net_cost": float(compare_df["strategy_external_net_cost"].sum()),
        "total_community_saving": float(compare_df["community_saving"].sum()),
        "avg_slot_saving": float(compare_df["community_saving"].mean()),

        # benefit split
        "total_buyer_savings": float(compare_df["buyer_savings"].sum()),
        "total_seller_gains": float(compare_df["seller_gains"].sum()),
        "total_social_benefit": float(compare_df["total_social_benefit"].sum()),
        "max_abs_saving_gap_check": float(compare_df["saving_gap_check"].abs().max()),

        # grid reliance
        "total_grid_only_import": total_grid_only_import,
        "total_grid_only_export": total_grid_only_export,
        "total_strategy_external_import": total_strategy_external_import,
        "total_strategy_external_export": total_strategy_external_export,
        "total_external_import_reduction": total_external_import_reduction,
        "total_external_export_reduction": total_external_export_reduction,

        # market activity
        "total_p2p_matched_volume": total_p2p_matched_volume,
        "total_num_trades": total_num_trades,

        # normalized indicators
        "matched_volume_ratio": (
            total_p2p_matched_volume / min(total_grid_only_import, total_grid_only_export)
            if min(total_grid_only_import, total_grid_only_export) > 1e-12 else 0.0
        ),
        "external_import_reduction_ratio": (
            total_external_import_reduction / total_grid_only_import
            if total_grid_only_import > 1e-12 else 0.0
        ),
        "external_export_reduction_ratio": (
            total_external_export_reduction / total_grid_only_export
            if total_grid_only_export > 1e-12 else 0.0
        ),
        "avg_trade_size": (
            total_p2p_matched_volume / total_num_trades
            if total_num_trades > 0 else 0.0
        ),
    }


def print_compare_summary(summary: dict):
    print("\n=== GRID-ONLY COMPARISON SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")