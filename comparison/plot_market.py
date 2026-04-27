from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "results" / "market_mechanism" / "market_mechanism_summary.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "market_mechanism" / "market_mechanism_comparison_report_style.png"

BAR_WIDTH = 0.68


def load_plot_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["market_name", "total_matched_volume", "total_num_trades"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if "total_unmatched" not in df.columns:
        extra = ["total_unmatched_buy", "total_unmatched_sell"]
        missing_extra = [c for c in extra if c not in df.columns]
        if missing_extra:
            raise ValueError(
                "CSV needs either 'total_unmatched' or both "
                f"'total_unmatched_buy' and 'total_unmatched_sell'. Missing: {missing_extra}"
            )
        df["total_unmatched"] = (
            pd.to_numeric(df["total_unmatched_buy"], errors="coerce").fillna(0)
            + pd.to_numeric(df["total_unmatched_sell"], errors="coerce").fillna(0)
        )

    plot_df = df[["market_name", "total_matched_volume", "total_num_trades", "total_unmatched"]].copy()

    plot_df["market_name"] = plot_df["market_name"].replace(
        {
            "cda": "CDA",
            "call": "CALL",
            "uniform_price": "Uniform-price",
        }
    )

    order = ["CDA", "CALL", "Uniform-price"]
    plot_df["market_name"] = pd.Categorical(plot_df["market_name"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("market_name").reset_index(drop=True)

    return plot_df


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E9E9E9", linewidth=0.8)
    ax.xaxis.grid(False)

    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("")


def add_value_labels(ax, bars, values, is_count=False):
    ymax = max(values) if len(values) > 0 else 0
    offset = ymax * 0.025 if ymax > 0 else 0.5

    for bar, value in zip(bars, values):
        label = f"{int(value)}" if is_count else f"{value:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
        )


def plot_metric(ax, labels, values, title, is_count=False):
    bars = ax.bar(labels, values, width=BAR_WIDTH)

    ax.set_title(title, fontsize=18, pad=14)

    ymax = max(values) if len(values) > 0 else 0
    top_margin = ymax * 0.14 if ymax > 0 else 1.0
    ax.set_ylim(0, ymax + top_margin)

    style_axis(ax)
    add_value_labels(ax, bars, values, is_count=is_count)


def main():
    df = load_plot_data(CSV_PATH)

    labels = df["market_name"].astype(str).tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    #fig.suptitle("Annual summary for market mechanism comparison", fontsize=24, y=0.98)

    plot_metric(
        axes[0],
        labels,
        df["total_matched_volume"].tolist(),
        "Annual matched volume (kWh)",
        is_count=False,
    )

    plot_metric(
        axes[1],
        labels,
        df["total_num_trades"].tolist(),
        "Annual number of trades",
        is_count=True,
    )

    plot_metric(
        axes[2],
        labels,
        df["total_unmatched"].tolist(),
        "Annual unmatched volume (kWh)",
        is_count=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()