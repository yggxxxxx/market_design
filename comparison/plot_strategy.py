from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "results" / "bidding_strategy" / "annual_strategy_summary.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "bidding_strategy" / "annual_summary_bar.png"

STRATEGY_ORDER = ["static", "zip", "aa"]
STRATEGY_LABELS = {
    "static": "Static",
    "zip": "ZIP",
    "aa": "AA",
}

BAR_WIDTH = 0.68


def load_plot_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "strategy",
        "annual_total_p2p_matched_volume",
        "annual_total_num_trades",
        "annual_total_community_saving",
        "annual_total_strategy_external_net_cost",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.copy()
    df["strategy"] = pd.Categorical(df["strategy"], categories=STRATEGY_ORDER, ordered=True)
    df = df.sort_values("strategy").reset_index(drop=True)
    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS)

    return df


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E9E9E9", linewidth=0.8)
    ax.xaxis.grid(False)

    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("")


def add_value_labels(ax, bars, values, decimals=2):
    ymax = max(values) if len(values) > 0 else 0
    offset = ymax * 0.02 if ymax > 0 else 0.5

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )


def plot_metric(ax, labels, values, title, decimals=2):
    bars = ax.bar(labels, values, width=BAR_WIDTH)

    ax.set_title(title, fontsize=18, pad=14)

    ymax = max(values) if len(values) > 0 else 0
    top_margin = ymax * 0.14 if ymax > 0 else 1.0
    ax.set_ylim(0, ymax + top_margin)

    style_axis(ax)
    add_value_labels(ax, bars, values, decimals=decimals)


def main():
    df = load_plot_data(CSV_PATH)
    labels = df["strategy_label"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    axes = axes.flatten()

    #fig.suptitle("Annual summary for bidding strategy comparison", fontsize=24, y=0.98)

    plot_metric(
        axes[0],
        labels,
        df["annual_total_p2p_matched_volume"].tolist(),
        "Annual matched volume (kWh)",
        decimals=2,
    )

    plot_metric(
        axes[1],
        labels,
        df["annual_total_num_trades"].tolist(),
        "Annual number of trades",
        decimals=0,
    )

    plot_metric(
        axes[2],
        labels,
        df["annual_total_community_saving"].tolist(),
        "Annual saving vs grid-only (£)",
        decimals=2,
    )

    plot_metric(
        axes[3],
        labels,
        df["annual_total_strategy_external_net_cost"].tolist(),
        "Annual external net cost (£)",
        decimals=2,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()