from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "results" / "pricing_mechanism" / "pricing_mechanism_summary.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "pricing_mechanism" / "pricing_mechanism_report_style.png"

BAR_WIDTH = 0.68


def build_pricing_label(row: pd.Series) -> str:
    pricing_name = str(row.get("pricing_name", "")).strip().lower()
    k = row.get("pricing_k", None)

    if pricing_name == "midpoint":
        return "Midpoint"
    if pricing_name in {"pay_as_bid", "pay-as-bid"}:
        return "Pay-as-bid"
    if pricing_name in {"pay_as_ask", "pay-as-ask"}:
        return "Pay-as-ask"
    if pricing_name in {"k_factor", "k-factor", "kfactor"}:
        if pd.notna(k):
            return f"k={float(k):.2f}"
        return "k-factor"

    return pricing_name or "Unknown"


def load_plot_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "pricing_name",
        "total_buyer_savings",
        "total_seller_gains",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.copy()
    df["pricing_label"] = df.apply(build_pricing_label, axis=1)

    preferred_order = ["Midpoint", "Pay-as-bid", "Pay-as-ask", "k=0.25", "k=0.75"]
    label_to_rank = {label: idx for idx, label in enumerate(preferred_order)}
    df["_sort_rank"] = df["pricing_label"].map(label_to_rank).fillna(999)
    df = df.sort_values(["_sort_rank", "pricing_label"]).reset_index(drop=True)

    return df


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E9E9E9", linewidth=0.8)
    ax.xaxis.grid(False)

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("")


def add_value_labels(ax, bars, values):
    ymax = max(values) if len(values) > 0 else 0
    offset = ymax * 0.02 if ymax > 0 else 0.5

    for bar, value in zip(bars, values):
        label = f"{value:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
        )


def plot_metric(ax, labels, values, title):
    bars = ax.bar(labels, values, width=BAR_WIDTH)

    ax.set_title(title, fontsize=18, pad=14)

    ymax = max(values) if len(values) > 0 else 0
    top_margin = ymax * 0.14 if ymax > 0 else 1.0
    ax.set_ylim(0, ymax + top_margin)

    style_axis(ax)
    add_value_labels(ax, bars, values)


def main():
    df = load_plot_data(CSV_PATH)
    labels = df["pricing_label"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    #fig.suptitle("Annual summary for pricing mechanism comparison", fontsize=24, y=0.98)

    plot_metric(
        axes[0],
        labels,
        pd.to_numeric(df["total_buyer_savings"], errors="coerce").fillna(0).tolist(),
        "Buyer savings (£)",
    )

    plot_metric(
        axes[1],
        labels,
        pd.to_numeric(df["total_seller_gains"], errors="coerce").fillna(0).tolist(),
        "Seller gains (£)",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()