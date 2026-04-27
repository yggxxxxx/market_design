from dataclasses import dataclass
from pathlib import Path
import os

import pandas as pd

SEASON_TO_MONTHS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}
 
PROJECT_ROOT = Path(__file__).resolve().parents[1]


TARIFF_DATA_DIR = PROJECT_ROOT / "data" / "tariffs"


def normalize_season(season: str | None) -> str | None:
    if season is None:
        return None

    season = str(season).strip().lower()

    if season in {"", "all", "none"}:
        return None

    if season == "fall":
        season = "autumn"

    if season not in SEASON_TO_MONTHS:
        raise ValueError(
            f"Unsupported season '{season}'. "
            f"Choose from: {sorted(SEASON_TO_MONTHS.keys())}"
        )

    return season


@dataclass
class TariffProfile:
    tariff_name: str
    target_year: int
    season: str | None
    aggregation: str
    hourly_prices_gbp_per_kwh: dict

    def get_price(self, hour: int) -> float:
        if not isinstance(hour, int):
            raise TypeError(f"hour must be int, got {type(hour).__name__}")

        if hour < 0 or hour > 23:
            raise ValueError(f"hour must be between 0 and 23, got {hour}")

        return float(self.hourly_prices_gbp_per_kwh[hour])

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for hour in range(24):
            rows.append(
                {
                    "hour": hour,
                    "price_gbp_per_kwh": self.hourly_prices_gbp_per_kwh[hour],
                }
            )
        return pd.DataFrame(rows)


class TariffLoader:
    COLUMN_NAMES = [
        "timestamp",
        "time_label",
        "region_code",
        "region_name",
        "price_pence_per_kwh",
    ]

    def resolve_csv_path(self, csv_path) -> Path:
        """
        解析 tariff CSV 路径：

        优先顺序：
        1) 绝对路径
        2) data/tariffs/<csv_path>
        3) 项目根目录下的相对路径
        4) 若都不存在，则返回 data/tariffs/<csv_path>，便于报错清晰
        """
        csv_path = Path(csv_path)

        if csv_path.is_absolute():
            return csv_path

        candidate = TARIFF_DATA_DIR / csv_path
        if candidate.exists():
            return candidate

        candidate = PROJECT_ROOT / csv_path
        if candidate.exists():
            return candidate

        return TARIFF_DATA_DIR / csv_path

    def load_raw_tariff_csv(self, csv_path) -> pd.DataFrame:
        csv_path = self.resolve_csv_path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"Tariff CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, header=None, names=self.COLUMN_NAMES)

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["price_pence_per_kwh"] = pd.to_numeric(
            df["price_pence_per_kwh"],
            errors="coerce",
        )

        df = df.dropna(subset=["timestamp", "price_pence_per_kwh"]).copy()
        df["price_gbp_per_kwh"] = df["price_pence_per_kwh"] / 100.0

        return df

    def build_representative_day_profile(
        self,
        csv_path,
        tariff_name,
        target_year=2025,
        season=None,
        agg="median",
    ) -> TariffProfile:
        df = self.load_raw_tariff_csv(csv_path)

        df_year = df[df["timestamp"].dt.year == target_year].copy()
        if df_year.empty:
            raise ValueError(
                f"No tariff data found for year {target_year} in file: {self.resolve_csv_path(csv_path)}"
            )

        season = normalize_season(season)
        if season is not None:
            months = SEASON_TO_MONTHS[season]
            df_year = df_year[df_year["timestamp"].dt.month.isin(months)].copy()

            if df_year.empty:
                raise ValueError(
                    f"No tariff data found for year {target_year} and season {season!r} "
                    f"in file: {self.resolve_csv_path(csv_path)}"
                )

        df_year["hour"] = df_year["timestamp"].dt.hour

        agg = str(agg).strip().lower()
        grouped = df_year.groupby("hour")["price_gbp_per_kwh"]

        if agg == "median":
            hourly_series = grouped.median().reindex(range(24))
        elif agg == "mean":
            hourly_series = grouped.mean().reindex(range(24))
        else:
            raise ValueError("agg must be either 'median' or 'mean'")

        if hourly_series.isna().any():
            missing_hours = hourly_series[hourly_series.isna()].index.tolist()
            raise ValueError(
                f"Missing tariff data for hours {missing_hours} "
                f"in year {target_year} and season {season!r}"
            )

        hourly_prices_gbp_per_kwh = {
            int(hour): float(price)
            for hour, price in hourly_series.items()
        }

        return TariffProfile(
            tariff_name=tariff_name,
            target_year=target_year,
            season=season,
            aggregation=agg,
            hourly_prices_gbp_per_kwh=hourly_prices_gbp_per_kwh,
        )


def load_tou_profile(csv_path=None, target_year=2025, season=None, agg="median") -> TariffProfile:
    loader = TariffLoader()

    if csv_path is None:
        csv_path = os.getenv("TOU_CSV_PATH", "ToU_London.csv")

    return loader.build_representative_day_profile(
        csv_path=csv_path,
        tariff_name="ToU",
        target_year=target_year,
        season=season,
        agg=agg,
    )


def load_fit_profile(csv_path=None, target_year=2025, season=None, agg="median") -> TariffProfile:
    loader = TariffLoader()

    if csv_path is None:
        csv_path = os.getenv("FIT_CSV_PATH", "FiT_London.csv")

    return loader.build_representative_day_profile(
        csv_path=csv_path,
        tariff_name="FiT",
        target_year=target_year,
        season=season,
        agg=agg,
    )