from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOUSEHOLD_DATA_DIR = PROJECT_ROOT / "data" / "household_data"

SEASON_TO_FOLDER = {
    "spring": "Data_spring",
    "summer": "Data_summer",
    "autumn": "Data_autumn",
    "winter": "Data_winter",
}


def normalize_season(season: str | None) -> str:
    """
    支持:
    - spring / summer / autumn / winter
    - fall -> autumn
    - all / year / annual -> all
    """
    if season is None:
        return "summer"

    season = str(season).strip().lower()

    if season == "fall":
        season = "autumn"

    if season in {"all", "year", "annual"}:
        return "all"

    if season not in SEASON_TO_FOLDER:
        raise ValueError(
            f"Unknown season: {season}. "
            f"Choose from: {sorted(SEASON_TO_FOLDER.keys()) + ['all']}"
        )

    return season


def load_season_folder(season: str = "summer") -> Path:
    season = normalize_season(season)

    if season == "all":
        raise ValueError("load_season_folder does not support 'all'; use load_season_folders instead.")

    folder = HOUSEHOLD_DATA_DIR / SEASON_TO_FOLDER[season]

    if not folder.exists():
        raise FileNotFoundError(f"Season folder not found: {folder}")

    return folder


def load_season_folders(season: str = "summer") -> list[Path]:
    """
    返回一个或多个季节文件夹。
    """
    season = normalize_season(season)

    if season == "all":
        folders = [HOUSEHOLD_DATA_DIR / SEASON_TO_FOLDER[s] for s in ["spring", "summer", "autumn", "winter"]]
    else:
        folders = [HOUSEHOLD_DATA_DIR / SEASON_TO_FOLDER[season]]

    missing = [str(folder) for folder in folders if not folder.exists()]
    if missing:
        raise FileNotFoundError(f"Season folder(s) not found: {missing}")

    return folders


def _load_one_csv(file: Path, required_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(file)

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{file.name} missing columns: {missing}")

    df = df[required_cols].copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df["import_energy"] = pd.to_numeric(df["import_energy"], errors="coerce")
    df["export_energy"] = pd.to_numeric(df["export_energy"], errors="coerce")
    df = df.dropna(subset=["DateTime", "import_energy", "export_energy"]).copy()

    # 用文件名覆盖 household id，避免文件内写错
    df["h_id"] = file.stem

    return df


def get_household_df(
    season: str = "summer",
    days: int | None = None,
    selected_households: list[str] | None = None,
) -> pd.DataFrame:
    """
    读取 household 数据并合并成一个 DataFrame。

    参数
    ----
    season:
        - 单季: spring / summer / autumn / winter
        - 全年: all / year / annual
    days:
        - None: 保留该 season 下全部数据
        - int : 按全局时间排序后，仅保留前 N 天
    selected_households:
        - 只保留指定 household 文件名列表，如 ["MAC000012", "MAC000034"]

    返回列
    ----
    - h_id
    - DateTime
    - import_energy
    - export_energy
    """
    folders = load_season_folders(season)
    required_cols = ["h_id", "DateTime", "import_energy", "export_energy"]
    selected_set = set(selected_households) if selected_households else None

    all_dfs = []

    for folder in folders:
        csv_files = sorted(folder.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {folder}")

        for file in csv_files:
            household_name = file.stem

            if selected_set is not None and household_name not in selected_set:
                continue

            df = _load_one_csv(file, required_cols)
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No household data loaded. Check season or selected_households.")

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df = merged_df.sort_values(["DateTime", "h_id"]).reset_index(drop=True)

    if days is not None:
        unique_days = sorted(merged_df["DateTime"].dt.normalize().unique())
        keep_days = set(unique_days[:days])
        merged_df = merged_df[
            merged_df["DateTime"].dt.normalize().isin(keep_days)
        ].reset_index(drop=True)

    return merged_df