import sys
from pathlib import Path

import pandas as pd

NAME_MAP = {
    "NCLE: 7. Dengue cases (any)": "disease_cases",
    "LSB: Population (Estimated-single age)": "population",
    "CCH - Precipitation (ERA5-Land)": "rainfall",
    "CCH - Air temperature (ERA5-Land)": "mean_temperature",
}

META = {
    "organisationunitid",
    "organisationunitname",
    "organisationunitcode",
    "organisationunitdescription",
    "dataid",
    "dataname",
    "datacode",
    "datadescription",
}


def wide_to_long(wide_path: Path, out_path: Path) -> None:
    df = pd.read_csv(wide_path)
    month_cols = [c for c in df.columns if c not in META]
    if not month_cols:
        raise SystemExit("No month columns found")
    long_df = df.melt(
        id_vars=["organisationunitname", "dataname"],
        value_vars=month_cols,
        var_name="month_label",
        value_name="value",
    )
    long_df["time_period"] = pd.to_datetime(
        long_df["month_label"], format="%B %Y", errors="coerce"
    ).dt.strftime("%Y-%m")
    long_df = long_df.dropna(subset=["time_period"])
    long_df["chap_col"] = long_df["dataname"].map(NAME_MAP)
    long_df = long_df.dropna(subset=["chap_col"])
    out = long_df.pivot_table(
        index=["organisationunitname", "time_period"],
        columns="chap_col",
        values="value",
        aggfunc="first",
    ).reset_index()
    out = out.rename(columns={"organisationunitname": "location"})
    for col in ["disease_cases", "population", "rainfall", "mean_temperature"]:
        if col not in out.columns:
            out[col] = float("nan")
    out = out[
        ["time_period", "location", "rainfall", "mean_temperature", "population", "disease_cases"]
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "training_data.csv"
    o = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "training_chap_long.csv"
    wide_to_long(inp, o)
    n = len(pd.read_csv(o))
    print(f"Wrote {o} ({n} rows)")
