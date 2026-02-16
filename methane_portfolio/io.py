# Autor: Ketney Otto
# Affiliation: „Lucian Blaga” University of Sibiu, Department of Agricultural Science and Food Engineering, Dr. I. Ratiu Street, no. 7-9, 550012 Sibiu, Romania
# Contact: otto.ketney@ulbsibiu.ro, orcid.org/0000-0003-1638-1154

"""Data loading and canonical dataframe construction.

Produces two canonical DataFrames:
  - ``long_df`` : species-level rows
  - ``agg_df``  : country-year aggregate rows
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from methane_portfolio import config


# ---------------------------------------------------------------------------
# dtype enforcement
# ---------------------------------------------------------------------------
_INTENSITY_DTYPES = {
    "country": str,
    "country_m49": np.int64,
    "year": np.int64,
    "milk_species": str,
    "milk_tonnes": np.float64,
    "ch4_ktco2e": np.float64,
    "kg_co2e_per_ton_milk": np.float64,
}

_STRUCTURE_DTYPES = {
    "country": str,
    "country_m49": np.int64,
    "year": np.int64,
    "milk_species": str,
    "milk_tonnes": np.float64,
    "species_share": np.float64,
}

_AGG_DTYPES = {
    "country": str,
    "country_m49": np.int64,
    "year": np.int64,
    "milk_total_tonnes": np.float64,
    "ch4_total_ktco2e": np.float64,
    "kg_co2e_per_ton_milk": np.float64,
}

_REGION_DTYPES = {
    "country_m49": "Int64",
    "region": "string",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw CSVs with enforced dtypes.

    Returns
    -------
    intensity_df, structure_df, agg_raw_df
    """
    d = Path(data_dir) if data_dir else config.DATA_DIR

    intensity = pd.read_csv(
        d / config.EMISSION_INTENSITY_FILE,
        dtype=_INTENSITY_DTYPES,
    )
    structure = pd.read_csv(
        d / config.SPECIES_STRUCTURE_FILE,
        dtype=_STRUCTURE_DTYPES,
    )
    agg_raw = pd.read_csv(
        d / config.COUNTRY_INTENSITY_FILE,
        dtype=_AGG_DTYPES,
    )
    return intensity, structure, agg_raw


def build_long_df(
    intensity: pd.DataFrame,
    structure: pd.DataFrame,
) -> pd.DataFrame:
    """Merge species intensity with species share into a single long table.

    Returns DataFrame with columns:
        country_m49, country, year, milk_species, species_share,
        milk_tonnes, kg_co2e_per_ton_milk
    """
    merge_keys = ["country_m49", "country", "year", "milk_species"]
    long = intensity.merge(
        structure[merge_keys + ["species_share"]],
        on=merge_keys,
        how="inner",
    )
    # Canonical species ordering
    long["milk_species"] = pd.Categorical(
        long["milk_species"],
        categories=sorted(long["milk_species"].unique()),
        ordered=True,
    )
    long.sort_values(
        ["country_m49", "year", "milk_species"],
        inplace=True,
    )
    long.reset_index(drop=True, inplace=True)
    return long[
        [
            "country_m49", "country", "year", "milk_species",
            "species_share", "milk_tonnes", "kg_co2e_per_ton_milk",
        ]
    ]


def build_agg_df(agg_raw: pd.DataFrame) -> pd.DataFrame:
    """Return the aggregate table with canonical column ordering.

    Columns: country_m49, country, year, milk_total_tonnes,
             ch4_total_ktco2e, kg_co2e_per_ton_milk
    """
    df = agg_raw.copy()
    df.sort_values(["country_m49", "year"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df[
        [
            "country_m49", "country", "year",
            "milk_total_tonnes", "ch4_total_ktco2e", "kg_co2e_per_ton_milk",
        ]
    ]


def canonical_species(long_df: pd.DataFrame) -> list[str]:
    """Return sorted unique species list."""
    return sorted(long_df["milk_species"].unique())


def load_all(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Convenience: load CSVs and return (long_df, agg_df, species_list).

    Parameters
    ----------
    data_dir : optional override for the data directory
    """
    intensity, structure, agg_raw = load_raw(data_dir)
    long_df = build_long_df(intensity, structure)
    agg_df = build_agg_df(agg_raw)
    species = canonical_species(long_df)
    return long_df, agg_df, species


def load_region_mapping(data_dir: Path | None = None) -> pd.DataFrame:
    """Load optional M49â†’region mapping.

    Returns a DataFrame with columns [country_m49, region].
    If the mapping file is missing, returns an empty DataFrame with the same columns.
    """
    d = Path(data_dir) if data_dir else config.DATA_DIR
    path = d / config.REGION_MAP_FILE
    if not path.exists():
        return pd.DataFrame(columns=["country_m49", "region"])

    regions = pd.read_csv(path, dtype=_REGION_DTYPES)
    missing_cols = {"country_m49", "region"} - set(regions.columns)
    if missing_cols:
        raise ValueError(
            f"Region mapping is missing required columns: {sorted(missing_cols)}"
        )

    out = regions[["country_m49", "region"]].copy()
    out["country_m49"] = out["country_m49"].astype("Int64")
    out.dropna(subset=["country_m49"], inplace=True)
    out["country_m49"] = out["country_m49"].astype(np.int64)
    out.drop_duplicates(subset=["country_m49"], keep="first", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out
