#!/usr/bin/env python3
"""
01_extract_faostat.py  -  reproducible re-extraction of the FAOSTAT panel (replaces the BigQuery SQL).

Source files (public FAOSTAT bulk downloads, Normalized format):
  QCL  https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip
  GLE  https://bulks-faostat.fao.org/production/Emissions_livestock_E_All_Data_(Normalized).zip

Output: ../R1_data/faostat_panel_2020_2023_long.csv  (country x year x species)

Columns
  country, area_code, m49, year, species,
  milk_t            QCL element 5510 Production, raw milk item (t)
  milk_animals      QCL element 5318 Milk Animals (head)  = PAS
  stock_qcl         QCL element 5111 Stocks (head)         = TS used for the primary allocation factor (FAO Emissions intensities)
  stock_gle         GLE element 5111 Stocks (head)         = used for the Tier 1 diagnostic and the boundary sensitivity (delta with GLE stock)
  ch4_enteric_kt    GLE 72254, ch4_manure_kt GLE 72256, ch4_whole_kt = sum (kt CH4)
  flag_*            FAOSTAT flags of each component
  note_*            FAOSTAT notes of each component

Species -> items
  cattle  : milk 882 | GLE 960 'Cattle, dairy' (milk cows only; non-dairy cattle is item 961)
  buffalo : milk 951 | GLE 946
  goats   : milk 1020| GLE 1016
  sheep   : milk 982 | GLE 976
  camel   : milk 1130| GLE 1126

Rules (identical to the originally frozen inputs, so results are comparable)
  * GLE rows with Source == 'FAO TIER 1' only (UNFCCC rows, present for 2020 only, are discarded).
  * Regional aggregates (area code >= 5000, 351 'China' aggregate, 420) are removed; 'China, mainland' kept.
  * CH4 mass is NOT converted to CO2e.
Usage: python 01_extract_faostat.py [path_to_bulk_dir]
"""
import sys, os, io, zipfile, urllib.request
import pandas as pd

BULK = sys.argv[1] if len(sys.argv) > 1 else "faostat_bulk"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "R1_data")
URL = "https://bulks-faostat.fao.org/production/"
FILES = {"QCL": "Production_Crops_Livestock_E_All_Data_(Normalized)",
         "GLE": "Emissions_livestock_E_All_Data_(Normalized)"}
YEARS = range(2020, 2024)
SPECIES = {  # species: (milk item, GLE item, QCL stock item)
    "cattle": ("882", "960", "866"), "buffalo": ("951", "946", "946"),
    "goats": ("1020", "1016", "1016"), "sheep": ("982", "976", "976"),
    "camel": ("1130", "1126", "1126")}


def load(key):
    csv = os.path.join(BULK, FILES[key] + ".csv")
    if not os.path.exists(csv):
        os.makedirs(BULK, exist_ok=True)
        data = urllib.request.urlopen(URL + FILES[key].replace("(", "(").replace(" ", "%20") + ".zip").read()
        zipfile.ZipFile(io.BytesIO(data)).extract(FILES[key] + ".csv", BULK)
    df = pd.read_csv(csv, encoding="utf-8", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["Year"] = df["Year"].astype(int)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df


def main():
    q, g = load("QCL"), load("GLE")
    q = q[q.Year.isin(YEARS)]
    g = g[g.Year.isin(YEARS) & (g.Source == "FAO TIER 1")]
    rows = []
    for sp, (mi, gi, qi) in SPECIES.items():
        def pick(df, item, el, name):
            d = df[(df["Item Code"] == item) & (df["Element Code"] == el)]
            d = d[["Area Code", "Area Code (M49)", "Area", "Year", "Value", "Flag", "Note"]].copy()
            d.columns = ["area_code", "m49", "country", "year", name, "flag_" + name, "note_" + name]
            return d
        parts = [pick(q, mi, "5510", "milk_t"), pick(q, mi, "5318", "milk_animals"),
                 pick(g, gi, "5111", "stock_gle"), pick(q, qi, "5111", "stock_qcl"),
                 pick(g, gi, "72254", "ch4_enteric_kt"), pick(g, gi, "72256", "ch4_manure_kt")]
        base = parts[0]
        for p in parts[1:]:
            base = base.merge(p.drop(columns=["m49", "country"]), on=["area_code", "year"], how="outer")
            base["m49"] = base["m49"].fillna(p["m49"].iloc[0] if len(p) else None)
        base["species"] = sp
        rows.append(base)
    # a simple outer merge loses country names for rows absent in the first part; rebuild from AreaCodes
    ac = g[["Area Code", "Area Code (M49)", "Area"]].drop_duplicates()
    ac.columns = ["area_code", "m49_ac", "country_ac"]
    df = pd.concat(rows, ignore_index=True).merge(ac, on="area_code", how="left")
    df["country"] = df["country"].fillna(df["country_ac"])
    df["m49"] = df["m49"].fillna(df["m49_ac"]).str.replace("'", "", regex=False)
    df = df.drop(columns=["m49_ac", "country_ac"])
    df["area_code"] = df["area_code"].astype(int)
    df = df[(df.area_code < 5000) & (~df.area_code.isin([351, 420]))]
    df["ch4_whole_kt"] = df[["ch4_enteric_kt", "ch4_manure_kt"]].sum(axis=1, min_count=2)
    df = df[df.milk_t.notna() & (df.milk_t > 0)]
    cols = ["country", "area_code", "m49", "year", "species", "milk_t", "milk_animals", "stock_gle", "stock_qcl",
            "ch4_enteric_kt", "ch4_manure_kt", "ch4_whole_kt", "flag_milk_t", "flag_milk_animals", "flag_stock_gle",
            "flag_ch4_enteric_kt", "flag_ch4_manure_kt", "note_milk_t", "note_milk_animals", "note_stock_gle",
            "flag_stock_qcl", "note_stock_qcl", "note_ch4_enteric_kt", "note_ch4_manure_kt"]
    df = df[cols].sort_values(["country", "species", "year"])
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "faostat_panel_2020_2023_long.csv"), index=False)
    print(df.groupby(["species", "year"]).size().unstack())
    print("rows", len(df), "countries", df.country.nunique())


if __name__ == "__main__":
    main()

