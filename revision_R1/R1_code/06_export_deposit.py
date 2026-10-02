#!/usr/bin/env python3
"""06_export_deposit.py - writes the public analytical panel with explicit column names (replaces the misnamed
'kg_co2e_per_ton_milk' column of the original files) and a data dictionary; computes SHA-256 checksums of the deposit."""
import os, hashlib, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "R1_data")
R = os.path.join(HERE, "..", "R1_results")
p = pd.read_csv(os.path.join(R, "analytical_panel_complete_case.csv"))
out = p.rename(columns={
    "m49": "country_m49", "milk_t": "raw_milk_t", "milk_animals": "milk_animals_head_PAS", "stock_gle": "total_stock_head_GLE",
    "stock_qcl": "total_stock_head_QCL", "ch4_enteric_kt": "ch4_enteric_kt", "ch4_manure_kt": "ch4_manure_kt", "ch4_whole_kt": "ch4_whole_herd_kt",
    "delta": "milk_allocation_factor_delta", "I_whole": "ch4_g_per_kg_raw_milk_whole_herd", "I_milk": "ch4_g_per_kg_raw_milk_milk_allocated"})
cols = ["country", "country_m49", "year", "species", "raw_milk_t", "milk_animals_head_PAS", "total_stock_head_QCL", "total_stock_head_GLE",
        "ch4_enteric_kt", "ch4_manure_kt", "ch4_whole_herd_kt", "milk_allocation_factor_delta", "ch4_g_per_kg_raw_milk_whole_herd",
        "ch4_g_per_kg_raw_milk_milk_allocated", "flag_milk_t", "flag_milk_animals", "flag_stock_gle", "flag_ch4_enteric_kt", "flag_ch4_manure_kt",
        "note_milk_t", "note_milk_animals", "note_stock_gle"]
out[cols].to_csv(os.path.join(D, "analytical_panel_2020_2023.csv"), index=False)
dd = """# Data dictionary - analytical_panel_2020_2023.csv

One row per country x species x year (complete-case series only, see Methods 2.3). Source: FAOSTAT QCL and GLE (bulk files), FAO Tier 1 series.

| Column | Meaning | Unit |
|---|---|---|
| country, country_m49 | FAOSTAT area name and UN M49 code | - |
| year | 2020-2023 | - |
| species | cattle (dairy herd), buffalo, goats, sheep, camel | - |
| raw_milk_t | QCL 5510 Production, raw milk of the species | t |
| milk_animals_head_PAS | QCL 5318 Milk animals (animals producing milk) | head |
| total_stock_head_QCL | QCL 5111 Stocks (cattle: total cattle, not used for delta) | head |
| total_stock_head_GLE | GLE 5111 Stocks of the emission category (cattle: Cattle, dairy) | head |
| ch4_enteric_kt, ch4_manure_kt | GLE 72254 and 72256, FAO Tier 1 | kt CH4 |
| ch4_whole_herd_kt | sum of the two elements (whole national herd of the species) | kt CH4 |
| milk_allocation_factor_delta | PAS / total_stock_head_QCL for non-bovine species; 1 for cattle | fraction |
| ch4_g_per_kg_raw_milk_whole_herd | 10^6 * ch4_whole_herd_kt / raw_milk_t | g CH4 per kg raw milk |
| ch4_g_per_kg_raw_milk_milk_allocated | whole-herd intensity * delta | g CH4 per kg raw milk |
| flag_*, note_* | FAOSTAT flags (A official, E estimated, I imputed, X external) and notes of each component | - |

No global-warming-potential conversion is applied. The former column name 'kg_co2e_per_ton_milk' of the original files was a misnomer for a CH4 mass ratio and is not used.
"""
open(os.path.join(D, "DATA_DICTIONARY.md"), "w", encoding="utf-8").write(dd)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


DEPOSITED_SCRIPTS = ("01_extract_faostat.py", "02_analysis.py", "04_legacy_smoother_summary.py", "06_export_deposit.py")   # data-generating scripts only
man = {}
for folder in (D, R, os.path.join(HERE)):
    for fn in sorted(os.listdir(folder)):
        fp = os.path.join(folder, fn)
        if not os.path.isfile(fp) or "pycache" in fp or fn == "reference_order.json" or fn == "SHA256_manifest.json":
            continue
        if folder == HERE and fn not in DEPOSITED_SCRIPTS:
            continue
        if fn.endswith((".csv", ".py", ".json", ".md")):
            man[os.path.relpath(fp, os.path.join(HERE, "..")).replace("\\", "/")] = sha(fp)
json.dump(man, open(os.path.join(R, "SHA256_manifest.json"), "w"), indent=1)
print(len(man), "files hashed")
