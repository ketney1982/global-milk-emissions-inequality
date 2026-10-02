# Revision R1 (release v4.0.0-R1) – FAOSTAT re-extraction and country-specific milk allocation

Data-generating code, data and result tables of revision 1 of the manuscript *Ruminant Species Composition and Reported Methane Intensity in Global Milk Production: A 186-Country Decomposition under Alternative Allocation Boundaries* (*Animals*).
Scripts that only draw figures or build the Word documents are not part of this repository.

## Contents
| Path | Content |
|---|---|
| `R1_code/01_extract_faostat.py` | Re-extraction from the public FAOSTAT bulk files (QCL and GLE); replaces the original BigQuery query |
| `R1_code/02_analysis.py` | Complete-case panel, allocation factor δ = milk animals / total stock, Tier 1 diagnostic, exact Shapley decompositions (two- and three-factor), influence analyses, functional-unit sensitivity, closed-form bounded counterfactual (checked against HiGHS) |
| `R1_code/04_legacy_smoother_summary.py` | Summary of the original hierarchical smoother (Supplement S6); reads `outputs_R2/portfolio_results_corrected.csv` |
| `R1_code/06_export_deposit.py` | Public analytical panel with explicit column names, data dictionary, SHA-256 manifest |
| `R1_data/` | `faostat_panel_2020_2023_long.csv` (extraction), `analytical_panel_2020_2023.csv` (analytical panel with FAOSTAT flags and notes), `DATA_DICTIONARY.md` |
| `R1_results/` | All result tables quoted in the manuscript and Supplement (CSV/JSON) and `SHA256_manifest.json` |

## Reproduction
Run from `revision_R1/R1_code/` (Python ≥ 3.10; pandas, numpy, scipy):
```
python 01_extract_faostat.py [dir_with_FAOSTAT_bulk_files]   # downloads the QCL and GLE bulk files if absent
python 02_analysis.py
python 04_legacy_smoother_summary.py
python 06_export_deposit.py
```
Starting from the deposited `R1_data/faostat_panel_2020_2023_long.csv`, step 02 reproduces every file of `R1_results/` (differences below 1e-12 in the bootstrap intervals of `T3_tier1_diagnostic.csv` are possible between linear-algebra libraries).

## Headline numbers (186 countries, 405 country–species series, 1,620 cells, 99.91 % of 2023 reported milk)
- Milk-allocated world ratio 2020–2023: −0.64 g CH4 kg⁻¹ = country shares +0.57, species mix −1.43, intensity +0.21 (three-factor); without the Mongolian sheep-milk series +0.48, −0.10, −1.02.
- Fixed-country-weight two-factor decomposition (a different estimand): −1.12 = structure −1.38 + within-species +0.26; without that series −0.10 and −1.02.
- Cattle have the lowest intensity in 89 (whole-herd) versus 53 (milk-allocated) of 107 multispecies countries.

## Note on the Mongolian series
National milk totals and livestock numbers of the National Statistics Office of Mongolia (https://data.1212.mn, tables DT_NSO_1001_041V1 and DT_NSO_1001_021V1, retrieved 2 October 2026) reconcile with FAOSTAT; the species split of milk in 2022–2023 is not published at national level and could not be reconciled. These NSO values are quoted in the manuscript and Supplement and are not an input of the scripts.
