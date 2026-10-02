# Data dictionary - analytical_panel_2020_2023.csv

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
| flag_milk_t, flag_milk_animals, flag_stock_qcl, flag_stock_gle, flag_ch4_enteric_kt, flag_ch4_manure_kt | FAOSTAT flag of each component (A official, E estimated, I imputed, X external) | - |
| note_milk_t, note_milk_animals, note_stock_qcl, note_stock_gle, note_ch4_enteric_kt, note_ch4_manure_kt | FAOSTAT note of each component (empty when FAOSTAT publishes no note; FAOSTAT publishes none for the methane elements in 2020-2023) | - |

No global-warming-potential conversion is applied. The former column name 'kg_co2e_per_ton_milk' of the original files was a misnomer for a CH4 mass ratio and is not used.
