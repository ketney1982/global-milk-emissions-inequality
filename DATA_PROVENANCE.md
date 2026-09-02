# DATA PROVENANCE

**Manuscript:** *Beyond the Dairy Cow: The Milk-Species Production Mix as a Structural Determinant of National Dairy Methane Intensity*

Ketney Otto — Department of Agricultural Science and Food Engineering, "Lucian Blaga" University of Sibiu, 7–9 Dr. I. Ratiu Street, 550012 Sibiu, Romania
`ketney.otto@ulbsibiu.ro` · ORCID 0000-0003-1638-1154

---

## 1. Purpose and status of this document

This file records the exact upstream provenance, units and construction rules of the three harmonised analytical datasets on which every reported result rests.

The three CSV files are treated as **frozen analytical inputs**: they are archived verbatim, identified by cryptographic hash in Section 7, and are the reproducibility entry point for the study. The FAOSTAT extraction that produced them is documented here element by element and has been re-verified against the live FAOSTAT API (Section 3.3). The analysis is not re-derived from raw FAOSTAT at run time.

---

## 2. Upstream sources

### 2.1 Milk production

| Field | Value |
| --- | --- |
| Provider | FAO — FAOSTAT |
| Domain | **QCL** — Crops and Livestock Products |
| Items | Raw milk of cattle; Raw milk of buffalo; Raw milk of goats; Raw milk of sheep; Raw milk of camel |
| Element | Production |
| Unit | tonnes (t) of raw milk, as reported (**not** fat- or protein-corrected) |
| Years | 2020–2023 |

### 2.2 Methane emissions

| Field | Value |
| --- | --- |
| Provider | FAO — FAOSTAT |
| Domain | **GLE** — Emissions from Livestock |
| Items | Cattle, dairy (960); Buffalo (946); Goats (1016); Sheep (976); Camels (1126) |
| Elements | Enteric fermentation (Emissions CH4) (**72254**) **+** Manure management (Emissions CH4) (**72256**) |
| Gas | CH4 |
| Unit | **kt of CH4 (methane mass)** |
| GWP conversion | **none applied** — see Section 4 |
| Years | 2020–2023 |
| Accounting tier | IPCC Tier 1 with FAOSTAT implicit emission factors |

The two elements are summed to give one farm-gate methane total per country–species–year. No other gas, source category or domain contributes to the numerator: N2O, manure applied to soils and manure left on pasture are **excluded**.

### 2.3 Country identifiers

UN M49 codes, via FAOSTAT area codes. The file `m49_regions.csv` supplies the regional grouping used for descriptive reporting only.

---

## 3. Construction rules

### 3.1 Derived intensity

For every country c, species s and year t:

    intensity = CH4_kt * 1000 / milk_tonnes

Because 1 kt = 1000 t, this ratio is **t CH4 per t raw milk**, equivalently **kg CH4 per kg raw milk**.

The stored column is named `kg_co2e_per_ton_milk`. **That column name is a misnomer, retained for archival fidelity with the deposited files.** The quantity it holds is neither CO2-equivalent nor per-tonne: it is a CH4 mass ratio. All manuscript reporting uses the correct unit, **g CH4 per kg raw milk**, obtained as `stored value * 1000`.

### 3.2 Aggregation and deduplication

| Rule | Detail |
| --- | --- |
| China | FAOSTAT reports both China (M49 159, aggregate) and China, mainland (M49 156) with near-identical values. The aggregate (159) is **dropped**; 156 is retained. |
| Country-year aggregate | Total CH4 and total milk summed over the five species, then divided — algebraically identical to the production-weighted mixture of species intensities (verified to 1e-10). |
| Species shares | w[c,s,t] = M[c,s,t] / M[c,t] — the share of **national milk output** supplied by species s. These are **production weights, not animal or herd counts**. |

### 3.3 Verification against live FAOSTAT

The numerator definition was re-verified by querying the FAOSTAT API directly and reconstructing archived values from the two constituent elements:

| Country | Year | Species | Archived (kt) | 72254 enteric | 72256 manure | Sum |
| --- | --- | --- | --- | --- | --- | --- |
| Afghanistan | 2020 | Cattle, dairy | 221.9665 | 204.3501 | 17.6164 | **221.9665** |
| Afghanistan | 2020 | Goats | 39.5652 | 38.7135 | 0.8517 | **39.5652** |
| Afghanistan | 2020 | Sheep | 65.3354 | 64.0543 | 1.2811 * | **65.3354** |
| Afghanistan | 2020 | Camels | 8.0097 | 7.7929 | 0.2168 * | **8.0097** |
| India | 2020 | Buffalo | 6579.9505 | 6031.6213 | 548.3292 | **6579.9505** |

All five reconstructions are exact to the last reported digit.
(*) manure component obtained by subtraction; the enteric element was queried directly.

Verification date: 2 September 2026. FAOSTAT values carry flag `E` (estimated).

---

## 4. Units — authoritative statement

| Quantity | Correct unit | Relation to stored column |
| --- | --- | --- |
| Species emission intensity | **g CH4 / kg raw milk** | stored * 1000 |
| National aggregate intensity | **g CH4 / kg raw milk** | stored * 1000 |
| Shapley components | **g CH4 / kg raw milk** | stored * 1000 |
| Absolute national reduction | **Mt CH4** | unchanged |

Identity: 1 g CH4 / kg milk = 1 kg CH4 / t milk = 0.001 kg CH4 / kg milk.

**No global-warming-potential conversion is applied anywhere in the analysis.** All results are expressed in methane mass, which makes them independent of the choice of GWP100, GWP* or any other metric. For readers who require a CO2-equivalent scale, 1 Mt CH4 corresponds to approximately 28 Mt CO2e on the AR5 100-year GWP that FAOSTAT itself applies to its CO2eq elements, or approximately 27 Mt CO2e on the AR6 value for non-fossil methane.

---

## 5. Allocation boundary for non-bovine species — important limitation

FAOSTAT resolves cattle into "Cattle, dairy" and "Cattle, non-dairy", so the cattle numerator is already restricted to the dairy herd. **FAOSTAT does not provide an equivalent dairy/non-dairy split for buffalo, goats, sheep or camels.** For those four species the numerator is therefore the methane of the **entire national herd of that species**, while the denominator is milk production alone.

Consequences, which must be carried into any interpretation:

1. Intensities for buffalo, goats, sheep and camels are **whole-herd methane charged against milk output only**. They are not milk-allocated intensities and are not directly comparable with the dairy-cattle figure.
2. The effect is largest where the milked fraction of the herd is smallest and where the species also yields meat, fibre, hides or draught power — most acutely for camels and for extensive small-ruminant systems.
3. The FAOSTAT *Emissions intensities* (EI) domain addresses this by scaling non-bovine emissions by the share of animals involved in milk production. That scaling is **not** applied here.
4. The species ordering reported in this study is therefore a statement about methane per unit of raw milk **under whole-herd allocation for the four non-bovine species**, not a species-level biological efficiency ranking.

---

## 6. Record counts

| Stage | Count |
| --- | --- |
| Reporting entities in raw extraction | 183 |
| After dropping the China aggregate (M49 159) | 182 |
| Species-level rows, raw | 1,637 |
| Species-level rows after China deduplication | 1,617 |
| Country-year rows, raw | 731 |
| Country-year rows after China deduplication | **727** |
| Countries with complete 2020–2023 series | **181** |
| Country-year-species rows entering the Bayesian model (share > 0 and intensity > 0) | **1,615** |
| Accounting checks executed | 2,181 (3 x 727), all passed |

**The 182 to 181 reduction.** Palestine (M49 275) reports 2020, 2021 and 2022 but has no 2023 record. It therefore cannot enter either interval analysis — the Shapley decomposition, which requires both endpoints, or the 2023-referenced optimisation — and is excluded from both. No country was excluded for a missing intensity on an active share: there are zero such records in the panel.

---

## 7. Frozen input fingerprints

SHA-256, computed on LF-normalised bytes so the value is invariant to the CRLF/LF difference between the GitHub checkout and the archived evidence folder:

| SHA-256 | File | Bytes (LF) |
| --- | --- | --- |
| `cfbd5c325b98ab91ffd276746be156209375f3567fc2cd3a47065fc167e3c265` | `cercetare-485010.faostat_clean.milk_emission_intensity_2020_2023.csv` | 126,240 |
| `e14285cb9e8e4f3e02fdd76707691ccf752cdf7f5e975978560c866dc9e8b4eb` | `cercetare-485010.faostat_clean.milk_intensity_country_year.csv` | 44,904 |
| `465ac85a24c01d618cc0990efb53e1eaf0dfb0085a9cfe9b15851f7ebe7424c9` | `cercetare-485010.faostat_clean.milk_species_structure.csv` | 104,754 |
| `756b97300b2326b020b9c3de5d600a5b555600ef7b2ad114f1fab7269a126d05` | `m49_regions.csv` | 1,645 |

Content equivalence between the GitHub `data/` copies and the archived `evidence/01_raw_data/` copies was confirmed field by field; the two differ only in line terminators.

---

## 8. Code and run identification

| Item | Value |
| --- | --- |
| Repository | https://github.com/ketney1982/global-milk-emissions-inequality |
| Release | `v2.0.0-R2` (2 September 2026) — the version that produces the reported results. A file inside a commit cannot carry that commit's own hash, so this record identifies the release by tag; resolve it with `git rev-parse v2.0.0-R2`. The preceding release is commit `57d1c4cd3b5110df6f1c511fb338d50bb6d50f90` (message "R1", 18 February 2026), which accompanied the preprint and differs in the two respects listed in `CHANGELOG.md`. |
| Pipeline | `methane_portfolio` v0.1.0, entry point `run_all` |
| Python | 3.13.5 (MSC v.1943, 64-bit), Windows 11 |
| Pipeline RNG seed | 20230101 |
| PyMC / NUTS seed | 42 |
| Sampler | 16 chains x (15,000 tune + 8,000 draws), target_accept 0.95 |
| Posterior artefact | `bayes_posterior.nc`, 1,629,605,755 bytes, retained by the author |

**BigQuery.** Extraction and harmonisation were carried out in Google BigQuery (project `cercetare-485010`, dataset `faostat_clean`). The extraction SQL is **not** part of the public repository. The public reproducibility entry point is the three frozen CSVs listed in Section 7 together with the pipeline code; every downstream result can be regenerated from them. Section 2 of this document specifies the extraction semantics completely enough for an independent analyst to rebuild the same inputs directly from FAOSTAT.

---

## 9. Interpretation guardrails

1. All reported reductions are **accounting counterfactuals** under scenario-based reallocation of the milk-species production mix. The panel contains no covariates that would support causal identification, and no causal claim is made.
2. The decision variable is the **share of national milk output supplied by each species**. It is not herd composition, animal numbers or stocking rate.
3. The reallocation budget delta is a **mathematical scenario bound on the production mix**, not a demonstrated feasible intervention. It does not represent feed resources, land suitability, breed availability, processing infrastructure, demand, product functionality, cultural role or transition cost.
4. Posterior species intensities are draws of a **latent central intensity**, exp(mu), with no residual noise added. They are not posterior predictive draws of realised country-species intensities.
5. Because the functional unit is raw milk tonnage, no comparison in this study is corrected for fat, protein or total solids.
