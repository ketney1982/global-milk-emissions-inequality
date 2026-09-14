# Evidence for the revised manuscript (tag `v3.1.0-R4`)

Analyses added after the R3 pipeline, carrying the boundary-sensitivity layer and the
supporting robustness checks reported in the Supplementary Materials. Nothing here
re-runs the sampler: every stage reads the deposited
`outputs_R3/posterior_intensity_draws.npz` (500 x 182 x 5 latent central
species-intensity draws) together with the three frozen analytical inputs in `data/`.

Both scripts resolve their input paths automatically, so they run unchanged from a
clean clone of this repository and from the manuscript's own evidence tree.

```bash
python evidence/06_boundary_sensitivity/boundary_sensitivity.py
python evidence/07_supporting_analyses/supporting_analyses.py
```

## `06_boundary_sensitivity/`

Allocation-boundary and functional-unit sensitivity, and the global estimand.
See `06_boundary_sensitivity/README.txt` for the full note. The script contains an
independent re-implementation of the Rockafellar–Uryasev mean–CVaR linear program and
prints two reproduction checks on startup:

* deposited country-level reductions reproduced to `max |diff| = 1.279e-13` percentage
  points, recovering the reported panel mean 11.9065% and median 2.3785%;
* deposited global Shapley components reproduced exactly (`-4.2813` / `+1.8501` g CH4
  per kg raw milk).

Panels A–H back Tables 2 and S4–S5 of the submission.

## `07_supporting_analyses/`

Robustness checks reported in Supplementary Sections S4.2–S4.8 and S4.12.

| File | Supplementary section | Content |
| --- | --- | --- |
| `robust_species_summaries.csv` | S4.2 | median, IQR, 20% trimmed mean, raw mean, and the raw mean recomputed without denominator-collapse cells, by species and year |
| `denominator_collapse_cells.csv` | S4.2 | the cells the rule flags (exactly one: Mongolia, sheep, 2023) |
| `coverage_by_species_year.csv`, `coverage_matrix_2023.csv`, `coverage_species_count_2023.csv` | S4.3 | reporting-entity counts by species and year, and the 2023 cross-section |
| `influence_summary_both_estimands.csv` | S4.4 | decompositions recomputed with the largest 1, 5 and 10 contributors removed, on both estimands |
| `finite_draw_stability.csv`, `finite_draw_stability_repeats.csv` | S4.5 | the linear program re-solved on subsamples of the deposited draws (100, 250, 500; 15 repeats at the first two) |
| `dirichlet_concentration_sensitivity.csv` | S4.6 | the reference mix perturbed as a Dirichlet draw at concentrations 50, 200, 1,000, 20,000 and infinity, 10 repeats at each finite concentration |
| `trend_step_separability.csv` | S4.7 | separability of the species trend from the post-2022 step |
| `zero_methane_cells.csv` | S4.8 | cells carrying positive milk output and a reported methane value of zero |
| `absent_reporting_entities.csv` | S4.12 | FAOSTAT reporting entities absent from the frozen analytical extraction |

### Two things these files are not

**They are not a pre-registration.** The rules they implement — the robust summaries, the
denominator-collapse rule, the influence thresholds and the exclusion modes — were fixed
before the corresponding analyses were run, but no separate protocol document was
deposited at that time and none is claimed here. The rules are specified in full in the
Supplementary Materials and in `supporting_analyses_config.json`.

**They are not a reconstruction of the missing entities.** Five FAOSTAT reporting
entities that carry milk and emissions records upstream are absent from the three frozen
analytical tables, and the criterion that removed them is not recoverable from the
archived artefacts. `absent_reporting_entities.csv` records them and verifies that none
of them appears in the frozen inputs. No post hoc reconstruction or imputation was
performed, and the analytical panel is therefore described throughout as a 181-country
global panel rather than an exhaustive census of FAOSTAT reporting entities.

### Provenance of this folder

`influence_summary_both_estimands.csv` and `trend_step_separability.csv` are copied
unchanged from the deposited R3 pipeline output. Every other file in
`07_supporting_analyses/` was regenerated at the R4 revision by
`supporting_analyses.py` from the frozen inputs and the deposited draw array; the
regenerated values reproduce those quoted in the manuscript (2023 species medians 37.1 /
181.5 / 204.7 / 555.4 / 1,159.2 g CH4 per kg; 20% trimmed means 46.9 / 232.3 / 225.8 /
613.8 / 1,808.2; the 2023 sheep mean falling from 5,252.9 to 1,531.8 when the single
flagged cell is excluded; 2023 coverage of 180 / 102 / 70 / 27 / 19 entities; and the
74 / 34 / 39 / 31 / 3 distribution of countries by number of species reported).
