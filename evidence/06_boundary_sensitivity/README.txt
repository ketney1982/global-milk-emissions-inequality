BOUNDARY AND FUNCTIONAL-UNIT SENSITIVITY
========================================
Added 3 September 2026 for manuscript Sections 2.13 and 3.7 and Tables 9 and 10.

Answers the objection that the inter-species comparison is not like-for-like: FAOSTAT
resolves only cattle into dairy and non-dairy herds, so the four non-bovine ratios
charge whole-herd methane against milk output alone, and the functional unit is
uncorrected raw milk.

Both conventions act multiplicatively on the species intensities, so the question
reduces to one ratio per species, r_s = phi_s / phi_cattle, where phi_s is the share of
a species' herd methane that a milk-allocated inventory would charge to milk. No new
FAOSTAT data is introduced; the analysis instead states what a milk-allocated boundary
would have to look like to change the conclusion. The species-differentiated ratios
(buffalo 0.70, goats 0.45, sheep 0.25, camels 0.35) are ILLUSTRATIVE assumed values
chosen to reflect how far each species' herd is kept for milk. They are not estimated
from herd data and must not be reported as a central empirical estimate.

  boundary_sensitivity.py        the analysis; run from the manuscript directory
  panelA_break_even.csv          r*_s = I_cattle / I_s on 2023 medians
  panelB_scenarios.csv           mean/median reduction, cattle-lowest count and
                                 reallocation-direction counts per scenario
  panelC_functional_unit.csv     FPCM and total-solids factors and corrected intensities
  panelD_shapley.csv             global Shapley components under every scenario

The script contains an independent re-implementation of the Rockafellar-Uryasev
mean-CVaR linear program of Section 2.7 (HiGHS via scipy.optimize.linprog), reading the
deposited 500 x 182 x 5 latent-intensity array from ../05_release_R3/. Run against the
unscaled draws it reproduces the deposited per-country reductions to a maximum absolute
difference of 1.279e-13 percentage points and recovers the reported panel mean 11.9065%
and median 2.3785%; the unscaled Shapley run likewise reproduces the deposited global
components -4.2813 / +1.8501 g CH4 per kg. It therefore also serves as an external check
on the released pipeline.

FOUR RESULTS, ON FOUR DIFFERENT ESTIMANDS. Keep them apart.

1. Global 2023 species medians. Cattle is the lowest-intensity species under every
   convention tested EXCEPT the uniform r = 0.10 boundary, where goats fall below
   cattle (18.2 against 37.1 g CH4/kg). That is exactly what the break-even ratios
   predict, since 0.10 lies below r* for both buffalo (0.181) and goats (0.205).

2. Country-level minimum species (observed 2023 ratios). Cattle is the minimum-intensity
   species in 89 of the 107 multi-species systems under the reported boundary, and in
   83, 74, 46 and 25 at uniform r = 0.75, 0.50, 0.25 and 0.10; 66 under the illustrative
   species-differentiated allocation; 89 per kg FPCM; 91 per kg total-solids equivalent;
   and 62 under FPCM combined with that allocation.

3. Direction of the optimal reallocation (posterior latent draws). Share moves toward
   cattle in all 107 multi-species systems under the reported boundary, per kg FPCM, per
   kg total solids, at r = 0.75 and r = 0.50 and under the illustrative differentiated
   allocation. It reverses only at uniform ratios of 0.25 or below: 67 of 107 move away
   from cattle at r = 0.25 and 84 of 107 at r = 0.10.

4. Magnitude of the mean reduction. 11.91% under the published whole-herd raw-milk
   convention, 10.20% per kg FPCM, 4.86% under the illustrative species-differentiated
   allocation and 3.76% under both.

Note that results 1 and 2 are computed on the OBSERVED 2023 species ratios whereas
results 3 and 4 are computed on the posterior latent-intensity draws. The two estimands
are reported side by side deliberately and must not be conflated.

SHAPLEY IS NOT INVARIANT. Only the additive identity is. Under every scenario the
reconstruction error of Delta_struct + Delta_within = Delta_obs stays below 3.2e-12,
but the components move substantially and the within-species term changes sign:

  convention                          struct    within      net
  whole-herd, raw milk               -4.2813   +1.8501  -2.4312
  uniform r = 0.75                   -3.1491   +1.1403  -2.0088
  uniform r = 0.50                   -2.0169   +0.4305  -1.5864
  uniform r = 0.25                   -0.8848   -0.2793  -1.1641
  uniform r = 0.10                   -0.2055   -0.7052  -0.9106
  illustrative differentiated        -1.2256   -0.5446  -1.7702
  per kg FPCM                        -2.9664   +0.9539  -2.0125
  per kg total-solids equivalent     -3.0672   +0.9512  -2.1160
  FPCM + illustrative differentiated -0.8173   -0.6804  -1.4978

An earlier version of this note and of manuscript Sections 3.7 and 4.4 stated that the
Shapley decomposition was "unaffected" by either convention. That was wrong: exactness
is invariant, the estimates are not. Corrected 3 September 2026.
