FIGURE CODE FOR THE SUBMITTED MANUSCRIPT
========================================

figures_corrected.py produced every figure embedded in the submitted manuscript.
Run it against the deposited figure inputs:

    python figures_corrected.py --evidence <dir with 01_raw_data, 02_processed_results,
                                            03_diagnostics> --out .

It writes Figure 1..6 as .png, .svg and .pdf. Regeneration is deterministic.

NUMBERING. The script emits the original generation numbers. The manuscript submitted
to Animals renumbers them, because the posterior-predictive-check figure moved to the
Supplementary Materials:

    generated        submitted
    ---------        ---------
    Figure 1    ->   Figure 1    global decomposition
    Figure 2    ->   Figure 2    country-level dispersion
    Figure 3    ->   Figure S1   posterior predictive checks (Supplementary)
    Figure 4    ->   Figure 3    budget response
    Figure 5    ->   Figure 4    concentration of the response
    Figure 6    ->   Figure 5    minority-species composition

Panel (b) of the last figure no longer annotates the Pearson correlation between the
non-cattle share and national intensity. The share enters the aggregate intensity by
construction, so the association is algebraic rather than independent evidence, and the
panel is presented as a descriptive structural visualisation.
