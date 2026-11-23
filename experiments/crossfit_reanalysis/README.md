# Compound-disjoint cross-fitting reanalysis

This is a retrospective correction and sensitivity analysis of the existing
historical archive. It does not create a new independent confirmation cohort.
The original archived implementation remains available at the repository root.

Use Python 3.10 and the pinned numerical environment:

```sh
python -m pip install -r requirements.txt
python verify.py
python runner.py audit
python runner.py run
python verify.py --refitted outputs/crossfit_results.json
```

Run these commands from this directory. `audit` tests one-task fitting,
omitted-label feature invariance, evaluation-label invariance, fault recovery,
checkpoint reuse and timeout termination. `run` uses one CPU thread, no GPU,
and atomic checkpoints. An interrupted run resumes completed folds. On systems
with GNU timeout, `timeout --kill-after=10s 1800s python runner.py run` bounds
the full process group. Do not use optimized Python (`python -O`) for validation.

The 693 archived molecular identities are traced through raw ChEMBL ID, parent
ID, standard InChIKey and standardized-parent InChIKey. Each evaluation pair
containing a molecule matched to calibration at any level is removed. Missing
identities fail qualification. The six calibration pairs stay fixed. The main
view requires six retained evaluation pairs and a single measurement type per
task. It contains 21 tasks: seven Ki and 14 IC50. A secondary view permits one
to six retained evaluation pairs and contains 32 tasks. Six mixed Ki/IC50 tasks
are excluded as complete tasks from both qualified donor pools. Worksheet-level
measurement type is traced for all four potency columns; original assay IDs are
unavailable and are not inferred. Standardized parent keys were generated with
RDKit fragment-parent, charge-normalization and canonical-tautomer operations;
the supplied identity table permits direct inspection without RDKit installation.

Each donor calibration feature row excludes that pair from local fits,
standardization, response normalization and task descriptors. Donor evaluation
rows use all six calibration pairs. Global response targets retain six-shot
normalization. Held evaluation labels are masked before candidate selection.
All other model hyperparameters, seeds and tie rules remain fixed.

There are five donor-pool analyses (39 original tasks, 21 strict tasks, 32 tasks
with reduced evaluation pools, seven Ki tasks and 14 IC50 tasks). All use the
original and two moderate learning states. The strict view also retains the
original median and 75th-percentile floors for its five-level joint curve.
Together these require 381 outer fits, each producing the six selectors.
Bootstrap intervals use 20,000 paired task draws with seed 20261117.

## Results

| Analysis | Tasks | Original / clipped-learning HIST-SAFE successes | B minus A, pp [95% CI] |
| --- | ---: | ---: | ---: |
| Strict main view | 21 | 10 / 10 | 0.0 [-23.8, 28.6] |
| Original task archive with cross-fitting | 39 | 26 / 21 | -12.8 [-28.2, 2.6] |
| Reduced evaluation pool | 32 | 22 / 20 | -6.3 [-25.0, 12.5] |
| Ki-only donors and evaluation | 7 | 1 / 2 | 14.3 [-28.6, 57.1] |
| IC50-only donors and evaluation | 14 | 10 / 9 | -7.1 [-28.6, 14.3] |

The table uses the fixed higher floor 5.61261017366127, affecting 26.6% of
strict-view potency slots and 24.9% of the original archive. Under original
evaluation labels, 14/21 strict-view choices change while total success remains
10/21; evaluation-only clipping yields 9/21. These corrected results supersede
the original implementation's aggregate deterioration claim. The numerical
record retains all successes, losses, zeros and unchanged choices.

`expected/` supplies complete task outcomes and every frozen choice. The
verification entry point reconstructs gains, flags, oracle counts, paired
statistics, mean-gain intervals and identity exclusions. `--refitted` additionally
compares each regenerated row and orientation against the archived selections.
The original frozen external Davis choices are kept unchanged; their plug-in
counts and reporting-limit identification are separate analyses.

Data and third-party attributions follow the repository's `DATA_LICENSES.md`.
