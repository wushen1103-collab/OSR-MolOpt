# OSR-MolOpt

Reproducible experiments on few-shot molecular prioritization, assay censoring,
and transfer across target pairs. The release contains the numerical methods,
fixed task splits, calibration/evaluation records, archived docking features,
frozen selections, complete positive and negative outcomes, and compressed
structural assets.

## Quick start

Use Python 3.13 and install the tested numerical dependencies:

```sh
python -m pip install -r requirements.txt
python tools/check_integrity.py
python -m osrmolopt verify
python -m osrmolopt smoke
python -m osrmolopt refit --timeout 1800
python tools/export_tables.py
```

No GPU, docking installation, account, or network access is needed after the
dependencies are installed. The executable commands limit numerical libraries
to one CPU thread. Outputs go to the ignored `outputs/` directory. Commands can
also use `--output PATH` to select a different output directory.

`verify` reconstructs 624 historical intervention outcomes, 54 external frozen
choices and 108 external endpoint intervals from potency records and archived
decisions. It performs no model fitting.

`smoke` fits one historical task, injects a failure before committing a
checkpoint, recovers, checks reuse of the checkpoint, and terminates an
intentionally stalled child process. It must pass before a full refit.

`refit` trains the frozen estimators again. It checks the selected row and
orientation against all archived 2-by-2 conditions, reproduces the six external
selectors, and evaluates the earlier independent local-calibration protocol.
Each completed fold has an atomic checkpoint. Repeating the same command
resumes completed folds. A timeout exits with code 124 and retains checkpoints.
The command normally takes a few minutes with the supplied feature cache.

See [VALIDATION.md](VALIDATION.md) for the checks executed on this release.

Add `--include-response-curve` to refit the two stronger clipping conditions
as well. Historical clipping parameters, model settings and seeds are fixed.
Theil--Sen may emit its original solver-limit warnings; the frozen solver
settings are preserved rather than tuned to remove them.

## Expected results

| Evaluation | HIST-SAFE | Vina-MO | Local Theil--Sen | Local Ridge |
| --- | ---: | ---: | ---: | ---: |
| Historical leave-one-task-out, 39 tasks | 28 | 20 | 23 | 21 |
| Earlier independent local-calibration test, 19 tasks | Not evaluated | 13 | 10 | 9 |
| External Davis, 9 tasks, recorded-point convention | 0 | 1 | 0 | 0 |

Counts are tasks with positive target **and** selectivity-margin gain. Global
Hist and Global Ridge, which omit the residual update, achieve 25/39 and 22/39
historically. The 19-task collection was later reused within the 39-task
historical analysis; those samples are not added as an independent extra set.

The crossed censoring experiment retains original or clipped learning and
evaluation records independently:

| Learning / evaluation | HIST-SAFE at 9.9% clipping | HIST-SAFE at 24.9% clipping |
| --- | ---: | ---: |
| Original / original | 28/39 | 28/39 |
| Clipped / original | 21/39 | 15/39 |
| Original / clipped | 28/39 | 27/39 |
| Clipped / clipped | 21/39 | 12/39 |

An improving direction remains available in 38/39 historical tasks. External
Davis evaluation contains 164/216 potency entries at the 10,000 nM ceiling.
Reporting-limit propagation and recorded-point evaluation are separate analyses.

## Contents

- `osrmolopt/`: numerical models, interventions, external selection,
  reporting-limit calculations and exploratory reliability controls.
- `data/historical_tasks.json`: 39 tasks, each with six calibration and six
  evaluation matched pairs; original potency slots, Vina differences and
  residue-type features are retained.
- `data/davis/`: separate selection and evaluation views, original benchmark
  matrix order, affinities, compound pairs, structures' identifiers and sequences.
- `data/records/`: archived features and complete experimental outcomes.
- `data/tables/`: long-form numerical exports for the crossed censoring
  experiment, external outcomes and endpoint bounds.
- `data/protocols/`: fixed splits, information access, structural exclusions,
  docking settings and frozen decisions.
- `data/source_records/`: source assay and compound identifiers, structures and
  record-level labels used to construct the tasks.
- `data/docking/`: compressed coordinate and pose assets, endpoint scores,
  seeds, residue-type count vectors and checksums.
- `data/provenance/`: source/published checksums and implementation extraction
  records. Portability changes affect paths, packaging and reporting metadata.
- `tools/`: integrity checks, result-table export and an explicit single-endpoint
  docking entry point.

Full all-residue distance maps are omitted from the compact endpoint cache;
the original poses and coordinates are supplied to recompute them. Stored
residue-type vectors and all features used by the prioritization experiments
are retained.

## Protocol details

The target gain is candidate minus reference log-potency. The margin gain is
the target gain minus the corresponding off-target gain. Both pair orientations
are evaluated, producing 12 possible directions from six evaluation pairs.
Strictly positive gains define dual improvement; zero gains do not pass.

Historical and external regression selectors maximize the smaller normalized
predicted gain, with a `1e-12` tie tolerance. The earlier independent test uses
its original prediction normalization and `1e-15` tolerance. The implementations
are kept separate. Local calibration uses only six within-task labeled pairs.
Historical donor training excludes the evaluated task. External selection reads
only its calibration records and zero-valued placeholders for evaluation gains.

HIST-SAFE uses two histogram-boosting regressors with 120 iterations, learning
rate 0.04, at most ten leaves, minimum leaf size 15, L2 regularization three and
seed 20261117. It adds half of the within-campaign median calibration residual.
The two-endpoint input has 15 features. Ridge and boosting controls use the
same available information. All numerical settings remain frozen.

## Docking assets and optional re-docking

The default reproduction uses the archived Vina features. Repeating docking
requires RDKit, Gemmi, Meeko preparation executables and a compatible Vina
binary. Frozen protocols retain the scoring options, box construction,
exhaustiveness, seeds, atom handling, chain normalization and tool hashes.
See [DOCKING.md](DOCKING.md) for the explicit single-endpoint workflow. This
entry point is sequential and never starts a GPU program. Docking binary and
preparation-tool differences can change scores; compare tool hashes before
treating a fresh docking run as the original feature calculation.

## Data sources and licensing

The historical collection derives from ChEMBL 20 selectivity-cliff records
([source dataset](https://doi.org/10.5281/zenodo.29983)). The external kinase
benchmark is Davis, distributed through the
[DeepDTA data repository](https://github.com/hkmztrk/DeepDTA/tree/master/data/davis).
Public structural coordinates originate from the Protein Data Bank.
Source identifiers and third-party attribution remain available for tracing
each measurement. Data terms are described in [DATA_LICENSES.md](DATA_LICENSES.md).
The implementation uses the [MIT license](LICENSE).

## Corrected cross-fitting and compound-isolation analysis

The original tables above describe the archived implementation. The corrected
[compound-disjoint cross-fitting reanalysis](experiments/crossfit_reanalysis/)
uses 21 strict tasks (seven Ki and 14 IC50), five prespecified donor-pool views,
and 381 CPU-only outer fits. It reports changed choices, gain means, paired
intervals and all negative results. In the strict view, original and
clipped-learning success are both 10/21, with 14/21 choices changed at the
fixed higher floor. These corrected aggregate results supersede the earlier
implementation's deterioration claim. Use the reanalysis-specific environment
and commands for that analysis.

The follow-up [decision-consequence and corrected Davis
reanalysis](experiments/decision_external_reanalysis/) shows that the 14 strict-view
changes comprise 13 changes of compound pair and one orientation reversal.
Six clipped-learning choices Pareto-dominate their original-learning choices,
five are dominated, and three form target--margin trade-offs. The paired mean
maximin-regret change is +0.144 (95% bootstrap interval -0.564 to +0.926), so
the evidence supports consequential reprioritization without a systematic
quality shift. Re-training on the 21 strict cross-fitted donors changes seven
of nine historical HIST-SAFE Davis choices. Eight corrected outcomes remain
unresolved under the 10,000 nM reporting limit and one is ruled out as a dual
improvement. Run `python verify.py` in that experiment directory for the
published checks, then `python runner.py audit` before any fresh refit.
