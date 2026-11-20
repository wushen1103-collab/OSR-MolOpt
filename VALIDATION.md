# Release validation

The numerical release was tested with Python 3.13.5, NumPy 2.1.3,
SciPy 1.15.3 and scikit-learn 1.6.1. Machine-readable results are in
`data/provenance/release_validation.json`.

The following commands were run on the release inputs:

```sh
python -m osrmolopt verify --output outputs/final_validation
python -m osrmolopt smoke --output outputs/final_validation
python -m osrmolopt refit --output outputs/final_validation --timeout 1800
python tools/export_tables.py
python tools/check_integrity.py
```

Verification reconstructs 624 crossed-censoring outcomes, 54 external frozen
choices and 108 external endpoint intervals. Fresh fitting starts from an
empty checkpoint directory and reproduces every archived row and orientation
in the two clipping conditions. It also reproduces all six external selectors
and the 19-task independent local-calibration comparison.

The resource check exercises an intentional failure before checkpoint commit,
recovery, completed-checkpoint reuse and timeout termination. Numerical jobs
use one CPU thread and no GPU. Core numerical modules import successfully,
and the exporter produces three result tables, also supplied in `data/tables/`.

Optional external docking executables were not installed or rerun during
release validation. Docking inputs, poses, scores and preparation source are
included. The two stronger auxiliary response-curve conditions are archived
and have an optional refitting switch; they were not part of this fresh refit.
