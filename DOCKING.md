# Optional feature regeneration

The tested default reproduction reuses the archived Vina score differences.
The compressed structures, original poses, endpoint scores and frozen docking
protocols are included for independent feature inspection and regeneration.

The implementation in `osrmolopt/docking.py` materializes the archived docking
runner. It retains parent-fragment selection, ETKDGv3 embedding, deterministic
compound seeds, 200-step MMFF/UFF optimization, chain normalization, uniform
hydrogen removal, HIE histidine templates, coordinate-based disulfide templates,
native-ligand box construction, Vina scoring and residue contacts.

Install RDKit, Gemmi, Meeko and a compatible Vina binary in a separate optional
environment. The required preparation commands are `mk_prepare_receptor.py`
and `mk_prepare_ligand.py`. Tool SHA256 values are recorded in the protocols.
Use `--require-tool-hashes` to reject any executable that differs from the
archived executable. This release includes software source and input data,
not those external executables.

Run exactly one endpoint explicitly:

```sh
python tools/redock_one.py --protocol davis_docking \
  --row-id Davis:AKT1:AKT2:00 --compound-index 1 --target-index 1 \
  --vina vina --prepare-receptor mk_prepare_receptor.py \
  --prepare-ligand mk_prepare_ligand.py
```

Each external command has a timeout. Docking uses one CPU thread and never
uses CUDA. Successful receptor, ligand and endpoint preparation have separate
atomic status files and are reused on restart. Inputs are read from the fixed
protocol; new outputs and unpacked structures are written under `artifacts/`.

Before any batch extension, repeat the single-endpoint resource, interruption
and resume checks in the destination environment. The package's `smoke` command
validates the numerical refitting launcher; it does not claim to validate an
external docking installation.

Endpoint cache records preserve scores, seeds, residue-type count vectors and
pose hashes. Full per-residue distance matrices can be recomputed using
`docking.contacts()` from a prepared receptor and its archived pose.
