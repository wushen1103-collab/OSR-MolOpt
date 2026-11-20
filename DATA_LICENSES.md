# Data attribution and terms

The MIT license in this repository covers the original implementation. It does
not replace the terms attached to source databases or third-party materials.

## Historical bioactivity records

The source is *Compounds with multi-target activities forming target cliffs and
selectivity cliffs*, DOI [10.5281/zenodo.29983](https://doi.org/10.5281/zenodo.29983).
The source describes high-confidence Ki and IC50 records assembled from
ChEMBL release 20. ChEMBL data are provided by EMBL-EBI under
[Creative Commons Attribution-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/).
See the [official ChEMBL licensing information](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions).
ChEMBL-derived data and their adaptations retain these attribution/share-alike
terms. This release reorganizes the records, materializes fixed task views and
removes execution-location metadata while retaining the scientific values.

## Davis benchmark

The source benchmark is the Davis kinase-inhibitor affinity panel, with the
archived matrix and identifier order obtained from
[DeepDTA's Davis directory](https://github.com/hkmztrk/DeepDTA/tree/master/data/davis).
The affinity matrix is converted from the benchmark's numeric pickle into a
non-object NumPy archive. The original row/column identifier order is stored
explicitly. Source-file checksums are retained. Database records and third-party
benchmark files remain subject to their source terms; no replacement MIT data
license is asserted for them.

## Structural coordinates

Structural identifiers are retained in the fixed input protocols. Coordinate
assets originate from the Protein Data Bank. PDB files retain coordinate and
connectivity records while nonessential header metadata is removed. Coordinates
are not rounded, rescaled or aligned by the release builder. Predicted poses,
scores and residue-type features are the archived computational outputs.

## Dependency software

NumPy, SciPy, scikit-learn, RDKit, Gemmi, Meeko and Vina are third-party software
with their own licenses. Their binaries, environments and source trees are not
redistributed here. The original preparation/scoring algorithms are called via
explicit executable paths when optional docking is requested.
