# zkfly data provenance

## Figshare v4 dataset

- DOI: https://doi.org/10.25378/janelia.25309105
- Landing page: https://janelia.figshare.com/articles/dataset/MuJoCo_fruit_fly_body_model_datasets_supporting_Whole-body_simulation_of_realistic_fruit_fly_locomotion_with_deep_reinforcement_learning_/25309105/4
- Version: 4
- Posted: 2025-05-07 18:19, as shown on the Figshare landing page
- Local directory: `data/raw/figshare-v4/`
- Manifest: `data/metadata/figshare-v4-manifest.tsv`

All seven Figshare v4 files were downloaded and validated with `unzip -t`.
`flight-controller-reuse-checkpoints.zip` was retrieved through the official
Figshare API download URL after the Janelia subdomain endpoint returned 403.

## Flybody code and MuJoCo model assets

- Repository: https://github.com/TuragaLab/flybody
- Local directory: `data/raw/flybody/`
- Commit: `d015e9bfe441bd90ae431bac24c55cb74bdbce26`
- Commit date: 2025-07-30T18:49:01-04:00
- Commit subject: `Update paper reference.`
- Asset files under `flybody/fruitfly/assets`: 87

The cloned repository contains the MuJoCo fruit fly model assets and the
official `flybody/download_data.py` script whose Figshare URLs match the file
ids in the manifest for trained policies, flight imitation, walking imitation,
and controller-reuse checkpoints.

## MaleCNS v1.0 connectome core dataset

- Project page: https://male-cns.janelia.org/
- Official downloads: https://male-cns.janelia.org/download/
- Version: MaleCNS v1.0
- Local directory: `data/raw/male-cns-v1.0/`
- Manifest: `data/metadata/male-cns-v1.0-manifest.tsv`

The local directory contains the four core inputs used by `fly.ai`: the full
flat neuron-to-neuron connection table, body annotations, neurotransmitter
predictions, and optic-column type assignments. It intentionally excludes the
much larger EM volumes, per-synapse point and partner tables, skeleton archive,
and Neo4j database.

The three files hosted in the official MaleCNS Google Cloud Storage bucket
match the sizes and MD5 hashes advertised by the server. All three Feather
tables were opened successfully: the connection table has 151,856,684 rows,
the annotation table has 211,577 rows, and the neurotransmitter table has
1,835,518 rows. The pinned optic-column XLSX passed ZIP container validation.

## Rust CSR projection

The Rust builder in `crates/zkfly-matrix` reads the three Feather tables in
streaming Arrow batches. It reproduces the structural `fly.ai` recipe: retain
non-empty superclass annotations, keep the first annotation per body ID, sort
body IDs, keep connections whose two endpoints are selected, and negate counts
for case-insensitive `gaba`, `glutamate`, or `histamine` consensus labels. The
generated matrix uses rows for postsynaptic neurons and columns for
presynaptic neurons. Every row is canonicalized by increasing column index and
validated for duplicate pairs and denominator consistency.

The full build completed on 2026-09-15 with 166,700 neurons and 25,582,938
retained connections. It scanned all 151,856,684 source connection rows twice;
the peak resident set was approximately 543 MB. The resulting files are in
`artifacts/male-cns-v1.0/`; `manifest.json` records source and artifact
SHA-256 digests. A second full build produced byte-identical binary and JSONL
payloads, and an independent verifier confirmed all CSR invariants.
