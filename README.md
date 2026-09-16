# zkfly

[Paper (PDF)](papers/zkfly-paper.pdf) · [LaTeX source](paper/main.tex)

`zkfly` builds a deterministic sparse representation of the official MaleCNS
v1.0 connectome. The source-of-truth data remains the downloaded Feather files;
the generated artifact is an explicitly versioned projection suitable for Rust
simulation and later fixed-point or proof-system work.

## Implementation stack

The proving path composes existing upstream systems instead of introducing a
new proof protocol:

| Layer | Upstream | Responsibility in `zkfly` |
| --- | --- | --- |
| recursive proof | [Microsoft Nova](https://github.com/microsoft/Nova), vendored as `nova-snark` 0.76 | Bellpepper R1CS synthesis, Nova folding, verification, IPA/HyperKZG |
| CUDA Rust | [NVIDIA Research cuda-oxide](https://github.com/NVlabs/cuda-oxide) | Rust GPU kernels, PTX generation, context/stream/device-buffer APIs |
| GPU MSM | [Blitzar](https://github.com/spaceandtimefdn/blitzar) through Nova | optional Linux BN254 multi-scalar multiplication |
| application relation | this workspace | canonical CSR topology, Poseidon roots, private weighted forward steps |

The vendored Nova copy adds a narrow arithmetic-provider hook and read-only MSM
telemetry. Microsoft Nova remains responsible for the proof protocol and
transcript; cuda-oxide executes selected arithmetic requested by that protocol.
See the [`cuda-nova` README](https://github.com/RyanKung/cuda-nova#readme) for the exact
host/device boundary, feature matrix, commands, and current measurement scope.

## Workspace layout

- `crates/zkfly-matrix`: deterministic MaleCNS-to-CSR artifact builder.
- `crates/zkfly-commitment`: canonical Poseidon topology encoding and root.
- `crates/zkfly-nova`: Nova topology and private weighted-forward relations.
- [`cuda-nova`](https://github.com/RyanKung/cuda-nova): revision-pinned Cargo Git dependency, CUDA Rust sidecar, and structured proof profiler.
- `crates/zkfly-bench`: CPU/CUDA execution benchmarks and capacity estimator.
- `vendor/nova-snark`: audited Nova 0.76 integration patch.
- `paper`: claim-bounded LaTeX paper and bibliography.

## Build the matrix

```sh
cargo run --release -p zkfly-matrix -- \
  --input-dir data/raw/male-cns-v1.0 \
  --output-dir artifacts/male-cns-v1.0
```

The destination must be new; this prevents an incomplete or stale build from
being silently overwritten.

The output stores the matrix as exact signed synapse counts rather than
normalized floating-point values:

```text
normalized_weight(row, edge) = signed_count(edge) / incoming_abs_sum(row)
```

Rows are postsynaptic neurons and columns are presynaptic neurons. Column
indices are sorted and unique within every row.

The recipe is compatible with the structural part of `fly.ai`'s
`flybrain/build.py`: superclass-annotated bodies are deduplicated by first
source occurrence, sorted by body ID, and edges are retained only when both
endpoints are selected; `gaba`, `glutamate`, and `histamine` labels make the
presynaptic count negative. The Rust artifact keeps the integer count and row
denominator exactly, so a consumer can choose its own fixed-point scale.

## Artifact files

- `row_offsets.u64le`: CSR row offsets.
- `column_indices.u32le`: presynaptic neuron indices.
- `signed_counts.i32le`: signed structural synapse counts.
- `incoming_abs_sums.u64le`: exact normalization denominator for every row.
- `neurons.jsonl`: stable index-to-MaleCNS-neuron mapping and annotations.
- `manifest.json`: recipe, source hashes, dimensions, and artifact hashes.

The checked-in source includes a tiny Arrow end-to-end fixture. Run
`cargo test --workspace --all-targets` for it, and use the release command
above to regenerate the full artifact from the downloaded source tables.

## Poseidon topology commitment

`zkfly-commitment` binds only the canonical CSR topology. It uses the
BN254/Circom x^5 Poseidon parameters at width 13, packs seven little-endian
`u32` values per field, and folds eleven fields at a time with the previous
accumulator. The current MaleCNS v1.0 root is recorded in the V100 receipts.
`verify_topology` is a deterministic host consistency check; it is not itself
a zero-knowledge proof. `commit_topology_with_trace` additionally streams each
Poseidon transition (`previous`, padded data fields, `next`) for a later proof
witness without retaining a second copy of the graph.

## Nova adapter

`zkfly-nova` implements the first Nova/R1CS step circuit with the official
`nova-snark` crate. It constrains the exact BN254/Circom Poseidon transition
used by the commitment and accepts transitions one at a time through
`TopologyNovaProver`. The streaming API keeps the accumulator dependency on the
host and does not commit to user weights. Each `prove_step` performs the full
official Bellpepper R1CS synthesis and Nova recursive fold; it is not a
transcript-only check.

The current proof binds the Poseidon transition and final accumulator.
`prove_for_root` and `verify_against_root` make the CSR-root claim explicit at
the application boundary. `CsrTopology` and `WeightedForwardProof` add the
bounded forward MVP: CSR row/column positions and the topology root are fixed
in the step-circuit shape, while weights and vectors remain private. Input and
output commitments are carried through the recursive state, so the verifier
can bind a claimed result without learning the vectors or weights. The MVP is
still a performance fixture, not the full MaleCNS executor.
`WeightedForwardParameters` can be prepared once and reused across proofs with
the same topology and circuit shape, keeping Nova's public-parameter setup
outside the proof hot path. Input and output commitments are folded in
Poseidon-rate chunks, so vectors can exceed one permutation block; the current
V100 numbers still use a small fixture.

The default build uses Nova's portable BN254 Pedersen/IPA primary commitment
engine. The official BN254 `HyperKZG` engine is available explicitly through the
`hyperkzg` feature; production setup must load a trusted, pruned Powers-of-Tau
directory with `setup_with_ptau_dir` (the loader selects files such as
`ppot_pruned_XX.ptau` according to the synthesized R1CS size). The ordinary
`setup`/`prove` methods intentionally return `HyperKzgSetupRequired` in a
non-test `hyperkzg` build, so a production proof cannot silently use random
parameters. Unit tests enable Nova's `test-utils` random setup only for local
regression coverage; those parameters are not production-safe.

## CUDA sidecar

`cuda-nova` is a standalone crate with a stable Cargo dependency boundary. The
workspace pins commit `8b69422` directly in `[workspace.dependencies]`; no Git
submodule or Cargo patch is required. It uses CUDA Rust through
NVIDIA Research's `cuda-oxide` and calls Microsoft Research's official Nova
implementation through `nova-snark`. It owns the cuda-oxide context, PTX
module, device buffers, and a one-lane-per-fold GPU preflight. The preflight
checks canonical indices, accumulator adjacency, data-length bounds, zero
padding, and the BN254/Circom Poseidon transition. It also registers a patched
Nova arithmetic backend: A/B/C CSR SpMV, NIFS cross-term evaluation, and
relaxed-witness vector folds are serialized to the device and returned to
Microsoft Nova. The Nova protocol, Bellpepper constraint-graph construction,
transcript/randomness control, and MSM provider remain host-orchestrated; MSM
has a CPU fallback and an optional official Blitzar path. This is complete Nova
R1CS synthesis and recursive folding with an explicit CUDA Rust arithmetic
boundary, not a replacement Nova protocol.

From a `cuda-nova` checkout on the V100, generate `cuda_nova.ptx` with
`cargo oxide`, then run the sidecar with both the toolkit path and `nvcc`
directory configured:

```sh
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_TOOLKIT_PATH=/usr/local/cuda
CUDA_OXIDE_TARGET=sm_70 cargo oxide run \
  --features cuda --arch sm_70 --bin cuda-nova -- --prove
```

For the official `HyperKZG` backend with Blitzar GPU MSM on supported Linux
hosts, enable both features and pass the directory that contains the trusted
pruned Powers-of-Tau files:

```sh
CUDA_OXIDE_TARGET=sm_70 cargo oxide run \
  --features cuda,hyperkzg,gpu-msm --arch sm_70 --bin cuda-nova -- \
  --prove --ptau-dir ./ptau_files
```

The `hyperkzg` feature changes Nova's primary commitment engine, while
`gpu-msm` selects the optional official Linux Blitzar MSM provider. Recursive
folding, the R1CS relation, Poseidon application commitments, and the CUDA
arithmetic boundary remain unchanged. No HyperNova implementation is used.
Without `--ptau-dir`, the production runner fails before setup by design.

The same feature exposes `HyperKzgVectorParameters` through a
`VectorCommitmentBackend` interface. It commits dense vectors, canonical sparse
rows, and row-wise sparse matrices with the trusted Nova HyperKZG key, and it
produces/verifies a batch of multilinear openings at one common point. Vectors
are zero-padded to the configured power-of-two capacity; opening witnesses are
checked against their commitments before proofs are generated. These standalone
commitment/opening helpers use Nova's official host evaluation engine today;
they do not change the CUDA arithmetic dispatch used by recursive Nova.

Omit `gpu-msm` only when the normal CPU MSM provider is intended.

The default feature-free build remains portable and CPU-backed. The existing
`zkfly-bench` cuda-oxide kernels continue to measure sparse forward execution.
The smoke runner prints upload, repeated-kernel, download, and end-to-end
microsecond measurements for its fixed two-step transcript, then runs an
official Nova topology proof followed by a two-step private weighted-forward
proof. It reports the forward circuit's constraint/variable counts and the
successful CUDA SpMV/fold/cross-term launches. Bellpepper synthesis and
recursive orchestration remain host-side; `gpu-msm` routes Nova's supported
BN254 MSM calls through the official Blitzar GPU provider.

To measure reusable weighted-forward proving at the audited recursive lengths,
keep the same environment and pass `--profile-forward` instead of `--prove`.
The standard profile performs one `PublicParams::setup`, then proves and
verifies 1, 8, 64, and 1,024-step prefixes with the same parameters. Use
`--profile-forward=smoke` for the previous bounded 1, 2, 8, and 16-step curve,
or `--profile-steps=1,4,32` for a custom positive, increasing sequence. Every
custom length is bounded at 1,024 steps before witness allocation. Every mode
also runs a 12-neuron single-step case whose commitment spans two Poseidon
chunks.

`--profile-json PATH` writes a versioned, create-new receipt containing setup,
prover initialization, recursive folding, finalization, verification,
constraint/variable, backend-feature, CUDA cache/allocation/synchronization,
and Blitzar MSM measurements. CUDA and MSM durations are accumulated provider
call time and can overlap under host parallelism; they must not be added to
wall time. Without the option the JSON is emitted to standard output. Pass
`--profile-revision REVISION` to record the exact `cuda-nova` source commit.
The runner refuses to overwrite an existing receipt.

The resident-workspace V100 result is recorded in
`receipts/cuda-nova-forward-profile-optimized-standard-2026-09-17.json`.
With HyperKZG, GPU MSM, and ComfyUI stopped, the 1,024-step proof took
`373.514 s`, down from the matching pre-optimization `612.609 s` receipt
(`1.64x`, `39.03%`). The 1,024-step proof plus verification performed the same
12,285 logical CUDA arithmetic operations; device-buffer allocations fell from
an inferred 53,241 to 22,515 and stream synchronizations from 36,855 to 12,285.

The follow-up fused A/B/C result is recorded in
`receipts/cuda-nova-forward-profile-fused-spmv-standard-2026-09-17.json`.
It vertically combines the three exact CSR matrices that share each Nova
dense vector, preserving 6,138 logical SpMV products while executing 2,046
physical SpMV batches during the 1,024-step proof. Proof time fell again to
`329.535 s` (`1.13x`, `11.77%` versus the resident-workspace result; `1.86x`,
`46.21%` versus the original exclusive baseline). Proof-phase SpMV provider
time fell `35.79%`, device-buffer allocations fell from 22,506 to 18,414, and
stream synchronizations fell from 12,276 to 8,184.

For paper step-count estimates against the built artifact, use:

```sh
cargo run --release -p zkfly-bench -- \
  --estimate-only --estimate-ticks 1000 --estimate-layers 1 \
  --estimate-tiles 1 --include-topology
```

This reports exact topology-fold steps and the parameterized forward-step
model, plus chunk counts, witness payload size, and conservative R1CS
constraint lower bounds for the current chunked circuit. For one full
MaleCNS forward step with topology registration included, the estimator reports
`30,310` input/output Poseidon permutations, `91,643,578` forward constraints
as a lower bound, and `818,655,266` total constraints as a lower bound. These
are capacity figures, not wall-clock or full-scale feasibility claims.

## License

This project is licensed under the GNU General Public License, version 3.0
only. See [LICENSE](LICENSE) for the complete text.
