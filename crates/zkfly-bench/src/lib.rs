//! A deterministic CSR forward pass with an optional CUDA implementation.
//!
//! The crate deliberately measures the graph computation only. It does not
//! claim that a measured output is a zero-knowledge proof, nor does it impose
//! a training rule on the user-supplied weights. The fixed relation for this
//! first benchmark is `y[row] = sum_e weights[e] * x[column[e]]` over the
//! `MaleCNS` CSR topology.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod artifact;
mod benchmark;
#[cfg(all(feature = "cuda", target_os = "linux"))]
mod cuda_runtime;
mod error;
mod estimate;

pub use artifact::{
    ArtifactDimensions, ArtifactSummary, CsrArtifact, load_artifact, load_artifact_dimensions,
};
pub use benchmark::{BenchmarkConfig, BenchmarkReport, run_benchmark};
pub use error::BenchError;
pub use estimate::{
    POSEIDON_PERMUTATION_CONSTRAINT_LOWER_BOUND, ProofStepEstimate, estimate_proof_steps,
};
pub use zkfly_commitment::{
    Commitment, CommitmentError, PACKED_U32_PER_FIELD, POSEIDON_INPUTS, POSEIDON_RATE,
    TOPOLOGY_DOMAIN, TOPOLOGY_SCHEME, commit_topology, topology_field_count,
    topology_fold_step_count, verify_topology,
};

/// Returns whether this build contains the Linux CUDA backend.
#[must_use]
pub const fn cuda_backend_enabled() -> bool {
    cfg!(all(feature = "cuda", target_os = "linux"))
}
