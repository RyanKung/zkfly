//! Nova proving adapters for the canonical zkfly topology transcript.
//!
//! The crate deliberately keeps the topology transcript independent from the
//! proving library. [`TopologyStepCircuit`] consumes the same fixed-width
//! transitions emitted by `zkfly-commitment`; [`TopologyNovaProof`] is the
//! bounded Nova integration used for the first performance measurements.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod forward;
#[cfg(feature = "hyperkzg")]
mod hyperkzg;
mod poseidon;
mod proof;

use thiserror::Error;
use zkfly_commitment::Commitment;

pub use forward::{
    CsrTopology, WeightedForwardParameters, WeightedForwardProof, WeightedForwardProver,
    WeightedForwardWitness,
};
#[cfg(feature = "hyperkzg")]
pub use hyperkzg::{
    HyperKzgBatchOpening, HyperKzgCommitment, HyperKzgError, HyperKzgSparseMatrixCommitment,
    HyperKzgVectorCommitment, HyperKzgVectorParameters, SparseMatrixRow, VectorCommitmentBackend,
};
pub use poseidon::TopologyStepCircuit;
pub use proof::{TopologyNovaProof, TopologyNovaProver};

/// Errors raised while constructing or verifying a topology step circuit.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TopologyNovaError {
    /// The trace contains no transitions, so no Nova base case exists.
    #[error("the topology trace must contain at least one step")]
    EmptyTrace,
    /// A trace index is not the canonical zero-based position.
    #[error("trace index {actual} is not the canonical position {expected}")]
    TraceIndex {
        /// The index found in the trace.
        actual: u64,
        /// The index required by the canonical encoding.
        expected: u64,
    },
    /// A trace transition does not connect to the previous accumulator.
    #[error("trace transition {index} does not connect to the previous accumulator")]
    TraceDiscontinuity {
        /// The transition that failed the chaining predicate.
        index: u64,
    },
    /// A trace field is not a canonical BN254 scalar encoding.
    #[error("trace field {field} is not a canonical BN254 scalar")]
    InvalidField {
        /// The field position being decoded.
        field: &'static str,
    },
    /// The trace's padding marker is inconsistent with its data bytes.
    #[error("trace transition {index} has non-zero padding")]
    NonZeroPadding {
        /// The transition containing invalid padding.
        index: u64,
    },
    /// Poseidon parameters could not be converted to the Nova field.
    #[error("Poseidon parameters could not be converted: {message}")]
    PoseidonParameters {
        /// The parameter conversion detail.
        message: String,
    },
    /// Nova rejected a circuit, witness, or recursive proof operation.
    #[error("Nova operation failed: {0}")]
    Nova(#[from] nova_snark::errors::NovaError),
    /// Nova returned no primary output for a supposedly valid proof.
    #[error("Nova returned no primary output")]
    MissingOutput,
    /// The trace's final root differs from the application-level claim.
    #[error("topology root mismatch: expected {expected}, produced {actual}")]
    RootMismatch {
        /// Root supplied as the public application-level claim.
        expected: Commitment,
        /// Root produced by the private trace and verified by Nova.
        actual: Commitment,
    },
    /// The verified primary output differs from the trace's final accumulator.
    #[error("verified output does not equal the trace final accumulator")]
    InvalidFinalState,
    /// The number of streamed transitions exceeded the host index range.
    #[error("the topology step count overflowed the host index range")]
    StepCountOverflow,
    /// The topology commitment could not be computed from the CSR input.
    #[error(transparent)]
    Commitment(#[from] zkfly_commitment::CommitmentError),
    /// The weighted-forward input vector has the wrong neuron count.
    #[error("weighted-forward input length {actual} does not equal neuron count {expected}")]
    ForwardInputLength {
        /// Number of supplied input values.
        actual: usize,
        /// Number of values required by the fixed topology.
        expected: usize,
    },
    /// The weighted-forward weight vector has the wrong edge count.
    #[error("weighted-forward weight length {actual} does not equal edge count {expected}")]
    ForwardWeightLength {
        /// Number of supplied weights.
        actual: usize,
        /// Number of weights required by the fixed topology.
        expected: usize,
    },
    /// `HyperKZG` production setup requires a trusted Powers-of-Tau directory.
    #[error("HyperKZG requires trusted Powers-of-Tau parameters; use setup_with_ptau_dir")]
    HyperKzgSetupRequired,
    /// A recursive forward step does not consume the previous output hash.
    #[error("weighted-forward step {index} input commitment does not match the previous output")]
    ForwardInputChain {
        /// Zero-based forward step index.
        index: usize,
    },
    /// A canonical `u32` topology index cannot be represented by this host.
    #[error("topology index does not fit the host representation: {target}")]
    TopologyIndexOverflow {
        /// Representation that overflowed.
        target: &'static str,
    },
}
