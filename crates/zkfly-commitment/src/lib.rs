//! Canonical commitments for the fixed `zkfly` graph relation.
//!
//! The topology commitment uses the BN254 scalar field and the Circom-style
//! x^5 Poseidon parameters supplied by `light-poseidon`. It binds the neuron
//! count and CSR structure while intentionally excluding edge weights, input
//! values, and output values. The encoding is versioned, packs seven
//! little-endian `u32` values into each field element, and folds fixed-width
//! Poseidon calls so a verifier can reproduce the same root in a circuit.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use light_poseidon::{Poseidon, PoseidonError, PoseidonHasher};
use serde::Serialize;
use serde::ser::Serializer;
use thiserror::Error;

/// Domain separator for the version-one topology commitment encoding.
pub const TOPOLOGY_DOMAIN: &[u8] = b"zkfly/poseidon/topology/v1";

/// Stable algorithm identifier emitted with benchmark artifact summaries.
pub const TOPOLOGY_SCHEME: &str = "poseidon-bn254-circom-x5-t13-fold-v1";

/// Number of bytes in a canonical BN254 scalar serialization.
pub const COMMITMENT_BYTES: usize = 32;

/// Number of values accepted by the selected width-thirteen Poseidon call.
pub const POSEIDON_INPUTS: usize = 12;

/// Number of data fields carried by one fold call after its accumulator.
pub const POSEIDON_RATE: usize = POSEIDON_INPUTS - 1;

/// Number of 32-bit topology values packed into one field element.
pub const PACKED_U32_PER_FIELD: usize = 7;

/// Number of metadata fields in the canonical topology stream.
const TOPOLOGY_METADATA_FIELDS: usize = 4;

/// A 32-byte commitment digest rendered as lowercase hexadecimal in JSON.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Commitment([u8; COMMITMENT_BYTES]);

/// One fixed-width Poseidon transition emitted for a topology witness.
///
/// `previous` and `next` are canonical little-endian BN254 field elements.
/// `data` always has [`POSEIDON_RATE`] slots; only the first `data_len` slots
/// are part of the canonical stream and the remaining slots are zero padding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyFoldStep {
    /// Zero-based fold transition number.
    pub index: u64,
    /// Accumulator before this Poseidon call.
    pub previous: [u8; COMMITMENT_BYTES],
    /// Canonical data fields and zero padding supplied to this call.
    pub data: [[u8; COMMITMENT_BYTES]; POSEIDON_RATE],
    /// Number of non-padding data fields in `data`.
    pub data_len: u8,
    /// Accumulator after this Poseidon call.
    pub next: [u8; COMMITMENT_BYTES],
}

impl Commitment {
    /// Constructs a commitment from an already decoded 32-byte digest.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; COMMITMENT_BYTES]) -> Self {
        Self(bytes)
    }

    /// Returns the raw digest bytes in canonical little-endian order.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; COMMITMENT_BYTES] {
        &self.0
    }

    /// Returns the lowercase hexadecimal digest used in reports and receipts.
    #[must_use]
    pub fn to_hex(self) -> String {
        hex::encode(self.0)
    }
}

impl Serialize for Commitment {
    /// Serializes the digest as a stable lowercase hexadecimal string.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl std::fmt::Display for Commitment {
    /// Formats the digest as lowercase hexadecimal.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

/// Errors proving that a CSR input is not a canonical topology.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CommitmentError {
    /// The row-offset vector does not have one boundary per neuron plus a terminal boundary.
    #[error("topology row offsets do not match neuron count: {proposition}")]
    RowShape {
        /// Failed shape proposition.
        proposition: &'static str,
    },
    /// A row offset violates monotonicity or edge-count bounds.
    #[error("topology row offsets are invalid: {proposition}")]
    RowBounds {
        /// Failed boundary proposition.
        proposition: &'static str,
    },
    /// A column index names a neuron outside the committed vertex set.
    #[error("topology column index is invalid: {proposition}")]
    ColumnBounds {
        /// Failed column proposition.
        proposition: &'static str,
    },
    /// A row's columns are not in the canonical sorted-unique order.
    #[error("topology row columns are not canonical: {proposition}")]
    ColumnOrder {
        /// Failed ordering proposition.
        proposition: &'static str,
    },
    /// A host integer cannot be represented in the canonical length field.
    #[error("topology length does not fit canonical encoding: {target}")]
    LengthOverflow {
        /// Canonical field that overflowed.
        target: &'static str,
    },
    /// The fixed Poseidon parameter set could not be initialized.
    #[error("Poseidon parameters could not be initialized: {message}")]
    PoseidonParameters {
        /// Library-provided parameter error.
        message: String,
    },
    /// A fixed-width Poseidon fold call failed unexpectedly.
    #[error("Poseidon fold failed: {message}")]
    PoseidonHash {
        /// Library-provided hashing error.
        message: String,
    },
    /// The field output did not have the expected canonical byte width.
    #[error("Poseidon output does not have the canonical 32-byte width")]
    PoseidonOutput,
}

/// Computes the topology-only commitment for a validated CSR graph.
///
/// The canonical field sequence is, in order, the encoded field count, neuron
/// count, row-offset count, edge count, packed row offsets, and packed column
/// indices. Seven little-endian `u32` values occupy one field element, leaving
/// 32 high bits zero below the BN254 modulus. Every eleven sequence fields are
/// folded together with the previous accumulator by a width-thirteen
/// Poseidon call. The fixed domain tag is configured in every call.
/// Weights are not read.
///
/// # Errors
///
/// Returns [`CommitmentError`] when the supplied CSR vectors do not describe a
/// complete sorted-unique topology, lengths overflow the canonical encoding,
/// or the fixed Poseidon library reports an error.
pub fn commit_topology(
    neuron_count: usize,
    row_offsets: &[u32],
    column_indices: &[u32],
) -> Result<Commitment, CommitmentError> {
    commit_topology_with_trace(neuron_count, row_offsets, column_indices, |_| {})
}

/// Estimates the number of canonical fields emitted for a CSR topology.
///
/// This count depends only on the neuron and edge lengths. It is intended for
/// capacity planning after the CSR has been validated; it does not validate a
/// topology or compute a Poseidon digest.
///
/// # Errors
///
/// Returns [`CommitmentError::LengthOverflow`] when a host length cannot be
/// represented by the checked arithmetic used by the estimator.
pub fn topology_field_count(
    neuron_count: usize,
    edge_count: usize,
) -> Result<usize, CommitmentError> {
    let row_count = neuron_count
        .checked_add(1)
        .ok_or(CommitmentError::LengthOverflow {
            target: "topology row-offset count",
        })?;
    let row_fields = packed_value_count(row_count, "topology row-offset field count")?;
    let column_fields = packed_value_count(edge_count, "topology column field count")?;
    TOPOLOGY_METADATA_FIELDS
        .checked_add(row_fields)
        .and_then(|value| value.checked_add(column_fields))
        .ok_or(CommitmentError::LengthOverflow {
            target: "topology field count",
        })
}

/// Estimates the number of fixed-width Poseidon fold steps for a CSR topology.
///
/// The result is the exact transition count produced by
/// [`commit_topology_with_trace`] for a validated topology with these lengths.
/// It is a paper estimate only: it does not measure Nova synthesis, recursive
/// folding, memory use, or prover wall-clock time.
///
/// # Errors
///
/// Returns [`CommitmentError::LengthOverflow`] when the field-count arithmetic
/// overflows the host representation.
pub fn topology_fold_step_count(
    neuron_count: usize,
    edge_count: usize,
) -> Result<usize, CommitmentError> {
    let field_count = topology_field_count(neuron_count, edge_count)?;
    ceil_div(field_count, POSEIDON_RATE, "topology fold step count")
}

/// Computes a topology commitment while streaming its Poseidon fold witness.
///
/// The callback receives one transition after each fixed-width Poseidon call.
/// It can persist these transitions as a prover witness without retaining the
/// complete graph or a second copy of the field stream. Callback order is
/// canonical and starts at index zero. The returned commitment is identical to
/// [`commit_topology`] when the callback has no side effects.
///
/// # Errors
///
/// Returns [`CommitmentError`] when the supplied CSR vectors do not describe a
/// complete sorted-unique topology, lengths overflow the canonical encoding,
/// or the fixed Poseidon library reports an error.
pub fn commit_topology_with_trace<F>(
    neuron_count: usize,
    row_offsets: &[u32],
    column_indices: &[u32],
    mut on_step: F,
) -> Result<Commitment, CommitmentError>
where
    F: FnMut(TopologyFoldStep),
{
    validate_topology(neuron_count, row_offsets, column_indices)?;
    let neuron_count = canonical_length(neuron_count, "neuron count")?;
    let offset_count = canonical_length(row_offsets.len(), "row-offset count")?;
    let edge_count = canonical_length(column_indices.len(), "edge count")?;
    let row_field_count = packed_field_count(row_offsets.len(), "row-offset field count")?;
    let column_field_count = packed_field_count(column_indices.len(), "column field count")?;
    let field_count = 4_u64
        .checked_add(row_field_count)
        .and_then(|value| value.checked_add(column_field_count))
        .ok_or(CommitmentError::LengthOverflow {
            target: "encoded field count",
        })?;

    let mut fold = PoseidonFold::new()?;
    fold.absorb(Fr::from(field_count), &mut on_step)?;
    fold.absorb(Fr::from(neuron_count), &mut on_step)?;
    fold.absorb(Fr::from(offset_count), &mut on_step)?;
    fold.absorb(Fr::from(edge_count), &mut on_step)?;
    for chunk in row_offsets.chunks(PACKED_U32_PER_FIELD) {
        fold.absorb(pack_u32_chunk(chunk), &mut on_step)?;
    }
    for chunk in column_indices.chunks(PACKED_U32_PER_FIELD) {
        fold.absorb(pack_u32_chunk(chunk), &mut on_step)?;
    }
    fold.finish(&mut on_step)
}

/// Recomputes a topology commitment and compares it with a claimed root.
///
/// This is a deterministic host-side consistency check, not a zero-knowledge
/// verifier. A proof circuit can use the same canonical field sequence and
/// Poseidon parameters while exposing the claimed commitment as a public input.
///
/// # Errors
///
/// Returns the same validation or Poseidon errors as [`commit_topology`].
pub fn verify_topology(
    claimed: Commitment,
    neuron_count: usize,
    row_offsets: &[u32],
    column_indices: &[u32],
) -> Result<bool, CommitmentError> {
    Ok(commit_topology(neuron_count, row_offsets, column_indices)? == claimed)
}

/// Checks the canonical CSR propositions without hashing any bytes.
fn validate_topology(
    neuron_count: usize,
    row_offsets: &[u32],
    column_indices: &[u32],
) -> Result<(), CommitmentError> {
    let expected_offsets = neuron_count
        .checked_add(1)
        .ok_or(CommitmentError::RowShape {
            proposition: "neuron count has a terminal boundary",
        })?;
    if row_offsets.len() != expected_offsets {
        return Err(CommitmentError::RowShape {
            proposition: "row_offsets.len() == neuron_count + 1",
        });
    }
    if row_offsets.first().copied() != Some(0) {
        return Err(CommitmentError::RowBounds {
            proposition: "the first row offset is zero",
        });
    }
    let edge_count =
        u32::try_from(column_indices.len()).map_err(|_| CommitmentError::LengthOverflow {
            target: "u32 edge offset",
        })?;
    if row_offsets.last().copied() != Some(edge_count) {
        return Err(CommitmentError::RowBounds {
            proposition: "the terminal row offset equals the edge count",
        });
    }
    let mut previous = 0_u32;
    for window in row_offsets.windows(2) {
        let start = *window.first().ok_or(CommitmentError::RowBounds {
            proposition: "each row has a start boundary",
        })?;
        let end = *window.get(1).ok_or(CommitmentError::RowBounds {
            proposition: "each row has an end boundary",
        })?;
        if end < start {
            return Err(CommitmentError::RowBounds {
                proposition: "row offsets are monotonically nondecreasing",
            });
        }
        if start < previous {
            return Err(CommitmentError::RowBounds {
                proposition: "row boundaries preserve global order",
            });
        }
        previous = end;
        let start_index = usize::try_from(start).map_err(|_| CommitmentError::RowBounds {
            proposition: "row start fits host index type",
        })?;
        let end_index = usize::try_from(end).map_err(|_| CommitmentError::RowBounds {
            proposition: "row end fits host index type",
        })?;
        let row_columns =
            column_indices
                .get(start_index..end_index)
                .ok_or(CommitmentError::RowBounds {
                    proposition: "row ranges fit the column vector",
                })?;
        let mut prior_column = None;
        for column in row_columns {
            if usize::try_from(*column).map_or(true, |value| value >= neuron_count) {
                return Err(CommitmentError::ColumnBounds {
                    proposition: "every column is below neuron_count",
                });
            }
            if prior_column.is_some_and(|prior| *column <= prior) {
                return Err(CommitmentError::ColumnOrder {
                    proposition: "columns are sorted and unique within each row",
                });
            }
            prior_column = Some(*column);
        }
    }
    Ok(())
}

/// Counts packed field elements without rounding through an overflowing sum.
fn packed_value_count(value: usize, target: &'static str) -> Result<usize, CommitmentError> {
    let adjusted = value
        .checked_add(PACKED_U32_PER_FIELD - 1)
        .ok_or(CommitmentError::LengthOverflow { target })?;
    Ok(adjusted / PACKED_U32_PER_FIELD)
}

/// Computes a checked ceiling division for a non-zero divisor.
fn ceil_div(value: usize, divisor: usize, target: &'static str) -> Result<usize, CommitmentError> {
    let adjusted = value
        .checked_add(divisor - 1)
        .ok_or(CommitmentError::LengthOverflow { target })?;
    Ok(adjusted / divisor)
}

/// Converts a host length to the canonical 64-bit length field.
fn canonical_length(value: usize, target: &'static str) -> Result<u64, CommitmentError> {
    u64::try_from(value).map_err(|_| CommitmentError::LengthOverflow { target })
}

/// Computes the number of packed field elements for one `u32` sequence.
fn packed_field_count(value_count: usize, target: &'static str) -> Result<u64, CommitmentError> {
    let rounded = value_count
        .checked_add(PACKED_U32_PER_FIELD - 1)
        .ok_or(CommitmentError::LengthOverflow { target })?;
    let fields = rounded / PACKED_U32_PER_FIELD;
    canonical_length(fields, target)
}

/// Packs up to seven little-endian `u32` values into one BN254 field element.
fn pack_u32_chunk(values: &[u32]) -> Fr {
    let mut bytes = [0_u8; COMMITMENT_BYTES];
    let mut slots = bytes.chunks_mut(std::mem::size_of::<u32>());
    for value in values {
        let Some(slot) = slots.next() else {
            break;
        };
        slot.copy_from_slice(&value.to_le_bytes());
    }
    Fr::from_le_bytes_mod_order(&bytes)
}

/// Stateful fixed-width Poseidon fold used by the canonical topology encoder.
struct PoseidonFold {
    /// Poseidon hasher with the topology domain tag configured.
    hasher: Poseidon<Fr>,
    /// Digest of all complete blocks seen so far.
    accumulator: Fr,
    /// Uncompressed sequence fields waiting for the next call.
    block: Vec<Fr>,
    /// Reused fixed-width input buffer to avoid per-block allocations.
    inputs: Vec<Fr>,
    /// Number of transitions already emitted to the witness callback.
    step_index: usize,
}

impl PoseidonFold {
    /// Initializes the fixed BN254 Circom parameter set.
    fn new() -> Result<Self, CommitmentError> {
        let domain_tag = Fr::from_le_bytes_mod_order(TOPOLOGY_DOMAIN);
        let hasher = Poseidon::<Fr>::with_domain_tag_circom(POSEIDON_INPUTS, domain_tag)
            .map_err(|error| poseidon_parameters_error(&error))?;
        Ok(Self {
            hasher,
            accumulator: Fr::from(0_u64),
            block: Vec::with_capacity(POSEIDON_RATE),
            inputs: Vec::with_capacity(POSEIDON_INPUTS),
            step_index: 0,
        })
    }

    /// Adds one canonical field and compresses a full block immediately.
    fn absorb<F>(&mut self, value: Fr, on_step: &mut F) -> Result<(), CommitmentError>
    where
        F: FnMut(TopologyFoldStep),
    {
        self.block.push(value);
        if self.block.len() == POSEIDON_RATE {
            self.compress(on_step)?;
        }
        Ok(())
    }

    /// Hashes the accumulator and one padded data block with Poseidon.
    fn compress<F>(&mut self, on_step: &mut F) -> Result<(), CommitmentError>
    where
        F: FnMut(TopologyFoldStep),
    {
        let previous = field_to_bytes(self.accumulator)?;
        let data_len =
            u8::try_from(self.block.len()).map_err(|_| CommitmentError::LengthOverflow {
                target: "Poseidon fold data length",
            })?;
        let mut data = [[0_u8; COMMITMENT_BYTES]; POSEIDON_RATE];
        let mut slots = data.iter_mut();
        for field in self.block.iter().copied() {
            let Some(slot) = slots.next() else {
                break;
            };
            *slot = field_to_bytes(field)?;
        }
        self.inputs.clear();
        self.inputs.push(self.accumulator);
        self.inputs.append(&mut self.block);
        self.inputs.resize(POSEIDON_INPUTS, Fr::from(0_u64));
        let next = self
            .hasher
            .hash(&self.inputs)
            .map_err(|error| poseidon_hash_error(&error))?;
        let next_bytes = field_to_bytes(next)?;
        let index =
            u64::try_from(self.step_index).map_err(|_| CommitmentError::LengthOverflow {
                target: "Poseidon fold step index",
            })?;
        on_step(TopologyFoldStep {
            index,
            previous,
            data,
            data_len,
            next: next_bytes,
        });
        self.accumulator = next;
        self.step_index =
            self.step_index
                .checked_add(1)
                .ok_or(CommitmentError::LengthOverflow {
                    target: "Poseidon fold step count",
                })?;
        Ok(())
    }

    /// Flushes the final padded block and serializes its field digest.
    fn finish<F>(mut self, on_step: &mut F) -> Result<Commitment, CommitmentError>
    where
        F: FnMut(TopologyFoldStep),
    {
        if !self.block.is_empty() {
            self.compress(on_step)?;
        }
        let bytes = field_to_bytes(self.accumulator)?;
        Ok(Commitment::from_bytes(bytes))
    }
}

/// Serializes one BN254 scalar as a fixed-width little-endian field element.
fn field_to_bytes(field: Fr) -> Result<[u8; COMMITMENT_BYTES], CommitmentError> {
    field
        .into_bigint()
        .to_bytes_le()
        .try_into()
        .map_err(|_| CommitmentError::PoseidonOutput)
}

/// Converts a parameter-construction error without exposing a library type in the API.
fn poseidon_parameters_error(error: &PoseidonError) -> CommitmentError {
    CommitmentError::PoseidonParameters {
        message: error.to_string(),
    }
}

/// Converts a fold error without exposing a library type in the API.
fn poseidon_hash_error(error: &PoseidonError) -> CommitmentError {
    CommitmentError::PoseidonHash {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Commitment, CommitmentError, PACKED_U32_PER_FIELD, POSEIDON_INPUTS, POSEIDON_RATE,
        TOPOLOGY_DOMAIN, TOPOLOGY_SCHEME, commit_topology, commit_topology_with_trace,
        topology_field_count, topology_fold_step_count, verify_topology,
    };

    #[test]
    fn topology_commitment_is_deterministic_for_same_csr() {
        let first = commit_topology(2, &[0, 1, 2], &[0, 1]);
        let second = commit_topology(2, &[0, 1, 2], &[0, 1]);
        assert_eq!(first, second);
    }

    #[test]
    fn topology_commitment_changes_when_an_edge_changes() {
        let first = commit_topology(3, &[0, 1, 2, 2], &[0, 2]);
        let second = commit_topology(3, &[0, 1, 2, 2], &[0, 1]);
        assert!(first.is_ok());
        assert!(second.is_ok());
        if let (Ok(first), Ok(second)) = (first, second) {
            assert_ne!(first, second);
        }
    }

    #[test]
    fn topology_commitment_matches_poseidon_vector() {
        let result = commit_topology(2, &[0, 1, 2], &[0, 1]);
        assert!(result.is_ok());
        if let Ok(commitment) = result {
            assert_eq!(
                commitment.to_hex(),
                "b3708686ddc07e592c34924cfcf6bfb3bc07691824ff93c61f5f2657fc8e9617"
            );
        }
    }

    #[test]
    fn topology_commitment_verifies_claimed_root() {
        let result = commit_topology(2, &[0, 1, 2], &[0, 1]);
        assert!(result.is_ok());
        if let Ok(commitment) = result {
            let verified = verify_topology(commitment, 2, &[0, 1, 2], &[0, 1]);
            assert_eq!(verified, Ok(true));
            let rejected = verify_topology(Commitment::from_bytes([0; 32]), 2, &[0, 1, 2], &[0, 1]);
            assert_eq!(rejected, Ok(false));
        }
    }

    #[test]
    fn topology_commitment_trace_has_one_canonical_final_step() {
        let mut steps = Vec::new();
        let result = commit_topology_with_trace(2, &[0, 1, 2], &[0, 1], |step| {
            steps.push(step);
        });
        assert!(result.is_ok());
        assert_eq!(steps.len(), 1);
        if let (Ok(commitment), Some(step)) = (result, steps.first()) {
            assert_eq!(step.index, 0);
            assert_eq!(step.data_len, 6);
            assert_eq!(step.previous, [0; 32]);
            assert_eq!(step.next, *commitment.as_bytes());
            assert!(
                step.data
                    .get(6..)
                    .is_some_and(|padding| { padding.iter().all(|field| *field == [0; 32]) })
            );
        }
    }

    #[test]
    fn topology_step_estimate_matches_small_trace_shape() {
        assert_eq!(topology_field_count(50, 50), Ok(20));
        assert_eq!(topology_fold_step_count(50, 50), Ok(2));
    }

    #[test]
    fn topology_step_estimate_matches_malecns_dimensions() {
        assert_eq!(topology_field_count(166_700, 25_582_938), Ok(3_678_525));
        assert_eq!(topology_fold_step_count(166_700, 25_582_938), Ok(334_412));
    }

    #[test]
    fn topology_commitment_rejects_unsorted_row() {
        let result = commit_topology(2, &[0, 2, 2], &[1, 0]);
        assert!(matches!(result, Err(CommitmentError::ColumnOrder { .. })));
    }

    #[test]
    fn topology_commitment_uses_versioned_poseidon_domain() {
        assert_eq!(TOPOLOGY_DOMAIN, b"zkfly/poseidon/topology/v1");
        assert_eq!(TOPOLOGY_SCHEME, "poseidon-bn254-circom-x5-t13-fold-v1");
        assert_eq!(POSEIDON_INPUTS, 12);
        assert_eq!(POSEIDON_RATE, 11);
        assert_eq!(PACKED_U32_PER_FIELD, 7);
    }
}
