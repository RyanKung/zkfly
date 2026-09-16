//! Loading and validating the deterministic `MaleCNS` CSR artifact.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use serde::Serialize;
use zkfly_commitment::{Commitment, TOPOLOGY_SCHEME, commit_topology};

use crate::error::{ArtifactError, BenchError, io_error};

/// Compact CSR data needed for one graph forward pass.
#[derive(Debug)]
pub struct CsrArtifact {
    /// Number of neurons represented by the matrix.
    pub neuron_count: usize,
    /// CSR row boundaries, with one extra terminal boundary.
    pub row_offsets: Vec<u32>,
    /// Presynaptic column for each retained edge.
    pub column_indices: Vec<u32>,
    /// Signed count divided by the absolute incoming count of its row.
    pub weights: Vec<f32>,
    /// Commitment to neuron count, row offsets, and column indices only.
    pub topology_commitment: Commitment,
}

/// CSR dimensions loaded without decoding weights or recomputing a root.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArtifactDimensions {
    /// Number of neurons represented by the matrix.
    pub neuron_count: usize,
    /// Number of retained directed edges.
    pub edge_count: usize,
}

/// Stable size and orientation information for reporting and audit logs.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ArtifactSummary {
    /// Number of neurons in the matrix.
    pub neurons: usize,
    /// Number of retained directed edges.
    pub edges: usize,
    /// Bytes transferred for row offsets, columns, and weights.
    pub matrix_bytes: u64,
    /// Canonical commitment to the fixed graph topology.
    pub topology_commitment: Commitment,
    /// Exact hash and encoding parameters used for the topology commitment.
    pub topology_commitment_scheme: &'static str,
}

impl CsrArtifact {
    /// Returns the compact size summary used by the benchmark report.
    #[must_use]
    pub fn summary(&self) -> ArtifactSummary {
        let offset_bytes = self
            .row_offsets
            .len()
            .saturating_mul(std::mem::size_of::<u32>());
        let column_bytes = self
            .column_indices
            .len()
            .saturating_mul(std::mem::size_of::<u32>());
        let weight_bytes = self
            .weights
            .len()
            .saturating_mul(std::mem::size_of::<f32>());
        let matrix_bytes = saturating_u64(offset_bytes)
            .saturating_add(saturating_u64(column_bytes))
            .saturating_add(saturating_u64(weight_bytes));
        ArtifactSummary {
            neurons: self.neuron_count,
            edges: self.column_indices.len(),
            matrix_bytes,
            topology_commitment: self.topology_commitment,
            topology_commitment_scheme: TOPOLOGY_SCHEME,
        }
    }
}

/// Loads the five binary values needed from a built CSR artifact directory.
///
/// The loader converts 64-bit on-disk offsets to 32-bit offsets after checking
/// that the complete `MaleCNS` artifact fits. This is safe for the current
/// 25,582,938-edge matrix and saves device bandwidth on every benchmark run.
///
/// # Errors
///
/// Returns [`BenchError::Artifact`] when a file is missing, malformed, or
/// violates a CSR invariant.
pub fn load_artifact(directory: impl AsRef<Path>) -> Result<CsrArtifact, BenchError> {
    let directory = directory.as_ref();
    let offsets_path = directory.join("row_offsets.u64le");
    let columns_path = directory.join("column_indices.u32le");
    let counts_path = directory.join("signed_counts.i32le");
    let sums_path = directory.join("incoming_abs_sums.u64le");
    let offsets = read_u64(&offsets_path)?;
    let columns = read_u32(&columns_path)?;
    let counts = read_i32(&counts_path)?;
    let sums = read_u64(&sums_path)?;
    validate_lengths(&offsets, &columns, &counts, &sums)?;
    let neuron_count =
        offsets
            .len()
            .checked_sub(1)
            .ok_or(BenchError::Artifact(ArtifactError::Invariant {
                proposition: "row offsets contain a terminal boundary",
            }))?;
    let row_offsets = offsets
        .iter()
        .copied()
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                BenchError::Artifact(ArtifactError::Overflow {
                    target: "u32 row offset",
                    value,
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let weights = normalized_weights(&row_offsets, &counts, &sums)?;
    let topology_commitment = commit_topology(neuron_count, &row_offsets, &columns)?;
    Ok(CsrArtifact {
        neuron_count,
        row_offsets,
        column_indices: columns,
        weights,
        topology_commitment,
    })
}

/// Loads only the binary dimensions needed by the paper step estimator.
///
/// This intentionally avoids decoding signed counts, normalized weights, or
/// recomputing the Poseidon topology root. Full benchmark and proof paths must
/// continue to use [`load_artifact`] for complete CSR validation.
///
/// # Errors
///
/// Returns [`BenchError::Artifact`] when a required file is missing, has a
/// misaligned byte length, or cannot represent its value count on this host.
pub fn load_artifact_dimensions(
    directory: impl AsRef<Path>,
) -> Result<ArtifactDimensions, BenchError> {
    let directory = directory.as_ref();
    let offsets = value_count(&directory.join("row_offsets.u64le"), 8)?;
    let neuron_count =
        offsets
            .checked_sub(1)
            .ok_or(BenchError::Artifact(ArtifactError::Invariant {
                proposition: "row offsets contain a terminal boundary",
            }))?;
    let edge_count = value_count(&directory.join("column_indices.u32le"), 4)?;
    Ok(ArtifactDimensions {
        neuron_count,
        edge_count,
    })
}

/// Reads a little-endian unsigned 64-bit vector.
fn read_u64(path: &Path) -> Result<Vec<u64>, BenchError> {
    read_values(path, 8, "u64", |bytes| {
        let array: [u8; 8] = bytes.try_into().map_err(|_| ())?;
        Ok(u64::from_le_bytes(array))
    })
}

/// Reads a little-endian unsigned 32-bit vector.
fn read_u32(path: &Path) -> Result<Vec<u32>, BenchError> {
    read_values(path, 4, "u32", |bytes| {
        let array: [u8; 4] = bytes.try_into().map_err(|_| ())?;
        Ok(u32::from_le_bytes(array))
    })
}

/// Reads a little-endian signed 32-bit vector.
fn read_i32(path: &Path) -> Result<Vec<i32>, BenchError> {
    read_values(path, 4, "i32", |bytes| {
        let array: [u8; 4] = bytes.try_into().map_err(|_| ())?;
        Ok(i32::from_le_bytes(array))
    })
}

/// Reads and decodes a fixed-width little-endian vector without unsafe casts.
fn read_values<T, F>(
    path: &Path,
    width: usize,
    kind: &'static str,
    decode: F,
) -> Result<Vec<T>, BenchError>
where
    F: Fn(&[u8]) -> Result<T, ()>,
{
    let file = File::open(path).map_err(|source| io_error("open", path, source))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|source| io_error("read", path, source))?;
    if !bytes.len().is_multiple_of(width) {
        return Err(BenchError::Artifact(ArtifactError::Misaligned {
            path: path.to_path_buf(),
            bytes: bytes.len(),
            width,
        }));
    }
    bytes
        .chunks_exact(width)
        .map(|chunk| {
            decode(chunk).map_err(|()| {
                BenchError::Artifact(ArtifactError::Decode {
                    path: path.to_path_buf(),
                    kind,
                })
            })
        })
        .collect()
}

/// Reads a binary vector length from filesystem metadata without loading it.
fn value_count(path: &Path, width: usize) -> Result<usize, BenchError> {
    let bytes = std::fs::metadata(path)
        .map_err(|source| io_error("stat", path, source))?
        .len();
    let bytes_usize = usize::try_from(bytes).map_err(|_| {
        BenchError::Artifact(ArtifactError::Overflow {
            target: "host file byte count",
            value: bytes,
        })
    })?;
    if !bytes_usize.is_multiple_of(width) {
        return Err(BenchError::Artifact(ArtifactError::Misaligned {
            path: path.to_path_buf(),
            bytes: bytes_usize,
            width,
        }));
    }
    Ok(bytes_usize / width)
}

/// Checks that all four binary vectors describe one complete CSR matrix.
fn validate_lengths(
    offsets: &[u64],
    columns: &[u32],
    counts: &[i32],
    sums: &[u64],
) -> Result<(), BenchError> {
    let neuron_count =
        offsets
            .len()
            .checked_sub(1)
            .ok_or(BenchError::Artifact(ArtifactError::Invariant {
                proposition: "row offsets contain a terminal boundary",
            }))?;
    if sums.len() != neuron_count {
        return Err(BenchError::Artifact(ArtifactError::Invariant {
            proposition: "incoming sums have one value per neuron",
        }));
    }
    if columns.len() != counts.len() {
        return Err(BenchError::Artifact(ArtifactError::Invariant {
            proposition: "column and signed-count vectors have equal length",
        }));
    }
    let mut previous = 0_u64;
    for offset in offsets {
        if *offset < previous {
            return Err(BenchError::Artifact(ArtifactError::Invariant {
                proposition: "row offsets are monotonically nondecreasing",
            }));
        }
        previous = *offset;
    }
    let edge_count = u64::try_from(columns.len()).map_err(|_| {
        BenchError::Artifact(ArtifactError::Overflow {
            target: "u64 edge count",
            value: u64::MAX,
        })
    })?;
    if offsets.last().copied() != Some(edge_count) {
        return Err(BenchError::Artifact(ArtifactError::Invariant {
            proposition: "terminal row offset equals edge count",
        }));
    }
    if columns
        .iter()
        .any(|column| usize::try_from(*column).map_or(true, |value| value >= neuron_count))
    {
        return Err(BenchError::Artifact(ArtifactError::Invariant {
            proposition: "every column index names a selected neuron",
        }));
    }
    Ok(())
}

/// Converts signed integer counts into the documented normalized weights.
fn normalized_weights(
    offsets: &[u32],
    counts: &[i32],
    sums: &[u64],
) -> Result<Vec<f32>, BenchError> {
    let mut weights = Vec::with_capacity(counts.len());
    for (row, window) in offsets.windows(2).enumerate() {
        let start_value =
            *window
                .first()
                .ok_or(BenchError::Artifact(ArtifactError::Invariant {
                    proposition: "row window has a start",
                }))?;
        let start = usize::try_from(start_value).map_err(|_| {
            BenchError::Artifact(ArtifactError::Overflow {
                target: "usize edge offset",
                value: u64::from(start_value),
            })
        })?;
        let end_value = *window
            .get(1)
            .ok_or(BenchError::Artifact(ArtifactError::Invariant {
                proposition: "row window has an end",
            }))?;
        let end = usize::try_from(end_value).map_err(|_| {
            BenchError::Artifact(ArtifactError::Overflow {
                target: "usize edge offset",
                value: u64::from(end_value),
            })
        })?;
        let denominator = *sums
            .get(row)
            .ok_or(BenchError::Artifact(ArtifactError::Invariant {
                proposition: "row denominator exists",
            }))?;
        for edge in start..end {
            let count =
                *counts
                    .get(edge)
                    .ok_or(BenchError::Artifact(ArtifactError::Invariant {
                        proposition: "row range stays within signed counts",
                    }))?;
            let weight = if denominator == 0 {
                0.0
            } else {
                #[allow(clippy::cast_precision_loss)]
                {
                    count as f32 / denominator as f32
                }
            };
            weights.push(weight);
        }
    }
    Ok(weights)
}

/// Converts a host byte count to `u64`, saturating only on an exotic wider
/// host index type where the report cannot represent the exact value.
#[allow(clippy::manual_unwrap_or)]
fn saturating_u64(value: usize) -> u64 {
    match u64::try_from(value) {
        Ok(value) => value,
        Err(_) => u64::MAX,
    }
}
