//! Two-pass construction and validation of the deterministic `CSR` matrix.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::MatrixError;
use crate::arrow_io::{InputPaths, read_annotations, read_neurotransmitters, visit_edges};
use crate::artifact::{ArtifactContext, write_artifacts};
use crate::model::{EdgeRow, NeuronCatalog};

/// Inputs and destination for one deterministic matrix build.
#[derive(Debug)]
pub struct BuildConfig {
    /// Directory containing the three official `MaleCNS` v1.0 Feather files.
    pub input_dir: PathBuf,
    /// New directory that will receive the matrix artifact.
    pub output_dir: PathBuf,
}

impl BuildConfig {
    /// Creates a build configuration from input and output directories.
    pub fn new(input_dir: impl Into<PathBuf>, output_dir: impl Into<PathBuf>) -> Self {
        Self {
            input_dir: input_dir.into(),
            output_dir: output_dir.into(),
        }
    }
}

/// A coarse progress event emitted at stable transformation boundaries.
#[derive(Clone, Copy, Debug)]
pub enum BuildProgress {
    /// The annotation filter and deterministic index assignment completed.
    NeuronsSelected {
        /// Raw annotation rows scanned.
        raw_rows: u64,
        /// Unique superclass-annotated neurons retained.
        selected: u64,
    },
    /// Neurotransmitter signs were attached to selected neurons.
    NeurotransmittersApplied {
        /// Raw neurotransmitter rows scanned.
        raw_rows: u64,
    },
    /// One batch of a connection-table pass completed.
    ConnectionRowsScanned {
        /// Pass number: one counts, two fills.
        pass: u8,
        /// Raw connection rows scanned so far.
        raw_rows: u64,
    },
    /// CSR construction and invariant checks completed.
    MatrixValidated {
        /// Retained directed connections.
        connections: u64,
    },
    /// Artifact files are being written.
    WritingArtifacts,
}

/// Counts that identify the exact source projection produced by a build.
#[derive(Debug, Serialize)]
pub struct BuildReport {
    /// Raw rows in the annotation file.
    pub raw_annotation_rows: u64,
    /// Selected unique neurons.
    pub neurons: u64,
    /// Raw rows in the neurotransmitter file.
    pub raw_neurotransmitter_rows: u64,
    /// Raw rows in the connection file.
    pub raw_connection_rows: u64,
    /// Directed connections whose endpoints are both selected neurons.
    pub connections: u64,
    /// Output artifact directory.
    pub output_dir: PathBuf,
}

/// One signed CSR payload entry after source IDs become compact indices.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EdgeEntry {
    /// Presynaptic matrix column.
    pub(crate) column: u32,
    /// Signed structural synapse count.
    pub(crate) signed_count: i32,
}

/// Complete compact sparse matrix before binary serialization.
#[derive(Debug)]
pub(crate) struct CsrMatrix {
    /// `neuron_count + 1` row boundaries.
    pub(crate) row_offsets: Vec<u64>,
    /// Row-major entries sorted by column.
    pub(crate) entries: Vec<EdgeEntry>,
    /// Exact absolute-count denominator for each postsynaptic row.
    pub(crate) incoming_abs_sums: Vec<u64>,
}

/// First-pass counts used to allocate the second-pass CSR payload exactly.
#[derive(Debug)]
struct CsrPlan {
    /// Number of retained edges per postsynaptic row.
    row_counts: Vec<u64>,
    /// Absolute signed-count sum per postsynaptic row.
    incoming_abs_sums: Vec<u64>,
    /// Total retained directed edges.
    retained: u64,
}

/// Builds a `MaleCNS` matrix without progress notifications.
///
/// # Errors
///
/// Returns [`MatrixError`] when an input cannot be decoded, a checked matrix
/// invariant fails, or an artifact cannot be written.
pub fn build_matrix(config: &BuildConfig) -> Result<BuildReport, MatrixError> {
    build_matrix_with_progress(config, |_| {})
}

/// Builds a `MaleCNS` matrix and reports coarse deterministic progress events.
///
/// # Errors
///
/// Returns [`MatrixError`] when an input cannot be decoded, a checked matrix
/// invariant fails, or an artifact cannot be written.
pub fn build_matrix_with_progress(
    config: &BuildConfig,
    mut progress: impl FnMut(BuildProgress),
) -> Result<BuildReport, MatrixError> {
    let paths = InputPaths::in_directory(&config.input_dir);
    let loaded_annotations = read_annotations(&paths.annotations)?;
    let mut catalog = NeuronCatalog::from_annotations(loaded_annotations.rows)?;
    let neurons = u64::try_from(catalog.len()).map_err(|_| MatrixError::Overflow {
        context: "selected neuron count",
    })?;
    progress(BuildProgress::NeuronsSelected {
        raw_rows: loaded_annotations.raw_count,
        selected: neurons,
    });

    let loaded_nt = read_neurotransmitters(&paths.neurotransmitters)?;
    catalog.apply_neurotransmitters(loaded_nt.rows)?;
    progress(BuildProgress::NeurotransmittersApplied {
        raw_rows: loaded_nt.raw_count,
    });

    let (plan, raw_connection_rows) = plan_matrix(&paths.connections, &catalog, &mut progress)?;
    let matrix = fill_matrix(&paths.connections, &catalog, plan, &mut progress)?;
    progress(BuildProgress::MatrixValidated {
        connections: u64::try_from(matrix.entries.len()).map_err(|_| MatrixError::Overflow {
            context: "retained connection count",
        })?,
    });
    progress(BuildProgress::WritingArtifacts);

    let report = BuildReport {
        raw_annotation_rows: loaded_annotations.raw_count,
        neurons,
        raw_neurotransmitter_rows: loaded_nt.raw_count,
        raw_connection_rows,
        connections: u64::try_from(matrix.entries.len()).map_err(|_| MatrixError::Overflow {
            context: "retained connection count",
        })?,
        output_dir: config.output_dir.clone(),
    };
    write_artifacts(ArtifactContext {
        output_dir: &config.output_dir,
        input_paths: &paths,
        catalog: &catalog,
        matrix: &matrix,
        report: &report,
    })?;
    Ok(report)
}

/// Counts retained edges and row denominators in one streaming source pass.
fn plan_matrix(
    path: &Path,
    catalog: &NeuronCatalog,
    progress: &mut impl FnMut(BuildProgress),
) -> Result<(CsrPlan, u64), MatrixError> {
    let mut plan = CsrPlan {
        row_counts: vec![0; catalog.len()],
        incoming_abs_sums: vec![0; catalog.len()],
        retained: 0,
    };
    let raw_rows = visit_edges(
        path,
        |edge| plan.observe(catalog, edge),
        |raw_rows| progress(BuildProgress::ConnectionRowsScanned { pass: 1, raw_rows }),
    )?;
    Ok((plan, raw_rows))
}

/// Fills, canonicalizes, and validates CSR entries in a second source pass.
fn fill_matrix(
    path: &Path,
    catalog: &NeuronCatalog,
    plan: CsrPlan,
    progress: &mut impl FnMut(BuildProgress),
) -> Result<CsrMatrix, MatrixError> {
    let row_offsets = prefix_sum(&plan.row_counts)?;
    let entry_count = usize::try_from(plan.retained).map_err(|_| MatrixError::Overflow {
        context: "CSR allocation length",
    })?;
    let mut entries = vec![EdgeEntry::default(); entry_count];
    let mut cursors: Vec<u64> = row_offsets.iter().take(catalog.len()).copied().collect();
    let mut filled = 0_u64;
    visit_edges(
        path,
        |edge| {
            if let Some((row, entry)) = retained_entry(catalog, edge)? {
                insert_entry(&mut entries, &mut cursors, row, entry)?;
                filled = filled.checked_add(1).ok_or(MatrixError::Overflow {
                    context: "filled connection count",
                })?;
            }
            Ok(())
        },
        |raw_rows| progress(BuildProgress::ConnectionRowsScanned { pass: 2, raw_rows }),
    )?;
    if filled != plan.retained {
        return Err(MatrixError::SourceChanged {
            planned: plan.retained,
            filled,
        });
    }
    canonicalize_rows(&row_offsets, &mut entries)?;
    let matrix = CsrMatrix {
        row_offsets,
        entries,
        incoming_abs_sums: plan.incoming_abs_sums,
    };
    matrix.validate(catalog.len())?;
    Ok(matrix)
}

impl CsrPlan {
    /// Observes one source edge and updates its retained row statistics.
    fn observe(&mut self, catalog: &NeuronCatalog, edge: EdgeRow) -> Result<(), MatrixError> {
        let Some((row, entry)) = retained_entry(catalog, edge)? else {
            return Ok(());
        };
        let row_index = usize::try_from(row).map_err(|_| MatrixError::Overflow {
            context: "CSR row index",
        })?;
        let row_count = self
            .row_counts
            .get_mut(row_index)
            .ok_or(MatrixError::CsrInvariant {
                proposition: "retained postsynaptic index is within the catalog",
            })?;
        *row_count = row_count.checked_add(1).ok_or(MatrixError::Overflow {
            context: "CSR row connection count",
        })?;
        let row_sum =
            self.incoming_abs_sums
                .get_mut(row_index)
                .ok_or(MatrixError::CsrInvariant {
                    proposition: "normalization row is within the catalog",
                })?;
        *row_sum = row_sum
            .checked_add(u64::from(entry.signed_count.unsigned_abs()))
            .ok_or(MatrixError::Overflow {
                context: "incoming absolute synapse sum",
            })?;
        self.retained = self.retained.checked_add(1).ok_or(MatrixError::Overflow {
            context: "retained connection count",
        })?;
        Ok(())
    }
}

/// Converts a source edge into a compact row and signed entry when both endpoints are selected.
fn retained_entry(
    catalog: &NeuronCatalog,
    edge: EdgeRow,
) -> Result<Option<(u32, EdgeEntry)>, MatrixError> {
    let Some(column) = catalog.index_of(edge.pre) else {
        return Ok(None);
    };
    let Some(row) = catalog.index_of(edge.post) else {
        return Ok(None);
    };
    let signed_count = catalog.sign_of(column)?.apply(edge.count, edge)?;
    Ok(Some((
        row,
        EdgeEntry {
            column,
            signed_count,
        },
    )))
}

/// Converts per-row counts into checked cumulative CSR offsets.
fn prefix_sum(counts: &[u64]) -> Result<Vec<u64>, MatrixError> {
    let mut offsets =
        Vec::with_capacity(counts.len().checked_add(1).ok_or(MatrixError::Overflow {
            context: "CSR row offset length",
        })?);
    let mut running = 0_u64;
    offsets.push(running);
    for count in counts {
        running = running.checked_add(*count).ok_or(MatrixError::Overflow {
            context: "CSR row offsets",
        })?;
        offsets.push(running);
    }
    Ok(offsets)
}

/// Writes one entry at the current cursor for its postsynaptic row.
fn insert_entry(
    entries: &mut [EdgeEntry],
    cursors: &mut [u64],
    row: u32,
    entry: EdgeEntry,
) -> Result<(), MatrixError> {
    let row = usize::try_from(row).map_err(|_| MatrixError::Overflow {
        context: "CSR row index",
    })?;
    let cursor = cursors.get_mut(row).ok_or(MatrixError::CsrInvariant {
        proposition: "fill cursor exists for every row",
    })?;
    let slot = usize::try_from(*cursor).map_err(|_| MatrixError::Overflow {
        context: "CSR entry offset",
    })?;
    let destination = entries.get_mut(slot).ok_or(MatrixError::CsrInvariant {
        proposition: "fill cursor stays inside the planned allocation",
    })?;
    *destination = entry;
    *cursor = cursor.checked_add(1).ok_or(MatrixError::Overflow {
        context: "CSR fill cursor",
    })?;
    Ok(())
}

/// Sorts every row by column and rejects duplicate directed pairs.
fn canonicalize_rows(offsets: &[u64], entries: &mut [EdgeEntry]) -> Result<(), MatrixError> {
    for (row, bounds) in offsets.windows(2).enumerate() {
        let [start, end] = bounds else {
            return Err(MatrixError::CsrInvariant {
                proposition: "every CSR row window contains two offsets",
            });
        };
        let range = usize::try_from(*start).map_err(|_| MatrixError::Overflow {
            context: "CSR row start",
        })?..usize::try_from(*end).map_err(|_| MatrixError::Overflow {
            context: "CSR row end",
        })?;
        let row_entries = entries.get_mut(range).ok_or(MatrixError::CsrInvariant {
            proposition: "every CSR row range lies inside the entry array",
        })?;
        row_entries.sort_unstable_by_key(|entry| entry.column);
        for pair in row_entries.windows(2) {
            let [left, right] = pair else {
                return Err(MatrixError::CsrInvariant {
                    proposition: "every adjacent-entry window contains two entries",
                });
            };
            if left.column == right.column {
                return Err(MatrixError::DuplicateConnection {
                    row: u32::try_from(row).map_err(|_| MatrixError::Overflow {
                        context: "duplicate connection row",
                    })?,
                    column: left.column,
                });
            }
        }
    }
    Ok(())
}

impl CsrMatrix {
    /// Checks global CSR shape and row normalization invariants.
    fn validate(&self, neuron_count: usize) -> Result<(), MatrixError> {
        if self.row_offsets.len() != neuron_count.saturating_add(1) {
            return Err(MatrixError::CsrInvariant {
                proposition: "row_offsets.len() equals neuron_count + 1",
            });
        }
        if self.incoming_abs_sums.len() != neuron_count {
            return Err(MatrixError::CsrInvariant {
                proposition: "one normalization denominator exists per row",
            });
        }
        if self.row_offsets.first().copied() != Some(0) {
            return Err(MatrixError::CsrInvariant {
                proposition: "the first row offset is zero",
            });
        }
        let entry_count = u64::try_from(self.entries.len()).map_err(|_| MatrixError::Overflow {
            context: "CSR entry count",
        })?;
        if self.row_offsets.last().copied() != Some(entry_count) {
            return Err(MatrixError::CsrInvariant {
                proposition: "the final row offset equals the entry count",
            });
        }
        validate_rows(self)
    }
}

/// Checks each row's denominator against its serialized signed entries.
fn validate_rows(matrix: &CsrMatrix) -> Result<(), MatrixError> {
    for (bounds, denominator) in matrix
        .row_offsets
        .windows(2)
        .zip(matrix.incoming_abs_sums.iter())
    {
        let [start, end] = bounds else {
            return Err(MatrixError::CsrInvariant {
                proposition: "every validation row has two offsets",
            });
        };
        let range = usize::try_from(*start).map_err(|_| MatrixError::Overflow {
            context: "validation row start",
        })?..usize::try_from(*end).map_err(|_| MatrixError::Overflow {
            context: "validation row end",
        })?;
        let entries = matrix.entries.get(range).ok_or(MatrixError::CsrInvariant {
            proposition: "validated row range lies inside the entry array",
        })?;
        let sum = entries.iter().try_fold(0_u64, |sum, entry| {
            sum.checked_add(u64::from(entry.signed_count.unsigned_abs()))
                .ok_or(MatrixError::Overflow {
                    context: "validated row normalization sum",
                })
        })?;
        if sum != *denominator {
            return Err(MatrixError::CsrInvariant {
                proposition: "every denominator equals its row absolute-count sum",
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::model::{AnnotationRow, NeurotransmitterRow};

    use super::*;

    fn catalog() -> Result<NeuronCatalog, MatrixError> {
        let mut catalog = NeuronCatalog::from_annotations([
            AnnotationRow {
                body_id: 10,
                superclass: "sensory".to_owned(),
                cell_type: "R1-6".to_owned(),
                side: "L".to_owned(),
            },
            AnnotationRow {
                body_id: 20,
                superclass: "interneuron".to_owned(),
                cell_type: "L1".to_owned(),
                side: "L".to_owned(),
            },
        ])?;
        catalog.apply_neurotransmitters([NeurotransmitterRow {
            body_id: 10,
            consensus: Some("histamine".to_owned()),
        }])?;
        Ok(catalog)
    }

    #[test]
    fn plan_preserves_exact_signed_counts_and_denominator() -> Result<(), MatrixError> {
        let catalog = catalog()?;
        let edge = EdgeRow {
            pre: 10,
            post: 20,
            count: 7,
        };
        let mut plan = CsrPlan {
            row_counts: vec![0; catalog.len()],
            incoming_abs_sums: vec![0; catalog.len()],
            retained: 0,
        };
        plan.observe(&catalog, edge)?;
        assert_eq!(plan.row_counts, [0, 1]);
        assert_eq!(plan.incoming_abs_sums, [0, 7]);
        let retained = retained_entry(&catalog, edge)?.ok_or(MatrixError::CsrInvariant {
            proposition: "the selected test edge is retained",
        })?;
        assert_eq!(retained.1.signed_count, -7);
        Ok(())
    }

    #[test]
    fn plan_ignores_edges_with_an_unselected_endpoint() -> Result<(), MatrixError> {
        let catalog = catalog()?;
        let mut plan = CsrPlan {
            row_counts: vec![0; catalog.len()],
            incoming_abs_sums: vec![0; catalog.len()],
            retained: 0,
        };
        plan.observe(
            &catalog,
            EdgeRow {
                pre: 99,
                post: 20,
                count: 4,
            },
        )?;
        assert_eq!(plan.retained, 0);
        Ok(())
    }
}
