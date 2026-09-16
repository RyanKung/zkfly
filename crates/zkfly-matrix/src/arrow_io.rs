//! Strict, streaming readers for the three `MaleCNS` Feather tables.

use std::fs::File;
use std::path::{Path, PathBuf};

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_ipc::reader::FileReader;

use crate::MatrixError;
use crate::model::{AnnotationRow, EdgeRow, NeurotransmitterRow};

/// Rows loaded into memory together with the number of physical source rows.
pub(crate) struct LoadedRows<T> {
    /// Number of rows encountered before filtering or deduplication.
    pub(crate) raw_count: u64,
    /// Typed rows retained for the catalog-building phase.
    pub(crate) rows: Vec<T>,
}

/// Opens a Feather file through Arrow's buffered IPC reader.
fn open(path: &Path) -> Result<FileReader<std::io::BufReader<File>>, MatrixError> {
    let file = File::open(path).map_err(|source| MatrixError::io("open", path, source))?;
    FileReader::try_new_buffered(file, None).map_err(|source| MatrixError::arrow(path, source))
}

/// Fetches and type-checks a nullable `int64` column by name.
fn int64_column<'a>(
    batch: &'a RecordBatch,
    path: &Path,
    name: &'static str,
) -> Result<&'a Int64Array, MatrixError> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| MatrixError::MissingColumn {
            path: path.to_path_buf(),
            column: name,
        })?;
    column
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| MatrixError::ColumnType {
            path: path.to_path_buf(),
            column: name,
            expected: "int64",
            actual: column.data_type().to_string(),
        })
}

/// Fetches and type-checks a nullable UTF-8 column by name.
fn string_column<'a>(
    batch: &'a RecordBatch,
    path: &Path,
    name: &'static str,
) -> Result<&'a StringArray, MatrixError> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| MatrixError::MissingColumn {
            path: path.to_path_buf(),
            column: name,
        })?;
    column
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| MatrixError::ColumnType {
            path: path.to_path_buf(),
            column: name,
            expected: "utf8",
            actual: column.data_type().to_string(),
        })
}

/// Converts one nullable integer cell into a required value with row context.
fn required_i64(
    value: Option<i64>,
    path: &Path,
    column: &'static str,
    row: u64,
) -> Result<i64, MatrixError> {
    value.ok_or_else(|| MatrixError::NullValue {
        path: path.to_path_buf(),
        column,
        row,
    })
}

/// Loads annotation rows while preserving physical order for first-row semantics.
pub(crate) fn read_annotations(path: &Path) -> Result<LoadedRows<AnnotationRow>, MatrixError> {
    let mut rows = Vec::new();
    let mut raw_count = 0_u64;
    for batch in open(path)? {
        let batch = batch.map_err(|source| MatrixError::arrow(path, source))?;
        let body = int64_column(&batch, path, "bodyId")?;
        let superclass = string_column(&batch, path, "superclass")?;
        let flywire_type = string_column(&batch, path, "flywireType")?;
        let fallback_type = string_column(&batch, path, "type")?;
        let soma_side = string_column(&batch, path, "somaSide")?;
        let root_side = string_column(&batch, path, "rootSide")?;
        let values = body
            .iter()
            .zip(superclass.iter())
            .zip(flywire_type.iter())
            .zip(fallback_type.iter())
            .zip(soma_side.iter())
            .zip(root_side.iter());
        for (local_row, (((((body_id, superclass), flywire), fallback), soma), root)) in
            values.enumerate()
        {
            let absolute_row = raw_count
                .checked_add(u64::try_from(local_row).map_err(|_| MatrixError::Overflow {
                    context: "annotation row number",
                })?)
                .ok_or(MatrixError::Overflow {
                    context: "annotation row number",
                })?;
            let body_id = required_i64(body_id, path, "bodyId", absolute_row)?;
            let Some(superclass) = superclass else {
                continue;
            };
            rows.push(AnnotationRow {
                body_id,
                superclass: superclass.to_owned(),
                cell_type: flywire.or(fallback).unwrap_or_default().to_owned(),
                side: soma.or(root).unwrap_or_default().to_ascii_uppercase(),
            });
        }
        raw_count = raw_count
            .checked_add(
                u64::try_from(batch.num_rows()).map_err(|_| MatrixError::Overflow {
                    context: "annotation row count",
                })?,
            )
            .ok_or(MatrixError::Overflow {
                context: "annotation row count",
            })?;
    }
    Ok(LoadedRows { raw_count, rows })
}

/// Loads neurotransmitter rows while preserving physical order for first-row semantics.
pub(crate) fn read_neurotransmitters(
    path: &Path,
) -> Result<LoadedRows<NeurotransmitterRow>, MatrixError> {
    let mut rows = Vec::new();
    let mut raw_count = 0_u64;
    for batch in open(path)? {
        let batch = batch.map_err(|source| MatrixError::arrow(path, source))?;
        let body = int64_column(&batch, path, "body")?;
        let consensus = string_column(&batch, path, "consensus_nt")?;
        for (local_row, (body_id, label)) in body.iter().zip(consensus.iter()).enumerate() {
            let absolute_row = raw_count
                .checked_add(u64::try_from(local_row).map_err(|_| MatrixError::Overflow {
                    context: "neurotransmitter row number",
                })?)
                .ok_or(MatrixError::Overflow {
                    context: "neurotransmitter row number",
                })?;
            rows.push(NeurotransmitterRow {
                body_id: required_i64(body_id, path, "body", absolute_row)?,
                consensus: label.map(str::to_owned),
            });
        }
        raw_count = raw_count
            .checked_add(
                u64::try_from(batch.num_rows()).map_err(|_| MatrixError::Overflow {
                    context: "neurotransmitter row count",
                })?,
            )
            .ok_or(MatrixError::Overflow {
                context: "neurotransmitter row count",
            })?;
    }
    Ok(LoadedRows { raw_count, rows })
}

/// Visits every connection row without retaining the 1 GB source table.
pub(crate) fn visit_edges(
    path: &Path,
    mut visitor: impl FnMut(EdgeRow) -> Result<(), MatrixError>,
    mut batch_complete: impl FnMut(u64),
) -> Result<u64, MatrixError> {
    let mut raw_count = 0_u64;
    for batch in open(path)? {
        let batch = batch.map_err(|source| MatrixError::arrow(path, source))?;
        visit_edge_batch(&batch, path, raw_count, &mut visitor)?;
        raw_count = raw_count
            .checked_add(
                u64::try_from(batch.num_rows()).map_err(|_| MatrixError::Overflow {
                    context: "connection row count",
                })?,
            )
            .ok_or(MatrixError::Overflow {
                context: "connection row count",
            })?;
        batch_complete(raw_count);
    }
    Ok(raw_count)
}

/// Visits one Arrow record batch from the connection table.
fn visit_edge_batch(
    batch: &RecordBatch,
    path: &Path,
    base_row: u64,
    visitor: &mut impl FnMut(EdgeRow) -> Result<(), MatrixError>,
) -> Result<(), MatrixError> {
    let pre = int64_column(batch, path, "body_pre")?;
    let post = int64_column(batch, path, "body_post")?;
    let weight = int64_column(batch, path, "weight")?;
    for (local_row, ((pre, post), count)) in
        pre.iter().zip(post.iter()).zip(weight.iter()).enumerate()
    {
        let row = base_row
            .checked_add(u64::try_from(local_row).map_err(|_| MatrixError::Overflow {
                context: "connection row number",
            })?)
            .ok_or(MatrixError::Overflow {
                context: "connection row number",
            })?;
        visitor(EdgeRow {
            pre: required_i64(pre, path, "body_pre", row)?,
            post: required_i64(post, path, "body_post", row)?,
            count: required_i64(count, path, "weight", row)?,
        })?;
    }
    Ok(())
}

/// Resolved locations of the three source files required by the build recipe.
pub(crate) struct InputPaths {
    /// Body annotation Feather file.
    pub(crate) annotations: PathBuf,
    /// Body neurotransmitter prediction Feather file.
    pub(crate) neurotransmitters: PathBuf,
    /// Aggregated connection-weight Feather file.
    pub(crate) connections: PathBuf,
}

impl InputPaths {
    /// Resolves the canonical filenames below one downloaded data directory.
    pub(crate) fn in_directory(directory: &Path) -> Self {
        Self {
            annotations: directory.join("body-annotations-male-cns-v1.0-minconf-0.5.feather"),
            neurotransmitters: directory.join("body-neurotransmitters-male-cns-v1.0.feather"),
            connections: directory.join("connectome-weights-male-cns-v1.0-minconf-0.5.feather"),
        }
    }
}
