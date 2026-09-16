//! End-to-end verification using tiny Arrow IPC files with production schemas.

use std::error::Error;
use std::fs::{self, File};
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};
use serde_json::Value;
use tempfile::tempdir;
use zkfly_matrix::{BuildConfig, build_matrix};

/// Writes one uncompressed Feather-compatible Arrow IPC file.
fn write_batch(path: &Path, batch: &RecordBatch) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut writer = FileWriter::try_new(file, batch.schema().as_ref())?;
    writer.write(batch)?;
    writer.finish()?;
    Ok(())
}

/// Creates one record batch while preserving the array order supplied by the test.
fn batch(fields: Vec<Field>, arrays: Vec<ArrayRef>) -> Result<RecordBatch, Box<dyn Error>> {
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?)
}

/// Decodes a complete little-endian `u64` artifact without native-endian assumptions.
fn read_u64(path: &Path) -> Result<Vec<u64>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    bytes
        .chunks_exact(8)
        .map(|chunk| {
            let encoded: [u8; 8] = chunk.try_into()?;
            Ok(u64::from_le_bytes(encoded))
        })
        .collect()
}

/// Decodes a complete little-endian `u32` artifact without native-endian assumptions.
fn read_u32(path: &Path) -> Result<Vec<u32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let encoded: [u8; 4] = chunk.try_into()?;
            Ok(u32::from_le_bytes(encoded))
        })
        .collect()
}

/// Decodes a complete little-endian `i32` artifact without native-endian assumptions.
fn read_i32(path: &Path) -> Result<Vec<i32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let encoded: [u8; 4] = chunk.try_into()?;
            Ok(i32::from_le_bytes(encoded))
        })
        .collect()
}

/// Builds fixture files that cover filtering, duplicate annotations, signs, and endpoints.
fn write_fixture(input: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir(input)?;
    let annotations = batch(
        vec![
            Field::new("bodyId", DataType::Int64, false),
            Field::new("superclass", DataType::Utf8, true),
            Field::new("flywireType", DataType::Utf8, true),
            Field::new("type", DataType::Utf8, true),
            Field::new("somaSide", DataType::Utf8, true),
            Field::new("rootSide", DataType::Utf8, true),
        ],
        vec![
            Arc::new(Int64Array::from(vec![20, 10, 30, 40, 20])),
            Arc::new(StringArray::from(vec![
                Some("interneuron"),
                Some("sensory"),
                None,
                Some(""),
                Some("duplicate"),
            ])),
            Arc::new(StringArray::from(vec![
                Some("L1"),
                Some("R1-6"),
                None,
                None,
                Some("wrong"),
            ])),
            Arc::new(StringArray::from(vec![None::<&str>; 5])),
            Arc::new(StringArray::from(vec![
                Some("left"),
                None,
                None,
                None,
                Some("right"),
            ])),
            Arc::new(StringArray::from(vec![
                None,
                Some("right"),
                None,
                None,
                None,
            ])),
        ],
    )?;
    write_batch(
        &input.join("body-annotations-male-cns-v1.0-minconf-0.5.feather"),
        &annotations,
    )?;

    let neurotransmitters = batch(
        vec![
            Field::new("body", DataType::Int64, false),
            Field::new("consensus_nt", DataType::Utf8, true),
        ],
        vec![
            Arc::new(Int64Array::from(vec![10, 10, 20, 99])),
            Arc::new(StringArray::from(vec![
                Some("histamine"),
                Some("acetylcholine"),
                None,
                Some("gaba"),
            ])),
        ],
    )?;
    write_batch(
        &input.join("body-neurotransmitters-male-cns-v1.0.feather"),
        &neurotransmitters,
    )?;

    let connections = batch(
        vec![
            Field::new("body_pre", DataType::Int64, false),
            Field::new("body_post", DataType::Int64, false),
            Field::new("weight", DataType::Int64, false),
        ],
        vec![
            Arc::new(Int64Array::from(vec![10, 20, 20, 10, 99])),
            Arc::new(Int64Array::from(vec![20, 10, 20, 99, 20])),
            Arc::new(Int64Array::from(vec![7, 5, 3, 11, 13])),
        ],
    )?;
    write_batch(
        &input.join("connectome-weights-male-cns-v1.0-minconf-0.5.feather"),
        &connections,
    )
}

/// Verifies the complete projection and every binary artifact on a hand-checkable graph.
#[test]
fn builds_expected_postsynaptic_csr() -> Result<(), Box<dyn Error>> {
    let temporary = tempdir()?;
    let input = temporary.path().join("input");
    let output = temporary.path().join("output");
    write_fixture(&input)?;

    let report = build_matrix(&BuildConfig::new(&input, &output))?;
    assert_eq!(report.raw_annotation_rows, 5);
    assert_eq!(report.neurons, 2);
    assert_eq!(report.raw_neurotransmitter_rows, 4);
    assert_eq!(report.raw_connection_rows, 5);
    assert_eq!(report.connections, 3);
    assert_eq!(read_u64(&output.join("row_offsets.u64le"))?, [0, 1, 3]);
    assert_eq!(read_u32(&output.join("column_indices.u32le"))?, [1, 0, 1]);
    assert_eq!(read_i32(&output.join("signed_counts.i32le"))?, [5, -7, 3]);
    assert_eq!(read_u64(&output.join("incoming_abs_sums.u64le"))?, [5, 10]);

    let neurons = fs::read_to_string(output.join("neurons.jsonl"))?;
    let body_ids = neurons
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter_map(|neuron| neuron.get("body_id").and_then(Value::as_i64))
        .collect::<Vec<_>>();
    assert_eq!(body_ids, [10, 20]);

    let manifest: Value = serde_json::from_slice(&fs::read(output.join("manifest.json"))?)?;
    assert_eq!(
        manifest.get("matrix_orientation").and_then(Value::as_str),
        Some("rows=postsynaptic, columns=presynaptic")
    );
    Ok(())
}
