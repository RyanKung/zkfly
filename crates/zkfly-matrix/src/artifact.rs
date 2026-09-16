//! Deterministic binary, JSONL, and manifest serialization for the matrix.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::MatrixError;
use crate::arrow_io::InputPaths;
use crate::builder::{BuildReport, CsrMatrix};
use crate::model::NeuronCatalog;

/// Recipe identifier embedded in every manifest.
const RECIPE: &str = "male-cns-superclass-signed-counts-v1";
/// Version of the on-disk artifact format.
const FORMAT_VERSION: u32 = 1;
/// Number of integer values encoded per buffered write.
const ENCODE_CHUNK_VALUES: usize = 65_536;

/// Borrowed build state passed to the artifact writer.
#[derive(Clone, Copy)]
pub(crate) struct ArtifactContext<'a> {
    /// Destination directory for all output files.
    pub(crate) output_dir: &'a Path,
    /// Canonical source paths used to compute provenance hashes.
    pub(crate) input_paths: &'a InputPaths,
    /// Stable neuron catalog to serialize as JSONL.
    pub(crate) catalog: &'a NeuronCatalog,
    /// Validated sparse matrix to serialize.
    pub(crate) matrix: &'a CsrMatrix,
    /// Build counts included in the manifest.
    pub(crate) report: &'a BuildReport,
}

/// SHA-256 and byte length for one source or artifact file.
#[derive(Debug, Serialize)]
struct FileDigest {
    /// Basename recorded in the manifest.
    file: String,
    /// Exact file length in bytes.
    bytes: u64,
    /// Lower-case hexadecimal SHA-256 digest.
    sha256: String,
}

/// Self-describing recipe and provenance record for one matrix artifact.
#[derive(Debug, Serialize)]
struct Manifest<'a> {
    /// Stable artifact format name.
    format: &'static str,
    /// Binary layout version.
    format_version: u32,
    /// Transformation recipe identifier.
    recipe: &'static str,
    /// Meaning of CSR row and column indices.
    matrix_orientation: &'static str,
    /// Exact rational interpretation of every stored edge.
    normalized_weight: &'static str,
    /// Case-insensitive substrings that produce a negative sign.
    inhibitory_consensus_substrings: [&'static str; 3],
    /// Sign assigned when no consensus label is present.
    missing_neurotransmitter_sign: &'static str,
    /// Source row and retained-edge counts.
    report: &'a BuildReport,
    /// Digests of the three source Feather files.
    sources: Vec<FileDigest>,
    /// Digests of the five generated data files.
    artifacts: Vec<FileDigest>,
}

/// Buffered file writer that hashes bytes before an atomic same-directory rename.
struct ArtifactWriter {
    /// Temporary path receiving the current file.
    path: PathBuf,
    /// Buffered destination writer.
    writer: BufWriter<File>,
    /// Incremental SHA-256 state.
    hasher: Sha256,
    /// Number of bytes written so far.
    bytes: u64,
}

impl ArtifactWriter {
    /// Creates a new temporary artifact file.
    fn create(path: &Path) -> Result<Self, MatrixError> {
        let file = File::create(path).map_err(|source| MatrixError::io("create", path, source))?;
        Ok(Self {
            path: path.to_path_buf(),
            writer: BufWriter::with_capacity(1 << 20, file),
            hasher: Sha256::new(),
            bytes: 0,
        })
    }

    /// Writes bytes, updates the digest, and checks the byte counter.
    fn write(&mut self, bytes: &[u8]) -> Result<(), MatrixError> {
        self.writer
            .write_all(bytes)
            .map_err(|source| MatrixError::io("write", &self.path, source))?;
        self.hasher.update(bytes);
        self.bytes = self
            .bytes
            .checked_add(
                u64::try_from(bytes.len()).map_err(|_| MatrixError::Overflow {
                    context: "artifact byte count",
                })?,
            )
            .ok_or(MatrixError::Overflow {
                context: "artifact byte count",
            })?;
        Ok(())
    }

    /// Flushes, renames the temporary file, and returns its digest record.
    fn finish(mut self, final_path: &Path) -> Result<FileDigest, MatrixError> {
        self.writer
            .flush()
            .map_err(|source| MatrixError::io("flush", &self.path, source))?;
        drop(self.writer);
        fs::rename(&self.path, final_path)
            .map_err(|source| MatrixError::io("rename", final_path, source))?;
        Ok(FileDigest {
            file: file_name(final_path)?,
            bytes: self.bytes,
            sha256: hex::encode(self.hasher.finalize()),
        })
    }
}

/// Writes every generated matrix file and then the manifest.
pub(crate) fn write_artifacts(context: ArtifactContext<'_>) -> Result<(), MatrixError> {
    create_output_directory(context.output_dir)?;
    let mut artifacts = Vec::with_capacity(5);
    artifacts.push(write_u64_artifact(
        context.output_dir,
        "row_offsets.u64le",
        &context.matrix.row_offsets,
    )?);
    let columns: Vec<u32> = context
        .matrix
        .entries
        .iter()
        .map(|entry| entry.column)
        .collect();
    artifacts.push(write_u32_artifact(
        context.output_dir,
        "column_indices.u32le",
        &columns,
    )?);
    drop(columns);
    let counts: Vec<i32> = context
        .matrix
        .entries
        .iter()
        .map(|entry| entry.signed_count)
        .collect();
    artifacts.push(write_i32_artifact(
        context.output_dir,
        "signed_counts.i32le",
        &counts,
    )?);
    drop(counts);
    artifacts.push(write_u64_artifact(
        context.output_dir,
        "incoming_abs_sums.u64le",
        &context.matrix.incoming_abs_sums,
    )?);
    artifacts.push(write_neurons(context.output_dir, context.catalog)?);

    let sources = [
        &context.input_paths.annotations,
        &context.input_paths.neurotransmitters,
        &context.input_paths.connections,
    ]
    .into_iter()
    .map(|path| hash_file(path))
    .collect::<Result<Vec<_>, _>>()?;
    write_manifest(
        context.output_dir,
        &Manifest {
            format: "zkfly-csr",
            format_version: FORMAT_VERSION,
            recipe: RECIPE,
            matrix_orientation: "rows=postsynaptic, columns=presynaptic",
            normalized_weight: "signed_counts[edge] / incoming_abs_sums[row]",
            inhibitory_consensus_substrings: ["gaba", "glutamate", "histamine"],
            missing_neurotransmitter_sign: "excitatory",
            report: context.report,
            sources,
            artifacts,
        },
    )
}

/// Creates a new output directory and refuses to overwrite an existing one.
fn create_output_directory(path: &Path) -> Result<(), MatrixError> {
    match fs::create_dir(path) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
            Err(MatrixError::OutputExists {
                path: path.to_path_buf(),
            })
        }
        Err(source) => Err(MatrixError::io("create directory", path, source)),
    }
}

/// Defines a chunked little-endian writer for one integer element type.
macro_rules! define_integer_writer {
    ($function:ident, $integer:ty) => {
        fn $function(
            directory: &Path,
            name: &str,
            values: &[$integer],
        ) -> Result<FileDigest, MatrixError> {
            let final_path = directory.join(name);
            let partial_path = directory.join(format!("{name}.part"));
            let mut writer = ArtifactWriter::create(&partial_path)?;
            let mut encoded =
                Vec::with_capacity(ENCODE_CHUNK_VALUES * std::mem::size_of::<$integer>());
            for chunk in values.chunks(ENCODE_CHUNK_VALUES) {
                encoded.clear();
                for value in chunk {
                    encoded.extend_from_slice(&value.to_le_bytes());
                }
                writer.write(&encoded)?;
            }
            writer.finish(&final_path)
        }
    };
}

define_integer_writer!(write_u64_artifact, u64);
define_integer_writer!(write_u32_artifact, u32);
define_integer_writer!(write_i32_artifact, i32);

/// Serializes the stable catalog as newline-delimited JSON.
fn write_neurons(directory: &Path, catalog: &NeuronCatalog) -> Result<FileDigest, MatrixError> {
    let name = "neurons.jsonl";
    let final_path = directory.join(name);
    let partial_path = directory.join(format!("{name}.part"));
    let mut writer = ArtifactWriter::create(&partial_path)?;
    for neuron in catalog.neurons() {
        let encoded = serde_json::to_vec(neuron).map_err(|source| MatrixError::Json {
            artifact: "neurons.jsonl",
            source,
        })?;
        writer.write(&encoded)?;
        writer.write(b"\n")?;
    }
    writer.finish(&final_path)
}

/// Serializes the recipe, counts, and all source/artifact digests.
fn write_manifest(directory: &Path, manifest: &Manifest<'_>) -> Result<(), MatrixError> {
    let final_path = directory.join("manifest.json");
    let partial_path = directory.join("manifest.json.part");
    let mut writer = ArtifactWriter::create(&partial_path)?;
    let mut encoded = serde_json::to_vec_pretty(manifest).map_err(|source| MatrixError::Json {
        artifact: "manifest.json",
        source,
    })?;
    encoded.push(b'\n');
    writer.write(&encoded)?;
    let _manifest_digest = writer.finish(&final_path)?;
    Ok(())
}

/// Hashes one existing file with bounded memory.
fn hash_file(path: &Path) -> Result<FileDigest, MatrixError> {
    let file = File::open(path).map_err(|source| MatrixError::io("open", path, source))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut buffer = vec![0_u8; 1 << 20];
    let mut hasher = Sha256::new();
    let mut bytes = 0_u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|source| MatrixError::io("read", path, source))?;
        if read == 0 {
            break;
        }
        let chunk = buffer.get(..read).ok_or(MatrixError::CsrInvariant {
            proposition: "read byte count lies inside the hash buffer",
        })?;
        hasher.update(chunk);
        bytes = bytes
            .checked_add(u64::try_from(read).map_err(|_| MatrixError::Overflow {
                context: "source byte count",
            })?)
            .ok_or(MatrixError::Overflow {
                context: "source byte count",
            })?;
    }
    Ok(FileDigest {
        file: file_name(path)?,
        bytes,
        sha256: hex::encode(hasher.finalize()),
    })
}

/// Converts a path's basename to UTF-8 for manifest serialization.
fn file_name(path: &Path) -> Result<String, MatrixError> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .ok_or(MatrixError::CsrInvariant {
            proposition: "artifact paths have UTF-8 file names",
        })
}
