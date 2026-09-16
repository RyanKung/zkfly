//! Typed failures returned by every matrix build boundary.

use std::io;
use std::path::PathBuf;

use arrow_schema::ArrowError;
use thiserror::Error;

/// A named failure while reading, transforming, validating, or writing a matrix.
#[derive(Debug, Error)]
pub enum MatrixError {
    /// A filesystem operation failed.
    #[error("cannot {operation} {path}: {source}")]
    Io {
        /// Operation being attempted.
        operation: &'static str,
        /// Path involved in the operation.
        path: PathBuf,
        /// Underlying IO failure.
        #[source]
        source: io::Error,
    },

    /// An Arrow IPC file or record batch could not be decoded.
    #[error("cannot decode Arrow IPC file {path}: {source}")]
    Arrow {
        /// Feather/Arrow file being decoded.
        path: PathBuf,
        /// Underlying Arrow failure.
        #[source]
        source: ArrowError,
    },

    /// A required Arrow column was absent.
    #[error("required column {column} is missing from {path}")]
    MissingColumn {
        /// Feather/Arrow file being decoded.
        path: PathBuf,
        /// Expected column name.
        column: &'static str,
    },

    /// An Arrow column did not have the required physical type.
    #[error("column {column} in {path} has type {actual}; expected {expected}")]
    ColumnType {
        /// Feather/Arrow file being decoded.
        path: PathBuf,
        /// Column name.
        column: &'static str,
        /// Expected Arrow type.
        expected: &'static str,
        /// Actual Arrow type.
        actual: String,
    },

    /// A required row value was null.
    #[error("required value {column} is null at row {row} in {path}")]
    NullValue {
        /// Feather/Arrow file being decoded.
        path: PathBuf,
        /// Column name.
        column: &'static str,
        /// Zero-based absolute row number.
        row: u64,
    },

    /// A synapse count was zero, negative, or too large for the artifact format.
    #[error("invalid synapse count {count} for {pre}->{post}")]
    InvalidSynapseCount {
        /// Presynaptic `MaleCNS` body ID.
        pre: i64,
        /// Postsynaptic `MaleCNS` body ID.
        post: i64,
        /// Invalid count.
        count: i64,
    },

    /// A checked dimension or accumulation overflowed its representation.
    #[error("numeric overflow while computing {context}")]
    Overflow {
        /// Operation whose result could not be represented.
        context: &'static str,
    },

    /// The source changed between the counting and filling passes.
    #[error("connection source changed between passes: planned {planned}, filled {filled}")]
    SourceChanged {
        /// Retained edge count from the first pass.
        planned: u64,
        /// Retained edge count from the second pass.
        filled: u64,
    },

    /// Two source rows represented the same directed neuron pair.
    #[error("duplicate directed connection in row {row}, column {column}")]
    DuplicateConnection {
        /// Postsynaptic matrix row.
        row: u32,
        /// Presynaptic matrix column.
        column: u32,
    },

    /// A constructed CSR invariant did not hold.
    #[error("CSR invariant failed: {proposition}")]
    CsrInvariant {
        /// Failed proposition.
        proposition: &'static str,
    },

    /// The requested output directory already existed.
    #[error("output directory already exists: {path}")]
    OutputExists {
        /// Existing output directory.
        path: PathBuf,
    },

    /// JSON serialization failed.
    #[error("cannot serialize {artifact}: {source}")]
    Json {
        /// Artifact being serialized.
        artifact: &'static str,
        /// Underlying JSON failure.
        #[source]
        source: serde_json::Error,
    },
}

impl MatrixError {
    /// Constructs a filesystem error while attaching the attempted operation.
    pub(crate) fn io(operation: &'static str, path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            operation,
            path: path.into(),
            source,
        }
    }

    /// Constructs an Arrow decode error while attaching its source path.
    pub(crate) fn arrow(path: impl Into<PathBuf>, source: ArrowError) -> Self {
        Self::Arrow {
            path: path.into(),
            source,
        }
    }
}
