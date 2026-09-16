//! Errors returned by artifact loading, CPU execution, and CUDA execution.

use std::io;
use std::path::PathBuf;

use thiserror::Error;

/// Failure while loading or validating a CSR artifact.
#[derive(Debug, Error)]
pub enum ArtifactError {
    /// A required file could not be opened or read.
    #[error("cannot {operation} artifact file {path}: {source}")]
    Io {
        /// Operation attempted on the file.
        operation: &'static str,
        /// Path of the file involved.
        path: PathBuf,
        /// Underlying operating-system error.
        source: io::Error,
    },
    /// A binary file has a byte length that is not a whole number of values.
    #[error("artifact file {path} has {bytes} bytes, not a multiple of {width}")]
    Misaligned {
        /// Path of the malformed file.
        path: PathBuf,
        /// Number of bytes found.
        bytes: usize,
        /// Width of one encoded value.
        width: usize,
    },
    /// An encoded integer could not be decoded.
    #[error("cannot decode {kind} value in {path}")]
    Decode {
        /// Path of the malformed file.
        path: PathBuf,
        /// Encoded value type.
        kind: &'static str,
    },
    /// A structural invariant of the CSR artifact is false.
    #[error("invalid CSR artifact: {proposition}")]
    Invariant {
        /// Invariant that failed.
        proposition: &'static str,
    },
    /// A value does not fit the compact type used by the CUDA kernel.
    #[error("artifact value does not fit {target}: {value}")]
    Overflow {
        /// Compact target type.
        target: &'static str,
        /// Original value.
        value: u64,
    },
}

/// Failure while running a benchmark.
#[derive(Debug, Error)]
pub enum BenchError {
    /// The artifact could not be loaded or validated.
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    /// The artifact topology could not be encoded canonically.
    #[error(transparent)]
    Commitment(#[from] zkfly_commitment::CommitmentError),
    /// The benchmark configuration is not meaningful.
    #[error("invalid benchmark configuration: {proposition}")]
    Configuration {
        /// Configuration proposition that failed.
        proposition: &'static str,
    },
    /// The CUDA backend was requested but this binary cannot use it.
    #[error("CUDA backend is unavailable in this build or on this host")]
    CudaUnavailable,
    /// A CUDA driver operation failed.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[error(transparent)]
    Driver(#[from] cuda_core::DriverError),
    /// The PTX file emitted by cuda-oxide could not be read.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[error("cannot read generated PTX {path}: {source}")]
    PtxIo {
        /// Path of the generated PTX file.
        path: PathBuf,
        /// Underlying operating-system error.
        source: io::Error,
    },
    /// The embedded CUDA module could not be loaded.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[error(transparent)]
    Module(#[from] cuda_host::EmbeddedModuleError),
    /// The CUDA execution lock was poisoned after a host panic.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[error("CUDA execution state was poisoned")]
    RuntimeStatePoisoned,
}

/// Adds an operation and path to a standard I/O error.
pub(crate) fn io_error(
    operation: &'static str,
    path: &std::path::Path,
    source: io::Error,
) -> BenchError {
    BenchError::Artifact(ArtifactError::Io {
        operation,
        path: path.to_path_buf(),
        source,
    })
}
