//! Deterministic construction of a sparse `MaleCNS` connectivity matrix.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod arrow_io;
mod artifact;
mod builder;
mod error;
mod model;

pub use builder::{
    BuildConfig, BuildProgress, BuildReport, build_matrix, build_matrix_with_progress,
};
pub use error::MatrixError;
