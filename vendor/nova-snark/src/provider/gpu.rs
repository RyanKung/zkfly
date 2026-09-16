//! Optional device backend hooks for Nova's arithmetic-heavy prover paths.
//!
//! The protocol remains implemented by Nova itself. A backend only receives
//! canonical field encodings and returns canonical field encodings; callers
//! validate every returned length and fall back to the reference implementation
//! when the backend declines an operation. This keeps the hook fail-closed:
//! a CUDA runtime can be unavailable without changing proof semantics.

use std::sync::{Arc, OnceLock, RwLock};

/// Process-wide arithmetic backend installed by an accelerator adapter.
static GPU_BACKEND: OnceLock<RwLock<Option<Arc<dyn GpuBackend>>>> = OnceLock::new();

/// A backend for field-vector and sparse-matrix operations used by Nova.
pub trait GpuBackend: Send + Sync {
  /// Computes one CSR sparse matrix multiplied by a dense field vector.
  ///
  /// `values`, `vector`, and the returned vector are packed in fixed-width
  /// little-endian field representations. `field_width` and `modulus` identify
  /// the field without exposing a concrete curve type to this crate.
  fn spmv(
    &self,
    row_offsets: &[usize],
    column_indices: &[usize],
    values: &[u8],
    vector: &[u8],
    field_width: usize,
    modulus: &[u8],
  ) -> Option<Vec<u8>>;

  /// Computes three CSR products that share one dense field vector.
  ///
  /// The default preserves compatibility by invoking [`Self::spmv`] once per
  /// matrix. Backends with a fused kernel can override this method to share
  /// the vector upload, launch, and result synchronization.
  fn spmv_three(
    &self,
    row_offsets: [&[usize]; 3],
    column_indices: [&[usize]; 3],
    values: [&[u8]; 3],
    vector: &[u8],
    field_width: usize,
    modulus: &[u8],
  ) -> Option<[Vec<u8>; 3]> {
    let [a_rows, b_rows, c_rows] = row_offsets;
    let [a_columns, b_columns, c_columns] = column_indices;
    let [a_values, b_values, c_values] = values;
    Some([
      self.spmv(a_rows, a_columns, a_values, vector, field_width, modulus)?,
      self.spmv(b_rows, b_columns, b_values, vector, field_width, modulus)?,
      self.spmv(c_rows, c_columns, c_values, vector, field_width, modulus)?,
    ])
  }

  /// Computes `left + scalar * right` element by element.
  fn vector_linear_combination(
    &self,
    left: &[u8],
    right: &[u8],
    scalar: &[u8],
    field_width: usize,
    modulus: &[u8],
  ) -> Option<Vec<u8>>;

  /// Computes `az * bz - u * cz - error` element by element.
  fn cross_term(
    &self,
    az: &[u8],
    bz: &[u8],
    cz: &[u8],
    error: &[u8],
    u: &[u8],
    field_width: usize,
    modulus: &[u8],
  ) -> Option<Vec<u8>>;
}

/// Installs or replaces the process-wide device backend.
pub fn install_gpu_backend(backend: Arc<dyn GpuBackend>) {
  let lock = GPU_BACKEND.get_or_init(|| RwLock::new(None));
  if let Ok(mut slot) = lock.write() {
    *slot = Some(backend);
  }
}

/// Removes the process-wide device backend.
pub fn clear_gpu_backend() {
  if let Some(lock) = GPU_BACKEND.get() {
    if let Ok(mut slot) = lock.write() {
      *slot = None;
    }
  }
}

/// Returns the currently installed device backend, if any.
pub(crate) fn gpu_backend() -> Option<Arc<dyn GpuBackend>> {
  GPU_BACKEND
    .get()
    .and_then(|lock| lock.read().ok())
    .and_then(|slot| slot.as_ref().map(Arc::clone))
}
