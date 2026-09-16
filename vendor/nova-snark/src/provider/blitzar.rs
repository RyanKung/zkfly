//! This module implements variable time multi-scalar multiplication using Blitzar's GPU acceleration
use std::{
  sync::atomic::{AtomicU64, Ordering},
  time::{Duration, Instant},
};

use blitzar;
use halo2curves::bn256::{Fr as Scalar, G1Affine as Affine, G1 as Point};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// Number of completed calls through this provider.
static MSM_CALLS: AtomicU64 = AtomicU64::new(0);

/// Number of completed commitment outputs across all provider calls.
static MSM_BATCHES: AtomicU64 = AtomicU64::new(0);

/// Number of scalar-base pairs consumed across all commitment outputs.
static MSM_SCALARS: AtomicU64 = AtomicU64::new(0);

/// End-to-end provider time accumulated in microseconds.
static MSM_MICROSECONDS: AtomicU64 = AtomicU64::new(0);

/// Cumulative telemetry for the optional Blitzar MSM provider.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MsmStats {
  /// Number of single or batched provider calls.
  pub calls: u64,
  /// Number of commitment outputs requested across those calls.
  pub batches: u64,
  /// Number of scalar-base pairs processed across all outputs.
  pub scalars: u64,
  /// End-to-end microseconds spent inside the provider boundary.
  pub microseconds: u64,
}

/// Returns a relaxed snapshot of cumulative provider telemetry.
pub fn stats() -> MsmStats {
  MsmStats {
    calls: MSM_CALLS.load(Ordering::Relaxed),
    batches: MSM_BATCHES.load(Ordering::Relaxed),
    scalars: MSM_SCALARS.load(Ordering::Relaxed),
    microseconds: MSM_MICROSECONDS.load(Ordering::Relaxed),
  }
}

/// A trait that provides the ability to perform multi-scalar multiplication in variable time
pub fn vartime_multiscalar_mul(scalars: &[Scalar], bases: &[Affine]) -> Point {
  let start = Instant::now();
  let mut blitzar_commitments = vec![Point::default(); 1];

  let scalar_bytes: Vec<[u8; 32]> = scalars.par_iter().map(|s| s.to_bytes()).collect();

  blitzar::compute::compute_bn254_g1_uncompressed_commitments_with_halo2_generators(
    &mut blitzar_commitments,
    &[(&scalar_bytes).into()],
    bases,
  );

  record_msm(1, scalars.len(), start.elapsed());
  blitzar_commitments[0]
}

/// A trait that provides the ability to perform a batch of multi-scalar multiplication in variable time
pub fn batch_vartime_multiscalar_mul(scalars: &[Vec<Scalar>], bases: &[Affine]) -> Vec<Point> {
  let start = Instant::now();
  let mut blitzar_commitments = vec![Point::default(); scalars.len()];

  let scalar_bytes: Vec<Vec<[u8; 32]>> = scalars
    .par_iter()
    .map(|s| s.par_iter().map(|v| v.to_bytes()).collect())
    .collect();

  let scalars_table: Vec<blitzar::sequence::Sequence<'_>> =
    scalar_bytes.par_iter().map(|s| s.into()).collect();

  blitzar::compute::compute_bn254_g1_uncompressed_commitments_with_halo2_generators(
    &mut blitzar_commitments,
    &scalars_table,
    bases,
  );

  let scalar_count = scalars
    .iter()
    .map(Vec::len)
    .fold(0_usize, usize::saturating_add);
  record_msm(scalars.len(), scalar_count, start.elapsed());
  blitzar_commitments
}

/// Records one completed provider call without imposing synchronization.
fn record_msm(batches: usize, scalars: usize, elapsed: Duration) {
  MSM_CALLS.fetch_add(1, Ordering::Relaxed);
  MSM_BATCHES.fetch_add(usize_to_u64(batches), Ordering::Relaxed);
  MSM_SCALARS.fetch_add(usize_to_u64(scalars), Ordering::Relaxed);
  MSM_MICROSECONDS.fetch_add(duration_microseconds(elapsed), Ordering::Relaxed);
}

/// Saturates a host size at the telemetry counter's representation limit.
fn usize_to_u64(value: usize) -> u64 {
  u64::try_from(value).unwrap_or(u64::MAX)
}

/// Saturates a duration at the telemetry counter's representation limit.
fn duration_microseconds(duration: Duration) -> u64 {
  u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;
  use halo2curves::msm::msm_best;

  #[test]
  fn test_vartime_multiscalar_mul_empty() {
    let scalars = vec![];
    let bases = vec![];

    let result = vartime_multiscalar_mul(&scalars, &bases);

    assert_eq!(result, Point::default());
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul_empty() {
    let scalars = vec![vec![]];
    let bases = vec![];

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    assert_eq!(result, [Point::default(); 1]);
  }

  #[test]
  fn test_vartime_multiscalar_mul_simple() {
    let mut rng = rand::thread_rng();

    let scalars = vec![Scalar::random(&mut rng), Scalar::random(&mut rng)];
    let bases = vec![Affine::random(&mut rng), Affine::random(&mut rng)];

    let result = vartime_multiscalar_mul(&scalars, &bases);

    let expected = bases[0] * scalars[0] + bases[1] * scalars[1];

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul_simple() {
    let mut rng = rand::thread_rng();

    let scalars = vec![
      vec![Scalar::random(&mut rng), Scalar::random(&mut rng)],
      vec![Scalar::random(&mut rng), Scalar::random(&mut rng)],
    ];
    let bases = vec![Affine::random(&mut rng), Affine::random(&mut rng)];

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    assert_eq!(
      result[0],
      bases[0] * scalars[0][0] + bases[1] * scalars[0][1]
    );
    assert_eq!(
      result[1],
      bases[0] * scalars[1][0] + bases[1] * scalars[1][1]
    );
  }

  #[test]
  fn test_vartime_multiscalar_mul() {
    let mut rng = rand::thread_rng();
    let sample_len = 100;

    let (scalars, bases): (Vec<_>, Vec<_>) = (0..sample_len)
      .map(|_| (Scalar::random(&mut rng), Affine::random(&mut rng)))
      .unzip();

    let result = vartime_multiscalar_mul(&scalars, &bases);

    let mut expected = Point::default();
    for i in 0..sample_len {
      expected += bases[i] * scalars[i];
    }

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul() {
    let mut rng = rand::thread_rng();
    let batch_len = 20;
    let sample_len = 100;

    let scalars: Vec<Vec<Scalar>> = (0..batch_len)
      .map(|_| (0..sample_len).map(|_| Scalar::random(&mut rng)).collect())
      .collect();

    let bases: Vec<Affine> = (0..sample_len).map(|_| Affine::random(&mut rng)).collect();

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    let expected: Vec<Point> = scalars
      .iter()
      .map(|scalar_row| {
        scalar_row
          .iter()
          .enumerate()
          .map(|(i, scalar)| bases[i] * scalar)
          .sum()
      })
      .collect();

    assert_eq!(result, expected);
  }

  #[test]
  fn test_vartime_multiscalar_mul_with_msm_best() {
    let mut rng = rand::thread_rng();
    let sample_len = 100;

    let (scalars, bases): (Vec<_>, Vec<_>) = (0..sample_len)
      .map(|_| (Scalar::random(&mut rng), Affine::random(&mut rng)))
      .unzip();

    let result = vartime_multiscalar_mul(&scalars, &bases);
    let expected = msm_best(&scalars, &bases);

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul_with_msm_best() {
    let mut rng = rand::thread_rng();
    let batch_len = 20;
    let sample_len = 100;

    let scalars: Vec<Vec<Scalar>> = (0..batch_len)
      .map(|_| (0..sample_len).map(|_| Scalar::random(&mut rng)).collect())
      .collect();

    let bases: Vec<Affine> = (0..sample_len).map(|_| Affine::random(&mut rng)).collect();

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    let expected = scalars
      .iter()
      .map(|scalar| msm_best(scalar, &bases))
      .collect::<Vec<_>>();

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul_of_varying_sized_scalars_with_msm_best() {
    let mut rng = rand::thread_rng();
    let batch_len = 20;
    let sample_lens: Vec<usize> = (0..batch_len).map(|i| i * 100 / (batch_len - 1)).collect();

    let scalars: Vec<Vec<Scalar>> = (0..batch_len)
      .map(|i| {
        (0..sample_lens[i])
          .map(|_| Scalar::random(&mut rng))
          .collect()
      })
      .collect();

    let bases: Vec<Affine> = (0..sample_lens[batch_len - 1])
      .map(|_| Affine::random(&mut rng))
      .collect();

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    let expected = scalars
      .iter()
      .map(|scalar| msm_best(scalar, &bases[..scalar.len()]))
      .collect::<Vec<_>>();

    assert_eq!(result, expected);
  }
}
