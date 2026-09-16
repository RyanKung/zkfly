//! This module provides a multi-scalar multiplication routine
/// Adapted from zcash/halo2
use ff::PrimeField;
use group::{prime::PrimeCurveAffine, Group as GroupTrait};
use itertools::Itertools as _;
use pasta_curves::{self, arithmetic::CurveAffine, group::Group as AnotherGroup};
use rayon::{current_num_threads, prelude::*};

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
use ark_ff::{BigInteger, PrimeField as ArkPrimeField};

fn cpu_msm_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  let c = if bases.len() < 4 {
    1
  } else if bases.len() < 32 {
    3
  } else {
    (f64::from(bases.len() as u32)).ln().ceil() as usize
  };

  fn get_at<F: PrimeField>(segment: usize, c: usize, bytes: &F::Repr) -> usize {
    let skip_bits = segment * c;
    let skip_bytes = skip_bits / 8;

    if skip_bytes >= 32 {
      return 0;
    }

    let mut v = [0; 8];
    #[allow(clippy::disallowed_methods)]
    for (v, o) in v.iter_mut().zip(bytes.as_ref()[skip_bytes..].iter()) {
      *v = *o;
    }

    let mut tmp = u64::from_le_bytes(v);
    tmp >>= skip_bits - (skip_bytes * 8);
    tmp %= 1 << c;

    tmp as usize
  }

  let segments = (256 / c) + 1;

  (0..segments)
    .rev()
    .fold(C::Curve::identity(), |mut acc, segment| {
      (0..c).for_each(|_| acc = acc.double());

      #[derive(Clone, Copy)]
      enum Bucket<C: CurveAffine> {
        None,
        Affine(C),
        Projective(C::Curve),
      }

      impl<C: CurveAffine> Bucket<C> {
        fn add_assign(&mut self, other: &C) {
          *self = match *self {
            Bucket::None => Bucket::Affine(*other),
            Bucket::Affine(a) => Bucket::Projective(a + *other),
            Bucket::Projective(a) => Bucket::Projective(a + other),
          }
        }

        fn add(self, other: C::Curve) -> C::Curve {
          match self {
            Bucket::None => other,
            Bucket::Affine(a) => other + a,
            Bucket::Projective(a) => other + a,
          }
        }
      }

      let mut buckets = vec![Bucket::None; (1 << c) - 1];

      for (coeff, base) in coeffs.iter().zip_eq(bases.iter()) {
        let coeff = get_at::<C::Scalar>(segment, c, &coeff.to_repr());
        if coeff != 0 {
          buckets[coeff - 1].add_assign(base);
        }
      }

      // Summation by parts
      // e.g. 3a + 2b + 1c = a +
      //                    (a) + b +
      //                    ((a) + b) + c
      let mut running_sum = C::Curve::identity();
      for exp in buckets.into_iter().rev() {
        running_sum = exp.add(running_sum);
        acc += &running_sum;
      }
      acc
    })
}

/// Performs a multi-scalar-multiplication operation without GPU acceleration.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
/// Adapted from zcash/halo2
pub(crate) fn cpu_best_msm<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  assert_eq!(coeffs.len(), bases.len());

  let num_threads = current_num_threads();
  if coeffs.len() > num_threads {
    let chunk = coeffs.len() / num_threads;
    coeffs
      .par_chunks(chunk)
      .zip_eq(bases.par_chunks(chunk))
      .map(|(coeffs, bases)| cpu_msm_serial(coeffs, bases))
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    cpu_msm_serial(coeffs, bases)
  }
}

/// Selects a GPU implementation for the curve-specific MSM when the CUDA
/// provider is compiled for a Linux x86_64 target.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn gpu_best_msm<C: CurveAffine + GpuMsmCurve>(
  coeffs: &[C::Scalar],
  bases: &[C],
) -> Option<C::Curve> {
  if coeffs.len() != bases.len() || coeffs.len() < GPU_MSM_MIN_POINTS {
    return None;
  }
  C::gpu_msm(coeffs, bases)
}

/// Runs a deterministic BN256 MSM through the CUDA provider and compares it
/// with the CPU reference implementation.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn gpu_msm_self_test() -> bool {
  let scalars = (0..GPU_MSM_MIN_POINTS)
    .map(|index| bn256_scalar_from_index(index, 1))
    .collect::<Vec<_>>();
  let bases = (0..GPU_MSM_MIN_POINTS)
    .map(|index| {
      let scalar = bn256_scalar_from_index(index, 2);
      halo2curves::bn256::G1Affine::from(halo2curves::bn256::G1::generator() * scalar)
    })
    .collect::<Vec<_>>();
  let expected = cpu_best_msm(&scalars, &bases);
  gpu_best_msm::<halo2curves::bn256::G1Affine>(&scalars, &bases)
    .is_some_and(|actual| actual == expected)
}

/// Runs a deterministic Grumpkin MSM through the CUDA provider and compares it
/// with the CPU reference implementation.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn gpu_grumpkin_msm_self_test() -> bool {
  let scalars = (0..GPU_MSM_MIN_POINTS)
    .map(|index| grumpkin_scalar_from_index(index, 1))
    .collect::<Vec<_>>();
  let bases = (0..GPU_MSM_MIN_POINTS)
    .map(|index| {
      let scalar = grumpkin_scalar_from_index(index, 2);
      halo2curves::grumpkin::G1Affine::from(halo2curves::grumpkin::G1::generator() * scalar)
    })
    .collect::<Vec<_>>();
  let expected = cpu_best_msm(&scalars, &bases);
  gpu_best_msm::<halo2curves::grumpkin::G1Affine>(&scalars, &bases)
    .is_some_and(|actual| actual == expected)
}

/// Builds a deterministic scalar for the CUDA self-test without randomness.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
fn bn256_scalar_from_index(index: usize, offset: u64) -> halo2curves::bn256::Fr {
  halo2curves::bn256::Fr::from(
    u64::try_from(index)
      .unwrap_or(u64::MAX)
      .saturating_add(offset),
  )
}

/// Builds a deterministic Grumpkin scalar for the CUDA self-test.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
fn grumpkin_scalar_from_index(index: usize, offset: u64) -> halo2curves::grumpkin::Fr {
  halo2curves::grumpkin::Fr::from(
    u64::try_from(index)
      .unwrap_or(u64::MAX)
      .saturating_add(offset),
  )
}

/// Minimum MSM length at which the CUDA provider is eligible.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
const GPU_MSM_MIN_POINTS: usize = 1;

/// Provides curve-specific conversion and dispatch for the Blitzar CUDA MSM.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub(crate) trait GpuMsmCurve: CurveAffine {
  /// Runs one MSM on the configured GPU and converts the result back.
  fn gpu_msm(coeffs: &[Self::Scalar], bases: &[Self]) -> Option<Self::CurveExt>;
}

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
impl GpuMsmCurve for halo2curves::bn256::G1Affine {
  fn gpu_msm(coeffs: &[Self::Scalar], bases: &[Self]) -> Option<Self::CurveExt> {
    let ark_bases = coeffs_and_bases_to_ark_bn254(bases)?;
    let scalar_bytes = coeffs
      .iter()
      .map(|scalar| scalar.to_repr())
      .collect::<Vec<_>>();
    let sequence = blitzar::sequence::Sequence::from(scalar_bytes.as_slice());
    let mut output = vec![ark_bn254::G1Affine::default()];
    blitzar::compute::compute_bn254_g1_uncompressed_commitments_with_generators(
      &mut output,
      &[sequence],
      &ark_bases,
    );
    output.first().and_then(ark_bn254_to_halo2)
  }
}

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
impl GpuMsmCurve for halo2curves::grumpkin::G1Affine {
  fn gpu_msm(coeffs: &[Self::Scalar], bases: &[Self]) -> Option<Self::CurveExt> {
    let ark_bases = coeffs_and_bases_to_ark_grumpkin(bases)?;
    let scalar_bytes = coeffs
      .iter()
      .map(|scalar| scalar.to_repr())
      .collect::<Vec<_>>();
    let sequence = blitzar::sequence::Sequence::from(scalar_bytes.as_slice());
    let mut output = vec![ark_grumpkin::Affine::default()];
    blitzar::compute::compute_grumpkin_uncompressed_commitments_with_generators(
      &mut output,
      &[sequence],
      &ark_bases,
    );
    output.first().and_then(ark_grumpkin_to_halo2)
  }
}

/// Converts halo2 BN256 affine points into the arkworks representation used by
/// the Blitzar C ABI without relying on either crate's private field layout.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
fn coeffs_and_bases_to_ark_bn254(
  bases: &[halo2curves::bn256::G1Affine],
) -> Option<Vec<ark_bn254::G1Affine>> {
  bases
    .iter()
    .map(|base| {
      if *base == halo2curves::bn256::G1Affine::identity() {
        return Some(ark_bn254::G1Affine::default());
      }
      Some(ark_bn254::G1Affine::new_unchecked(
        ark_field_from_repr(&base.x.to_repr())?,
        ark_field_from_repr(&base.y.to_repr())?,
      ))
    })
    .collect()
}

/// Converts halo2 Grumpkin affine points into the arkworks representation used
/// by the Blitzar C ABI without relying on either crate's private field layout.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
fn coeffs_and_bases_to_ark_grumpkin(
  bases: &[halo2curves::grumpkin::G1Affine],
) -> Option<Vec<ark_grumpkin::Affine>> {
  bases
    .iter()
    .map(|base| {
      if *base == halo2curves::grumpkin::G1Affine::identity() {
        return Some(ark_grumpkin::Affine::default());
      }
      Some(ark_grumpkin::Affine::new_unchecked(
        ark_grumpkin_field_from_repr(&base.x.to_repr())?,
        ark_grumpkin_field_from_repr(&base.y.to_repr())?,
      ))
    })
    .collect()
}

/// Converts one canonical BN254 base-field representation into arkworks.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
fn ark_field_from_repr(bytes: &[u8; 32]) -> Option<ark_bn254::Fq> {
  Some(<ark_bn254::Fq as ArkPrimeField>::from_le_bytes_mod_order(
    bytes,
  ))
}

/// Converts one canonical Grumpkin base-field representation into arkworks.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
fn ark_grumpkin_field_from_repr(bytes: &[u8; 32]) -> Option<ark_grumpkin::Fq> {
  Some(<ark_grumpkin::Fq as ArkPrimeField>::from_le_bytes_mod_order(bytes))
}

/// Converts one arkworks BN254 result into a halo2 projective point.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
fn ark_bn254_to_halo2(point: &ark_bn254::G1Affine) -> Option<halo2curves::bn256::G1> {
  if point.infinity {
    return Some(halo2curves::bn256::G1::identity());
  }
  let x = halo2_field_from_repr(&point.x.into_bigint().to_bytes_le())?;
  let y = halo2_field_from_repr(&point.y.into_bigint().to_bytes_le())?;
  Some(halo2curves::bn256::G1Affine { x, y }.into())
}

/// Converts one arkworks Grumpkin result into a halo2 projective point.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
fn ark_grumpkin_to_halo2(point: &ark_grumpkin::Affine) -> Option<halo2curves::grumpkin::G1> {
  if point.infinity {
    return Some(halo2curves::grumpkin::G1::identity());
  }
  let x = halo2_grumpkin_field_from_repr(&point.x.into_bigint().to_bytes_le())?;
  let y = halo2_grumpkin_field_from_repr(&point.y.into_bigint().to_bytes_le())?;
  Some(halo2curves::grumpkin::G1Affine { x, y }.into())
}

/// Converts one canonical representation into halo2 BN256's base field.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
fn halo2_field_from_repr(bytes: &[u8]) -> Option<halo2curves::bn256::Fq> {
  let repr = <[u8; 32]>::try_from(bytes).ok()?;
  Option::from(halo2curves::bn256::Fq::from_repr(repr))
}

/// Converts one canonical representation into halo2 Grumpkin's base field.
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
fn halo2_grumpkin_field_from_repr(bytes: &[u8]) -> Option<halo2curves::grumpkin::Fq> {
  let repr = <[u8; 32]>::try_from(bytes).ok()?;
  Option::from(halo2curves::grumpkin::Fq::from_repr(repr))
}

#[cfg(test)]
mod tests {
  use super::cpu_best_msm;

  use crate::provider::{
    bn256_grumpkin::{bn256, grumpkin},
    secp_secq::{secp256k1, secq256k1},
  };
  use group::{ff::Field, Group};
  use halo2curves::CurveAffine;
  use itertools::Itertools as _;
  use pasta_curves::{pallas, vesta};
  use rand_core::OsRng;

  fn test_msm_with<F: Field, A: CurveAffine<ScalarExt = F>>() {
    let n = 8;
    let coeffs = (0..n).map(|_| F::random(OsRng)).collect::<Vec<_>>();
    let bases = (0..n)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect::<Vec<_>>();
    let naive = coeffs
      .iter()
      .zip_eq(bases.iter())
      .fold(A::CurveExt::identity(), |acc, (coeff, base)| {
        acc + *base * coeff
      });
    let msm = cpu_best_msm(&coeffs, &bases);

    assert_eq!(naive, msm)
  }

  #[test]
  fn test_msm() {
    test_msm_with::<pallas::Scalar, pallas::Affine>();
    test_msm_with::<vesta::Scalar, vesta::Affine>();
    test_msm_with::<bn256::Scalar, bn256::Affine>();
    test_msm_with::<grumpkin::Scalar, grumpkin::Affine>();
    test_msm_with::<secp256k1::Scalar, secp256k1::Affine>();
    test_msm_with::<secq256k1::Scalar, secq256k1::Affine>();
  }

  /// Checks that the CUDA BN254 provider agrees with the CPU reference.
  #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
  #[test]
  fn test_gpu_bn256_msm_matches_cpu() {
    let scalars = (0..GPU_MSM_MIN_POINTS)
      .map(|index| bn256::Scalar::from(u64::try_from(index + 1).unwrap_or(u64::MAX)))
      .collect::<Vec<_>>();
    let bases = (0..GPU_MSM_MIN_POINTS)
      .map(|index| {
        let scalar = bn256::Scalar::from(u64::try_from(index + 2).unwrap_or(u64::MAX));
        bn256::Affine::from(bn256::Point::generator() * scalar)
      })
      .collect::<Vec<_>>();
    let expected = cpu_best_msm(&scalars, &bases);
    let actual = super::gpu_best_msm::<bn256::Affine>(&scalars, &bases);
    assert!(actual.is_some());
    if let Some(actual) = actual {
      assert_eq!(actual, expected);
    }
  }
}
