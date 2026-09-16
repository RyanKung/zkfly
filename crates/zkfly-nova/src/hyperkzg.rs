//! Standalone vector and sparse-row commitments backed by Nova's `HyperKZG`.
//!
//! Nova uses the same primary commitment engine for its R1CS vectors. This
//! module exposes that engine for callers that need a separately committed
//! input vector or sparse matrix row, while keeping the application-level
//! Poseidon topology commitment unchanged. A vector is encoded as a multilinear
//! polynomial in evaluation order; short vectors are padded with zeroes to the
//! configured power-of-two capacity.

use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use ff::Field;
use halo2curves::bn256::Fr;
use nova_snark::{
    errors::NovaError,
    provider::{
        Bn256EngineKZG,
        hyperkzg::{
            CommitmentEngine, EvaluationArgument, EvaluationEngine, ProverKey, VerifierKey,
        },
        keccak::Keccak256Transcript,
        ptau::MAX_PPOT_POWER,
    },
    traits::{
        Engine, TranscriptEngineTrait, commitment::CommitmentEngineTrait,
        evaluation::EvaluationEngineTrait,
    },
};

/// Nova's BN254 `HyperKZG` engine used for primary commitments.
type KzgEngine = Bn256EngineKZG;
/// The scalar field used by the BN254 primary circuit.
type KzgScalar = <KzgEngine as Engine>::Scalar;
/// The `HyperKZG` commitment engine implementation.
type KzgCommitmentEngine = CommitmentEngine<KzgEngine>;
/// The `HyperKZG` multilinear evaluation engine implementation.
type KzgEvaluationEngine = EvaluationEngine<KzgEngine>;
/// `HyperKZG`'s commitment-key type.
type KzgCommitmentKey = <KzgCommitmentEngine as CommitmentEngineTrait<KzgEngine>>::CommitmentKey;

/// A raw BN254 `HyperKZG` vector commitment.
pub type HyperKzgCommitment = <KzgCommitmentEngine as CommitmentEngineTrait<KzgEngine>>::Commitment;
/// A serialized `HyperKZG` multilinear evaluation argument.
pub type HyperKzgEvaluationArgument = EvaluationArgument<KzgEngine>;

/// Domain separator for standalone vector evaluation transcripts.
const VECTOR_OPENING_DOMAIN: &[u8] = b"zkfly/hyperkzg/vector-opening/v1";
/// Domain separator used when deriving the vector commitment key.
const VECTOR_COMMITMENT_LABEL: &[u8] = b"zkfly/hyperkzg/vector/v1";

/// Errors returned by the standalone `HyperKZG` vector API.
#[derive(Debug, thiserror::Error)]
pub enum HyperKzgError {
    /// A vector or matrix dimension is zero, overflows, or exceeds capacity.
    #[error("invalid HyperKZG vector length {length} for capacity {capacity}")]
    InvalidVectorLength {
        /// Supplied logical vector length.
        length: usize,
        /// Configured power-of-two capacity.
        capacity: usize,
    },
    /// A sparse row's indices and values have different lengths.
    #[error("sparse row has {indices} indices but {values} values")]
    SparseLengthMismatch {
        /// Number of sparse indices.
        indices: usize,
        /// Number of sparse values.
        values: usize,
    },
    /// A sparse row is not in canonical strictly increasing order.
    #[error("sparse row indices must be strictly increasing")]
    SparseIndicesNotCanonical,
    /// A sparse index lies outside the row's logical length.
    #[error("sparse index {index} is outside logical length {length}")]
    SparseIndexOutOfBounds {
        /// Invalid sparse index.
        index: usize,
        /// Logical row length.
        length: usize,
    },
    /// The opening point has the wrong number of multilinear coordinates.
    #[error("opening point has {actual} coordinates; expected {expected}")]
    PointLength {
        /// Number of supplied coordinates.
        actual: usize,
        /// Required number of coordinates.
        expected: usize,
    },
    /// The supplied batch has inconsistent lengths.
    #[error("batch has {commitments} commitments but {vectors} vectors")]
    BatchLengthMismatch {
        /// Number of commitments.
        commitments: usize,
        /// Number of vectors.
        vectors: usize,
    },
    /// The sparse matrix row has the wrong logical column count.
    #[error("matrix row has logical length {actual}; expected {expected}")]
    MatrixColumnMismatch {
        /// Supplied row length.
        actual: usize,
        /// Matrix column count.
        expected: usize,
    },
    /// A row commitment does not correspond to the supplied witness vector.
    #[error("vector commitment does not match its opening witness at batch index {index}")]
    CommitmentMismatch {
        /// Position in the opening batch.
        index: usize,
    },
    /// The opening argument batch does not match the commitment batch.
    #[error("opening batch contains {openings} arguments for {commitments} commitments")]
    OpeningLengthMismatch {
        /// Number of opening arguments.
        openings: usize,
        /// Number of commitments.
        commitments: usize,
    },
    /// The trusted setup directory contains no suitable SRS file.
    #[error("no suitable Powers-of-Tau file found in {directory} for {capacity} generators")]
    MissingPtau {
        /// Directory searched for `PTau` files.
        directory: PathBuf,
        /// Minimum number of generators required.
        capacity: usize,
    },
    /// File-system failure while loading the trusted setup.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Nova rejected setup, commitment, or evaluation proof construction.
    #[error(transparent)]
    Nova(#[from] NovaError),
    /// The `PTau` reader rejected the trusted setup file.
    #[error("invalid Powers-of-Tau setup: {message}")]
    Ptau {
        /// `PTau` parser detail.
        message: String,
    },
}

/// A sparse row supplied to [`HyperKzgVectorParameters::commit_sparse_matrix`].
#[derive(Clone, Debug)]
pub struct SparseMatrixRow {
    /// Logical number of columns in this row.
    pub logical_len: usize,
    /// Strictly increasing non-zero column positions.
    pub indices: Vec<usize>,
    /// Values corresponding to [`Self::indices`].
    pub values: Vec<Fr>,
}

/// A row-wise `HyperKZG` commitment to a sparse matrix.
#[derive(Clone, Debug)]
pub struct HyperKzgSparseMatrixCommitment {
    /// Logical number of columns in every committed row.
    column_count: usize,
    /// One vector commitment per matrix row.
    rows: Vec<HyperKzgVectorCommitment>,
}

impl HyperKzgSparseMatrixCommitment {
    /// Returns the logical number of matrix columns.
    #[must_use]
    pub const fn column_count(&self) -> usize {
        self.column_count
    }

    /// Returns the number of committed matrix rows.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Returns the row commitments in canonical row order.
    #[must_use]
    pub fn rows(&self) -> &[HyperKzgVectorCommitment] {
        &self.rows
    }
}

/// A committed vector together with its logical (unpadded) length.
#[derive(Clone, Debug)]
pub struct HyperKzgVectorCommitment {
    /// Raw `HyperKZG` group commitment.
    commitment: HyperKzgCommitment,
    /// Number of values supplied before zero padding.
    logical_len: usize,
}

impl HyperKzgVectorCommitment {
    /// Returns the raw `HyperKZG` commitment used by Nova.
    #[must_use]
    pub fn commitment(&self) -> HyperKzgCommitment {
        self.commitment
    }

    /// Returns the logical vector length before capacity padding.
    #[must_use]
    pub const fn logical_len(&self) -> usize {
        self.logical_len
    }
}

/// A batch of `HyperKZG` evaluation arguments at one common multilinear point.
#[derive(Clone, Debug)]
pub struct HyperKzgBatchOpening {
    /// Common multilinear evaluation point.
    point: Vec<Fr>,
    /// Claimed evaluations, one per committed vector.
    evaluations: Vec<Fr>,
    /// `HyperKZG` evaluation arguments, one per committed vector.
    arguments: Vec<HyperKzgEvaluationArgument>,
}

impl HyperKzgBatchOpening {
    /// Returns the common multilinear evaluation point.
    #[must_use]
    pub fn point(&self) -> &[Fr] {
        &self.point
    }

    /// Returns the claimed evaluations in batch order.
    #[must_use]
    pub fn evaluations(&self) -> &[Fr] {
        &self.evaluations
    }

    /// Returns the underlying `HyperKZG` arguments in batch order.
    #[must_use]
    pub fn arguments(&self) -> &[HyperKzgEvaluationArgument] {
        &self.arguments
    }
}

/// Backend-neutral interface for vector commitments and batched openings.
pub trait VectorCommitmentBackend {
    /// Commitment representation returned by this backend.
    type Commitment: Clone;
    /// Batched opening representation returned by this backend.
    type BatchOpening: Clone;
    /// Error type returned by this backend.
    type Error: std::error::Error;

    /// Returns the power-of-two vector capacity supported by the setup.
    fn capacity(&self) -> usize;

    /// Commits to a dense vector, padding it with zeroes to the configured capacity.
    ///
    /// # Errors
    ///
    /// Returns the backend-specific error when the vector is invalid.
    fn commit_vector(&self, values: &[Fr]) -> Result<Self::Commitment, Self::Error>;

    /// Commits to a canonical sparse vector without materializing zero entries.
    ///
    /// # Errors
    ///
    /// Returns the backend-specific error when the sparse vector is invalid.
    fn commit_sparse_vector(
        &self,
        logical_len: usize,
        indices: &[usize],
        values: &[Fr],
    ) -> Result<Self::Commitment, Self::Error>;

    /// Commits to a batch of dense vectors using the backend's batch MSM path.
    ///
    /// # Errors
    ///
    /// Returns the backend-specific error when a vector is invalid.
    fn commit_batch(&self, vectors: &[Vec<Fr>]) -> Result<Vec<Self::Commitment>, Self::Error>;

    /// Produces batched multilinear openings for the supplied vector witnesses.
    ///
    /// # Errors
    ///
    /// Returns the backend-specific error when a commitment, witness, or point
    /// is invalid.
    fn open_batch(
        &self,
        commitments: &[Self::Commitment],
        vectors: &[Vec<Fr>],
        point: &[Fr],
    ) -> Result<Self::BatchOpening, Self::Error>;

    /// Verifies the batched openings against their commitments and evaluations.
    ///
    /// # Errors
    ///
    /// Returns the backend-specific error when the batch is malformed or Nova
    /// rejects an evaluation argument.
    fn verify_batch(
        &self,
        commitments: &[Self::Commitment],
        opening: &Self::BatchOpening,
    ) -> Result<bool, Self::Error>;
}

/// Reusable `HyperKZG` setup for dense vectors and sparse matrix rows.
#[derive(Clone, Debug)]
pub struct HyperKzgVectorParameters {
    /// Trusted `HyperKZG` commitment key.
    commitment_key: KzgCommitmentKey,
    /// `HyperKZG` prover key for multilinear evaluations.
    prover_key: ProverKey<KzgEngine>,
    /// `HyperKZG` verifier key for multilinear evaluations.
    verifier_key: VerifierKey<KzgEngine>,
    /// Power-of-two number of supported vector entries.
    capacity: usize,
}

impl HyperKzgVectorParameters {
    /// Loads a trusted setup from the first suitable `PTau` file in `ptau_dir`.
    ///
    /// The loader accepts the same file naming conventions as Nova's R1CS
    /// loader: `ppot_pruned_XX.ptau`, `ppot_0080_XX.ptau`, and
    /// `ppot_0080_final.ptau`.
    ///
    /// # Errors
    ///
    /// Returns an error when `capacity` is not a non-zero power of two, no
    /// suitable file exists, the `PTau` file is invalid, or Nova rejects setup.
    pub fn from_ptau_dir(ptau_dir: &Path, capacity: usize) -> Result<Self, HyperKzgError> {
        validate_capacity(capacity)?;
        let path = find_ptau_file(ptau_dir, capacity)?;
        Self::from_ptau_file(&path, capacity)
    }

    /// Loads a trusted setup from an explicit `PTau` file.
    ///
    /// # Errors
    ///
    /// Returns an error when `capacity` is invalid, the file cannot be read, or
    /// the `PTau` parser or `HyperKZG` setup rejects it.
    pub fn from_ptau_file(path: &Path, capacity: usize) -> Result<Self, HyperKzgError> {
        validate_capacity(capacity)?;
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let commitment_key = <KzgCommitmentEngine as CommitmentEngineTrait<KzgEngine>>::load_setup(
            &mut reader,
            VECTOR_COMMITMENT_LABEL,
            capacity,
        )
        .map_err(|error| HyperKzgError::Ptau {
            message: error.to_string(),
        })?;
        let (prover_key, verifier_key) =
            <KzgEvaluationEngine as EvaluationEngineTrait<KzgEngine>>::setup(&commitment_key)?;
        Ok(Self {
            commitment_key,
            prover_key,
            verifier_key,
            capacity,
        })
    }

    /// Returns a dense vector commitment.
    ///
    /// # Errors
    ///
    /// Returns an error when the vector is empty or exceeds the configured
    /// capacity.
    pub fn commit_vector(&self, values: &[Fr]) -> Result<HyperKzgVectorCommitment, HyperKzgError> {
        let padded = self.pad_vector(values)?;
        let commitment = <KzgCommitmentEngine as CommitmentEngineTrait<KzgEngine>>::commit(
            &self.commitment_key,
            &padded,
            &KzgScalar::ZERO,
        );
        Ok(HyperKzgVectorCommitment {
            commitment,
            logical_len: values.len(),
        })
    }

    /// Returns a canonical sparse vector commitment.
    ///
    /// # Errors
    ///
    /// Returns an error when lengths differ, indices are not strictly
    /// increasing, or an index is outside `logical_len`/the setup capacity.
    pub fn commit_sparse_vector(
        &self,
        logical_len: usize,
        indices: &[usize],
        values: &[Fr],
    ) -> Result<HyperKzgVectorCommitment, HyperKzgError> {
        validate_logical_len(logical_len, self.capacity)?;
        validate_sparse_indices(logical_len, indices, values)?;
        let commitment = <KzgCommitmentEngine as CommitmentEngineTrait<KzgEngine>>::commit_sparse(
            &self.commitment_key,
            indices,
            values,
            &KzgScalar::ZERO,
        );
        Ok(HyperKzgVectorCommitment {
            commitment,
            logical_len,
        })
    }

    /// Commits each sparse row using the same trusted `HyperKZG` setup.
    ///
    /// # Errors
    ///
    /// Returns an error when a row's logical column count differs from
    /// `column_count`, or when a sparse row is malformed.
    pub fn commit_sparse_matrix(
        &self,
        column_count: usize,
        rows: &[SparseMatrixRow],
    ) -> Result<HyperKzgSparseMatrixCommitment, HyperKzgError> {
        validate_logical_len(column_count, self.capacity)?;
        let mut commitments = Vec::with_capacity(rows.len());
        for row in rows {
            if row.logical_len != column_count {
                return Err(HyperKzgError::MatrixColumnMismatch {
                    actual: row.logical_len,
                    expected: column_count,
                });
            }
            commitments.push(self.commit_sparse_vector(
                row.logical_len,
                &row.indices,
                &row.values,
            )?);
        }
        Ok(HyperKzgSparseMatrixCommitment {
            column_count,
            rows: commitments,
        })
    }

    /// Returns the configured power-of-two vector capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Loads and pads a dense witness to the setup capacity.
    fn pad_vector(&self, values: &[Fr]) -> Result<Vec<Fr>, HyperKzgError> {
        validate_logical_len(values.len(), self.capacity)?;
        let mut padded = values.to_vec();
        padded.resize(self.capacity, Fr::ZERO);
        Ok(padded)
    }

    /// Computes a multilinear evaluation and a `HyperKZG` proof batch.
    fn open_batch_inner(
        &self,
        commitments: &[HyperKzgVectorCommitment],
        vectors: &[Vec<Fr>],
        point: &[Fr],
    ) -> Result<HyperKzgBatchOpening, HyperKzgError> {
        if commitments.len() != vectors.len() {
            return Err(HyperKzgError::BatchLengthMismatch {
                commitments: commitments.len(),
                vectors: vectors.len(),
            });
        }
        validate_point(point, self.capacity)?;
        let mut evaluations = Vec::with_capacity(vectors.len());
        let mut arguments = Vec::with_capacity(vectors.len());
        let mut vectors_iter = vectors.iter();
        for (index, commitment) in commitments.iter().enumerate() {
            let vector = vectors_iter
                .next()
                .ok_or(HyperKzgError::BatchLengthMismatch {
                    commitments: commitments.len(),
                    vectors: vectors.len(),
                })?;
            if commitment.logical_len != vector.len() {
                return Err(HyperKzgError::InvalidVectorLength {
                    length: vector.len(),
                    capacity: commitment.logical_len,
                });
            }
            let padded = self.pad_vector(vector)?;
            let expected = self.commit_vector(vector)?;
            if expected.commitment != commitment.commitment {
                return Err(HyperKzgError::CommitmentMismatch { index });
            }
            let evaluation = multilinear_evaluate(&padded, point)?;
            let mut transcript = KzgTranscript::new(VECTOR_OPENING_DOMAIN);
            absorb_opening_instance(
                &mut transcript,
                index,
                &commitment.commitment,
                point,
                &evaluation,
            );
            let argument = <KzgEvaluationEngine as EvaluationEngineTrait<KzgEngine>>::prove(
                &self.commitment_key,
                &self.prover_key,
                &mut transcript,
                &commitment.commitment,
                &padded,
                point,
                &evaluation,
            )?;
            evaluations.push(evaluation);
            arguments.push(argument);
        }
        Ok(HyperKzgBatchOpening {
            point: point.to_vec(),
            evaluations,
            arguments,
        })
    }

    /// Verifies a `HyperKZG` proof batch.
    fn verify_batch_inner(
        &self,
        commitments: &[HyperKzgVectorCommitment],
        opening: &HyperKzgBatchOpening,
    ) -> Result<bool, HyperKzgError> {
        if commitments.len() != opening.arguments.len()
            || commitments.len() != opening.evaluations.len()
        {
            return Err(HyperKzgError::OpeningLengthMismatch {
                openings: opening.arguments.len(),
                commitments: commitments.len(),
            });
        }
        validate_point(&opening.point, self.capacity)?;
        for (index, commitment) in commitments.iter().enumerate() {
            let evaluation =
                opening
                    .evaluations
                    .get(index)
                    .ok_or(HyperKzgError::OpeningLengthMismatch {
                        openings: opening.evaluations.len(),
                        commitments: commitments.len(),
                    })?;
            let argument =
                opening
                    .arguments
                    .get(index)
                    .ok_or(HyperKzgError::OpeningLengthMismatch {
                        openings: opening.arguments.len(),
                        commitments: commitments.len(),
                    })?;
            let mut transcript = KzgTranscript::new(VECTOR_OPENING_DOMAIN);
            absorb_opening_instance(
                &mut transcript,
                index,
                &commitment.commitment,
                &opening.point,
                evaluation,
            );
            <KzgEvaluationEngine as EvaluationEngineTrait<KzgEngine>>::verify(
                &self.verifier_key,
                &mut transcript,
                &commitment.commitment,
                &opening.point,
                evaluation,
                argument,
            )?;
        }
        Ok(true)
    }
}

impl VectorCommitmentBackend for HyperKzgVectorParameters {
    type Commitment = HyperKzgVectorCommitment;
    type BatchOpening = HyperKzgBatchOpening;
    type Error = HyperKzgError;

    fn capacity(&self) -> usize {
        self.capacity()
    }

    fn commit_vector(&self, values: &[Fr]) -> Result<Self::Commitment, Self::Error> {
        self.commit_vector(values)
    }

    fn commit_sparse_vector(
        &self,
        logical_len: usize,
        indices: &[usize],
        values: &[Fr],
    ) -> Result<Self::Commitment, Self::Error> {
        self.commit_sparse_vector(logical_len, indices, values)
    }

    fn commit_batch(&self, vectors: &[Vec<Fr>]) -> Result<Vec<Self::Commitment>, Self::Error> {
        let padded = vectors
            .iter()
            .map(|vector| self.pad_vector(vector))
            .collect::<Result<Vec<_>, _>>()?;
        let blinds = vec![KzgScalar::ZERO; padded.len()];
        let commitments = <KzgCommitmentEngine as CommitmentEngineTrait<KzgEngine>>::batch_commit(
            &self.commitment_key,
            &padded,
            &blinds,
        );
        let mut result = Vec::with_capacity(vectors.len());
        let mut commitment_iter = commitments.into_iter();
        for vector in vectors {
            let commitment = commitment_iter
                .next()
                .ok_or(HyperKzgError::BatchLengthMismatch {
                    commitments: vectors.len(),
                    vectors: vectors.len(),
                })?;
            result.push(HyperKzgVectorCommitment {
                commitment,
                logical_len: vector.len(),
            });
        }
        Ok(result)
    }

    fn open_batch(
        &self,
        commitments: &[Self::Commitment],
        vectors: &[Vec<Fr>],
        point: &[Fr],
    ) -> Result<Self::BatchOpening, Self::Error> {
        self.open_batch_inner(commitments, vectors, point)
    }

    fn verify_batch(
        &self,
        commitments: &[Self::Commitment],
        opening: &Self::BatchOpening,
    ) -> Result<bool, Self::Error> {
        self.verify_batch_inner(commitments, opening)
    }
}

/// Transcript type used by standalone `HyperKZG` openings.
type KzgTranscript = Keccak256Transcript<KzgEngine>;

/// Binds one opening instance before calling Nova's evaluation engine.
fn absorb_opening_instance(
    transcript: &mut KzgTranscript,
    index: usize,
    commitment: &HyperKzgCommitment,
    point: &[Fr],
    evaluation: &Fr,
) {
    transcript.dom_sep(VECTOR_OPENING_DOMAIN);
    let index = KzgScalar::from(u64::try_from(index).unwrap_or(u64::MAX));
    transcript.absorb(b"index", &index);
    transcript.absorb(b"commitment", commitment);
    transcript.absorb(b"point", &point);
    transcript.absorb(b"evaluation", evaluation);
}

/// Evaluates an evaluation-form multilinear polynomial at `point`.
fn multilinear_evaluate(values: &[Fr], point: &[Fr]) -> Result<Fr, HyperKzgError> {
    let point_bits = u32::try_from(point.len()).unwrap_or(u32::MAX);
    if values.len() != 1_usize.checked_shl(point_bits).unwrap_or(0) {
        return Err(HyperKzgError::PointLength {
            actual: point.len(),
            expected: usize::try_from(values.len().ilog2()).unwrap_or(usize::MAX),
        });
    }
    let mut layer = values.to_vec();
    for coordinate in point.iter().rev() {
        let mut next = Vec::with_capacity(layer.len() / 2);
        let mut pairs = layer.chunks_exact(2);
        for pair in &mut pairs {
            let left = pair.first().copied().unwrap_or(Fr::ZERO);
            let right = pair.get(1).copied().unwrap_or(Fr::ZERO);
            next.push(left * (Fr::ONE - *coordinate) + right * *coordinate);
        }
        layer = next;
    }
    layer
        .first()
        .copied()
        .ok_or(HyperKzgError::InvalidVectorLength {
            length: 0,
            capacity: values.len(),
        })
}

/// Validates a power-of-two setup capacity.
fn validate_capacity(capacity: usize) -> Result<(), HyperKzgError> {
    if capacity == 0 || !capacity.is_power_of_two() {
        return Err(HyperKzgError::InvalidVectorLength {
            length: capacity,
            capacity,
        });
    }
    Ok(())
}

/// Validates a logical vector length against setup capacity.
fn validate_logical_len(length: usize, capacity: usize) -> Result<(), HyperKzgError> {
    if length == 0 || length > capacity {
        return Err(HyperKzgError::InvalidVectorLength { length, capacity });
    }
    Ok(())
}

/// Validates canonical sparse row indices and values.
fn validate_sparse_indices(
    logical_len: usize,
    indices: &[usize],
    values: &[Fr],
) -> Result<(), HyperKzgError> {
    if indices.len() != values.len() {
        return Err(HyperKzgError::SparseLengthMismatch {
            indices: indices.len(),
            values: values.len(),
        });
    }
    let mut previous = None;
    for &index in indices {
        if index >= logical_len {
            return Err(HyperKzgError::SparseIndexOutOfBounds {
                index,
                length: logical_len,
            });
        }
        if previous.is_some_and(|last| index <= last) {
            return Err(HyperKzgError::SparseIndicesNotCanonical);
        }
        previous = Some(index);
    }
    Ok(())
}

/// Validates that the point dimension matches the configured capacity.
fn validate_point(point: &[Fr], capacity: usize) -> Result<(), HyperKzgError> {
    let expected = usize::try_from(capacity.trailing_zeros()).unwrap_or(usize::MAX);
    if point.len() != expected {
        return Err(HyperKzgError::PointLength {
            actual: point.len(),
            expected,
        });
    }
    Ok(())
}

/// Selects the smallest supported `PTau` file containing `capacity` generators.
fn find_ptau_file(ptau_dir: &Path, capacity: usize) -> Result<PathBuf, HyperKzgError> {
    let min_power = capacity.trailing_zeros();
    for power in min_power..=MAX_PPOT_POWER {
        let pruned = ptau_dir.join(format!("ppot_pruned_{power:02}.ptau"));
        if pruned.is_file() {
            return Ok(pruned);
        }
        let original = if power == MAX_PPOT_POWER {
            ptau_dir.join("ppot_0080_final.ptau")
        } else {
            ptau_dir.join(format!("ppot_0080_{power:02}.ptau"))
        };
        if original.is_file() {
            return Ok(original);
        }
    }
    Err(HyperKzgError::MissingPtau {
        directory: ptau_dir.to_path_buf(),
        capacity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parameters() -> Result<HyperKzgVectorParameters, HyperKzgError> {
        let commitment_key = <KzgCommitmentEngine as CommitmentEngineTrait<KzgEngine>>::setup(
            VECTOR_COMMITMENT_LABEL,
            4,
        )?;
        let (prover_key, verifier_key) =
            <KzgEvaluationEngine as EvaluationEngineTrait<KzgEngine>>::setup(&commitment_key)?;
        Ok(HyperKzgVectorParameters {
            commitment_key,
            prover_key,
            verifier_key,
            capacity: 4,
        })
    }

    #[test]
    fn vector_batch_opening_verifies_and_rejects_tampering() -> Result<(), HyperKzgError> {
        let parameters = test_parameters()?;
        let vectors = vec![
            vec![Fr::from(1_u64), Fr::from(2_u64), Fr::from(3_u64)],
            vec![Fr::from(4_u64), Fr::from(5_u64)],
        ];
        let commitments = parameters.commit_batch(&vectors)?;
        let point = vec![Fr::from(2_u64), Fr::from(3_u64)];
        let opening = parameters.open_batch(&commitments, &vectors, &point)?;
        assert!(parameters.verify_batch(&commitments, &opening)?);
        let mut bad = opening.clone();
        if let Some(value) = bad.evaluations.first_mut() {
            *value += Fr::ONE;
        }
        assert!(parameters.verify_batch(&commitments, &bad).is_err());
        Ok(())
    }

    #[test]
    fn sparse_matrix_rows_use_canonical_sparse_commitments() -> Result<(), HyperKzgError> {
        let parameters = test_parameters()?;
        let matrix = parameters.commit_sparse_matrix(
            4,
            &[SparseMatrixRow {
                logical_len: 4,
                indices: vec![0, 3],
                values: vec![Fr::from(2_u64), Fr::from(5_u64)],
            }],
        )?;
        assert_eq!(matrix.row_count(), 1);
        assert_eq!(matrix.column_count(), 4);
        Ok(())
    }
}
