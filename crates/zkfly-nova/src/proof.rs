//! Nova recursive-proof wrapper for topology fold transitions.

use crate::poseidon::PoseidonParameters;
use crate::{TopologyNovaError, TopologyStepCircuit};
use ff::{Field, PrimeField};
use halo2curves::bn256::Fr;
use nova_snark::{
    nova::{PublicParams, RecursiveSNARK},
    provider::GrumpkinEngine,
    traits::snark::default_ck_hint,
};
use zkfly_commitment::{COMMITMENT_BYTES, Commitment, TopologyFoldStep};

#[cfg(not(feature = "hyperkzg"))]
use nova_snark::provider::Bn256EngineIPA;
#[cfg(feature = "hyperkzg")]
use nova_snark::provider::Bn256EngineKZG;

#[cfg(feature = "hyperkzg")]
type PrimaryEngine = Bn256EngineKZG;
#[cfg(not(feature = "hyperkzg"))]
type PrimaryEngine = Bn256EngineIPA;
type SecondaryEngine = GrumpkinEngine;
type Parameters = PublicParams<PrimaryEngine, SecondaryEngine, TopologyStepCircuit>;
type RecursiveProof = RecursiveSNARK<PrimaryEngine, SecondaryEngine, TopologyStepCircuit>;

/// Creates topology public parameters using the selected Nova backend.
fn setup_parameters(circuit: &TopologyStepCircuit) -> Result<Parameters, TopologyNovaError> {
    #[cfg(all(feature = "hyperkzg", not(test)))]
    {
        let _ = circuit;
        Err(TopologyNovaError::HyperKzgSetupRequired)
    }
    #[cfg(any(not(feature = "hyperkzg"), test))]
    {
        Ok(Parameters::setup(
            circuit,
            &*default_ck_hint(),
            &*default_ck_hint(),
        )?)
    }
}

/// A Nova recursive proof for a sequence of Poseidon topology transitions.
///
/// The proof stores public parameters because the first MVP is a self-contained
/// benchmark artifact. A production deployment should cache and version these
/// parameters separately from each proof.
pub struct TopologyNovaProof {
    /// Nova public parameters for the fixed step circuit shape.
    parameters: Parameters,
    /// The recursive proof object produced by the official `nova-snark` crate.
    recursive: RecursiveProof,
    /// Number of transitions folded into the proof.
    steps: usize,
    /// Final accumulator encoded in the canonical commitment format.
    final_root: [u8; COMMITMENT_BYTES],
}

/// A streaming Nova prover for topology fold transitions.
///
/// The prover owns one immutable Poseidon parameter table and accepts one trace
/// transition at a time. This is the boundary used by a future CUDA trace
/// producer: device work can fill or validate the next transition while the
/// host folds the previous one, without retaining the entire `MaleCNS` trace.
pub struct TopologyNovaProver {
    /// Nova public parameters for the fixed step circuit shape.
    parameters: Parameters,
    /// The recursive proof being updated.
    recursive: RecursiveProof,
    /// Number of transitions accepted so far.
    steps: usize,
    /// Canonical accumulator expected at the next transition.
    previous: [u8; COMMITMENT_BYTES],
    /// Final accumulator after the last accepted transition.
    final_root: Option<[u8; COMMITMENT_BYTES]>,
    /// Shared converted Poseidon constants for all step circuits.
    poseidon_parameters: std::sync::Arc<PoseidonParameters>,
}

impl TopologyNovaProver {
    /// Creates a streaming prover from the first canonical trace transition.
    ///
    /// The first transition is used to synthesize the Nova circuit shape. It is
    /// still passed to [`Self::push_step`] so the public step counter and final
    /// state follow one uniform rule.
    ///
    /// # Errors
    ///
    /// Returns an error when the first transition is not index zero, does not
    /// start at the zero accumulator, or contains malformed field data.
    pub fn new(first: &TopologyFoldStep) -> Result<Self, TopologyNovaError> {
        validate_step_header(first, 0, &[0_u8; COMMITMENT_BYTES])?;
        let poseidon_parameters = std::sync::Arc::new(PoseidonParameters::new()?);
        let first_circuit = TopologyStepCircuit::from_fold_step_with_parameters(
            first,
            poseidon_parameters.clone(),
        )?;
        let parameters = setup_parameters(&first_circuit)?;
        let initial_primary = vec![Fr::ZERO];
        let recursive = RecursiveProof::new(&parameters, &first_circuit, &initial_primary)?;
        Ok(Self {
            parameters,
            recursive,
            steps: 0,
            previous: [0_u8; COMMITMENT_BYTES],
            final_root: None,
            poseidon_parameters,
        })
    }

    /// Creates a streaming prover from trusted `HyperKZG` setup parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when the transition is malformed, the setup directory
    /// is unsuitable, or Nova rejects the base case.
    #[cfg(feature = "hyperkzg")]
    pub fn new_with_ptau_dir(
        first: &TopologyFoldStep,
        ptau_dir: &std::path::Path,
    ) -> Result<Self, TopologyNovaError> {
        validate_step_header(first, 0, &[0_u8; COMMITMENT_BYTES])?;
        let poseidon_parameters = std::sync::Arc::new(PoseidonParameters::new()?);
        let first_circuit = TopologyStepCircuit::from_fold_step_with_parameters(
            first,
            poseidon_parameters.clone(),
        )?;
        let parameters = Parameters::setup_with_ptau_dir(
            &first_circuit,
            &*default_ck_hint(),
            &*default_ck_hint(),
            ptau_dir,
        )?;
        let initial_primary = vec![Fr::ZERO];
        let recursive = RecursiveProof::new(&parameters, &first_circuit, &initial_primary)?;
        Ok(Self {
            parameters,
            recursive,
            steps: 0,
            previous: [0_u8; COMMITMENT_BYTES],
            final_root: None,
            poseidon_parameters,
        })
    }

    /// Folds one canonical transition into the running Nova proof.
    ///
    /// The transition is checked before synthesis. The accumulator dependency
    /// remains sequential, while data decoding and GPU witness preparation can
    /// happen outside this method.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-canonical index, a broken accumulator chain,
    /// malformed fields, or a Nova synthesis failure.
    pub fn push_step(&mut self, step: &TopologyFoldStep) -> Result<(), TopologyNovaError> {
        let expected_index =
            u64::try_from(self.steps).map_err(|_| TopologyNovaError::TraceIndex {
                actual: step.index,
                expected: u64::MAX,
            })?;
        validate_step_header(step, expected_index, &self.previous)?;
        let circuit = TopologyStepCircuit::from_fold_step_with_parameters(
            step,
            self.poseidon_parameters.clone(),
        )?;
        self.recursive.prove_step(&self.parameters, &circuit)?;
        self.steps = self
            .steps
            .checked_add(1)
            .ok_or(TopologyNovaError::StepCountOverflow)?;
        self.previous = step.next;
        self.final_root = Some(step.next);
        Ok(())
    }

    /// Completes the streaming prover and returns the recursive proof object.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyNovaError::EmptyTrace`] when no transition was pushed.
    pub fn finish(self) -> Result<TopologyNovaProof, TopologyNovaError> {
        let final_root = self.final_root.ok_or(TopologyNovaError::EmptyTrace)?;
        Ok(TopologyNovaProof {
            parameters: self.parameters,
            recursive: self.recursive,
            steps: self.steps,
            final_root,
        })
    }

    /// Returns the number of transitions accepted so far.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }
}

impl TopologyNovaProof {
    /// Proves a non-empty canonical topology fold trace with Nova.
    ///
    /// The input transitions are checked for canonical indices and accumulator
    /// chaining before Nova synthesis starts. Each transition's data remains a
    /// private step witness; binding that sequence to a public topology root is
    /// a separate circuit composition boundary.
    ///
    /// # Errors
    ///
    /// Returns an error when the trace is empty, discontinuous, malformed, or
    /// rejected by the Nova implementation.
    pub fn prove(steps: &[TopologyFoldStep]) -> Result<Self, TopologyNovaError> {
        let first = steps.first().ok_or(TopologyNovaError::EmptyTrace)?;
        let mut prover = TopologyNovaProver::new(first)?;
        for step in steps {
            prover.push_step(step)?;
        }
        prover.finish()
    }

    /// Proves a topology trace using trusted `HyperKZG` setup parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when the trace is empty or malformed, the setup
    /// directory is unsuitable, or Nova rejects synthesis/folding.
    #[cfg(feature = "hyperkzg")]
    pub fn prove_with_ptau_dir(
        steps: &[TopologyFoldStep],
        ptau_dir: &std::path::Path,
    ) -> Result<Self, TopologyNovaError> {
        let first = steps.first().ok_or(TopologyNovaError::EmptyTrace)?;
        let mut prover = TopologyNovaProver::new_with_ptau_dir(first, ptau_dir)?;
        for step in steps {
            prover.push_step(step)?;
        }
        prover.finish()
    }

    /// Proves a trace and checks its final root against an external claim.
    ///
    /// The root check is performed after the full Nova proof is built, so a
    /// caller cannot accidentally publish a proof for a different topology.
    /// Use [`Self::verify_against_root`] at the verifier boundary as well.
    ///
    /// # Errors
    ///
    /// Returns an error when the trace is malformed, Nova rejects the proof,
    /// or the produced root differs from `claimed_root`.
    pub fn prove_for_root(
        steps: &[TopologyFoldStep],
        claimed_root: Commitment,
    ) -> Result<Self, TopologyNovaError> {
        let proof = Self::prove(steps)?;
        let actual = proof.final_root();
        if actual != claimed_root {
            return Err(TopologyNovaError::RootMismatch {
                expected: claimed_root,
                actual,
            });
        }
        Ok(proof)
    }

    /// Proves a topology trace and checks its root using trusted `HyperKZG` setup.
    ///
    /// # Errors
    ///
    /// Returns an error when the trace or setup directory is invalid, Nova
    /// rejects the proof, or the produced root differs from the claim.
    #[cfg(feature = "hyperkzg")]
    pub fn prove_for_root_with_ptau_dir(
        steps: &[TopologyFoldStep],
        claimed_root: Commitment,
        ptau_dir: &std::path::Path,
    ) -> Result<Self, TopologyNovaError> {
        let proof = Self::prove_with_ptau_dir(steps, ptau_dir)?;
        let actual = proof.final_root();
        if actual != claimed_root {
            return Err(TopologyNovaError::RootMismatch {
                expected: claimed_root,
                actual,
            });
        }
        Ok(proof)
    }

    /// Verifies the recursive proof and checks its final accumulator.
    ///
    /// This verifies the Nova relation and the canonical final root, but does
    /// not assert that the private step data came from a particular CSR unless
    /// a separate topology-binding relation is composed with this proof.
    ///
    /// # Errors
    ///
    /// Returns an error when Nova rejects the proof or returns an unexpected
    /// output shape.
    pub fn verify(&self) -> Result<bool, TopologyNovaError> {
        let initial_primary = vec![Fr::ZERO];
        let outputs = self
            .recursive
            .verify(&self.parameters, self.steps, &initial_primary)?;
        let final_state = outputs
            .first()
            .copied()
            .ok_or(TopologyNovaError::MissingOutput)?;
        let expected = field_from_bytes(&self.final_root)?;
        Ok(final_state == expected)
    }

    /// Verifies the recursive proof and matches it against an external root.
    ///
    /// The caller can use this method when the canonical CSR root is already
    /// known as a public instance. Nova verifies the recursive relation first;
    /// the method then compares that verified final state with `claimed_root`.
    /// The root remains a public application-level claim and is not revealed as
    /// an additional private witness.
    ///
    /// # Errors
    ///
    /// Returns an error when Nova rejects the proof or the final accumulator is
    /// not a canonical field encoding.
    pub fn verify_against_root(&self, claimed_root: Commitment) -> Result<bool, TopologyNovaError> {
        Ok(self.verify()? && self.final_root() == claimed_root)
    }

    /// Returns the number of folded topology transitions.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }

    /// Returns the final topology commitment represented by this proof.
    #[must_use]
    pub const fn final_root(&self) -> Commitment {
        Commitment::from_bytes(self.final_root)
    }

    /// Returns the primary R1CS constraint count for one Nova step.
    #[must_use]
    pub fn primary_constraints(&self) -> usize {
        self.parameters.num_constraints().0
    }

    /// Returns the primary R1CS variable count for one Nova step.
    #[must_use]
    pub fn primary_variables(&self) -> usize {
        self.parameters.num_variables().0
    }
}

/// Validates the index and accumulator chaining predicates of one transition.
fn validate_step_header(
    step: &TopologyFoldStep,
    expected_index: u64,
    previous: &[u8; COMMITMENT_BYTES],
) -> Result<(), TopologyNovaError> {
    if step.index != expected_index {
        return Err(TopologyNovaError::TraceIndex {
            actual: step.index,
            expected: expected_index,
        });
    }
    if step.previous != *previous {
        return Err(TopologyNovaError::TraceDiscontinuity { index: step.index });
    }
    Ok(())
}

/// Decodes one canonical little-endian BN254 scalar into Nova's equivalent
/// field representation.
fn field_from_bytes(bytes: &[u8; COMMITMENT_BYTES]) -> Result<Fr, TopologyNovaError> {
    Option::from(Fr::from_repr((*bytes).into())).ok_or(TopologyNovaError::InvalidField {
        field: "final accumulator",
    })
}

#[cfg(test)]
mod tests {
    use super::TopologyNovaProof;
    use zkfly_commitment::{Commitment, commit_topology_with_trace};

    #[test]
    fn nova_proof_verifies_the_canonical_single_step_vector() {
        let mut steps = Vec::new();
        let root_result = commit_topology_with_trace(2, &[0, 1, 2], &[0, 1], |step| {
            steps.push(step);
        });
        assert!(root_result.is_ok());
        let Some(root) = root_result.ok() else {
            return;
        };
        let proof_result = TopologyNovaProof::prove(&steps);
        assert!(proof_result.is_ok());
        let Some(proof) = proof_result.ok() else {
            return;
        };
        assert_eq!(proof.steps(), 1);
        assert_eq!(proof.final_root(), root);
        assert!(proof.verify().unwrap_or(false));
        assert!(proof.verify_against_root(root).unwrap_or(false));
        assert!(proof.primary_constraints() > 0);
        assert!(proof.primary_variables() > 0);
    }

    #[test]
    fn nova_proof_verifies_multiple_recursive_steps_and_rejects_wrong_root() {
        let row_offsets: Vec<u32> = (0_u32..=50).collect();
        let column_indices: Vec<u32> = (0_u32..50).collect();
        let mut steps = Vec::new();
        let root_result =
            commit_topology_with_trace(50, &row_offsets, &column_indices, |step| steps.push(step));
        assert!(root_result.is_ok());
        let Some(root) = root_result.ok() else {
            return;
        };
        let proof_result = TopologyNovaProof::prove(&steps);
        assert!(proof_result.is_ok());
        let Some(proof) = proof_result.ok() else {
            return;
        };
        assert_eq!(proof.steps(), 2);
        assert!(proof.verify().unwrap_or(false));
        assert!(proof.verify_against_root(root).unwrap_or(false));
        assert!(
            !proof
                .verify_against_root(Commitment::from_bytes([1_u8; 32]))
                .unwrap_or(true)
        );
    }
}
