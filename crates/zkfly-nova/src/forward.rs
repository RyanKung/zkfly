//! Minimal CSR-bound weighted forward relation for the Nova adapter.
//!
//! The relation fixes the CSR row/column topology in the step-circuit shape,
//! while edge weights and input vectors are private witnesses. A step proves
//! `y[row] = sum(weight[e] * x[column[e]])`, binds the private input vector to
//! the previous state by a Poseidon commitment, and returns a Poseidon
//! commitment to the output vector. Vector commitments are folded in fixed
//! Poseidon-rate chunks, so the circuit shape remains deterministic while the
//! supported neuron count can exceed one permutation block.

#[cfg(feature = "hyperkzg")]
use std::path::Path;
use std::sync::Arc;

use ff::{Field, PrimeField};
use halo2curves::bn256::Fr;
use nova_snark::{
    frontend::{ConstraintSystem, SynthesisError, num::AllocatedNum},
    nova::{PublicParams, RecursiveSNARK},
    provider::GrumpkinEngine,
    traits::{circuit::StepCircuit, snark::default_ck_hint},
};
use zkfly_commitment::{Commitment, POSEIDON_RATE, commit_topology};

use crate::TopologyNovaError;
use crate::poseidon::{
    PoseidonParameters, poseidon_domain, poseidon_hash_fields, synthesize_poseidon_with_domain,
};

/// Domain separator for private input and output vector commitments.
const FORWARD_DOMAIN: &[u8] = b"zkfly/poseidon/forward/v1";

#[cfg(not(feature = "hyperkzg"))]
use nova_snark::provider::Bn256EngineIPA;
#[cfg(feature = "hyperkzg")]
use nova_snark::provider::Bn256EngineKZG;

#[cfg(feature = "hyperkzg")]
type PrimaryEngine = Bn256EngineKZG;
#[cfg(not(feature = "hyperkzg"))]
type PrimaryEngine = Bn256EngineIPA;
type SecondaryEngine = GrumpkinEngine;
type Parameters = PublicParams<PrimaryEngine, SecondaryEngine, WeightedForwardCircuit>;
type RecursiveProof = RecursiveSNARK<PrimaryEngine, SecondaryEngine, WeightedForwardCircuit>;

/// Creates public parameters using the selected Nova commitment backend.
fn setup_parameters(circuit: &WeightedForwardCircuit) -> Result<Parameters, TopologyNovaError> {
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

/// A validated CSR topology whose row/column positions are fixed in the
/// Nova circuit shape.
#[derive(Clone, Debug)]
pub struct CsrTopology {
    /// Number of rows and columns in the square neuron graph.
    neuron_count: usize,
    /// CSR row boundaries with one terminal entry.
    row_offsets: Vec<usize>,
    /// Presynaptic column for every edge in CSR order.
    column_indices: Vec<usize>,
    /// Canonical Poseidon commitment to the topology.
    root: Commitment,
    /// The same root decoded into Nova's native field for the circuit state.
    root_field: Fr,
}

impl CsrTopology {
    /// Validates a CSR topology and computes its canonical public root.
    ///
    /// The forward circuit folds vector commitments in width-thirteen Poseidon
    /// chunks. The edge count and neuron count remain otherwise unconstrained.
    ///
    /// # Errors
    ///
    /// Returns an error when the CSR is non-canonical or an index cannot be
    /// represented by the host-side circuit shape.
    pub fn new(
        neuron_count: usize,
        row_offsets: &[u32],
        column_indices: &[u32],
    ) -> Result<Self, TopologyNovaError> {
        let root = commit_topology(neuron_count, row_offsets, column_indices)?;
        let row_offsets = row_offsets
            .iter()
            .copied()
            .map(|value| {
                usize::try_from(value).map_err(|_| TopologyNovaError::TopologyIndexOverflow {
                    target: "CSR row offset",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let column_indices = column_indices
            .iter()
            .copied()
            .map(|value| {
                usize::try_from(value).map_err(|_| TopologyNovaError::TopologyIndexOverflow {
                    target: "CSR column index",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let root_field = field_from_commitment(root, "topology root")?;
        Ok(Self {
            neuron_count,
            row_offsets,
            column_indices,
            root,
            root_field,
        })
    }

    /// Returns the number of neurons represented by this topology.
    #[must_use]
    pub const fn neuron_count(&self) -> usize {
        self.neuron_count
    }

    /// Returns the number of retained CSR edges.
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.column_indices.len()
    }

    /// Returns the canonical topology root.
    #[must_use]
    pub const fn root(&self) -> Commitment {
        self.root
    }
}

/// One private weighted forward witness for a fixed CSR topology.
#[derive(Clone, Debug)]
pub struct WeightedForwardWitness {
    /// Private input vector consumed by this forward step.
    input: Vec<Fr>,
    /// Private edge weights in canonical CSR order.
    weights: Vec<Fr>,
    /// Host-computed output vector used for commitment assignment.
    output: Vec<Fr>,
    /// Commitment to `input` under [`FORWARD_DOMAIN`].
    input_commitment: Commitment,
    /// Commitment to `output` under [`FORWARD_DOMAIN`].
    output_commitment: Commitment,
    /// Topology root used when the witness was constructed.
    topology_root: Commitment,
}

impl WeightedForwardWitness {
    /// Computes a private witness and its input/output commitments.
    ///
    /// Weights are used by the circuit but are not committed separately and no
    /// training rule is checked. The output is deterministic for the supplied
    /// topology, input, and weights.
    ///
    /// # Errors
    ///
    /// Returns an error when input or weight lengths do not match the topology
    /// or a commitment field cannot be encoded.
    pub fn new(
        topology: &CsrTopology,
        input: &[Fr],
        weights: &[Fr],
    ) -> Result<Self, TopologyNovaError> {
        if input.len() != topology.neuron_count {
            return Err(TopologyNovaError::ForwardInputLength {
                actual: input.len(),
                expected: topology.neuron_count,
            });
        }
        if weights.len() != topology.edge_count() {
            return Err(TopologyNovaError::ForwardWeightLength {
                actual: weights.len(),
                expected: topology.edge_count(),
            });
        }
        let poseidon_parameters = PoseidonParameters::new()?;
        let input_commitment = commitment_for_vector(input, &poseidon_parameters)?;
        let output = evaluate_forward(topology, input, weights)?;
        let output_commitment = commitment_for_vector(&output, &poseidon_parameters)?;
        Ok(Self {
            input: input.to_vec(),
            weights: weights.to_vec(),
            output,
            input_commitment,
            output_commitment,
            topology_root: topology.root,
        })
    }

    /// Returns the private input commitment for this step.
    #[must_use]
    pub const fn input_commitment(&self) -> Commitment {
        self.input_commitment
    }

    /// Returns the output commitment produced by this step.
    #[must_use]
    pub const fn output_commitment(&self) -> Commitment {
        self.output_commitment
    }

    /// Returns the topology root associated with this witness.
    #[must_use]
    pub const fn topology_root(&self) -> Commitment {
        self.topology_root
    }
}

/// Nova step circuit for one fixed-topology weighted forward pass.
#[derive(Clone, Debug)]
struct WeightedForwardCircuit {
    /// Fixed row/column positions and public topology root.
    topology: Arc<CsrTopology>,
    /// Private input, weights, and output assignment for this step.
    witness: WeightedForwardWitness,
    /// Shared Poseidon constants used by input/output commitments.
    parameters: Arc<PoseidonParameters>,
    /// Forward-vector Poseidon domain in Nova's native field.
    domain: Fr,
}

impl WeightedForwardCircuit {
    /// Creates a step circuit with validated topology and witness lengths.
    fn new(
        topology: Arc<CsrTopology>,
        witness: WeightedForwardWitness,
        parameters: Arc<PoseidonParameters>,
    ) -> Result<Self, TopologyNovaError> {
        if witness.topology_root != topology.root {
            return Err(TopologyNovaError::RootMismatch {
                expected: topology.root,
                actual: witness.topology_root,
            });
        }
        if witness.input.len() != topology.neuron_count {
            return Err(TopologyNovaError::ForwardInputLength {
                actual: witness.input.len(),
                expected: topology.neuron_count,
            });
        }
        if witness.weights.len() != topology.edge_count() {
            return Err(TopologyNovaError::ForwardWeightLength {
                actual: witness.weights.len(),
                expected: topology.edge_count(),
            });
        }
        Ok(Self {
            topology,
            witness,
            parameters,
            domain: poseidon_domain(FORWARD_DOMAIN)?,
        })
    }
}

impl StepCircuit<Fr> for WeightedForwardCircuit {
    /// The state carries the current input commitment and fixed topology root.
    fn arity(&self) -> usize {
        2
    }

    /// Constrains input commitment, fixed-topology weighted sums, and output
    /// commitment for one recursive forward step.
    fn synthesize<CS: ConstraintSystem<Fr>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<Fr>],
    ) -> Result<Vec<AllocatedNum<Fr>>, SynthesisError> {
        let input_state = z.first().ok_or_else(unsatisfiable)?;
        let topology_state = z.get(1).ok_or_else(unsatisfiable)?;
        if z.len() != self.arity()
            || self.witness.input.len() != self.topology.neuron_count
            || self.witness.output.len() != self.topology.neuron_count
        {
            return Err(unsatisfiable());
        }

        let input = allocate_private_vector(cs, &self.witness.input, "input")?;
        let zero = allocate_linear(cs, &[], Fr::ZERO, "input_previous")?;
        let input_digest = synthesize_vector_commitment(
            cs,
            &input,
            &zero,
            &self.parameters,
            self.domain,
            "input_commitment",
        )?;
        cs.enforce(
            || "input_commitment_matches_state",
            |lc| lc + input_digest.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + input_state.get_variable(),
        );

        let topology_root =
            allocate_linear(cs, &[], self.topology.root_field, "topology_root_constant")?;
        cs.enforce(
            || "topology_root_matches_state",
            |lc| lc + topology_root.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + topology_state.get_variable(),
        );

        let weights = allocate_private_vector(cs, &self.witness.weights, "weight")?;
        let mut output = Vec::with_capacity(self.topology.neuron_count);
        for row in 0..self.topology.neuron_count {
            let start = self
                .topology
                .row_offsets
                .get(row)
                .copied()
                .ok_or_else(unsatisfiable)?;
            let end = self
                .topology
                .row_offsets
                .get(row.saturating_add(1))
                .copied()
                .ok_or_else(unsatisfiable)?;
            let mut products = Vec::with_capacity(end.saturating_sub(start));
            for edge in start..end {
                let column = self
                    .topology
                    .column_indices
                    .get(edge)
                    .copied()
                    .ok_or_else(unsatisfiable)?;
                let weight = weights.get(edge).ok_or_else(unsatisfiable)?;
                let input_value = input.get(column).ok_or_else(unsatisfiable)?;
                products.push(
                    weight.mul(cs.namespace(|| format!("edge_{edge}_product")), input_value)?,
                );
            }
            output.push(allocate_linear(
                cs,
                &products
                    .iter()
                    .map(|product| (Fr::ONE, product))
                    .collect::<Vec<_>>(),
                Fr::ZERO,
                &format!("output_{row}"),
            )?);
        }

        let output_digest = synthesize_vector_commitment(
            cs,
            &output,
            &zero,
            &self.parameters,
            self.domain,
            "output_commitment",
        )?;
        Ok(vec![output_digest, topology_state.clone()])
    }
}

/// Reusable Nova public parameters for one fixed-topology forward circuit.
///
/// Construct this once and pass an [`Arc`] clone to multiple proving jobs with
/// the same CSR shape. The setup witness is used only to synthesize the fixed
/// R1CS shape; its values are not retained by this object.
pub struct WeightedForwardParameters {
    /// Fixed topology shared by every proof using these parameters.
    topology: Arc<CsrTopology>,
    /// Nova public parameters containing the fixed CSR R1CS shape.
    parameters: Arc<Parameters>,
    /// Shared Poseidon constants used by each step circuit.
    poseidon_parameters: Arc<PoseidonParameters>,
}

impl WeightedForwardParameters {
    /// Sets up reusable public parameters for a fixed CSR topology.
    ///
    /// # Errors
    ///
    /// Returns an error when the setup witness does not match the topology or
    /// Nova rejects the fixed circuit shape.
    pub fn setup(
        topology: Arc<CsrTopology>,
        shape_witness: &WeightedForwardWitness,
    ) -> Result<Self, TopologyNovaError> {
        let poseidon_parameters = Arc::new(PoseidonParameters::new()?);
        let circuit = WeightedForwardCircuit::new(
            topology.clone(),
            shape_witness.clone(),
            poseidon_parameters.clone(),
        )?;
        let parameters = Arc::new(setup_parameters(&circuit)?);
        Ok(Self {
            topology,
            parameters,
            poseidon_parameters,
        })
    }

    /// Sets up reusable parameters from a trusted Powers-of-Tau directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the topology or witness is invalid, the directory
    /// has no suitable Powers-of-Tau file, or Nova rejects setup.
    #[cfg(feature = "hyperkzg")]
    pub fn setup_with_ptau_dir(
        topology: Arc<CsrTopology>,
        shape_witness: &WeightedForwardWitness,
        ptau_dir: &Path,
    ) -> Result<Self, TopologyNovaError> {
        let poseidon_parameters = Arc::new(PoseidonParameters::new()?);
        let circuit = WeightedForwardCircuit::new(
            topology.clone(),
            shape_witness.clone(),
            poseidon_parameters.clone(),
        )?;
        let parameters = Arc::new(Parameters::setup_with_ptau_dir(
            &circuit,
            &*default_ck_hint(),
            &*default_ck_hint(),
            ptau_dir,
        )?);
        Ok(Self {
            topology,
            parameters,
            poseidon_parameters,
        })
    }

    /// Returns the topology root fixed by this parameter set.
    #[must_use]
    pub fn topology_root(&self) -> Commitment {
        self.topology.root
    }

    /// Returns the primary R1CS constraint count for one forward step.
    #[must_use]
    pub fn primary_constraints(&self) -> usize {
        self.parameters.num_constraints().0
    }

    /// Returns the primary R1CS variable count for one forward step.
    #[must_use]
    pub fn primary_variables(&self) -> usize {
        self.parameters.num_variables().0
    }
}

/// A recursive Nova proof for one or more fixed-topology forward passes.
pub struct WeightedForwardProof {
    /// Nova public parameters containing the fixed CSR R1CS shape.
    parameters: Arc<Parameters>,
    /// Recursive proof object from the official Nova implementation.
    recursive: RecursiveProof,
    /// Number of forward passes folded into the proof.
    steps: usize,
    /// Public initial input commitment.
    initial_input: Commitment,
    /// Public fixed topology root.
    topology_root: Commitment,
    /// Public final output commitment.
    final_output: Commitment,
}

/// A streaming prover for fixed-topology weighted forward passes.
pub struct WeightedForwardProver {
    /// Fixed topology shared by every step circuit.
    topology: Arc<CsrTopology>,
    /// Nova public parameters for the fixed circuit shape.
    parameters: Arc<Parameters>,
    /// Recursive proof being updated.
    recursive: RecursiveProof,
    /// Number of forward passes accepted so far.
    steps: usize,
    /// Public input commitment used by the first recursive state.
    initial_input: Commitment,
    /// Input commitment expected by the next private witness.
    previous_input: Commitment,
    /// Final output commitment after the last accepted witness.
    final_output: Option<Commitment>,
    /// Shared Poseidon constants reused by every step circuit.
    poseidon_parameters: Arc<PoseidonParameters>,
}

impl WeightedForwardProver {
    /// Creates a streaming prover from the first private forward witness.
    ///
    /// The first witness determines the public initial input commitment and the
    /// Nova public parameters are synthesized from the fixed topology shape.
    ///
    /// # Errors
    ///
    /// Returns an error for a topology mismatch, malformed witness, or Nova
    /// public-parameter setup failure.
    pub fn new(
        topology: Arc<CsrTopology>,
        first: &WeightedForwardWitness,
    ) -> Result<Self, TopologyNovaError> {
        let parameters = Arc::new(WeightedForwardParameters::setup(topology, first)?);
        Self::new_with_parameters(&parameters, first)
    }

    /// Creates a streaming prover from trusted `HyperKZG` setup parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when the witness is invalid or Nova rejects the base
    /// case constructed from the trusted setup.
    #[cfg(feature = "hyperkzg")]
    pub fn new_with_ptau_dir(
        topology: Arc<CsrTopology>,
        first: &WeightedForwardWitness,
        ptau_dir: &Path,
    ) -> Result<Self, TopologyNovaError> {
        let parameters = Arc::new(WeightedForwardParameters::setup_with_ptau_dir(
            topology, first, ptau_dir,
        )?);
        Self::new_with_parameters(&parameters, first)
    }

    /// Creates a streaming prover using reusable public parameters.
    ///
    /// Reusing the parameter object avoids repeating Nova's expensive shape
    /// setup when several proofs share one topology and circuit shape.
    ///
    /// # Errors
    ///
    /// Returns an error for a topology mismatch, malformed witness, or Nova
    /// base-case construction failure.
    pub fn new_with_parameters(
        parameters: &WeightedForwardParameters,
        first: &WeightedForwardWitness,
    ) -> Result<Self, TopologyNovaError> {
        let first_circuit = WeightedForwardCircuit::new(
            parameters.topology.clone(),
            first.clone(),
            parameters.poseidon_parameters.clone(),
        )?;
        let initial_state = vec![
            field_from_commitment(first.input_commitment, "forward input commitment")?,
            parameters.topology.root_field,
        ];
        let recursive =
            RecursiveProof::new(&parameters.parameters, &first_circuit, &initial_state)?;
        Ok(Self {
            topology: parameters.topology.clone(),
            parameters: parameters.parameters.clone(),
            recursive,
            steps: 0,
            initial_input: first.input_commitment,
            previous_input: first.input_commitment,
            final_output: None,
            poseidon_parameters: parameters.poseidon_parameters.clone(),
        })
    }

    /// Folds one private weighted forward pass into the running proof.
    ///
    /// The input commitment must equal the previous output commitment. The
    /// circuit independently enforces the same relation against its Nova state.
    ///
    /// # Errors
    ///
    /// Returns an error when the input chain, topology, witness, or Nova fold
    /// operation is invalid.
    pub fn push_step(&mut self, witness: &WeightedForwardWitness) -> Result<(), TopologyNovaError> {
        if witness.topology_root != self.topology.root {
            return Err(TopologyNovaError::RootMismatch {
                expected: self.topology.root,
                actual: witness.topology_root,
            });
        }
        if witness.input_commitment != self.previous_input {
            return Err(TopologyNovaError::ForwardInputChain { index: self.steps });
        }
        let circuit = WeightedForwardCircuit::new(
            self.topology.clone(),
            witness.clone(),
            self.poseidon_parameters.clone(),
        )?;
        self.recursive.prove_step(&self.parameters, &circuit)?;
        self.steps = self
            .steps
            .checked_add(1)
            .ok_or(TopologyNovaError::StepCountOverflow)?;
        self.previous_input = witness.output_commitment;
        self.final_output = Some(witness.output_commitment);
        Ok(())
    }

    /// Finishes the streaming proof.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyNovaError::EmptyTrace`] when no witness was pushed.
    pub fn finish(self) -> Result<WeightedForwardProof, TopologyNovaError> {
        let final_output = self.final_output.ok_or(TopologyNovaError::EmptyTrace)?;
        Ok(WeightedForwardProof {
            parameters: self.parameters,
            recursive: self.recursive,
            steps: self.steps,
            initial_input: self.initial_input,
            topology_root: self.topology.root,
            final_output,
        })
    }

    /// Returns the number of forward passes accepted so far.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }
}

impl WeightedForwardProof {
    /// Proves a non-empty sequence of fixed-topology forward passes.
    ///
    /// # Errors
    ///
    /// Returns an error when the witness sequence is empty, its input chain is
    /// broken, or Nova rejects synthesis/folding.
    pub fn prove(
        topology: Arc<CsrTopology>,
        witnesses: &[WeightedForwardWitness],
    ) -> Result<Self, TopologyNovaError> {
        let first = witnesses.first().ok_or(TopologyNovaError::EmptyTrace)?;
        let parameters = Arc::new(WeightedForwardParameters::setup(topology, first)?);
        Self::prove_with_parameters(&parameters, witnesses)
    }

    /// Proves a sequence using trusted `HyperKZG` setup parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when the witness sequence is empty or malformed, the
    /// setup directory is unsuitable, or Nova rejects synthesis/folding.
    #[cfg(feature = "hyperkzg")]
    pub fn prove_with_ptau_dir(
        topology: Arc<CsrTopology>,
        witnesses: &[WeightedForwardWitness],
        ptau_dir: &Path,
    ) -> Result<Self, TopologyNovaError> {
        let first = witnesses.first().ok_or(TopologyNovaError::EmptyTrace)?;
        let parameters = Arc::new(WeightedForwardParameters::setup_with_ptau_dir(
            topology, first, ptau_dir,
        )?);
        Self::prove_with_parameters(&parameters, witnesses)
    }

    /// Proves a sequence using a reusable fixed-topology parameter set.
    ///
    /// # Errors
    ///
    /// Returns an error when the witness sequence is empty, its input chain is
    /// broken, or Nova rejects synthesis/folding.
    pub fn prove_with_parameters(
        parameters: &WeightedForwardParameters,
        witnesses: &[WeightedForwardWitness],
    ) -> Result<Self, TopologyNovaError> {
        let first = witnesses.first().ok_or(TopologyNovaError::EmptyTrace)?;
        let mut prover = WeightedForwardProver::new_with_parameters(parameters, first)?;
        for witness in witnesses {
            prover.push_step(witness)?;
        }
        prover.finish()
    }

    /// Verifies Nova and checks the final output/topology state.
    ///
    /// # Errors
    ///
    /// Returns an error when Nova rejects the recursive proof or a state is not
    /// a canonical field encoding.
    pub fn verify(&self) -> Result<bool, TopologyNovaError> {
        let initial_state = vec![
            field_from_commitment(self.initial_input, "forward input commitment")?,
            field_from_commitment(self.topology_root, "topology root")?,
        ];
        let outputs = self
            .recursive
            .verify(&self.parameters, self.steps, &initial_state)?;
        let output = outputs
            .first()
            .copied()
            .ok_or(TopologyNovaError::MissingOutput)?;
        let topology = outputs
            .get(1)
            .copied()
            .ok_or(TopologyNovaError::MissingOutput)?;
        let expected_output =
            field_from_commitment(self.final_output, "forward output commitment")?;
        let expected_topology = field_from_commitment(self.topology_root, "topology root")?;
        Ok(output == expected_output && topology == expected_topology)
    }

    /// Verifies Nova against external topology, input, and output commitments.
    ///
    /// # Errors
    ///
    /// Returns an error when the recursive proof or a commitment encoding is
    /// invalid. A valid proof with a different public claim returns `false`.
    pub fn verify_against(
        &self,
        topology_root: Commitment,
        input_commitment: Commitment,
        output_commitment: Commitment,
    ) -> Result<bool, TopologyNovaError> {
        Ok(self.verify()?
            && self.topology_root == topology_root
            && self.initial_input == input_commitment
            && self.final_output == output_commitment)
    }

    /// Returns the number of folded forward passes.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }

    /// Returns the public topology root carried by this proof.
    #[must_use]
    pub const fn topology_root(&self) -> Commitment {
        self.topology_root
    }

    /// Returns the public initial input commitment.
    #[must_use]
    pub const fn initial_input(&self) -> Commitment {
        self.initial_input
    }

    /// Returns the public final output commitment.
    #[must_use]
    pub const fn final_output(&self) -> Commitment {
        self.final_output
    }

    /// Returns the primary R1CS constraint count for one forward step.
    #[must_use]
    pub fn primary_constraints(&self) -> usize {
        self.parameters.num_constraints().0
    }

    /// Returns the primary R1CS variable count for one forward step.
    #[must_use]
    pub fn primary_variables(&self) -> usize {
        self.parameters.num_variables().0
    }
}

/// Computes the fixed-topology weighted forward result in Nova's field.
fn evaluate_forward(
    topology: &CsrTopology,
    input: &[Fr],
    weights: &[Fr],
) -> Result<Vec<Fr>, TopologyNovaError> {
    let mut output = Vec::with_capacity(topology.neuron_count);
    for row in 0..topology.neuron_count {
        let start = topology.row_offsets.get(row).copied().ok_or(
            TopologyNovaError::TopologyIndexOverflow {
                target: "CSR row start",
            },
        )?;
        let end = topology
            .row_offsets
            .get(row.saturating_add(1))
            .copied()
            .ok_or(TopologyNovaError::TopologyIndexOverflow {
                target: "CSR row end",
            })?;
        let mut value = Fr::ZERO;
        for edge in start..end {
            let column = topology.column_indices.get(edge).copied().ok_or(
                TopologyNovaError::TopologyIndexOverflow {
                    target: "CSR edge column",
                },
            )?;
            let input_value =
                input
                    .get(column)
                    .copied()
                    .ok_or(TopologyNovaError::ForwardInputLength {
                        actual: input.len(),
                        expected: topology.neuron_count,
                    })?;
            let weight =
                weights
                    .get(edge)
                    .copied()
                    .ok_or(TopologyNovaError::ForwardWeightLength {
                        actual: weights.len(),
                        expected: topology.edge_count(),
                    })?;
            value += weight * input_value;
        }
        output.push(value);
    }
    Ok(output)
}

/// Computes the private vector commitment used by the forward circuit.
fn commitment_for_vector(
    values: &[Fr],
    parameters: &PoseidonParameters,
) -> Result<Commitment, TopologyNovaError> {
    let digest = poseidon_hash_fields(values, parameters, FORWARD_DOMAIN)?;
    Ok(Commitment::from_bytes(digest.to_repr().into()))
}

/// Decodes one canonical commitment into Nova's native field.
fn field_from_commitment(
    commitment: Commitment,
    field: &'static str,
) -> Result<Fr, TopologyNovaError> {
    Option::from(Fr::from_repr((*commitment.as_bytes()).into()))
        .ok_or(TopologyNovaError::InvalidField { field })
}

/// Allocates private witness fields without changing the circuit shape.
fn allocate_private_vector<CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
    values: &[Fr],
    label: &str,
) -> Result<Vec<AllocatedNum<Fr>>, SynthesisError> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            AllocatedNum::alloc(cs.namespace(|| format!("{label}_{index}")), || Ok(*value))
        })
        .collect()
}

/// Pads one vector chunk to the eleven data fields accepted by Poseidon.
fn pad_vector<CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
    values: &[AllocatedNum<Fr>],
    label: &str,
) -> Result<Vec<AllocatedNum<Fr>>, SynthesisError> {
    if values.len() > POSEIDON_RATE {
        return Err(unsatisfiable());
    }
    let mut padded = values.to_vec();
    while padded.len() < POSEIDON_RATE {
        let index = padded.len();
        padded.push(allocate_linear(
            cs,
            &[],
            Fr::ZERO,
            &format!("{label}_{index}"),
        )?);
    }
    Ok(padded)
}

/// Folds a private vector into a chained fixed-width Poseidon commitment.
fn synthesize_vector_commitment<CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
    values: &[AllocatedNum<Fr>],
    zero: &AllocatedNum<Fr>,
    parameters: &PoseidonParameters,
    domain: Fr,
    label: &str,
) -> Result<AllocatedNum<Fr>, SynthesisError> {
    let mut accumulator = zero.clone();
    if values.is_empty() {
        let data = pad_vector(cs, &[], &format!("{label}_chunk_0_padding"))?;
        return synthesize_poseidon_with_domain(cs, &accumulator, &data, parameters, domain);
    }
    for (chunk_index, chunk) in values.chunks(POSEIDON_RATE).enumerate() {
        let data = pad_vector(cs, chunk, &format!("{label}_chunk_{chunk_index}_padding"))?;
        accumulator = synthesize_poseidon_with_domain(cs, &accumulator, &data, parameters, domain)?;
    }
    Ok(accumulator)
}

/// Allocates and constrains an affine field combination.
fn allocate_linear<CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
    terms: &[(Fr, &AllocatedNum<Fr>)],
    constant: Fr,
    label: &str,
) -> Result<AllocatedNum<Fr>, SynthesisError> {
    let value = terms
        .iter()
        .try_fold(constant, |value, (coefficient, term)| {
            term.get_value()
                .map(|term_value| value + (*coefficient * term_value))
        });
    let result = AllocatedNum::alloc(cs.namespace(|| label.to_owned()), || {
        value.ok_or(SynthesisError::AssignmentMissing)
    })?;
    cs.enforce(
        || label.to_owned(),
        |lc| {
            terms
                .iter()
                .fold(lc + (constant, CS::one()), |lc, (coefficient, term)| {
                    lc + (*coefficient, term.get_variable())
                })
        },
        |lc| lc + CS::one(),
        |lc| lc + result.get_variable(),
    );
    Ok(result)
}

/// Constructs a deterministic synthesis error for malformed fixed shapes.
fn unsatisfiable() -> SynthesisError {
    SynthesisError::Unsatisfiable("invalid fixed CSR weighted-forward circuit shape".to_owned())
}

#[cfg(test)]
mod tests {
    use super::{
        CsrTopology, WeightedForwardParameters, WeightedForwardProof, WeightedForwardWitness,
        evaluate_forward,
    };
    use ff::Field;
    use halo2curves::bn256::Fr;
    use std::sync::Arc;
    use zkfly_commitment::Commitment;

    fn topology() -> Option<CsrTopology> {
        CsrTopology::new(3, &[0, 2, 3, 4], &[0, 2, 1, 0]).ok()
    }

    #[test]
    fn weighted_forward_reference_uses_fixed_csr_positions() {
        let Some(topology) = topology() else {
            return;
        };
        let input = vec![Fr::ONE, Fr::from(2_u64), Fr::from(5_u64)];
        let weights = vec![Fr::from(2_u64), -Fr::ONE, Fr::from(3_u64), Fr::from(4_u64)];
        let output = evaluate_forward(&topology, &input, &weights);
        assert_eq!(
            output,
            Ok(vec![-Fr::from(3_u64), Fr::from(6_u64), Fr::from(4_u64)])
        );
    }

    #[test]
    fn weighted_forward_proof_folds_private_layers_and_binds_commitments() {
        let Some(topology) = topology() else {
            return;
        };
        let first_input = vec![Fr::ONE, Fr::from(2_u64), Fr::from(5_u64)];
        let weights = vec![Fr::from(2_u64), -Fr::ONE, Fr::from(3_u64), Fr::from(4_u64)];
        let first_result = WeightedForwardWitness::new(&topology, &first_input, &weights);
        assert!(first_result.is_ok());
        let Some(first) = first_result.ok() else {
            return;
        };
        let second_input = first.output.clone();
        let second_result = WeightedForwardWitness::new(&topology, &second_input, &weights);
        assert!(second_result.is_ok());
        let Some(second) = second_result.ok() else {
            return;
        };
        let parameters = WeightedForwardParameters::setup(Arc::new(topology.clone()), &first);
        assert!(parameters.is_ok());
        let Some(parameters) = parameters.ok().map(Arc::new) else {
            return;
        };
        let proof_result = WeightedForwardProof::prove_with_parameters(
            &parameters,
            &[first.clone(), second.clone()],
        );
        assert!(proof_result.is_ok());
        let Some(proof) = proof_result.ok() else {
            return;
        };
        assert_eq!(proof.steps(), 2);
        assert!(proof.verify().unwrap_or(false));
        assert!(
            proof
                .verify_against(
                    topology.root(),
                    first.input_commitment(),
                    second.output_commitment(),
                )
                .unwrap_or(false)
        );
        assert!(
            !proof
                .verify_against(
                    topology.root(),
                    first.input_commitment(),
                    Commitment::from_bytes([0_u8; 32]),
                )
                .unwrap_or(true)
        );
    }

    #[test]
    fn weighted_forward_proof_supports_chunked_vector_commitment() {
        let row_offsets = (0_u32..=12).collect::<Vec<_>>();
        let column_indices = (0_u32..12).collect::<Vec<_>>();
        let Some(topology) = CsrTopology::new(12, &row_offsets, &column_indices).ok() else {
            return;
        };
        let input = (1_u64..=12).map(Fr::from).collect::<Vec<_>>();
        let weights = vec![Fr::ONE; 12];
        let Ok(first) = WeightedForwardWitness::new(&topology, &input, &weights) else {
            return;
        };
        let topology = Arc::new(topology);
        let Ok(parameters) = WeightedForwardParameters::setup(topology.clone(), &first) else {
            return;
        };
        let Ok(proof) =
            WeightedForwardProof::prove_with_parameters(&parameters, std::slice::from_ref(&first))
        else {
            return;
        };
        assert!(
            proof
                .verify_against(
                    topology.root(),
                    first.input_commitment(),
                    first.output_commitment()
                )
                .unwrap_or(false)
        );
    }
}
