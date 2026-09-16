//! Paper step-count estimates for topology registration and execution traces.

use serde::Serialize;
use zkfly_commitment::{
    COMMITMENT_BYTES, POSEIDON_RATE, topology_field_count, topology_fold_step_count,
};

use crate::error::BenchError;

/// Lower-bound primary constraints for one width-thirteen Poseidon permutation.
///
/// The count follows the current Nova gadget: 8 full rounds, 57 partial
/// rounds, 13 affine additions and 13 MDS combinations per round, three
/// multiplication constraints per x^5 S-box, and one domain allocation. It is
/// deliberately a lower bound and excludes padding, witness plumbing, and
/// Nova's protocol overhead.
pub const POSEIDON_PERMUTATION_CONSTRAINT_LOWER_BOUND: usize = 2_174;

/// A parameterized proof-step estimate with no wall-clock claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ProofStepEstimate {
    /// Number of neurons in the committed CSR topology.
    pub neurons: usize,
    /// Number of retained CSR edges.
    pub edges: usize,
    /// Number of canonical BN254 fields in the topology stream.
    pub topology_fields: usize,
    /// Number of Poseidon fold steps needed to derive the topology root.
    pub topology_steps: usize,
    /// Number of game or simulation ticks in the execution trace.
    pub ticks: usize,
    /// Number of network updates per tick.
    pub layers_per_tick: usize,
    /// Number of graph tiles per network update.
    pub tiles_per_update: usize,
    /// Forward steps under the supplied execution model.
    pub forward_steps: usize,
    /// Whether the one-time topology registration proof is included.
    pub includes_topology: bool,
    /// Topology plus forward steps under the supplied model.
    pub total_steps: usize,
    /// Poseidon-rate chunks required by one input or output vector.
    pub forward_vector_chunks: usize,
    /// Input and output Poseidon permutations required by one forward step.
    pub forward_poseidon_permutations: usize,
    /// Lower-bound Poseidon constraints in one forward step.
    pub forward_poseidon_constraints_lower_bound: usize,
    /// One multiplication relation per retained CSR edge.
    pub forward_edge_constraints_lower_bound: usize,
    /// One row-combination relation per neuron output.
    pub forward_row_constraints_lower_bound: usize,
    /// Lower-bound primary constraints in one forward step.
    pub forward_constraints_lower_bound: usize,
    /// Private input, output, and weight fields retained by one host witness.
    pub forward_witness_fields: usize,
    /// Bytes occupied by those one-step witness fields in canonical 32-byte form.
    pub forward_witness_bytes: usize,
    /// Lower-bound Poseidon constraints in the one-time topology proof.
    pub topology_constraints_lower_bound: usize,
    /// Lower-bound primary constraints for all included proof steps.
    pub total_constraints_lower_bound: usize,
}

/// Computes a paper estimate for one topology and a parameterized execution.
///
/// The model assumes one Nova step per network update per graph tile. It does
/// uses the current fixed Poseidon-rate chunking for vector capacity figures;
/// cross-tile consistency and a scalable PCS remain future circuit milestones.
/// The topology proof is treated as one-time registration when
/// `includes_topology` is false. The CLI's `--estimate-only` mode reads only the
/// artifact dimensions, so it does not recompute the root.
///
/// # Errors
///
/// Returns [`BenchError::Configuration`] for zero execution dimensions or
/// checked-arithmetic overflow, and forwards topology length errors.
pub fn estimate_proof_steps(
    neurons: usize,
    edges: usize,
    ticks: usize,
    layers_per_tick: usize,
    tiles_per_update: usize,
    includes_topology: bool,
) -> Result<ProofStepEstimate, BenchError> {
    if ticks == 0 || layers_per_tick == 0 || tiles_per_update == 0 {
        return Err(BenchError::Configuration {
            proposition: "ticks, layers_per_tick, and tiles_per_update must be nonzero",
        });
    }
    let topology_fields = topology_field_count(neurons, edges)?;
    let topology_steps = topology_fold_step_count(neurons, edges)?;
    let forward_capacity = estimate_forward_capacity(neurons, edges)?;
    let topology_constraints_lower_bound = topology_steps
        .checked_mul(POSEIDON_PERMUTATION_CONSTRAINT_LOWER_BOUND)
        .ok_or(BenchError::Configuration {
            proposition: "topology constraint lower bound fits host arithmetic",
        })?;
    let forward_steps = ticks
        .checked_mul(layers_per_tick)
        .and_then(|value| value.checked_mul(tiles_per_update))
        .ok_or(BenchError::Configuration {
            proposition: "forward step count fits host arithmetic",
        })?;
    let total_steps = if includes_topology {
        topology_steps
            .checked_add(forward_steps)
            .ok_or(BenchError::Configuration {
                proposition: "total proof step count fits host arithmetic",
            })?
    } else {
        forward_steps
    };
    let forward_constraints = forward_capacity
        .constraints_lower_bound
        .checked_mul(forward_steps)
        .ok_or(BenchError::Configuration {
            proposition: "total forward constraint lower bound fits host arithmetic",
        })?;
    let total_constraints_lower_bound = if includes_topology {
        topology_constraints_lower_bound
            .checked_add(forward_constraints)
            .ok_or(BenchError::Configuration {
                proposition: "total constraint lower bound fits host arithmetic",
            })?
    } else {
        forward_constraints
    };
    Ok(ProofStepEstimate {
        neurons,
        edges,
        topology_fields,
        topology_steps,
        ticks,
        layers_per_tick,
        tiles_per_update,
        forward_steps,
        includes_topology,
        total_steps,
        forward_vector_chunks: forward_capacity.vector_chunks,
        forward_poseidon_permutations: forward_capacity.poseidon_permutations,
        forward_poseidon_constraints_lower_bound: forward_capacity.poseidon_constraints,
        forward_edge_constraints_lower_bound: edges,
        forward_row_constraints_lower_bound: neurons,
        forward_constraints_lower_bound: forward_capacity.constraints_lower_bound,
        forward_witness_fields: forward_capacity.witness_fields,
        forward_witness_bytes: forward_capacity.witness_bytes,
        topology_constraints_lower_bound,
        total_constraints_lower_bound,
    })
}

/// Intermediate capacity figures for one forward circuit step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ForwardCapacity {
    /// Poseidon-rate chunks required by one vector.
    vector_chunks: usize,
    /// Input and output permutations required by one step.
    poseidon_permutations: usize,
    /// Lower-bound Poseidon constraints.
    poseidon_constraints: usize,
    /// Lower-bound constraints including edge and row relations.
    constraints_lower_bound: usize,
    /// Private input, output, and weight fields.
    witness_fields: usize,
    /// Canonical 32-byte storage for those fields.
    witness_bytes: usize,
}

/// Computes one-step forward capacity without making a timing claim.
fn estimate_forward_capacity(neurons: usize, edges: usize) -> Result<ForwardCapacity, BenchError> {
    let vector_chunks = if neurons == 0 {
        1
    } else {
        neurons
            .checked_add(POSEIDON_RATE - 1)
            .ok_or(BenchError::Configuration {
                proposition: "forward vector chunk count fits host arithmetic",
            })?
            / POSEIDON_RATE
    };
    let poseidon_permutations = vector_chunks
        .checked_mul(2)
        .ok_or(BenchError::Configuration {
            proposition: "forward Poseidon permutation count fits host arithmetic",
        })?;
    let poseidon_constraints = poseidon_permutations
        .checked_mul(POSEIDON_PERMUTATION_CONSTRAINT_LOWER_BOUND)
        .ok_or(BenchError::Configuration {
            proposition: "forward Poseidon constraint count fits host arithmetic",
        })?;
    let constraints_lower_bound = poseidon_constraints
        .checked_add(edges)
        .and_then(|value| value.checked_add(neurons))
        .ok_or(BenchError::Configuration {
            proposition: "forward constraint lower bound fits host arithmetic",
        })?;
    let witness_fields = neurons
        .checked_mul(2)
        .and_then(|value| value.checked_add(edges))
        .ok_or(BenchError::Configuration {
            proposition: "forward witness field count fits host arithmetic",
        })?;
    let witness_bytes =
        witness_fields
            .checked_mul(COMMITMENT_BYTES)
            .ok_or(BenchError::Configuration {
                proposition: "forward witness byte count fits host arithmetic",
            })?;
    Ok(ForwardCapacity {
        vector_chunks,
        poseidon_permutations,
        poseidon_constraints,
        constraints_lower_bound,
        witness_fields,
        witness_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::estimate_proof_steps;

    #[test]
    fn estimates_registered_topology_and_one_thousand_ticks() {
        let estimate = estimate_proof_steps(166_700, 25_582_938, 1_000, 1, 1, true);
        assert!(estimate.is_ok());
        if let Ok(estimate) = estimate {
            assert_eq!(estimate.topology_fields, 3_678_525);
            assert_eq!(estimate.topology_steps, 334_412);
            assert_eq!(estimate.forward_steps, 1_000);
            assert_eq!(estimate.total_steps, 335_412);
            assert_eq!(estimate.forward_vector_chunks, 15_155);
            assert_eq!(estimate.forward_poseidon_permutations, 30_310);
            assert_eq!(
                estimate.forward_poseidon_constraints_lower_bound,
                65_893_940
            );
            assert_eq!(estimate.forward_constraints_lower_bound, 91_643_578);
            assert_eq!(estimate.forward_witness_fields, 25_916_338);
            assert_eq!(estimate.forward_witness_bytes, 829_322_816);
            assert_eq!(estimate.topology_constraints_lower_bound, 727_011_688);
            assert_eq!(estimate.total_constraints_lower_bound, 92_370_589_688);
        }
    }

    #[test]
    fn excludes_one_time_topology_by_default() {
        let estimate = estimate_proof_steps(50, 50, 8, 2, 3, false);
        assert!(estimate.is_ok());
        if let Ok(estimate) = estimate {
            assert_eq!(estimate.topology_steps, 2);
            assert_eq!(estimate.forward_steps, 48);
            assert_eq!(estimate.total_steps, 48);
        }
    }
}
