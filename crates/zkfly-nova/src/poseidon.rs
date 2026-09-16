//! Bellpepper constraints for the fixed BN254 Poseidon fold transition.

use crate::TopologyNovaError;
use ark_bn254::Fr as ArkFr;
use ark_ff::{BigInteger, PrimeField as ArkPrimeField};
use ff::{Field, PrimeField};
use halo2curves::bn256::Fr;
use light_poseidon::parameters::bn254_x5::get_poseidon_parameters;
use nova_snark::frontend::{ConstraintSystem, SynthesisError, num::AllocatedNum};
use zkfly_commitment::{
    COMMITMENT_BYTES, POSEIDON_INPUTS, POSEIDON_RATE, TOPOLOGY_DOMAIN, TopologyFoldStep,
};

/// Parameters for the width-thirteen Circom-compatible Poseidon permutation.
#[derive(Clone, Debug)]
pub(crate) struct PoseidonParameters {
    /// Round constants grouped by permutation round.
    pub(crate) ark: Vec<Vec<Fr>>,
    /// The MDS matrix in row-major form.
    pub(crate) mds: Vec<Vec<Fr>>,
    /// Number of full S-box rounds.
    pub(crate) full_rounds: usize,
    /// Number of partial S-box rounds.
    pub(crate) partial_rounds: usize,
}

impl PoseidonParameters {
    /// Converts the canonical `light-poseidon` BN254 parameters to the Nova
    /// implementation's equivalent `ff` field representation.
    pub(crate) fn new() -> Result<Self, TopologyNovaError> {
        let width = u8::try_from(POSEIDON_INPUTS + 1).map_err(|_| {
            TopologyNovaError::PoseidonParameters {
                message: "Poseidon input width does not fit u8".to_owned(),
            }
        })?;
        let raw = get_poseidon_parameters::<ArkFr>(width).map_err(|error| {
            TopologyNovaError::PoseidonParameters {
                message: error.to_string(),
            }
        })?;
        let (ark_chunks, ark_remainder) = raw.ark.as_slice().as_chunks::<{ POSEIDON_INPUTS + 1 }>();
        let ark = ark_chunks
            .iter()
            .map(|row| convert_row(row))
            .collect::<Result<Vec<_>, _>>()?;
        if !ark_remainder.is_empty() {
            return Err(TopologyNovaError::PoseidonParameters {
                message: "round constants are not divisible by the state width".to_owned(),
            });
        }
        let mds = raw
            .mds
            .iter()
            .map(|row| convert_row(row.as_slice()))
            .collect::<Result<Vec<_>, _>>()?;
        if mds.len() != POSEIDON_INPUTS + 1
            || mds.iter().any(|row| row.len() != POSEIDON_INPUTS + 1)
        {
            return Err(TopologyNovaError::PoseidonParameters {
                message: "MDS matrix does not have width-thirteen shape".to_owned(),
            });
        }
        Ok(Self {
            ark,
            mds,
            full_rounds: raw.full_rounds,
            partial_rounds: raw.partial_rounds,
        })
    }

    /// Returns the number of permutation rounds represented by this table.
    pub(crate) fn round_count(&self) -> usize {
        self.full_rounds + self.partial_rounds
    }
}

/// A Nova primary step whose private witness is one canonical topology block.
///
/// The public state is a single accumulator. The eleven data fields are
/// circuit witnesses supplied by the prover for this step. Binding the entire
/// data sequence to a public topology commitment is intentionally a separate
/// composition boundary; this circuit only proves the Poseidon transition.
#[derive(Clone, Debug)]
pub struct TopologyStepCircuit {
    /// The eleven padded data fields consumed by this transition.
    data: Vec<Fr>,
    /// Shared immutable Poseidon constants used by every step instance.
    parameters: std::sync::Arc<PoseidonParameters>,
}

impl TopologyStepCircuit {
    /// Builds a step circuit from one canonical commitment trace transition.
    ///
    /// # Errors
    ///
    /// Returns an error when a field is not a canonical BN254 encoding or when
    /// the transition's padding bytes are non-zero.
    pub fn from_fold_step(step: &TopologyFoldStep) -> Result<Self, TopologyNovaError> {
        Self::from_fold_step_with_parameters(step, std::sync::Arc::new(PoseidonParameters::new()?))
    }

    /// Builds a step circuit while reusing a previously converted parameter
    /// table. The Nova streaming prover uses this path for every later step.
    pub(crate) fn from_fold_step_with_parameters(
        step: &TopologyFoldStep,
        parameters: std::sync::Arc<PoseidonParameters>,
    ) -> Result<Self, TopologyNovaError> {
        let data_len = usize::from(step.data_len);
        if data_len > POSEIDON_RATE {
            return Err(TopologyNovaError::NonZeroPadding { index: step.index });
        }
        let mut data = Vec::with_capacity(POSEIDON_RATE);
        for (position, bytes) in step.data.iter().enumerate() {
            if position >= data_len && bytes.iter().any(|byte| *byte != 0) {
                return Err(TopologyNovaError::NonZeroPadding { index: step.index });
            }
            let value = field_from_bytes(bytes, "topology data")?;
            data.push(value);
        }
        Ok(Self { data, parameters })
    }

    /// Returns the number of primary state elements expected by Nova.
    #[must_use]
    pub const fn state_arity() -> usize {
        1
    }

    /// Returns the number of data fields carried by one transition.
    #[must_use]
    pub const fn data_arity() -> usize {
        POSEIDON_RATE
    }

    /// Returns the configured Poseidon state width.
    #[must_use]
    pub const fn poseidon_width() -> usize {
        POSEIDON_INPUTS + 1
    }
}

impl nova_snark::traits::circuit::StepCircuit<Fr> for TopologyStepCircuit {
    /// Return the single accumulator state element.
    fn arity(&self) -> usize {
        Self::state_arity()
    }

    /// Constrain one `previous -> next` Poseidon fold transition.
    fn synthesize<CS: ConstraintSystem<Fr>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<Fr>],
    ) -> Result<Vec<AllocatedNum<Fr>>, SynthesisError> {
        let previous = z.first().ok_or_else(unsatisfiable)?;
        if z.len() != Self::state_arity() || self.data.len() != POSEIDON_RATE {
            return Err(unsatisfiable());
        }
        let data = self
            .data
            .iter()
            .enumerate()
            .map(|(position, value)| {
                AllocatedNum::alloc(cs.namespace(|| format!("data_{position}")), || Ok(*value))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let next = synthesize_poseidon(cs, previous, &data, &self.parameters)?;
        Ok(vec![next])
    }
}

/// Converts one fixed-width arkworks field element to the Nova field.
fn convert_field(value: ArkFr) -> Result<Fr, TopologyNovaError> {
    let bytes = value.into_bigint().to_bytes_le();
    let repr = <[u8; COMMITMENT_BYTES]>::try_from(bytes.as_slice()).map_err(|_| {
        TopologyNovaError::PoseidonParameters {
            message: "arkworks field encoding is not 32 bytes".to_owned(),
        }
    })?;
    Option::from(Fr::from_repr(repr.into())).ok_or(TopologyNovaError::PoseidonParameters {
        message: "arkworks field encoding is not canonical in the Nova field".to_owned(),
    })
}

/// Converts a row of arkworks constants to the Nova field.
fn convert_row(row: &[ArkFr]) -> Result<Vec<Fr>, TopologyNovaError> {
    row.iter().copied().map(convert_field).collect()
}

/// Decodes one canonical little-endian field element.
fn field_from_bytes(
    bytes: &[u8; COMMITMENT_BYTES],
    field: &'static str,
) -> Result<Fr, TopologyNovaError> {
    Option::from(Fr::from_repr((*bytes).into())).ok_or(TopologyNovaError::InvalidField { field })
}

/// Converts the ASCII domain separator into a canonical Nova field element.
fn domain_tag() -> Result<Fr, TopologyNovaError> {
    poseidon_domain(TOPOLOGY_DOMAIN)
}

/// Converts a domain separator into a canonical Nova field element.
pub(crate) fn poseidon_domain(domain: &[u8]) -> Result<Fr, TopologyNovaError> {
    let mut bytes = [0_u8; COMMITMENT_BYTES];
    let target = bytes
        .get_mut(..domain.len())
        .ok_or(TopologyNovaError::PoseidonParameters {
            message: "Poseidon domain separator exceeds field width".to_owned(),
        })?;
    target.copy_from_slice(domain);
    field_from_bytes(&bytes, "Poseidon domain")
}

/// Constructs the synthesis error used when the fixed Poseidon shape is invalid.
fn unsatisfiable() -> SynthesisError {
    SynthesisError::Unsatisfiable("invalid fixed Poseidon topology circuit shape".to_owned())
}

/// Synthesizes the complete fixed-width Poseidon permutation.
fn synthesize_poseidon<CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
    previous: &AllocatedNum<Fr>,
    data: &[AllocatedNum<Fr>],
    parameters: &PoseidonParameters,
) -> Result<AllocatedNum<Fr>, SynthesisError> {
    let domain = domain_tag().map_err(|_| unsatisfiable())?;
    synthesize_poseidon_with_domain(cs, previous, data, parameters, domain)
}

/// Synthesizes a fixed-width Poseidon permutation with an explicit domain.
pub(crate) fn synthesize_poseidon_with_domain<CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
    previous: &AllocatedNum<Fr>,
    data: &[AllocatedNum<Fr>],
    parameters: &PoseidonParameters,
    domain: Fr,
) -> Result<AllocatedNum<Fr>, SynthesisError> {
    let width = POSEIDON_INPUTS + 1;
    if data.len() != POSEIDON_RATE
        || parameters.ark.len() != parameters.round_count()
        || parameters.mds.len() != width
    {
        return Err(unsatisfiable());
    }
    let mut state = Vec::with_capacity(width);
    state.push(allocate_linear(cs, &[], domain, "domain")?);
    state.push(previous.clone());
    state.extend(data.iter().cloned());

    let half_rounds = parameters.full_rounds / 2;
    for round in 0..half_rounds {
        apply_round(cs, &mut state, parameters, round, true)?;
    }
    for round in half_rounds..half_rounds + parameters.partial_rounds {
        apply_round(cs, &mut state, parameters, round, false)?;
    }
    for round in half_rounds + parameters.partial_rounds..parameters.round_count() {
        apply_round(cs, &mut state, parameters, round, true)?;
    }
    state.first().cloned().ok_or_else(unsatisfiable)
}

/// Evaluates one fixed-width Poseidon call in Nova's native field.
pub(crate) fn poseidon_hash_fields(
    values: &[Fr],
    parameters: &PoseidonParameters,
    domain: &[u8],
) -> Result<Fr, TopologyNovaError> {
    let mut accumulator = Fr::ZERO;
    if values.is_empty() {
        return poseidon_hash_fields_block(&[], accumulator, parameters, domain);
    }
    for chunk in values.chunks(POSEIDON_RATE) {
        accumulator = poseidon_hash_fields_block(chunk, accumulator, parameters, domain)?;
    }
    Ok(accumulator)
}

/// Evaluates one fixed-width Poseidon block with an explicit accumulator.
fn poseidon_hash_fields_block(
    values: &[Fr],
    previous: Fr,
    parameters: &PoseidonParameters,
    domain: &[u8],
) -> Result<Fr, TopologyNovaError> {
    if values.len() > POSEIDON_RATE {
        return Err(TopologyNovaError::PoseidonParameters {
            message: "field block exceeds the fixed Poseidon rate".to_owned(),
        });
    }
    let width = POSEIDON_INPUTS + 1;
    if parameters.ark.len() != parameters.round_count()
        || parameters.mds.len() != width
        || parameters.mds.iter().any(|row| row.len() != width)
    {
        return Err(TopologyNovaError::PoseidonParameters {
            message: "Poseidon parameters do not have width-thirteen shape".to_owned(),
        });
    }
    let mut state = Vec::with_capacity(width);
    state.push(poseidon_domain(domain)?);
    state.push(previous);
    state.extend_from_slice(values);
    state.resize(width, Fr::ZERO);
    let half_rounds = parameters.full_rounds / 2;
    for round in 0..parameters.round_count() {
        let constants =
            parameters
                .ark
                .get(round)
                .ok_or_else(|| TopologyNovaError::PoseidonParameters {
                    message: "Poseidon round constants are incomplete".to_owned(),
                })?;
        let full_sbox = round < half_rounds || round >= half_rounds + parameters.partial_rounds;
        for position in 0..width {
            let value =
                state
                    .get_mut(position)
                    .ok_or_else(|| TopologyNovaError::PoseidonParameters {
                        message: "Poseidon state width is inconsistent".to_owned(),
                    })?;
            let constant =
                constants
                    .get(position)
                    .ok_or_else(|| TopologyNovaError::PoseidonParameters {
                        message: "Poseidon round width is inconsistent".to_owned(),
                    })?;
            *value += *constant;
            if full_sbox || position == 0 {
                let square = *value * *value;
                let fourth = square * square;
                *value = fourth * *value;
            }
        }
        let before_mds = state.clone();
        for row in 0..width {
            let coefficients =
                parameters
                    .mds
                    .get(row)
                    .ok_or_else(|| TopologyNovaError::PoseidonParameters {
                        message: "Poseidon MDS row is missing".to_owned(),
                    })?;
            let mut value = Fr::ZERO;
            for column in 0..width {
                let coefficient = coefficients.get(column).ok_or_else(|| {
                    TopologyNovaError::PoseidonParameters {
                        message: "Poseidon MDS column is missing".to_owned(),
                    }
                })?;
                let state_value = before_mds.get(column).ok_or_else(|| {
                    TopologyNovaError::PoseidonParameters {
                        message: "Poseidon state column is missing".to_owned(),
                    }
                })?;
                value += *coefficient * *state_value;
            }
            let output =
                state
                    .get_mut(row)
                    .ok_or_else(|| TopologyNovaError::PoseidonParameters {
                        message: "Poseidon state row is missing".to_owned(),
                    })?;
            *output = value;
        }
    }
    state
        .first()
        .copied()
        .ok_or_else(|| TopologyNovaError::PoseidonParameters {
            message: "Poseidon state is empty".to_owned(),
        })
}

/// Applies one Ark, S-box, and MDS round.
fn apply_round<CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
    state: &mut Vec<AllocatedNum<Fr>>,
    parameters: &PoseidonParameters,
    round: usize,
    full_sbox: bool,
) -> Result<(), SynthesisError> {
    let constants = parameters.ark.get(round).ok_or_else(unsatisfiable)?;
    if constants.len() != state.len() {
        return Err(unsatisfiable());
    }
    let ark_state = state
        .iter()
        .enumerate()
        .map(|(position, value)| {
            let constant = constants.get(position).ok_or_else(unsatisfiable)?;
            allocate_linear(
                cs,
                &[(Fr::ONE, value)],
                *constant,
                &format!("round_{round}_ark_{position}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let sboxed = ark_state
        .iter()
        .enumerate()
        .map(|(position, value)| {
            if full_sbox || position == 0 {
                sbox_fifth(cs, value, round, position)
            } else {
                Ok(value.clone())
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mds_state = parameters
        .mds
        .iter()
        .enumerate()
        .map(|(row, coefficients)| {
            if coefficients.len() != sboxed.len() {
                return Err(unsatisfiable());
            }
            let mut values = sboxed.iter();
            let mut terms = Vec::with_capacity(coefficients.len());
            for coefficient in coefficients {
                let value = values.next().ok_or_else(unsatisfiable)?;
                terms.push((*coefficient, value));
            }
            allocate_linear(cs, &terms, Fr::ZERO, &format!("round_{round}_mds_{row}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    *state = mds_state;
    Ok(())
}

/// Constrains an x^5 S-box using three multiplication constraints.
fn sbox_fifth<CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
    value: &AllocatedNum<Fr>,
    round: usize,
    position: usize,
) -> Result<AllocatedNum<Fr>, SynthesisError> {
    let square = value.square(cs.namespace(|| format!("round_{round}_sbox_{position}_x2")))?;
    let fourth = square.square(cs.namespace(|| format!("round_{round}_sbox_{position}_x4")))?;
    fourth.mul(
        cs.namespace(|| format!("round_{round}_sbox_{position}_x5")),
        value,
    )
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
