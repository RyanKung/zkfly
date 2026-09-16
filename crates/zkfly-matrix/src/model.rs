//! In-memory neuron metadata and presynaptic sign classification.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry as BTreeEntry;

use rustc_hash::FxHashMap;
use serde::Serialize;

use crate::MatrixError;

/// One annotation row needed by the deterministic neuron filter.
#[derive(Debug)]
pub(crate) struct AnnotationRow {
    /// Stable body identifier from the source table.
    pub(crate) body_id: i64,
    /// Source superclass; empty values are filtered later.
    pub(crate) superclass: String,
    /// Preferred `FlyWire` cell-type label.
    pub(crate) cell_type: String,
    /// Upper-cased soma-side or root-side fallback.
    pub(crate) side: String,
}

/// One body-to-neurotransmitter prediction row.
#[derive(Debug)]
pub(crate) struct NeurotransmitterRow {
    /// Stable body identifier from the source table.
    pub(crate) body_id: i64,
    /// Nullable consensus label; the first source row wins.
    pub(crate) consensus: Option<String>,
}

/// One aggregated directed source connection.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeRow {
    /// Presynaptic body identifier.
    pub(crate) pre: i64,
    /// Postsynaptic body identifier.
    pub(crate) post: i64,
    /// Positive structural synapse count.
    pub(crate) count: i64,
}

/// Sign assigned to a presynaptic neuron before matrix insertion.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SynapseSign {
    Excitatory,
    Inhibitory,
}

impl SynapseSign {
    /// Applies the fly.ai inhibitory-substring recipe to one label.
    pub(crate) fn from_consensus(label: Option<&str>) -> Self {
        let Some(label) = label else {
            return Self::Excitatory;
        };
        let lower = label.to_ascii_lowercase();
        if lower.contains("gaba") || lower.contains("glutamate") || lower.contains("histamine") {
            Self::Inhibitory
        } else {
            Self::Excitatory
        }
    }

    /// Applies this sign and checks that the source count fits the artifact type.
    pub(crate) fn apply(self, count: i64, edge: EdgeRow) -> Result<i32, MatrixError> {
        let magnitude = i32::try_from(count).map_err(|_| MatrixError::InvalidSynapseCount {
            pre: edge.pre,
            post: edge.post,
            count,
        })?;
        if magnitude <= 0 {
            return Err(MatrixError::InvalidSynapseCount {
                pre: edge.pre,
                post: edge.post,
                count,
            });
        }
        match self {
            Self::Excitatory => Ok(magnitude),
            Self::Inhibitory => magnitude.checked_neg().ok_or(MatrixError::Overflow {
                context: "inhibitory synapse count",
            }),
        }
    }
}

/// Stable metadata serialized one neuron per JSONL line.
#[derive(Debug, Serialize)]
pub(crate) struct Neuron {
    /// Zero-based matrix index.
    pub(crate) index: u32,
    /// Stable source body identifier.
    pub(crate) body_id: i64,
    /// Non-empty source superclass.
    pub(crate) superclass: String,
    /// `FlyWire` type or source type fallback.
    pub(crate) cell_type: String,
    /// Source soma-side or root-side fallback.
    pub(crate) side: String,
    /// First matching source neurotransmitter label, if any.
    pub(crate) consensus_nt: Option<String>,
    /// Sign derived from `consensus_nt`.
    pub(crate) sign: SynapseSign,
}

/// Deterministic body-ID-to-matrix-index catalog.
#[derive(Debug)]
pub(crate) struct NeuronCatalog {
    /// Sorted, unique neuron records.
    neurons: Vec<Neuron>,
    /// Constant-time body-ID lookup used by both CSR passes.
    by_id: FxHashMap<i64, u32>,
}

impl NeuronCatalog {
    /// Filters superclass-annotated rows, keeps the first duplicate, and sorts IDs.
    pub(crate) fn from_annotations(
        rows: impl IntoIterator<Item = AnnotationRow>,
    ) -> Result<Self, MatrixError> {
        let mut unique = BTreeMap::new();
        for row in rows {
            if !row.superclass.is_empty() {
                match unique.entry(row.body_id) {
                    BTreeEntry::Vacant(entry) => {
                        entry.insert(row);
                    }
                    BTreeEntry::Occupied(_) => {}
                }
            }
        }

        let mut neurons = Vec::with_capacity(unique.len());
        let mut by_id = FxHashMap::default();
        for row in unique.into_values() {
            let index = u32::try_from(neurons.len()).map_err(|_| MatrixError::Overflow {
                context: "neuron index",
            })?;
            by_id.insert(row.body_id, index);
            neurons.push(Neuron {
                index,
                body_id: row.body_id,
                superclass: row.superclass,
                cell_type: row.cell_type,
                side: row.side,
                consensus_nt: None,
                sign: SynapseSign::Excitatory,
            });
        }
        Ok(Self { neurons, by_id })
    }

    /// Attaches the first neurotransmitter row seen for every selected body.
    pub(crate) fn apply_neurotransmitters(
        &mut self,
        rows: impl IntoIterator<Item = NeurotransmitterRow>,
    ) -> Result<(), MatrixError> {
        let mut seen = rustc_hash::FxHashSet::default();
        for row in rows {
            let Some(index) = self.by_id.get(&row.body_id).copied() else {
                continue;
            };
            if !seen.insert(row.body_id) {
                continue;
            }
            let neuron = self.neuron_mut(index)?;
            neuron.sign = SynapseSign::from_consensus(row.consensus.as_deref());
            neuron.consensus_nt = row.consensus;
        }
        Ok(())
    }

    /// Returns the number of selected neurons.
    pub(crate) fn len(&self) -> usize {
        self.neurons.len()
    }

    /// Returns neurons in their stable matrix-index order.
    pub(crate) fn neurons(&self) -> &[Neuron] {
        &self.neurons
    }

    /// Resolves one source body ID to its matrix index.
    pub(crate) fn index_of(&self, body_id: i64) -> Option<u32> {
        self.by_id.get(&body_id).copied()
    }

    /// Reads a sign by matrix index with an explicit bounds error.
    pub(crate) fn sign_of(&self, index: u32) -> Result<SynapseSign, MatrixError> {
        let index = usize::try_from(index).map_err(|_| MatrixError::Overflow {
            context: "neuron index conversion",
        })?;
        self.neurons
            .get(index)
            .map(|neuron| neuron.sign)
            .ok_or(MatrixError::CsrInvariant {
                proposition: "every matrix index names a selected neuron",
            })
    }

    /// Mutably resolves a matrix index while preserving the catalog invariant.
    fn neuron_mut(&mut self, index: u32) -> Result<&mut Neuron, MatrixError> {
        let index = usize::try_from(index).map_err(|_| MatrixError::Overflow {
            context: "neuron index conversion",
        })?;
        self.neurons
            .get_mut(index)
            .ok_or(MatrixError::CsrInvariant {
                proposition: "every catalog index names a neuron",
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn annotation(body_id: i64, superclass: &str) -> AnnotationRow {
        AnnotationRow {
            body_id,
            superclass: superclass.to_owned(),
            cell_type: String::new(),
            side: String::new(),
        }
    }

    #[test]
    fn catalog_keeps_first_annotated_body_and_sorts_ids() -> Result<(), MatrixError> {
        let catalog = NeuronCatalog::from_annotations([
            annotation(9, "descending_neuron"),
            annotation(3, ""),
            annotation(7, "sensory"),
            annotation(9, "duplicate"),
        ])?;

        let ids: Vec<_> = catalog
            .neurons()
            .iter()
            .map(|neuron| neuron.body_id)
            .collect();
        assert_eq!(ids, [7, 9]);
        let second = catalog.neurons().get(1).ok_or(MatrixError::CsrInvariant {
            proposition: "the test catalog has a second neuron",
        })?;
        assert_eq!(second.superclass, "descending_neuron");
        Ok(())
    }

    #[test]
    fn inhibitory_classification_matches_fly_ai_recipe() {
        assert_eq!(
            SynapseSign::from_consensus(Some("GABA")),
            SynapseSign::Inhibitory
        );
        assert_eq!(
            SynapseSign::from_consensus(Some("glutamate")),
            SynapseSign::Inhibitory
        );
        assert_eq!(
            SynapseSign::from_consensus(Some("histamine")),
            SynapseSign::Inhibitory
        );
        assert_eq!(
            SynapseSign::from_consensus(Some("acetylcholine")),
            SynapseSign::Excitatory
        );
        assert_eq!(SynapseSign::from_consensus(None), SynapseSign::Excitatory);
    }

    #[test]
    fn first_neurotransmitter_row_wins_even_when_null() -> Result<(), MatrixError> {
        let mut catalog = NeuronCatalog::from_annotations([annotation(7, "sensory")])?;
        catalog.apply_neurotransmitters([
            NeurotransmitterRow {
                body_id: 7,
                consensus: None,
            },
            NeurotransmitterRow {
                body_id: 7,
                consensus: Some("gaba".to_owned()),
            },
        ])?;
        assert_eq!(catalog.sign_of(0)?, SynapseSign::Excitatory);
        Ok(())
    }
}
