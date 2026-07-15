//! Permutation factorization queries and policy choices.

use crate::{
    action::{Action, DefinitionPosition, QueryError},
    repr::{Coefficient, Index, IndexId, TensorDef, TensorId, TensorRef, Term},
    state::{State, StateError},
};
use std::collections::{BTreeMap, BTreeSet};

/// The policy interface for one permutation factorization query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermutationSpace {
    target: DefinitionPosition,
    choices: Vec<PermutationAction>,
}

impl PermutationSpace {
    pub fn target(&self) -> DefinitionPosition {
        self.target
    }

    pub fn candidate_count(&self) -> usize {
        self.choices.len()
    }

    pub fn shape(&self, candidate: usize) -> Option<(usize, usize)> {
        self.choices
            .get(candidate)
            .map(|choice| (choice.intermediate.rhs.len(), choice.pattern.len()))
    }

    pub fn select(&self, candidate: usize) -> Result<Action, PermutationChoiceError> {
        self.choices
            .get(candidate)
            .cloned()
            .map(Action::Permutation)
            .ok_or(PermutationChoiceError::CandidateOutOfBounds {
                index: candidate,
                len: self.choices.len(),
            })
    }
}

/// An invalid permutation-factorization choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermutationChoiceError {
    CandidateOutOfBounds { index: usize, len: usize },
}

/// One validated permutation factorization choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermutationAction {
    target: DefinitionPosition,
    terms: Vec<usize>,
    intermediate: TensorDef,
    pattern: Vec<(Vec<IndexId>, Coefficient)>,
}

impl PermutationAction {
    pub fn target(&self) -> DefinitionPosition {
        self.target
    }
}

struct BasePattern {
    tensor: TensorId,
    leading: Coefficient,
    terms: Vec<usize>,
}

pub(crate) fn query(
    state: &State,
    target: DefinitionPosition,
) -> Result<PermutationSpace, QueryError> {
    let definition = state
        .computation()
        .definitions
        .get(target.0)
        .ok_or(QueryError::DefinitionOutOfBounds { position: target })?;
    let external = definition
        .exts
        .iter()
        .map(|index| (index.id, *index))
        .collect::<BTreeMap<_, _>>();
    let mut by_tensor =
        BTreeMap::<TensorId, BTreeMap<Vec<IndexId>, (Coefficient, Vec<usize>)>>::new();

    for (position, term) in definition.rhs.iter().enumerate() {
        let [factor] = term.factors.as_slice() else {
            continue;
        };
        if !term.sums.is_empty()
            || factor.indices.iter().collect::<BTreeSet<_>>().len() != factor.indices.len()
            || factor
                .indices
                .iter()
                .any(|index| !external.contains_key(index))
        {
            continue;
        }

        let occurrence = by_tensor
            .entry(factor.tensor)
            .or_default()
            .entry(factor.indices.clone())
            .or_insert_with(|| (zero(), Vec::new()));
        occurrence.0 += &term.coeff;
        occurrence.1.push(position);
    }

    let mut groups = BTreeMap::<Vec<(Vec<IndexId>, Coefficient)>, Vec<BasePattern>>::new();
    for (tensor, occurrences) in by_tensor {
        let occurrences = occurrences
            .into_iter()
            .filter(|(_, (coefficient, _))| coefficient != &zero())
            .collect::<Vec<_>>();
        if occurrences.len() < 2 {
            continue;
        }

        let leading = occurrences[0].1.0.clone();
        let pattern = occurrences
            .iter()
            .map(|(indices, (coefficient, _))| (indices.clone(), coefficient / &leading))
            .collect::<Vec<_>>();
        let terms = occurrences
            .into_iter()
            .flat_map(|(_, (_, terms))| terms)
            .collect();
        groups.entry(pattern).or_default().push(BasePattern {
            tensor,
            leading,
            terms,
        });
    }

    let mut choices = Vec::new();
    for (pattern, bases) in groups {
        if bases.len() < 2 {
            continue;
        }
        let Some(exts) = pattern_exts(&pattern, &external) else {
            continue;
        };
        let pivot = pattern[0].0.clone();
        let mut terms = bases
            .iter()
            .flat_map(|base| base.terms.iter().copied())
            .collect::<Vec<_>>();
        terms.sort_unstable();
        let rhs = bases
            .into_iter()
            .map(|base| Term {
                sums: Vec::new(),
                coeff: base.leading,
                factors: vec![TensorRef {
                    tensor: base.tensor,
                    indices: pivot.clone(),
                }],
            })
            .collect();

        choices.push(PermutationAction {
            target,
            terms,
            intermediate: TensorDef {
                base: definition.base,
                exts,
                rhs,
            },
            pattern,
        });
    }

    Ok(PermutationSpace { target, choices })
}

fn pattern_exts(
    pattern: &[(Vec<IndexId>, Coefficient)],
    external: &BTreeMap<IndexId, Index>,
) -> Option<Vec<Index>> {
    let pivot = &pattern.first()?.0;
    if pivot.is_empty() {
        return None;
    }
    let exts = pivot
        .iter()
        .map(|index| external.get(index).copied())
        .collect::<Option<Vec<_>>>()?;

    if pattern.iter().any(|(indices, _)| {
        indices.len() != exts.len()
            || indices.iter().zip(&exts).any(|(index, pivot)| {
                external
                    .get(index)
                    .is_none_or(|index| index.range != pivot.range)
            })
    }) {
        None
    } else {
        Some(exts)
    }
}

pub(crate) fn apply(state: &mut State, action: PermutationAction) -> Result<(), StateError> {
    let pivot = action
        .intermediate
        .exts
        .iter()
        .map(|index| index.id)
        .collect::<Vec<_>>();
    let (intermediate_coeff, intermediate_ref) = state.add_intermediate(action.intermediate)?;
    let replacements = action
        .pattern
        .into_iter()
        .map(|(indices, coefficient)| {
            let substitution = pivot
                .iter()
                .copied()
                .zip(indices)
                .collect::<BTreeMap<_, _>>();
            Term {
                sums: Vec::new(),
                coeff: &intermediate_coeff * coefficient,
                factors: vec![TensorRef {
                    tensor: intermediate_ref.tensor,
                    indices: intermediate_ref
                        .indices
                        .iter()
                        .map(|index| substitution[index])
                        .collect(),
                }],
            }
        })
        .collect();

    state.replace_terms(action.target.0, &action.terms, replacements)
}

fn zero() -> Coefficient {
    Coefficient::from_integer(0.into())
}
