//! Validated symbolic rewrite state with canonical definition terms.

use crate::canon::{CanonError, canon_expr};
use crate::repr::{
    Coefficient, Computation, Index, IndexId, RangeId, TensorDef, TensorId, TensorInfo, TensorRef,
    Term,
};
use std::collections::{BTreeMap, BTreeSet};

/// A validation or canonicalization failure while constructing a [`State`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateError {
    UnknownRange {
        range: RangeId,
    },
    UnknownTensor {
        tensor: TensorId,
    },
    DuplicateDefinition {
        tensor: TensorId,
    },
    DuplicateProtectedOutput {
        tensor: TensorId,
    },
    MissingProtectedOutputDefinition {
        tensor: TensorId,
    },
    TensorArityMismatch {
        tensor: TensorId,
        expected: usize,
        got: usize,
    },
    InvalidSymmetryArity {
        tensor: TensorId,
        expected: usize,
        got: usize,
    },
    InvalidSymmetryPermutation {
        tensor: TensorId,
        perm: Vec<usize>,
    },
    DuplicateExternalIndex {
        definition: TensorId,
        index: IndexId,
    },
    DuplicateSumIndex {
        definition: TensorId,
        term: usize,
        index: IndexId,
    },
    ExternalAndSumIndexOverlap {
        definition: TensorId,
        term: usize,
        index: IndexId,
    },
    UnknownFactorIndex {
        definition: TensorId,
        term: usize,
        index: IndexId,
    },
    UnusedSumIndex {
        definition: TensorId,
        term: usize,
        index: IndexId,
    },
    DependencyCycle {
        tensor: TensorId,
    },
    DefinitionOutOfBounds {
        position: usize,
    },
    TermOutOfBounds {
        definition: usize,
        term: usize,
    },
    ZeroIntermediate,
    ExhaustedTensorIds,
    Canonicalization(CanonError),
}

/// A structurally valid computation with canonical terms and protected outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    computation: Computation,
    protected_outputs: Vec<TensorId>,
}

impl State {
    /// Validate and canonicalize a computation, then establish its outputs.
    pub fn new(
        computation: Computation,
        protected_outputs: Vec<TensorId>,
    ) -> Result<Self, StateError> {
        let definitions = validate_computation(&computation)?;
        validate_protected_outputs(&computation, &definitions, &protected_outputs)?;

        let mut state = Self {
            computation,
            protected_outputs,
        };
        state
            .canonicalize_definitions()
            .map_err(StateError::Canonicalization)?;
        validate_acyclic(&state.computation, &definitions)?;

        Ok(state)
    }

    pub fn computation(&self) -> &Computation {
        &self.computation
    }

    pub fn protected_outputs(&self) -> &[TensorId] {
        &self.protected_outputs
    }

    pub fn into_parts(self) -> (Computation, Vec<TensorId>) {
        (self.computation, self.protected_outputs)
    }

    /// Return a reference to an equivalent intermediate, adding one if needed.
    ///
    /// `candidate.base` is ignored. The returned coefficient and tensor
    /// reference are expressed in the candidate's index scope. Protected
    /// outputs are not eligible for reuse.
    #[allow(dead_code)] // Used by the transition implementation built next.
    pub(crate) fn add_intermediate(
        &mut self,
        mut candidate: TensorDef,
    ) -> Result<(Coefficient, TensorRef), StateError> {
        let exts = candidate.exts.clone();
        let mut insertion = None;

        for permutation in permutations(exts.len()) {
            candidate.exts = permutation.iter().map(|&position| exts[position]).collect();
            let canonical = canonicalize_definition(&self.computation, &candidate)
                .map_err(StateError::Canonicalization)?;

            if permutation.iter().copied().eq(0..exts.len()) {
                insertion = Some(canonical.clone());
            }

            for existing in &self.computation.definitions {
                if self.protected_outputs.contains(&existing.base) {
                    continue;
                }
                if canonical.exts != existing.exts {
                    continue;
                }
                if let Some(coeff) = proportional_rhs(&canonical.rhs, &existing.rhs) {
                    return Ok((
                        coeff,
                        TensorRef {
                            tensor: existing.base,
                            indices: candidate.exts.iter().map(|index| index.id).collect(),
                        },
                    ));
                }
            }
        }

        let mut definition = insertion.expect("the identity permutation is always generated");
        let Some(first) = definition.rhs.first() else {
            return Err(StateError::ZeroIntermediate);
        };
        let coeff = first.coeff.clone();
        for term in &mut definition.rhs {
            term.coeff = term.coeff.clone() / coeff.clone();
        }

        let tensor = fresh_tensor_id(&self.computation)?;
        definition.base = tensor;
        self.computation.tensors.insert(
            tensor,
            TensorInfo {
                rank: definition.exts.len(),
                symmetry: Vec::new(),
            },
        );
        self.computation.definitions.push(definition);

        Ok((
            coeff,
            TensorRef {
                tensor,
                indices: exts.iter().map(|index| index.id).collect(),
            },
        ))
    }

    /// Replace selected terms and recanonicalize their definition.
    #[allow(dead_code)] // Used by the transition implementation built next.
    pub(crate) fn replace_terms(
        &mut self,
        definition: usize,
        removed: &[usize],
        replacements: Vec<Term>,
    ) -> Result<(), StateError> {
        let mut updated = self
            .computation
            .definitions
            .get(definition)
            .cloned()
            .ok_or(StateError::DefinitionOutOfBounds {
                position: definition,
            })?;

        let mut remove = vec![false; updated.rhs.len()];
        for &term in removed {
            let selected = remove
                .get_mut(term)
                .ok_or(StateError::TermOutOfBounds { definition, term })?;
            *selected = true;
        }

        updated.rhs = updated
            .rhs
            .into_iter()
            .enumerate()
            .filter_map(|(position, term)| (!remove[position]).then_some(term))
            .chain(replacements)
            .collect();
        let updated = canonicalize_definition(&self.computation, &updated)
            .map_err(StateError::Canonicalization)?;
        self.computation.definitions[definition] = updated;

        Ok(())
    }

    fn canonicalize_definitions(&mut self) -> Result<(), CanonError> {
        for position in 0..self.computation.definitions.len() {
            let canonical = {
                let definition = &self.computation.definitions[position];
                canonicalize_definition(&self.computation, definition)?
            };
            self.computation.definitions[position] = canonical;
        }

        Ok(())
    }
}

fn permutations(len: usize) -> Vec<Vec<usize>> {
    fn generate(position: usize, current: &mut [usize], result: &mut Vec<Vec<usize>>) {
        if position == current.len() {
            result.push(current.to_vec());
            return;
        }
        for next in position..current.len() {
            current.swap(position, next);
            generate(position + 1, current, result);
            current.swap(position, next);
        }
    }

    let mut current = (0..len).collect::<Vec<_>>();
    let mut result = Vec::new();
    generate(0, &mut current, &mut result);
    result
}

fn proportional_rhs(candidate: &[Term], existing: &[Term]) -> Option<Coefficient> {
    let (Some(candidate_first), Some(existing_first)) = (candidate.first(), existing.first())
    else {
        return None;
    };
    if candidate.len() != existing.len() {
        return None;
    }

    let coeff = candidate_first.coeff.clone() / existing_first.coeff.clone();
    candidate
        .iter()
        .zip(existing)
        .all(|(candidate, existing)| {
            candidate.sums == existing.sums
                && candidate.factors == existing.factors
                && candidate.coeff == existing.coeff.clone() * coeff.clone()
        })
        .then_some(coeff)
}

fn fresh_tensor_id(computation: &Computation) -> Result<TensorId, StateError> {
    let mut candidate = 0;
    for tensor in computation.tensors.keys() {
        if tensor.0 > candidate {
            break;
        }
        candidate = candidate
            .checked_add(1)
            .ok_or(StateError::ExhaustedTensorIds)?;
    }
    Ok(TensorId(candidate))
}

fn canonicalize_definition(
    computation: &Computation,
    definition: &TensorDef,
) -> Result<TensorDef, CanonError> {
    let mut external_ids = BTreeMap::new();
    let mut exts = Vec::with_capacity(definition.exts.len());
    for (position, external) in definition.exts.iter().enumerate() {
        let id = index_id(position)?;
        external_ids.insert(external.id, id);
        exts.push(Index {
            id,
            range: external.range,
        });
    }

    let mut rhs = Vec::with_capacity(definition.rhs.len());
    for term in &definition.rhs {
        let mut index_ids = external_ids.clone();
        let mut sums = Vec::with_capacity(term.sums.len());
        for (position, sum) in term.sums.iter().enumerate() {
            let position = definition
                .exts
                .len()
                .checked_add(position)
                .ok_or(CanonError::ExhaustedIndexIds)?;
            let id = index_id(position)?;
            index_ids.insert(sum.id, id);
            sums.push(Index {
                id,
                range: sum.range,
            });
        }

        let factors = term
            .factors
            .iter()
            .map(|factor| TensorRef {
                tensor: factor.tensor,
                indices: factor
                    .indices
                    .iter()
                    .map(|index| index_ids[index])
                    .collect(),
            })
            .collect();
        rhs.push(Term {
            sums,
            coeff: term.coeff.clone(),
            factors,
        });
    }

    Ok(TensorDef {
        base: definition.base,
        rhs: canon_expr(computation, &exts, &rhs)?,
        exts,
    })
}

fn index_id(position: usize) -> Result<IndexId, CanonError> {
    u32::try_from(position)
        .map(IndexId)
        .map_err(|_| CanonError::ExhaustedIndexIds)
}

fn validate_computation(
    computation: &Computation,
) -> Result<BTreeMap<TensorId, usize>, StateError> {
    validate_tensor_symmetries(computation)?;

    let mut definitions = BTreeMap::new();
    for (position, definition) in computation.definitions.iter().enumerate() {
        let Some(tensor) = computation.tensors.get(&definition.base) else {
            return Err(StateError::UnknownTensor {
                tensor: definition.base,
            });
        };
        if definitions.insert(definition.base, position).is_some() {
            return Err(StateError::DuplicateDefinition {
                tensor: definition.base,
            });
        }
        if definition.exts.len() != tensor.rank {
            return Err(StateError::TensorArityMismatch {
                tensor: definition.base,
                expected: tensor.rank,
                got: definition.exts.len(),
            });
        }
        validate_definition(computation, definition)?;
    }

    Ok(definitions)
}

fn validate_tensor_symmetries(computation: &Computation) -> Result<(), StateError> {
    for (&tensor_id, tensor) in &computation.tensors {
        for generator in &tensor.symmetry {
            if generator.perm.len() != tensor.rank {
                return Err(StateError::InvalidSymmetryArity {
                    tensor: tensor_id,
                    expected: tensor.rank,
                    got: generator.perm.len(),
                });
            }

            let mut seen = vec![false; tensor.rank];
            for &position in &generator.perm {
                if position >= tensor.rank || seen[position] {
                    return Err(StateError::InvalidSymmetryPermutation {
                        tensor: tensor_id,
                        perm: generator.perm.clone(),
                    });
                }
                seen[position] = true;
            }
        }
    }
    Ok(())
}

fn validate_definition(
    computation: &Computation,
    definition: &crate::repr::TensorDef,
) -> Result<(), StateError> {
    let mut externals = BTreeSet::new();
    for external in &definition.exts {
        validate_range(computation, external.range)?;
        if !externals.insert(external.id) {
            return Err(StateError::DuplicateExternalIndex {
                definition: definition.base,
                index: external.id,
            });
        }
    }

    for (term_position, term) in definition.rhs.iter().enumerate() {
        let mut sums = BTreeSet::new();
        for sum in &term.sums {
            validate_range(computation, sum.range)?;
            if externals.contains(&sum.id) {
                return Err(StateError::ExternalAndSumIndexOverlap {
                    definition: definition.base,
                    term: term_position,
                    index: sum.id,
                });
            }
            if !sums.insert(sum.id) {
                return Err(StateError::DuplicateSumIndex {
                    definition: definition.base,
                    term: term_position,
                    index: sum.id,
                });
            }
        }

        let mut used_sums = BTreeSet::new();
        for factor in &term.factors {
            let Some(tensor) = computation.tensors.get(&factor.tensor) else {
                return Err(StateError::UnknownTensor {
                    tensor: factor.tensor,
                });
            };
            if factor.indices.len() != tensor.rank {
                return Err(StateError::TensorArityMismatch {
                    tensor: factor.tensor,
                    expected: tensor.rank,
                    got: factor.indices.len(),
                });
            }

            for &index in &factor.indices {
                if sums.contains(&index) {
                    used_sums.insert(index);
                } else if !externals.contains(&index) {
                    return Err(StateError::UnknownFactorIndex {
                        definition: definition.base,
                        term: term_position,
                        index,
                    });
                }
            }
        }

        if let Some(&index) = sums.iter().find(|index| !used_sums.contains(index)) {
            return Err(StateError::UnusedSumIndex {
                definition: definition.base,
                term: term_position,
                index,
            });
        }
    }

    Ok(())
}

fn validate_range(computation: &Computation, range: RangeId) -> Result<(), StateError> {
    if computation.ranges.contains(&range) {
        Ok(())
    } else {
        Err(StateError::UnknownRange { range })
    }
}

fn validate_protected_outputs(
    computation: &Computation,
    definitions: &BTreeMap<TensorId, usize>,
    protected_outputs: &[TensorId],
) -> Result<(), StateError> {
    let mut seen = BTreeSet::new();
    for &output in protected_outputs {
        if !computation.tensors.contains_key(&output) {
            return Err(StateError::UnknownTensor { tensor: output });
        }
        if !seen.insert(output) {
            return Err(StateError::DuplicateProtectedOutput { tensor: output });
        }
        if !definitions.contains_key(&output) {
            return Err(StateError::MissingProtectedOutputDefinition { tensor: output });
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Visit {
    Active,
    Complete,
}

fn validate_acyclic(
    computation: &Computation,
    definitions: &BTreeMap<TensorId, usize>,
) -> Result<(), StateError> {
    let mut visits = BTreeMap::new();
    for &tensor in definitions.keys() {
        visit_definition(computation, definitions, tensor, &mut visits)?;
    }
    Ok(())
}

fn visit_definition(
    computation: &Computation,
    definitions: &BTreeMap<TensorId, usize>,
    tensor: TensorId,
    visits: &mut BTreeMap<TensorId, Visit>,
) -> Result<(), StateError> {
    match visits.get(&tensor) {
        Some(Visit::Complete) => return Ok(()),
        Some(Visit::Active) => return Err(StateError::DependencyCycle { tensor }),
        None => {}
    }

    visits.insert(tensor, Visit::Active);
    let definition = &computation.definitions[definitions[&tensor]];
    for dependency in definition
        .rhs
        .iter()
        .flat_map(|term| term.factors.iter())
        .map(|factor| factor.tensor)
    {
        if definitions.contains_key(&dependency) {
            visit_definition(computation, definitions, dependency, visits)?;
        }
    }
    visits.insert(tensor, Visit::Complete);
    Ok(())
}

#[cfg(test)]
mod mutation_tests {
    use super::*;

    const INPUT: TensorId = TensorId(0);
    const INTERMEDIATE: TensorId = TensorId(1);
    const OUTPUT: TensorId = TensorId(2);
    const RANGE: RangeId = RangeId(0);

    fn index(id: u32) -> Index {
        Index {
            id: IndexId(id),
            range: RANGE,
        }
    }

    fn tensor() -> TensorInfo {
        TensorInfo {
            rank: 2,
            symmetry: Vec::new(),
        }
    }

    fn term(coeff: i64, indices: [u32; 2]) -> Term {
        Term {
            sums: Vec::new(),
            coeff: Coefficient::from_integer(coeff.into()),
            factors: vec![TensorRef {
                tensor: INPUT,
                indices: indices.into_iter().map(IndexId).collect(),
            }],
        }
    }

    #[test]
    fn adds_canonical_intermediate_then_reuses_it() {
        let computation = Computation {
            ranges: BTreeSet::from([RANGE]),
            tensors: BTreeMap::from([(INPUT, tensor()), (INTERMEDIATE, tensor())]),
            definitions: vec![TensorDef {
                base: INTERMEDIATE,
                exts: vec![index(0), index(1)],
                rhs: vec![term(1, [0, 1])],
            }],
        };
        let mut state = State::new(computation, vec![INTERMEDIATE]).unwrap();

        let (coeff, factor) = state
            .add_intermediate(TensorDef {
                base: INTERMEDIATE,
                exts: vec![index(0), index(1)],
                rhs: vec![term(6, [0, 1])],
            })
            .unwrap();

        assert_eq!(coeff, Coefficient::from_integer(6.into()));
        assert_eq!(factor.tensor, OUTPUT);
        assert_eq!(factor.indices, vec![IndexId(0), IndexId(1)]);
        assert_eq!(state.computation.definitions.len(), 2);
        assert_eq!(state.computation.definitions[1].rhs, vec![term(1, [0, 1])]);

        let (coeff, factor) = state
            .add_intermediate(TensorDef {
                base: INTERMEDIATE,
                exts: vec![index(0), index(1)],
                rhs: vec![term(12, [1, 0])],
            })
            .unwrap();

        assert_eq!(coeff, Coefficient::from_integer(12.into()));
        assert_eq!(factor.tensor, OUTPUT);
        assert_eq!(factor.indices, vec![IndexId(1), IndexId(0)]);
        assert_eq!(state.computation.definitions.len(), 2);
    }

    #[test]
    fn reuses_with_coefficient_and_external_permutation() {
        let computation = Computation {
            ranges: BTreeSet::from([RANGE]),
            tensors: BTreeMap::from([
                (INPUT, tensor()),
                (INTERMEDIATE, tensor()),
                (OUTPUT, tensor()),
            ]),
            definitions: vec![
                TensorDef {
                    base: INTERMEDIATE,
                    exts: vec![index(0), index(1)],
                    rhs: vec![term(2, [1, 0])],
                },
                TensorDef {
                    base: OUTPUT,
                    exts: vec![index(0), index(1)],
                    rhs: vec![term(1, [0, 1])],
                },
            ],
        };
        let mut state = State::new(computation, vec![OUTPUT]).unwrap();

        let (coeff, factor) = state
            .add_intermediate(TensorDef {
                base: OUTPUT,
                exts: vec![index(0), index(1)],
                rhs: vec![term(6, [0, 1])],
            })
            .unwrap();

        assert_eq!(coeff, Coefficient::from_integer(3.into()));
        assert_eq!(factor.tensor, INTERMEDIATE);
        assert_eq!(factor.indices, vec![IndexId(1), IndexId(0)]);
        assert_eq!(state.computation.definitions.len(), 2);
    }

    #[test]
    fn rejects_a_zero_intermediate() {
        let computation = Computation {
            ranges: BTreeSet::from([RANGE]),
            tensors: BTreeMap::from([(INPUT, tensor()), (INTERMEDIATE, tensor())]),
            definitions: vec![TensorDef {
                base: INTERMEDIATE,
                exts: vec![index(0), index(1)],
                rhs: vec![term(1, [0, 1])],
            }],
        };
        let mut state = State::new(computation, vec![INTERMEDIATE]).unwrap();

        assert_eq!(
            state.add_intermediate(TensorDef {
                base: INTERMEDIATE,
                exts: vec![index(0), index(1)],
                rhs: vec![term(0, [0, 1])],
            }),
            Err(StateError::ZeroIntermediate)
        );
    }

    #[test]
    fn replaces_terms_and_recanonicalizes_the_definition() {
        let computation = Computation {
            ranges: BTreeSet::from([RANGE]),
            tensors: BTreeMap::from([(INPUT, tensor()), (OUTPUT, tensor())]),
            definitions: vec![TensorDef {
                base: OUTPUT,
                exts: vec![index(0), index(1)],
                rhs: vec![term(1, [0, 1]), term(1, [1, 0])],
            }],
        };
        let mut state = State::new(computation, vec![OUTPUT]).unwrap();

        state
            .replace_terms(0, &[0, 1], vec![term(2, [1, 0]), term(3, [1, 0])])
            .unwrap();

        assert_eq!(state.computation.definitions[0].rhs, vec![term(5, [1, 0])]);
    }

    #[test]
    fn rejects_invalid_replacement_positions_without_changing_state() {
        let computation = Computation {
            ranges: BTreeSet::from([RANGE]),
            tensors: BTreeMap::from([(INPUT, tensor()), (OUTPUT, tensor())]),
            definitions: vec![TensorDef {
                base: OUTPUT,
                exts: vec![index(0), index(1)],
                rhs: vec![term(1, [0, 1])],
            }],
        };
        let mut state = State::new(computation, vec![OUTPUT]).unwrap();
        let original = state.clone();

        assert_eq!(
            state.replace_terms(1, &[], Vec::new()),
            Err(StateError::DefinitionOutOfBounds { position: 1 })
        );
        assert_eq!(state, original);

        assert_eq!(
            state.replace_terms(0, &[1], Vec::new()),
            Err(StateError::TermOutOfBounds {
                definition: 0,
                term: 1,
            })
        );
        assert_eq!(state, original);
    }
}
