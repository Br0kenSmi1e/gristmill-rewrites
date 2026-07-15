//! Validated symbolic rewrite state with canonical definition terms.

use crate::canon::{CanonError, canon_term};
use crate::repr::{Coefficient, Computation, IndexId, RangeId, TensorId, Term};
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

    fn canonicalize_definitions(&mut self) -> Result<(), CanonError> {
        for position in 0..self.computation.definitions.len() {
            let mut canonical = Vec::<Term>::new();
            {
                let definition = &self.computation.definitions[position];
                for term in &definition.rhs {
                    let Some(term) = canon_term(&self.computation, &definition.exts, term)? else {
                        continue;
                    };

                    if let Some(existing) = canonical.iter_mut().find(|existing| {
                        existing.sums == term.sums && existing.factors == term.factors
                    }) {
                        existing.coeff += term.coeff;
                    } else {
                        canonical.push(term);
                    }
                }
            }

            let zero = Coefficient::from_integer(0.into());
            canonical.retain(|term| term.coeff != zero);
            canonical.sort_by(|left, right| {
                left.sums
                    .cmp(&right.sums)
                    .then_with(|| left.factors.cmp(&right.factors))
            });
            self.computation.definitions[position].rhs = canonical;
        }

        Ok(())
    }
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
