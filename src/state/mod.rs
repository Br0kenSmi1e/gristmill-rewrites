//! Validated symbolic rewrite state with canonical definition terms.

mod canonicalize;
mod intermediate;
mod validate;

#[cfg(test)]
mod tests;

use self::canonicalize::canonicalize_definition;
use self::validate::{validate_acyclic, validate_computation, validate_protected_outputs};
use crate::canon::CanonError;
use crate::repr::{Computation, IndexId, RangeId, TensorId, Term};

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

    /// Replace selected terms and recanonicalize their definition.
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
}
