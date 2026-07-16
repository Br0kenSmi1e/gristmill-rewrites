//! Parenthesization queries and policy choices.

use crate::{
    action::{Action, QueryError, TermPosition},
    repr::{Coefficient, Index, IndexId, TensorDef, TensorId, TensorRef, Term},
    state::{State, StateError},
};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TermBipartition {
    pub(crate) coeff: Coefficient,
    pub(crate) left: Term,
    pub(crate) left_exts: Vec<Index>,
    pub(crate) right: Term,
    pub(crate) right_exts: Vec<Index>,
    pub(crate) contracted: Vec<Index>,
}

/// The non-symbolic policy interface for one parenthesization query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParenthesizeSpace {
    target: TermPosition,
    base: TensorId,
    exts: Vec<Index>,
    term: Term,
}

impl ParenthesizeSpace {
    pub fn target(&self) -> TermPosition {
        self.target
    }

    /// Return an owned semantic description of the parenthesization problem.
    pub fn snapshot(&self) -> (Vec<Index>, Term) {
        (self.exts.clone(), self.term.clone())
    }

    pub fn factor_count(&self) -> usize {
        self.term.factors.len()
    }

    /// Select an unordered binary partition of the target's factors.
    ///
    /// `left[i]` indicates that factor occurrence `i` belongs to the left
    /// child. Complementary masks describe the same partition and are
    /// normalized to the orientation containing factor zero on the left.
    pub fn select(&self, left: &[bool]) -> Result<Action, ParenthesizeChoiceError> {
        validate_choice(self.factor_count(), left)?;

        let mut left = left.to_vec();
        if !left[0] {
            for selected in &mut left {
                *selected = !*selected;
            }
        }

        let bipartition = bipartition_term(&self.exts, &self.term, &left);

        Ok(Action::Parenthesize(ParenthesizeAction {
            target: self.target,
            base: self.base,
            bipartition,
        }))
    }
}

fn validate_choice(factor_count: usize, left: &[bool]) -> Result<(), ParenthesizeChoiceError> {
    if factor_count <= 2 {
        return Err(ParenthesizeChoiceError::NoParenthesization { factor_count });
    }
    if left.len() != factor_count {
        return Err(ParenthesizeChoiceError::WrongPartitionLength {
            expected: factor_count,
            got: left.len(),
        });
    }
    if left.iter().all(|&selected| selected) || left.iter().all(|&selected| !selected) {
        return Err(ParenthesizeChoiceError::EmptyPartitionSide);
    }
    Ok(())
}

/// An invalid policy choice for a parenthesization space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParenthesizeChoiceError {
    NoParenthesization { factor_count: usize },
    WrongPartitionLength { expected: usize, got: usize },
    EmptyPartitionSide,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParenthesizeAction {
    target: TermPosition,
    base: TensorId,
    bipartition: TermBipartition,
}

impl ParenthesizeAction {
    pub fn target(&self) -> TermPosition {
        self.target
    }
}

pub(crate) fn query(state: &State, target: TermPosition) -> Result<ParenthesizeSpace, QueryError> {
    let definition = state
        .computation()
        .definitions
        .get(target.definition.0)
        .ok_or(QueryError::DefinitionOutOfBounds {
            position: target.definition,
        })?;
    let term = definition
        .rhs
        .get(target.term)
        .ok_or(QueryError::TermOutOfBounds { position: target })?;

    Ok(ParenthesizeSpace {
        target,
        base: definition.base,
        exts: definition.exts.clone(),
        term: term.clone(),
    })
}

pub(crate) fn apply(state: &mut State, action: ParenthesizeAction) -> Result<(), StateError> {
    let TermBipartition {
        coeff,
        left,
        left_exts,
        right,
        right_exts,
        contracted,
    } = action.bipartition;
    let left = TensorDef {
        base: action.base,
        exts: left_exts,
        rhs: vec![left],
    };
    let right = TensorDef {
        base: action.base,
        exts: right_exts,
        rhs: vec![right],
    };
    let (left_coeff, left_ref) = state.add_intermediate(left)?;
    let (right_coeff, right_ref) = state.add_intermediate(right)?;

    let replacement = Term {
        sums: contracted,
        coeff: coeff * left_coeff * right_coeff,
        factors: vec![left_ref, right_ref],
    };
    state.replace_terms(
        action.target.definition.0,
        &[action.target.term],
        vec![replacement],
    )?;

    Ok(())
}

pub(crate) fn bipartition_term(exts: &[Index], term: &Term, left: &[bool]) -> TermBipartition {
    let (left_factors, right_factors) = split_factors(&term.factors, left);
    let left_used = used_indices(&left_factors);
    let right_used = used_indices(&right_factors);

    let mut left_sums = Vec::new();
    let mut right_sums = Vec::new();
    let mut contracted = Vec::new();
    for &index in &term.sums {
        match (
            left_used.contains(&index.id),
            right_used.contains(&index.id),
        ) {
            (true, false) => left_sums.push(index),
            (false, true) => right_sums.push(index),
            (true, true) => contracted.push(index),
            (false, false) => unreachable!("validated summed indices are always used"),
        }
    }

    let left_exts = exts
        .iter()
        .chain(&contracted)
        .filter(|index| left_used.contains(&index.id))
        .copied()
        .collect();
    let right_exts = exts
        .iter()
        .chain(&contracted)
        .filter(|index| right_used.contains(&index.id))
        .copied()
        .collect();

    TermBipartition {
        coeff: term.coeff.clone(),
        left: Term {
            sums: left_sums,
            coeff: Coefficient::from_integer(1.into()),
            factors: left_factors,
        },
        left_exts,
        right: Term {
            sums: right_sums,
            coeff: Coefficient::from_integer(1.into()),
            factors: right_factors,
        },
        right_exts,
        contracted,
    }
}

fn split_factors(factors: &[TensorRef], left: &[bool]) -> (Vec<TensorRef>, Vec<TensorRef>) {
    let mut left_factors = Vec::new();
    let mut right_factors = Vec::new();
    for (factor, &selected) in factors.iter().zip(left) {
        if selected {
            left_factors.push(factor.clone());
        } else {
            right_factors.push(factor.clone());
        }
    }
    (left_factors, right_factors)
}

fn used_indices(factors: &[TensorRef]) -> BTreeSet<IndexId> {
    factors
        .iter()
        .flat_map(|factor| factor.indices.iter().copied())
        .collect()
}
