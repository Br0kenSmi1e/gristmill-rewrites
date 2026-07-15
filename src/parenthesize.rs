//! Parenthesization queries and policy choices.

use crate::{
    action::{Action, QueryError, TermPosition},
    repr::{Coefficient, Index, IndexId, TensorDef, TensorId, TensorRef, Term},
    state::{State, StateError},
};

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

        let (left_factors, right_factors) = split_factors(&self.term.factors, &left);
        let left_sums = self
            .term
            .sums
            .iter()
            .filter(|index| {
                uses_index(&left_factors, index.id) && !uses_index(&right_factors, index.id)
            })
            .copied()
            .collect::<Vec<_>>();
        let right_sums = self
            .term
            .sums
            .iter()
            .filter(|index| {
                uses_index(&right_factors, index.id) && !uses_index(&left_factors, index.id)
            })
            .copied()
            .collect::<Vec<_>>();
        let outer_sums = self
            .term
            .sums
            .iter()
            .filter(|index| !left_sums.contains(index) && !right_sums.contains(index))
            .copied()
            .collect::<Vec<_>>();

        let left_exts = self
            .exts
            .iter()
            .chain(&self.term.sums)
            .filter(|index| uses_index(&left_factors, index.id) && !left_sums.contains(index))
            .copied()
            .collect();
        let right_exts = self
            .exts
            .iter()
            .chain(&self.term.sums)
            .filter(|index| uses_index(&right_factors, index.id) && !right_sums.contains(index))
            .copied()
            .collect();

        let children = Box::new([
            TensorDef {
                base: self.base,
                exts: left_exts,
                rhs: vec![Term {
                    sums: left_sums,
                    coeff: Coefficient::from_integer(1.into()),
                    factors: left_factors,
                }],
            },
            TensorDef {
                base: self.base,
                exts: right_exts,
                rhs: vec![Term {
                    sums: right_sums,
                    coeff: Coefficient::from_integer(1.into()),
                    factors: right_factors,
                }],
            },
        ]);

        Ok(Action::Parenthesize(ParenthesizeAction {
            target: self.target,
            left: left.into_boxed_slice(),
            children,
            outer_sums,
            coeff: self.term.coeff.clone(),
        }))
    }
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
    left: Box<[bool]>,
    children: Box<[TensorDef; 2]>,
    outer_sums: Vec<Index>,
    coeff: Coefficient,
}

impl ParenthesizeAction {
    pub fn target(&self) -> TermPosition {
        self.target
    }

    pub fn left(&self) -> &[bool] {
        &self.left
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
    let [left, right] = *action.children;
    let (left_coeff, left_ref) = state.add_intermediate(left)?;
    let (right_coeff, right_ref) = state.add_intermediate(right)?;

    let replacement = Term {
        sums: action.outer_sums,
        coeff: action.coeff * left_coeff * right_coeff,
        factors: vec![left_ref, right_ref],
    };
    state.replace_terms(
        action.target.definition.0,
        &[action.target.term],
        vec![replacement],
    )?;

    Ok(())
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

fn uses_index(factors: &[TensorRef], index: IndexId) -> bool {
    factors.iter().any(|factor| factor.indices.contains(&index))
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
