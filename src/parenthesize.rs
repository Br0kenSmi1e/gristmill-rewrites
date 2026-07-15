//! Parenthesization queries and policy choices.

use crate::{
    action::{Action, QueryError, TermPosition},
    state::State,
};

/// The non-symbolic policy interface for one parenthesization query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParenthesizeSpace {
    target: TermPosition,
    factor_count: usize,
}

impl ParenthesizeSpace {
    pub fn target(&self) -> TermPosition {
        self.target
    }

    pub fn factor_count(&self) -> usize {
        self.factor_count
    }

    /// Select an unordered binary partition of the target's factors.
    ///
    /// `left[i]` indicates that factor occurrence `i` belongs to the left
    /// child. Complementary masks describe the same partition and are
    /// normalized to the orientation containing factor zero on the left.
    pub fn select(&self, left: &[bool]) -> Result<Action, ParenthesizeChoiceError> {
        validate_choice(self.factor_count, left)?;

        let mut left = left.to_vec();
        if !left[0] {
            for selected in &mut left {
                *selected = !*selected;
            }
        }

        Ok(Action::Parenthesize(ParenthesizeAction {
            target: self.target,
            left: left.into_boxed_slice(),
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
        factor_count: term.factors.len(),
    })
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
