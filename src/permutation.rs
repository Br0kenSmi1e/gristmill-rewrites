//! Permutation factorization queries and policy choices.

use crate::{
    action::{ActionQuery, DefinitionPosition, QueryError},
    state::State,
};

/// The policy interface for one permutation factorization query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermutationSpace {
    target: DefinitionPosition,
}

impl PermutationSpace {
    pub fn target(&self) -> DefinitionPosition {
        self.target
    }
}

/// One validated permutation factorization choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermutationAction {
    target: DefinitionPosition,
}

impl PermutationAction {
    pub fn target(&self) -> DefinitionPosition {
        self.target
    }
}

pub(crate) fn query(
    _state: &State,
    target: DefinitionPosition,
) -> Result<PermutationSpace, QueryError> {
    Err(QueryError::Unsupported {
        query: ActionQuery::PermutationFactor(target),
    })
}
