//! Biclique factorization queries and policy choices.

use crate::{
    action::{ActionQuery, DefinitionPosition, QueryError},
    state::State,
};

/// The policy interface for one biclique factorization query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BicliqueSpace {
    target: DefinitionPosition,
}

impl BicliqueSpace {
    pub fn target(&self) -> DefinitionPosition {
        self.target
    }
}

/// One validated biclique factorization choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BicliqueAction {
    target: DefinitionPosition,
}

impl BicliqueAction {
    pub fn target(&self) -> DefinitionPosition {
        self.target
    }
}

pub(crate) fn query(
    _state: &State,
    target: DefinitionPosition,
) -> Result<BicliqueSpace, QueryError> {
    Err(QueryError::Unsupported {
        query: ActionQuery::BicliqueFactor(target),
    })
}
