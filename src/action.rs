//! Shared symbolic rewrite query and action API.

use crate::{biclique, canon::CanonError, parenthesize, permutation, state::State};

pub use crate::biclique::{BicliqueAction, BicliqueChoiceError, BicliqueSnapshot, BicliqueSpace};
pub use crate::parenthesize::{ParenthesizeAction, ParenthesizeChoiceError, ParenthesizeSpace};
pub use crate::permutation::{PermutationAction, PermutationChoiceError, PermutationSpace};

/// A position in the state's definition sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DefinitionPosition(pub usize);

/// A term position within a definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TermPosition {
    pub definition: DefinitionPosition,
    pub term: usize,
}

/// A rewrite-family query with a target appropriate to that family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActionQuery {
    Parenthesize(TermPosition),
    BicliqueFactor(DefinitionPosition),
    PermutationFactor(DefinitionPosition),
}

/// A compact, family-specific description of valid policy choices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActionSpace {
    Parenthesize(ParenthesizeSpace),
    Biclique(BicliqueSpace),
    Permutation(PermutationSpace),
}

impl ActionSpace {
    pub fn query(&self) -> ActionQuery {
        match self {
            Self::Parenthesize(space) => ActionQuery::Parenthesize(space.target()),
            Self::Biclique(space) => ActionQuery::BicliqueFactor(space.target()),
            Self::Permutation(space) => ActionQuery::PermutationFactor(space.target()),
        }
    }
}

/// One validated policy choice.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    Parenthesize(ParenthesizeAction),
    Biclique(BicliqueAction),
    Permutation(PermutationAction),
}

impl Action {
    pub fn query(&self) -> ActionQuery {
        match self {
            Self::Parenthesize(action) => ActionQuery::Parenthesize(action.target()),
            Self::Biclique(action) => ActionQuery::BicliqueFactor(action.target()),
            Self::Permutation(action) => ActionQuery::PermutationFactor(action.target()),
        }
    }
}

/// Failure to construct an action space for a query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryError {
    DefinitionOutOfBounds { position: DefinitionPosition },
    TermOutOfBounds { position: TermPosition },
    Canonicalization(CanonError),
}

/// Query the legal actions for one rewrite family at one typed target.
pub fn query(state: &State, query: ActionQuery) -> Result<ActionSpace, QueryError> {
    match query {
        ActionQuery::Parenthesize(target) => {
            parenthesize::query(state, target).map(ActionSpace::Parenthesize)
        }
        ActionQuery::BicliqueFactor(target) => {
            biclique::query(state, target).map(ActionSpace::Biclique)
        }
        ActionQuery::PermutationFactor(target) => {
            permutation::query(state, target).map(ActionSpace::Permutation)
        }
    }
}
