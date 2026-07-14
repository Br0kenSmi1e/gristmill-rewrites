//! Symbolic rewrite queries and actions.

use crate::state::State;

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

/// A lazy representation of every legal action for one query.
///
/// Its policy-facing representation is intentionally deferred.
pub struct ActionSpace {
    _private: (),
}

/// One concrete choice produced from an [`ActionSpace`] by an external policy.
///
/// Its family-specific representation is intentionally deferred.
pub struct Action {
    _private: (),
}

/// Failure to construct an action space for a query.
///
/// Concrete error variants are intentionally deferred.
#[derive(Debug)]
pub struct QueryError {
    _private: (),
}

/// Query the legal actions for one rewrite family at one typed target.
pub fn query(_state: &State, _query: ActionQuery) -> Result<ActionSpace, QueryError> {
    todo!("action-space representation and rewrite discovery are not designed yet")
}
