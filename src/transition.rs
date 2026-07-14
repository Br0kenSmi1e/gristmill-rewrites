//! Symbolic state transitions.

use crate::{action::Action, state::State};

/// Failure to apply an action to a state.
///
/// Concrete error variants are intentionally deferred.
#[derive(Debug)]
pub struct ApplyError {
    _private: (),
}

/// Apply one policy-selected action and return the next canonical state.
pub fn apply(_state: &State, _action: Action) -> Result<State, ApplyError> {
    todo!("rewrite application and state normalization are not designed yet")
}
