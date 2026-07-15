//! Symbolic state transitions.

use crate::{
    action::{Action, ActionQuery},
    parenthesize,
    state::{State, StateError},
};

/// Failure to apply an action to a state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplyError {
    State(StateError),
    Unsupported { query: ActionQuery },
}

/// Apply an action selected from this state and return the next canonical state.
pub fn apply(state: &State, action: Action) -> Result<State, ApplyError> {
    let query = action.query();
    let mut next = state.clone();

    match action {
        Action::Parenthesize(action) => parenthesize::apply(&mut next, action)?,
        Action::Biclique(_) | Action::Permutation(_) => {
            return Err(ApplyError::Unsupported { query });
        }
    }

    Ok(next)
}

impl From<StateError> for ApplyError {
    fn from(error: StateError) -> Self {
        Self::State(error)
    }
}
