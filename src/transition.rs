//! Symbolic state transitions.

use crate::{
    action::Action,
    biclique, parenthesize, permutation,
    state::{State, StateError},
};

/// Failure to apply an action to a state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplyError {
    State(StateError),
}

/// Apply an action selected from this state and return the next canonical state.
pub fn apply(state: &State, action: Action) -> Result<State, ApplyError> {
    let mut next = state.clone();

    match action {
        Action::Parenthesize(action) => parenthesize::apply(&mut next, action)?,
        Action::Permutation(action) => permutation::apply(&mut next, action)?,
        Action::Biclique(action) => biclique::apply(&mut next, action)?,
    }

    Ok(next)
}

impl From<StateError> for ApplyError {
    fn from(error: StateError) -> Self {
        Self::State(error)
    }
}
