//! Cost-free symbolic rewrite environment.
//!
//! This crate currently defines only its high-level API shape. Symbolic state
//! representation, rewrite discovery, and transition behavior are deliberately
//! left for later design.

pub mod action;
pub mod state;
pub mod transition;

pub use action::{
    Action, ActionQuery, ActionSpace, DefinitionPosition, QueryError, TermPosition, query,
};
pub use state::State;
pub use transition::{ApplyError, apply};

#[cfg(test)]
mod tests {
    use super::*;

    const _: fn(&State, ActionQuery) -> Result<ActionSpace, QueryError> = query;
    const _: fn(&State, Action) -> Result<State, ApplyError> = apply;
}
