//! Cost-free symbolic rewrite environment.

pub mod action;
mod biclique;
pub mod canon;
mod parenthesize;
mod permutation;
pub mod repr;
pub mod state;
pub mod transition;

pub use action::{
    Action, ActionQuery, ActionSpace, BicliqueAction, BicliqueChoiceError, BicliqueSpace,
    DefinitionPosition, ParenthesizeAction, ParenthesizeChoiceError, ParenthesizeSpace,
    PermutationAction, PermutationChoiceError, PermutationSpace, QueryError, TermPosition, query,
};
pub use repr::{
    Coefficient, Computation, Index, IndexId, RangeId, SymmetryAction, SymmetryGenerator,
    TensorDef, TensorId, TensorInfo, TensorRef, Term,
};
pub use state::{State, StateError};
pub use transition::{ApplyError, apply};

#[cfg(test)]
mod tests {
    use super::*;

    const _: fn(&State, ActionQuery) -> Result<ActionSpace, QueryError> = query;
    const _: fn(&State, Action) -> Result<State, ApplyError> = apply;
}
