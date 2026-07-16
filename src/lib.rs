//! Cost-free symbolic rewrite environment.

pub mod action;
mod biclique;
mod canon;
pub mod cost;
pub mod io;
mod parenthesize;
mod permutation;
pub mod repr;
pub mod state;
pub mod transition;
pub mod verify;

pub use action::{
    Action, ActionQuery, ActionSpace, BicliqueAction, BicliqueChoiceError, BicliqueSnapshot,
    BicliqueSpace, DefinitionPosition, ParenthesizeAction, ParenthesizeChoiceError,
    ParenthesizeSpace, PermutationAction, PermutationChoiceError, PermutationSpace, QueryError,
    TermPosition, query,
};
pub use canon::CanonError;
pub use cost::{CostError, log_flops};
pub use repr::{
    Coefficient, Computation, Index, IndexId, RangeId, SymmetryAction, SymmetryGenerator,
    TensorDef, TensorId, TensorInfo, TensorRef, Term,
};
pub use state::{State, StateError};
pub use transition::{ApplyError, apply};
pub use verify::{VerifyError, equivalent_computations, equivalent_states};

#[cfg(test)]
mod tests {
    use super::*;

    const _: fn(&State, ActionQuery) -> Result<ActionSpace, QueryError> = query;
    const _: fn(&State, Action) -> Result<State, ApplyError> = apply;
}
