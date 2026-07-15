use gristmill_rewrites::{
    Action, ActionQuery, ActionSpace, Coefficient, Computation, DefinitionPosition,
    ParenthesizeChoiceError, ParenthesizeSpace, QueryError, State, TensorDef, TensorId, TensorInfo,
    TensorRef, Term, TermPosition, query,
};
use std::collections::{BTreeMap, BTreeSet};

fn state_with_factor_count(factor_count: usize) -> State {
    let output = TensorId(factor_count as u32);
    let tensors = (0..=factor_count)
        .map(|position| {
            (
                TensorId(position as u32),
                TensorInfo {
                    rank: 0,
                    symmetry: Vec::new(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let factors = (0..factor_count)
        .map(|position| TensorRef {
            tensor: TensorId(position as u32),
            indices: Vec::new(),
        })
        .collect();
    let computation = Computation {
        ranges: BTreeSet::new(),
        tensors,
        definitions: vec![TensorDef {
            base: output,
            exts: Vec::new(),
            rhs: vec![Term {
                sums: Vec::new(),
                coeff: Coefficient::from_integer(1.into()),
                factors,
            }],
        }],
    };

    State::new(computation, vec![output]).unwrap()
}

fn target() -> TermPosition {
    TermPosition {
        definition: DefinitionPosition(0),
        term: 0,
    }
}

fn parenthesize_space(state: &State) -> ParenthesizeSpace {
    match query(state, ActionQuery::Parenthesize(target())).unwrap() {
        ActionSpace::Parenthesize(space) => space,
        _ => panic!("expected a parenthesization space"),
    }
}

#[test]
fn exposes_a_non_symbolic_parenthesization_space() {
    let state = state_with_factor_count(3);
    let space = query(&state, ActionQuery::Parenthesize(target())).unwrap();

    assert_eq!(space.query(), ActionQuery::Parenthesize(target()));
    let ActionSpace::Parenthesize(space) = space else {
        panic!("expected a parenthesization space");
    };
    assert_eq!(space.target(), target());
    assert_eq!(space.factor_count(), 3);

    let action = space.select(&[true, true, false]).unwrap();
    assert_eq!(action.query(), ActionQuery::Parenthesize(target()));
    let Action::Parenthesize(action) = action else {
        panic!("expected a parenthesization action");
    };
    assert_eq!(action.target(), target());
    assert_eq!(action.left(), &[true, true, false]);
}

#[test]
fn treats_complementary_masks_as_the_same_partition() {
    let state = state_with_factor_count(3);
    let space = parenthesize_space(&state);

    assert_eq!(
        space.select(&[true, true, false]),
        space.select(&[false, false, true])
    );
}

#[test]
fn validates_parenthesization_choices() {
    let state = state_with_factor_count(3);
    let space = parenthesize_space(&state);

    assert_eq!(
        space.select(&[true, false]),
        Err(ParenthesizeChoiceError::WrongPartitionLength {
            expected: 3,
            got: 2,
        })
    );
    assert_eq!(
        space.select(&[true, true, true]),
        Err(ParenthesizeChoiceError::EmptyPartitionSide)
    );
    assert_eq!(
        space.select(&[false, false, false]),
        Err(ParenthesizeChoiceError::EmptyPartitionSide)
    );

    let binary_state = state_with_factor_count(2);
    let binary_space = parenthesize_space(&binary_state);
    assert_eq!(
        binary_space.select(&[true, false]),
        Err(ParenthesizeChoiceError::NoParenthesization { factor_count: 2 })
    );
}

#[test]
fn validates_query_targets_and_reports_deferred_families() {
    let state = state_with_factor_count(3);
    let missing_definition = TermPosition {
        definition: DefinitionPosition(4),
        term: 0,
    };
    assert_eq!(
        query(&state, ActionQuery::Parenthesize(missing_definition)),
        Err(QueryError::DefinitionOutOfBounds {
            position: DefinitionPosition(4),
        })
    );

    let missing_term = TermPosition {
        definition: DefinitionPosition(0),
        term: 3,
    };
    assert_eq!(
        query(&state, ActionQuery::Parenthesize(missing_term)),
        Err(QueryError::TermOutOfBounds {
            position: missing_term,
        })
    );

    let biclique = ActionQuery::BicliqueFactor(DefinitionPosition(0));
    assert_eq!(
        query(&state, biclique),
        Err(QueryError::Unsupported { query: biclique })
    );

    let permutation = ActionQuery::PermutationFactor(DefinitionPosition(0));
    assert_eq!(
        query(&state, permutation),
        Err(QueryError::Unsupported { query: permutation })
    );
}
