use gristmill_rewrites::{
    Action, ActionQuery, ActionSpace, Coefficient, Computation, DefinitionPosition, Index, IndexId,
    ParenthesizeChoiceError, ParenthesizeSpace, QueryError, RangeId, State, TensorDef, TensorId,
    TensorInfo, TensorRef, Term, TermPosition, apply, query,
};
use std::collections::{BTreeMap, BTreeSet};

fn state_with_factor_count(factor_count: usize) -> State {
    state_with_factor_count_and_coefficient(factor_count, 1)
}

fn state_with_factor_count_and_coefficient(factor_count: usize, coeff: i64) -> State {
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
                coeff: Coefficient::from_integer(coeff.into()),
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
fn validates_query_targets() {
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
    let ActionSpace::Biclique(space) = query(&state, biclique).unwrap() else {
        panic!("expected a biclique space");
    };
    assert_eq!(space.candidate_count(), 0);

    let permutation = ActionQuery::PermutationFactor(DefinitionPosition(0));
    let ActionSpace::Permutation(space) = query(&state, permutation).unwrap() else {
        panic!("expected a permutation space");
    };
    assert_eq!(space.candidate_count(), 0);
}

#[test]
fn applies_one_binary_split_without_a_single_factor_alias() {
    let state = state_with_factor_count(3);
    let action = parenthesize_space(&state)
        .select(&[true, true, false])
        .unwrap();

    let next = apply(&state, action).unwrap();

    assert_eq!(state.computation().definitions.len(), 1);
    assert_eq!(next.computation().definitions.len(), 2);
    assert_eq!(
        next.computation().definitions[0].rhs,
        vec![Term {
            sums: Vec::new(),
            coeff: Coefficient::from_integer(1.into()),
            factors: vec![
                TensorRef {
                    tensor: TensorId(2),
                    indices: Vec::new(),
                },
                TensorRef {
                    tensor: TensorId(4),
                    indices: Vec::new(),
                },
            ],
        }]
    );
    assert_eq!(
        next.computation().definitions[1],
        TensorDef {
            base: TensorId(4),
            exts: Vec::new(),
            rhs: vec![Term {
                sums: Vec::new(),
                coeff: Coefficient::from_integer(1.into()),
                factors: vec![
                    TensorRef {
                        tensor: TensorId(0),
                        indices: Vec::new(),
                    },
                    TensorRef {
                        tensor: TensorId(1),
                        indices: Vec::new(),
                    },
                ],
            }],
        }
    );
}

#[test]
fn applies_a_split_with_two_nontrivial_children() {
    let state = state_with_factor_count_and_coefficient(4, 6);
    let action = parenthesize_space(&state)
        .select(&[true, true, false, false])
        .unwrap();

    let next = apply(&state, action).unwrap();
    let definitions = &next.computation().definitions;

    assert_eq!(definitions.len(), 3);
    assert_eq!(
        definitions[0].rhs,
        vec![Term {
            sums: Vec::new(),
            coeff: Coefficient::from_integer(6.into()),
            factors: vec![
                TensorRef {
                    tensor: TensorId(5),
                    indices: Vec::new(),
                },
                TensorRef {
                    tensor: TensorId(6),
                    indices: Vec::new(),
                },
            ],
        }]
    );
    assert_eq!(
        definitions[1].rhs[0].factors,
        vec![
            TensorRef {
                tensor: TensorId(0),
                indices: Vec::new(),
            },
            TensorRef {
                tensor: TensorId(1),
                indices: Vec::new(),
            },
        ]
    );
    assert_eq!(
        definitions[2].rhs[0].factors,
        vec![
            TensorRef {
                tensor: TensorId(2),
                indices: Vec::new(),
            },
            TensorRef {
                tensor: TensorId(3),
                indices: Vec::new(),
            },
        ]
    );
}

#[test]
fn moves_child_only_sums_into_the_intermediate() {
    let range = RangeId(0);
    let tensor = |rank| TensorInfo {
        rank,
        symmetry: Vec::new(),
    };
    let index = |id| Index {
        id: IndexId(id),
        range,
    };
    let computation = Computation {
        ranges: BTreeSet::from([range]),
        tensors: BTreeMap::from([
            (TensorId(0), tensor(2)),
            (TensorId(1), tensor(2)),
            (TensorId(2), tensor(2)),
            (TensorId(3), tensor(2)),
        ]),
        definitions: vec![TensorDef {
            base: TensorId(3),
            exts: vec![index(0), index(1)],
            rhs: vec![Term {
                sums: vec![index(2), index(3)],
                coeff: Coefficient::from_integer(1.into()),
                factors: vec![
                    TensorRef {
                        tensor: TensorId(0),
                        indices: vec![IndexId(0), IndexId(2)],
                    },
                    TensorRef {
                        tensor: TensorId(1),
                        indices: vec![IndexId(2), IndexId(3)],
                    },
                    TensorRef {
                        tensor: TensorId(2),
                        indices: vec![IndexId(3), IndexId(1)],
                    },
                ],
            }],
        }],
    };
    let state = State::new(computation, vec![TensorId(3)]).unwrap();
    let action = parenthesize_space(&state)
        .select(&[true, true, false])
        .unwrap();

    let next = apply(&state, action).unwrap();

    assert_eq!(
        next.computation().definitions,
        vec![
            TensorDef {
                base: TensorId(3),
                exts: vec![index(0), index(1)],
                rhs: vec![Term {
                    sums: vec![index(2)],
                    coeff: Coefficient::from_integer(1.into()),
                    factors: vec![
                        TensorRef {
                            tensor: TensorId(2),
                            indices: vec![IndexId(2), IndexId(1)],
                        },
                        TensorRef {
                            tensor: TensorId(4),
                            indices: vec![IndexId(0), IndexId(2)],
                        },
                    ],
                }],
            },
            TensorDef {
                base: TensorId(4),
                exts: vec![index(0), index(1)],
                rhs: vec![Term {
                    sums: vec![index(2)],
                    coeff: Coefficient::from_integer(1.into()),
                    factors: vec![
                        TensorRef {
                            tensor: TensorId(0),
                            indices: vec![IndexId(0), IndexId(2)],
                        },
                        TensorRef {
                            tensor: TensorId(1),
                            indices: vec![IndexId(2), IndexId(1)],
                        },
                    ],
                }],
            },
        ]
    );
}
