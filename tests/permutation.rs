use gristmill_rewrites::{
    Action, ActionQuery, ActionSpace, Coefficient, Computation, DefinitionPosition, Index, IndexId,
    PermutationChoiceError, RangeId, State, TensorDef, TensorId, TensorInfo, TensorRef, Term,
    apply, equivalent_states, query,
};
use std::collections::{BTreeMap, BTreeSet};

const RANGE: RangeId = RangeId(0);
const X: TensorId = TensorId(0);
const Y: TensorId = TensorId(1);

#[test]
fn exposes_complete_normalized_permutation_patterns() {
    let state = state_without_existing_intermediate();
    let target = DefinitionPosition(0);
    let ActionSpace::Permutation(space) =
        query(&state, ActionQuery::PermutationFactor(target)).unwrap()
    else {
        panic!("expected a permutation space");
    };

    assert_eq!(space.target(), target);
    assert_eq!(space.candidate_count(), 1);
    assert_eq!(space.shape(0), Some((2, 2)));
    assert_eq!(space.shape(1), None);
    assert_eq!(
        space.snapshot(),
        vec![(
            vec![index(0), index(1)],
            vec![reference(2, X, &[0, 1]), reference(3, Y, &[1, 0])],
            vec![(vec![0, 1], rational(1, 1)), (vec![1, 0], rational(-2, 1)),],
        )]
    );
    assert_eq!(
        space.select(1, &[], &[]),
        Err(PermutationChoiceError::CandidateOutOfBounds { index: 1, len: 1 })
    );
    assert_eq!(
        space.select(0, &[true], &[true, true]),
        Err(PermutationChoiceError::WrongRootMaskLength {
            expected: 2,
            got: 1,
        })
    );
    assert_eq!(
        space.select(0, &[true, true], &[true]),
        Err(PermutationChoiceError::WrongUseMaskLength {
            expected: 2,
            got: 1,
        })
    );
    assert_eq!(
        space.select(0, &[true, false], &[true, true]),
        Err(PermutationChoiceError::TooFewRoots)
    );
    assert_eq!(
        space.select(0, &[true, true], &[true, false]),
        Err(PermutationChoiceError::TooFewUses)
    );
    assert_eq!(
        space.select(0, &[true, true], &[false, true]),
        Err(PermutationChoiceError::TooFewUses)
    );

    let action = space.select(0, &[true, true], &[true, true]).unwrap();
    assert_eq!(
        action.query(),
        ActionQuery::PermutationFactor(DefinitionPosition(0))
    );
    assert!(matches!(action, Action::Permutation(_)));
}

#[test]
fn factors_a_shared_permutation_pattern() {
    let state = state_without_existing_intermediate();
    let action = permutation_action(&state, DefinitionPosition(0));

    let next = apply(&state, action).unwrap();
    let definitions = &next.computation().definitions;

    assert_eq!(definitions.len(), 2);
    assert_eq!(
        definitions[0].rhs,
        vec![
            reference(2, TensorId(3), &[0, 1]),
            reference(-4, TensorId(3), &[1, 0]),
        ]
    );
    assert_eq!(
        definitions[1],
        TensorDef {
            base: TensorId(3),
            exts: vec![index(0), index(1)],
            rhs: vec![
                reference(1, X, &[0, 1]),
                Term {
                    sums: Vec::new(),
                    coeff: rational(3, 2),
                    factors: vec![tensor(Y, &[1, 0])],
                },
            ],
        }
    );
}

#[test]
fn reuses_an_intermediate_with_reversed_external_slots() {
    let state = state_with_reversed_intermediate();
    let action = permutation_action(&state, DefinitionPosition(1));

    let next = apply(&state, action).unwrap();
    let definitions = &next.computation().definitions;

    assert_eq!(definitions.len(), 2);
    assert_eq!(
        definitions[1].rhs,
        vec![
            reference(-4, TensorId(2), &[0, 1]),
            reference(2, TensorId(2), &[1, 0]),
        ]
    );
}

#[test]
fn composes_rank_three_permutations_relative_to_each_root() {
    let output = TensorId(2);
    let state = State::new(
        Computation {
            ranges: BTreeSet::from([RANGE]),
            tensors: tensor_infos_with_rank(&[X, Y, output], 3),
            definitions: vec![TensorDef {
                base: output,
                exts: vec![index(0), index(1), index(2)],
                rhs: vec![
                    reference(2, X, &[0, 1, 2]),
                    reference(-4, X, &[1, 2, 0]),
                    reference(6, X, &[2, 0, 1]),
                    reference(3, Y, &[1, 0, 2]),
                    reference(-6, Y, &[2, 1, 0]),
                    reference(9, Y, &[0, 2, 1]),
                ],
            }],
        },
        vec![output],
    )
    .unwrap();
    let ActionSpace::Permutation(space) = query(
        &state,
        ActionQuery::PermutationFactor(DefinitionPosition(0)),
    )
    .unwrap() else {
        panic!("expected a permutation space");
    };

    assert_eq!(space.candidate_count(), 1);
    assert_eq!(space.shape(0), Some((2, 3)));
    assert_eq!(
        space.snapshot(),
        vec![(
            vec![index(0), index(1), index(2)],
            vec![reference(2, X, &[0, 1, 2]), reference(3, Y, &[1, 0, 2]),],
            vec![
                (vec![0, 1, 2], rational(1, 1)),
                (vec![1, 2, 0], rational(-2, 1)),
                (vec![2, 0, 1], rational(3, 1)),
            ],
        )]
    );
    let partial = apply(
        &state,
        space
            .select(0, &[true, true], &[false, true, true])
            .unwrap(),
    )
    .unwrap();
    assert!(equivalent_states(&state, &partial).unwrap());

    let next = apply(
        &state,
        space.select(0, &[true, true], &[true, true, true]).unwrap(),
    )
    .unwrap();
    assert_eq!(
        next.computation().definitions,
        vec![
            TensorDef {
                base: output,
                exts: vec![index(0), index(1), index(2)],
                rhs: vec![
                    reference(2, TensorId(3), &[0, 1, 2]),
                    reference(-4, TensorId(3), &[1, 2, 0]),
                    reference(6, TensorId(3), &[2, 0, 1]),
                ],
            },
            TensorDef {
                base: TensorId(3),
                exts: vec![index(0), index(1), index(2)],
                rhs: vec![
                    reference(1, X, &[0, 1, 2]),
                    Term {
                        sums: Vec::new(),
                        coeff: rational(3, 2),
                        factors: vec![tensor(Y, &[1, 0, 2])],
                    },
                ],
            },
        ]
    );
}

#[test]
fn discovers_a_maximal_shared_partial_pattern() {
    let output = TensorId(2);
    let state = State::new(
        Computation {
            ranges: BTreeSet::from([RANGE]),
            tensors: tensor_infos_with_rank(&[X, Y, output], 3),
            definitions: vec![TensorDef {
                base: output,
                exts: vec![index(0), index(1), index(2)],
                rhs: vec![
                    reference(2, X, &[0, 1, 2]),
                    reference(-4, X, &[1, 2, 0]),
                    reference(10, X, &[2, 0, 1]),
                    reference(3, Y, &[1, 0, 2]),
                    reference(-6, Y, &[2, 1, 0]),
                    reference(21, Y, &[0, 2, 1]),
                ],
            }],
        },
        vec![output],
    )
    .unwrap();
    let ActionSpace::Permutation(space) = query(
        &state,
        ActionQuery::PermutationFactor(DefinitionPosition(0)),
    )
    .unwrap() else {
        panic!("expected a permutation space");
    };

    assert_eq!(space.candidate_count(), 1);
    assert_eq!(space.shape(0), Some((2, 2)));
    assert_eq!(
        space.snapshot()[0].2,
        vec![
            (vec![0, 1, 2], rational(1, 1)),
            (vec![1, 2, 0], rational(-2, 1)),
        ]
    );

    let next = apply(
        &state,
        space.select(0, &[true, true], &[true, true]).unwrap(),
    )
    .unwrap();
    assert!(equivalent_states(&state, &next).unwrap());
    assert_eq!(next.computation().definitions[0].rhs.len(), 4);
}

#[test]
fn rejects_permutations_between_different_ranges() {
    let other_range = RangeId(1);
    let output = TensorId(2);
    let state = State::new(
        Computation {
            ranges: BTreeSet::from([RANGE, other_range]),
            tensors: tensor_infos(&[X, Y, output]),
            definitions: vec![TensorDef {
                base: output,
                exts: vec![index(0), index_in_range(1, other_range)],
                rhs: repeated_terms(),
            }],
        },
        vec![output],
    )
    .unwrap();
    let ActionSpace::Permutation(space) = query(
        &state,
        ActionQuery::PermutationFactor(DefinitionPosition(0)),
    )
    .unwrap() else {
        panic!("expected a permutation space");
    };

    assert_eq!(space.candidate_count(), 0);
}

fn permutation_action(state: &State, target: DefinitionPosition) -> Action {
    let ActionSpace::Permutation(space) =
        query(state, ActionQuery::PermutationFactor(target)).unwrap()
    else {
        panic!("expected a permutation space");
    };
    let (roots, uses) = space.shape(0).unwrap();
    space
        .select(0, &vec![true; roots], &vec![true; uses])
        .unwrap()
}

fn state_without_existing_intermediate() -> State {
    let output = TensorId(2);
    State::new(
        Computation {
            ranges: BTreeSet::from([RANGE]),
            tensors: tensor_infos(&[X, Y, output]),
            definitions: vec![TensorDef {
                base: output,
                exts: vec![index(0), index(1)],
                rhs: repeated_terms(),
            }],
        },
        vec![output],
    )
    .unwrap()
}

fn state_with_reversed_intermediate() -> State {
    let intermediate = TensorId(2);
    let output = TensorId(3);
    State::new(
        Computation {
            ranges: BTreeSet::from([RANGE]),
            tensors: tensor_infos(&[X, Y, intermediate, output]),
            definitions: vec![
                TensorDef {
                    base: intermediate,
                    exts: vec![index(0), index(1)],
                    rhs: vec![
                        reference(1, X, &[1, 0]),
                        Term {
                            sums: Vec::new(),
                            coeff: rational(3, 2),
                            factors: vec![tensor(Y, &[0, 1])],
                        },
                    ],
                },
                TensorDef {
                    base: output,
                    exts: vec![index(0), index(1)],
                    rhs: repeated_terms(),
                },
            ],
        },
        vec![output],
    )
    .unwrap()
}

fn repeated_terms() -> Vec<Term> {
    vec![
        reference(2, X, &[0, 1]),
        reference(-4, X, &[1, 0]),
        reference(3, Y, &[1, 0]),
        reference(-6, Y, &[0, 1]),
    ]
}

fn tensor_infos(tensors: &[TensorId]) -> BTreeMap<TensorId, TensorInfo> {
    tensor_infos_with_rank(tensors, 2)
}

fn tensor_infos_with_rank(tensors: &[TensorId], rank: usize) -> BTreeMap<TensorId, TensorInfo> {
    tensors
        .iter()
        .copied()
        .map(|tensor| {
            (
                tensor,
                TensorInfo {
                    rank,
                    symmetry: Vec::new(),
                },
            )
        })
        .collect()
}

fn reference(coeff: i64, base: TensorId, indices: &[u32]) -> Term {
    Term {
        sums: Vec::new(),
        coeff: Coefficient::from_integer(coeff.into()),
        factors: vec![tensor(base, indices)],
    }
}

fn tensor(tensor: TensorId, indices: &[u32]) -> TensorRef {
    TensorRef {
        tensor,
        indices: indices.iter().copied().map(IndexId).collect(),
    }
}

fn index(id: u32) -> Index {
    index_in_range(id, RANGE)
}

fn index_in_range(id: u32, range: RangeId) -> Index {
    Index {
        id: IndexId(id),
        range,
    }
}

fn rational(numerator: i64, denominator: i64) -> Coefficient {
    Coefficient::new(numerator.into(), denominator.into())
}
