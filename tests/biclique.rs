use gristmill_rewrites::{
    Action, ActionQuery, ActionSpace, BicliqueChoiceError, Coefficient, Computation,
    DefinitionPosition, Index, IndexId, RangeId, State, TensorDef, TensorId, TensorInfo, TensorRef,
    Term, apply, query,
};
use std::collections::{BTreeMap, BTreeSet};

const A: TensorId = TensorId(0);
const B: TensorId = TensorId(1);
const C: TensorId = TensorId(2);
const D: TensorId = TensorId(3);
const RANGE: RangeId = RangeId(0);

#[test]
fn exposes_masked_maximal_bicliques() {
    let state = rank_one_rectangle();
    let target = DefinitionPosition(0);
    let ActionSpace::Biclique(space) = query(&state, ActionQuery::BicliqueFactor(target)).unwrap()
    else {
        panic!("expected a biclique space");
    };

    assert_eq!(space.target(), target);
    assert_eq!(space.candidate_count(), 1);
    assert_eq!(space.shape(0), Some((2, 2)));
    assert_eq!(space.shape(1), None);
    assert_eq!(
        space.select(1, &[true], &[true]),
        Err(BicliqueChoiceError::CandidateOutOfBounds { index: 1, len: 1 })
    );
    assert_eq!(
        space.select(0, &[true], &[true, true]),
        Err(BicliqueChoiceError::WrongLeftMaskLength {
            expected: 2,
            got: 1,
        })
    );
    assert_eq!(
        space.select(0, &[true, true], &[true]),
        Err(BicliqueChoiceError::WrongRightMaskLength {
            expected: 2,
            got: 1,
        })
    );
    assert_eq!(
        space.select(0, &[false, false], &[true, true]),
        Err(BicliqueChoiceError::EmptySide)
    );

    let action = space.select(0, &[true, false], &[true, true]).unwrap();
    assert_eq!(action.query(), ActionQuery::BicliqueFactor(target));
    assert!(matches!(action, Action::Biclique(_)));

    let next = apply(&state, action).unwrap();
    assert_eq!(state.computation().definitions.len(), 1);
    assert_eq!(next.computation().definitions.len(), 2);
    assert_eq!(next.computation().definitions[0].rhs.len(), 3);
    assert_eq!(next.computation().definitions[1].rhs.len(), 2);
}

#[test]
fn factors_a_biclique_through_the_public_api() {
    let output = D;
    let state = State::new(
        Computation {
            ranges: BTreeSet::new(),
            tensors: scalar_infos(&[A, B, C, output]),
            definitions: vec![TensorDef {
                base: output,
                exts: Vec::new(),
                rhs: vec![product(2, A, B), product(3, A, C)],
            }],
        },
        vec![output],
    )
    .unwrap();
    let target = DefinitionPosition(0);
    let ActionSpace::Biclique(space) = query(&state, ActionQuery::BicliqueFactor(target)).unwrap()
    else {
        panic!("expected a biclique space");
    };

    assert_eq!(space.candidate_count(), 1);
    let (left, right) = space.shape(0).unwrap();
    assert_eq!((left.min(right), left.max(right)), (1, 2));

    let action = space
        .select(0, &vec![true; left], &vec![true; right])
        .unwrap();
    let next = apply(&state, action).unwrap();

    assert_eq!(
        next.computation().definitions,
        vec![
            TensorDef {
                base: output,
                exts: Vec::new(),
                rhs: vec![Term {
                    sums: Vec::new(),
                    coeff: integer(2),
                    factors: vec![scalar(A), scalar(TensorId(4))],
                }],
            },
            TensorDef {
                base: TensorId(4),
                exts: Vec::new(),
                rhs: vec![
                    single(1, B),
                    Term {
                        sums: Vec::new(),
                        coeff: Coefficient::new(3.into(), 2.into()),
                        factors: vec![scalar(C)],
                    },
                ],
            },
        ]
    );
}

#[test]
fn preserves_the_contracted_interface_when_applying() {
    let output = D;
    let state = State::new(
        Computation {
            ranges: BTreeSet::from([RANGE]),
            tensors: rank_two_infos(&[A, B, C, output]),
            definitions: vec![TensorDef {
                base: output,
                exts: vec![index(0), index(1)],
                rhs: vec![contracted_product(2, A, B), contracted_product(3, A, C)],
            }],
        },
        vec![output],
    )
    .unwrap();
    let ActionSpace::Biclique(space) =
        query(&state, ActionQuery::BicliqueFactor(DefinitionPosition(0))).unwrap()
    else {
        panic!("expected a biclique space");
    };
    let (left, right) = space.shape(0).unwrap();

    let action = space
        .select(0, &vec![true; left], &vec![true; right])
        .unwrap();
    let next = apply(&state, action).unwrap();

    assert_eq!(
        next.computation().definitions,
        vec![
            TensorDef {
                base: output,
                exts: vec![index(0), index(1)],
                rhs: vec![Term {
                    sums: vec![index(2)],
                    coeff: integer(2),
                    factors: vec![tensor(A, &[0, 2]), tensor(TensorId(4), &[1, 2])],
                }],
            },
            TensorDef {
                base: TensorId(4),
                exts: vec![index(0), index(1)],
                rhs: vec![
                    indexed_single(integer(1), B, &[1, 0]),
                    indexed_single(Coefficient::new(3.into(), 2.into()), C, &[1, 0]),
                ],
            },
        ]
    );
}

fn rank_one_rectangle() -> State {
    let output = TensorId(4);
    State::new(
        Computation {
            ranges: BTreeSet::new(),
            tensors: scalar_infos(&[A, B, C, D, output]),
            definitions: vec![TensorDef {
                base: output,
                exts: Vec::new(),
                rhs: vec![
                    product(2, A, C),
                    product(3, A, D),
                    product(4, B, C),
                    product(6, B, D),
                ],
            }],
        },
        vec![output],
    )
    .unwrap()
}

fn scalar_infos(tensors: &[TensorId]) -> BTreeMap<TensorId, TensorInfo> {
    tensors
        .iter()
        .copied()
        .map(|tensor| {
            (
                tensor,
                TensorInfo {
                    rank: 0,
                    symmetry: Vec::new(),
                },
            )
        })
        .collect()
}

fn rank_two_infos(tensors: &[TensorId]) -> BTreeMap<TensorId, TensorInfo> {
    tensors
        .iter()
        .copied()
        .map(|tensor| {
            (
                tensor,
                TensorInfo {
                    rank: 2,
                    symmetry: Vec::new(),
                },
            )
        })
        .collect()
}

fn product(coeff: i64, left: TensorId, right: TensorId) -> Term {
    Term {
        sums: Vec::new(),
        coeff: integer(coeff),
        factors: vec![scalar(left), scalar(right)],
    }
}

fn single(coeff: i64, tensor: TensorId) -> Term {
    Term {
        sums: Vec::new(),
        coeff: integer(coeff),
        factors: vec![scalar(tensor)],
    }
}

fn contracted_product(coeff: i64, left: TensorId, right: TensorId) -> Term {
    Term {
        sums: vec![index(2)],
        coeff: integer(coeff),
        factors: vec![tensor(left, &[0, 2]), tensor(right, &[2, 1])],
    }
}

fn indexed_single(coeff: Coefficient, tensor_id: TensorId, indices: &[u32]) -> Term {
    Term {
        sums: Vec::new(),
        coeff,
        factors: vec![tensor(tensor_id, indices)],
    }
}

fn scalar(tensor: TensorId) -> TensorRef {
    TensorRef {
        tensor,
        indices: Vec::new(),
    }
}

fn tensor(tensor: TensorId, indices: &[u32]) -> TensorRef {
    TensorRef {
        tensor,
        indices: indices.iter().copied().map(IndexId).collect(),
    }
}

fn index(id: u32) -> Index {
    Index {
        id: IndexId(id),
        range: RANGE,
    }
}

fn integer(value: i64) -> Coefficient {
    Coefficient::from_integer(value.into())
}
