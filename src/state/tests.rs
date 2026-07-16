use super::*;
use crate::repr::{Coefficient, Index, TensorDef, TensorInfo, TensorRef};
use std::collections::{BTreeMap, BTreeSet};

const INPUT: TensorId = TensorId(0);
const INTERMEDIATE: TensorId = TensorId(1);
const OUTPUT: TensorId = TensorId(2);
const RANGE: RangeId = RangeId(0);

fn index(id: u32) -> Index {
    Index {
        id: IndexId(id),
        range: RANGE,
    }
}

fn tensor() -> TensorInfo {
    TensorInfo {
        rank: 2,
        symmetry: Vec::new(),
    }
}

fn term(coeff: i64, indices: [u32; 2]) -> Term {
    Term {
        sums: Vec::new(),
        coeff: Coefficient::from_integer(coeff.into()),
        factors: vec![TensorRef {
            tensor: INPUT,
            indices: indices.into_iter().map(IndexId).collect(),
        }],
    }
}

fn product_term(coeff: i64, indices: [u32; 2]) -> Term {
    Term {
        sums: Vec::new(),
        coeff: Coefficient::from_integer(coeff.into()),
        factors: vec![
            TensorRef {
                tensor: INPUT,
                indices: indices.into_iter().map(IndexId).collect(),
            },
            TensorRef {
                tensor: INPUT,
                indices: indices.into_iter().map(IndexId).collect(),
            },
        ],
    }
}

#[test]
fn adds_canonical_intermediate_then_reuses_it() {
    let computation = Computation {
        ranges: BTreeSet::from([RANGE]),
        tensors: BTreeMap::from([(INPUT, tensor()), (INTERMEDIATE, tensor())]),
        definitions: vec![TensorDef {
            base: INTERMEDIATE,
            exts: vec![index(0), index(1)],
            rhs: vec![term(1, [0, 1])],
        }],
    };
    let mut state = State::new(computation, vec![INTERMEDIATE]).unwrap();

    let (coeff, factor) = state
        .add_intermediate(TensorDef {
            base: INTERMEDIATE,
            exts: vec![index(0), index(1)],
            rhs: vec![product_term(6, [0, 1])],
        })
        .unwrap();

    assert_eq!(coeff, Coefficient::from_integer(6.into()));
    assert_eq!(factor.tensor, OUTPUT);
    assert_eq!(factor.indices, vec![IndexId(0), IndexId(1)]);
    assert_eq!(state.computation.definitions.len(), 2);
    assert_eq!(
        state.computation.definitions[1].rhs,
        vec![product_term(1, [0, 1])]
    );

    let (coeff, factor) = state
        .add_intermediate(TensorDef {
            base: INTERMEDIATE,
            exts: vec![index(0), index(1)],
            rhs: vec![product_term(12, [1, 0])],
        })
        .unwrap();

    assert_eq!(coeff, Coefficient::from_integer(12.into()));
    assert_eq!(factor.tensor, OUTPUT);
    assert_eq!(factor.indices, vec![IndexId(1), IndexId(0)]);
    assert_eq!(state.computation.definitions.len(), 2);
}

#[test]
fn reuses_with_coefficient_and_external_permutation() {
    let computation = Computation {
        ranges: BTreeSet::from([RANGE]),
        tensors: BTreeMap::from([
            (INPUT, tensor()),
            (INTERMEDIATE, tensor()),
            (OUTPUT, tensor()),
        ]),
        definitions: vec![
            TensorDef {
                base: INTERMEDIATE,
                exts: vec![index(0), index(1)],
                rhs: vec![product_term(2, [1, 0])],
            },
            TensorDef {
                base: OUTPUT,
                exts: vec![index(0), index(1)],
                rhs: vec![term(1, [0, 1])],
            },
        ],
    };
    let mut state = State::new(computation, vec![OUTPUT]).unwrap();

    let (coeff, factor) = state
        .add_intermediate(TensorDef {
            base: OUTPUT,
            exts: vec![index(0), index(1)],
            rhs: vec![product_term(6, [0, 1])],
        })
        .unwrap();

    assert_eq!(coeff, Coefficient::from_integer(3.into()));
    assert_eq!(factor.tensor, INTERMEDIATE);
    assert_eq!(factor.indices, vec![IndexId(1), IndexId(0)]);
    assert_eq!(state.computation.definitions.len(), 2);
}

#[test]
fn rejects_a_zero_intermediate() {
    let computation = Computation {
        ranges: BTreeSet::from([RANGE]),
        tensors: BTreeMap::from([(INPUT, tensor()), (INTERMEDIATE, tensor())]),
        definitions: vec![TensorDef {
            base: INTERMEDIATE,
            exts: vec![index(0), index(1)],
            rhs: vec![term(1, [0, 1])],
        }],
    };
    let mut state = State::new(computation, vec![INTERMEDIATE]).unwrap();

    assert_eq!(
        state.add_intermediate(TensorDef {
            base: INTERMEDIATE,
            exts: vec![index(0), index(1)],
            rhs: vec![term(0, [0, 1])],
        }),
        Err(StateError::ZeroIntermediate)
    );
}

#[test]
fn replaces_terms_and_recanonicalizes_the_definition() {
    let computation = Computation {
        ranges: BTreeSet::from([RANGE]),
        tensors: BTreeMap::from([(INPUT, tensor()), (OUTPUT, tensor())]),
        definitions: vec![TensorDef {
            base: OUTPUT,
            exts: vec![index(0), index(1)],
            rhs: vec![term(1, [0, 1]), term(1, [1, 0])],
        }],
    };
    let mut state = State::new(computation, vec![OUTPUT]).unwrap();

    state
        .replace_terms(0, &[0, 1], vec![term(2, [1, 0]), term(3, [1, 0])])
        .unwrap();

    assert_eq!(state.computation.definitions[0].rhs, vec![term(5, [1, 0])]);
}

#[test]
fn rejects_invalid_replacement_positions_without_changing_state() {
    let computation = Computation {
        ranges: BTreeSet::from([RANGE]),
        tensors: BTreeMap::from([(INPUT, tensor()), (OUTPUT, tensor())]),
        definitions: vec![TensorDef {
            base: OUTPUT,
            exts: vec![index(0), index(1)],
            rhs: vec![term(1, [0, 1])],
        }],
    };
    let mut state = State::new(computation, vec![OUTPUT]).unwrap();
    let original = state.clone();

    assert_eq!(
        state.replace_terms(1, &[], Vec::new()),
        Err(StateError::DefinitionOutOfBounds { position: 1 })
    );
    assert_eq!(state, original);

    assert_eq!(
        state.replace_terms(0, &[1], Vec::new()),
        Err(StateError::TermOutOfBounds {
            definition: 0,
            term: 1,
        })
    );
    assert_eq!(state, original);
}
