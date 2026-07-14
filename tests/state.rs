use gristmill_rewrites::{
    Coefficient, Computation, Index, IndexId, RangeId, State, StateError, SymmetryAction,
    SymmetryGenerator, TensorDef, TensorId, TensorInfo, TensorRef, Term,
};
use std::collections::{BTreeMap, BTreeSet};

const RANGE: RangeId = RangeId(0);
const INPUT: TensorId = TensorId(0);
const OUTPUT: TensorId = TensorId(1);

fn index(id: u32) -> Index {
    Index {
        id: IndexId(id),
        range: RANGE,
    }
}

fn tensor(rank: usize) -> TensorInfo {
    TensorInfo {
        rank,
        symmetry: Vec::new(),
    }
}

fn one() -> Coefficient {
    Coefficient::from_integer(1.into())
}

fn valid_computation() -> Computation {
    Computation {
        ranges: BTreeSet::from([RANGE]),
        tensors: BTreeMap::from([(INPUT, tensor(2)), (OUTPUT, tensor(2))]),
        definitions: vec![TensorDef {
            base: OUTPUT,
            exts: vec![index(0), index(1)],
            rhs: vec![Term {
                sums: Vec::new(),
                coeff: Coefficient::new(1.into(), 3.into()),
                factors: vec![TensorRef {
                    tensor: INPUT,
                    indices: vec![IndexId(0), IndexId(1)],
                }],
            }],
        }],
    }
}

#[test]
fn accepts_and_preserves_a_valid_computation() {
    let state = State::new(valid_computation(), vec![OUTPUT]).unwrap();

    assert_eq!(state.protected_outputs(), &[OUTPUT]);
    assert_eq!(
        state.computation().definitions[0].rhs[0].coeff,
        Coefficient::new(1.into(), 3.into())
    );

    let (computation, outputs) = state.into_parts();
    assert_eq!(computation, valid_computation());
    assert_eq!(outputs, vec![OUTPUT]);
}

#[test]
fn summed_index_ids_are_local_to_each_term() {
    let computation = Computation {
        ranges: BTreeSet::from([RANGE]),
        tensors: BTreeMap::from([(INPUT, tensor(1)), (OUTPUT, tensor(0))]),
        definitions: vec![TensorDef {
            base: OUTPUT,
            exts: Vec::new(),
            rhs: vec![
                Term {
                    sums: vec![index(0)],
                    coeff: one(),
                    factors: vec![TensorRef {
                        tensor: INPUT,
                        indices: vec![IndexId(0)],
                    }],
                },
                Term {
                    sums: vec![index(0)],
                    coeff: one(),
                    factors: vec![TensorRef {
                        tensor: INPUT,
                        indices: vec![IndexId(0)],
                    }],
                },
            ],
        }],
    };

    State::new(computation, vec![OUTPUT]).unwrap();
}

#[test]
fn definitions_need_not_be_topologically_ordered() {
    let intermediate = TensorId(2);
    let computation = Computation {
        ranges: BTreeSet::from([RANGE]),
        tensors: BTreeMap::from([
            (INPUT, tensor(1)),
            (OUTPUT, tensor(1)),
            (intermediate, tensor(1)),
        ]),
        definitions: vec![
            TensorDef {
                base: OUTPUT,
                exts: vec![index(0)],
                rhs: vec![Term {
                    sums: Vec::new(),
                    coeff: one(),
                    factors: vec![TensorRef {
                        tensor: intermediate,
                        indices: vec![IndexId(0)],
                    }],
                }],
            },
            TensorDef {
                base: intermediate,
                exts: vec![index(0)],
                rhs: vec![Term {
                    sums: Vec::new(),
                    coeff: one(),
                    factors: vec![TensorRef {
                        tensor: INPUT,
                        indices: vec![IndexId(0)],
                    }],
                }],
            },
        ],
    };

    State::new(computation, vec![OUTPUT]).unwrap();
}

#[test]
fn validates_tensor_and_range_references() {
    let mut computation = valid_computation();
    computation.definitions[0].exts[0].range = RangeId(99);
    assert_eq!(
        State::new(computation, vec![OUTPUT]),
        Err(StateError::UnknownRange { range: RangeId(99) })
    );

    let mut computation = valid_computation();
    computation.definitions[0].rhs[0].factors[0].tensor = TensorId(99);
    assert_eq!(
        State::new(computation, vec![OUTPUT]),
        Err(StateError::UnknownTensor {
            tensor: TensorId(99)
        })
    );
}

#[test]
fn validates_definition_shape() {
    let mut computation = valid_computation();
    computation
        .definitions
        .push(computation.definitions[0].clone());
    assert_eq!(
        State::new(computation, vec![OUTPUT]),
        Err(StateError::DuplicateDefinition { tensor: OUTPUT })
    );

    let mut computation = valid_computation();
    computation.definitions[0].exts.pop();
    assert_eq!(
        State::new(computation, vec![OUTPUT]),
        Err(StateError::TensorArityMismatch {
            tensor: OUTPUT,
            expected: 2,
            got: 1,
        })
    );

    let mut computation = valid_computation();
    computation.definitions[0].exts[1].id = IndexId(0);
    assert_eq!(
        State::new(computation, vec![OUTPUT]),
        Err(StateError::DuplicateExternalIndex {
            definition: OUTPUT,
            index: IndexId(0),
        })
    );
}

#[test]
fn validates_term_index_scopes_and_factor_arity() {
    let mut computation = valid_computation();
    computation.definitions[0].rhs[0].sums = vec![index(2), index(2)];
    assert_eq!(
        State::new(computation, vec![OUTPUT]),
        Err(StateError::DuplicateSumIndex {
            definition: OUTPUT,
            term: 0,
            index: IndexId(2),
        })
    );

    let mut computation = valid_computation();
    computation.definitions[0].rhs[0].sums = vec![index(0)];
    assert_eq!(
        State::new(computation, vec![OUTPUT]),
        Err(StateError::ExternalAndSumIndexOverlap {
            definition: OUTPUT,
            term: 0,
            index: IndexId(0),
        })
    );

    let mut computation = valid_computation();
    computation.definitions[0].rhs[0].factors[0].indices[1] = IndexId(9);
    assert_eq!(
        State::new(computation, vec![OUTPUT]),
        Err(StateError::UnknownFactorIndex {
            definition: OUTPUT,
            term: 0,
            index: IndexId(9),
        })
    );

    let mut computation = valid_computation();
    computation.definitions[0].rhs[0].sums = vec![index(2)];
    assert_eq!(
        State::new(computation, vec![OUTPUT]),
        Err(StateError::UnusedSumIndex {
            definition: OUTPUT,
            term: 0,
            index: IndexId(2),
        })
    );

    let mut computation = valid_computation();
    computation.definitions[0].rhs[0].factors[0].indices.pop();
    assert_eq!(
        State::new(computation, vec![OUTPUT]),
        Err(StateError::TensorArityMismatch {
            tensor: INPUT,
            expected: 2,
            got: 1,
        })
    );
}

#[test]
fn validates_symmetry_generators() {
    let mut computation = valid_computation();
    computation.tensors.get_mut(&INPUT).unwrap().symmetry = vec![SymmetryGenerator {
        perm: vec![0],
        action: SymmetryAction::Identity,
    }];
    assert_eq!(
        State::new(computation, vec![OUTPUT]),
        Err(StateError::InvalidSymmetryArity {
            tensor: INPUT,
            expected: 2,
            got: 1,
        })
    );

    let mut computation = valid_computation();
    computation.tensors.get_mut(&INPUT).unwrap().symmetry = vec![SymmetryGenerator {
        perm: vec![0, 0],
        action: SymmetryAction::Negate,
    }];
    assert_eq!(
        State::new(computation, vec![OUTPUT]),
        Err(StateError::InvalidSymmetryPermutation {
            tensor: INPUT,
            perm: vec![0, 0],
        })
    );
}

#[test]
fn validates_protected_outputs() {
    let computation = valid_computation();
    assert_eq!(
        State::new(computation.clone(), vec![OUTPUT, OUTPUT]),
        Err(StateError::DuplicateProtectedOutput { tensor: OUTPUT })
    );
    assert_eq!(
        State::new(computation.clone(), vec![INPUT]),
        Err(StateError::MissingProtectedOutputDefinition { tensor: INPUT })
    );
    assert_eq!(
        State::new(computation, vec![TensorId(99)]),
        Err(StateError::UnknownTensor {
            tensor: TensorId(99)
        })
    );
}

#[test]
fn rejects_dependency_cycles() {
    let first = TensorId(1);
    let second = TensorId(2);
    let computation = Computation {
        ranges: BTreeSet::from([RANGE]),
        tensors: BTreeMap::from([(first, tensor(1)), (second, tensor(1))]),
        definitions: vec![
            TensorDef {
                base: first,
                exts: vec![index(0)],
                rhs: vec![Term {
                    sums: Vec::new(),
                    coeff: one(),
                    factors: vec![TensorRef {
                        tensor: second,
                        indices: vec![IndexId(0)],
                    }],
                }],
            },
            TensorDef {
                base: second,
                exts: vec![index(0)],
                rhs: vec![Term {
                    sums: Vec::new(),
                    coeff: one(),
                    factors: vec![TensorRef {
                        tensor: first,
                        indices: vec![IndexId(0)],
                    }],
                }],
            },
        ],
    };

    assert!(matches!(
        State::new(computation, vec![first]),
        Err(StateError::DependencyCycle { tensor }) if tensor == first || tensor == second
    ));
}
