use gristmill_rewrites::{
    Coefficient, Computation, Index, IndexId, RangeId, State, SymmetryAction, SymmetryGenerator,
    TensorDef, TensorId, TensorInfo, TensorRef, Term, VerifyError, equivalent_computations,
    equivalent_states,
};
use std::collections::{BTreeMap, BTreeSet};

const RANGE: RangeId = RangeId(0);
const A: TensorId = TensorId(0);
const B: TensorId = TensorId(1);
const C: TensorId = TensorId(2);
const TEMP: TensorId = TensorId(3);
const OUTPUT: TensorId = TensorId(4);
const OTHER_OUTPUT: TensorId = TensorId(5);

#[test]
fn accepts_external_index_alpha_renaming() {
    let lhs = computation(
        &[(A, 1), (OUTPUT, 1)],
        vec![definition(
            OUTPUT,
            &[index(7)],
            vec![term(1, &[], vec![factor(A, &[7])])],
        )],
    );
    let rhs = computation(
        &[(A, 1), (OUTPUT, 1)],
        vec![definition(
            OUTPUT,
            &[index(11)],
            vec![term(1, &[], vec![factor(A, &[11])])],
        )],
    );

    assert!(equivalent_computations(&lhs, &rhs, &[OUTPUT]).unwrap());
}

#[test]
fn accepts_an_inlined_intermediate() {
    let lhs = computation(
        &[(A, 1), (B, 1), (C, 1), (TEMP, 1), (OUTPUT, 1)],
        vec![
            definition(
                TEMP,
                &[index(0)],
                vec![
                    term(1, &[], vec![factor(B, &[0])]),
                    term(1, &[], vec![factor(C, &[0])]),
                ],
            ),
            definition(
                OUTPUT,
                &[index(0)],
                vec![term(1, &[], vec![factor(A, &[0]), factor(TEMP, &[0])])],
            ),
        ],
    );
    let rhs = computation(
        &[(A, 1), (B, 1), (C, 1), (TEMP, 1), (OUTPUT, 1)],
        vec![definition(
            OUTPUT,
            &[index(0)],
            vec![
                term(1, &[], vec![factor(A, &[0]), factor(B, &[0])]),
                term(1, &[], vec![factor(A, &[0]), factor(C, &[0])]),
            ],
        )],
    );

    assert!(equivalent_computations(&lhs, &rhs, &[OUTPUT]).unwrap());
}

#[test]
fn gives_repeated_intermediate_uses_distinct_dummies() {
    let lhs = computation(
        &[(A, 1), (TEMP, 0), (OUTPUT, 0)],
        vec![
            definition(TEMP, &[], vec![term(1, &[index(0)], vec![factor(A, &[0])])]),
            definition(
                OUTPUT,
                &[],
                vec![term(1, &[], vec![factor(TEMP, &[]), factor(TEMP, &[])])],
            ),
        ],
    );
    let rhs = computation(
        &[(A, 1), (TEMP, 0), (OUTPUT, 0)],
        vec![definition(
            OUTPUT,
            &[],
            vec![term(
                1,
                &[index(0), index(1)],
                vec![factor(A, &[0]), factor(A, &[1])],
            )],
        )],
    );

    assert!(equivalent_computations(&lhs, &rhs, &[OUTPUT]).unwrap());
}

#[test]
fn rejects_different_output_expressions() {
    let lhs = computation(
        &[(A, 0), (OUTPUT, 0)],
        vec![definition(
            OUTPUT,
            &[],
            vec![term(1, &[], vec![factor(A, &[])])],
        )],
    );
    let rhs = computation(
        &[(A, 0), (OUTPUT, 0)],
        vec![definition(
            OUTPUT,
            &[],
            vec![term(2, &[], vec![factor(A, &[])])],
        )],
    );

    assert!(!equivalent_computations(&lhs, &rhs, &[OUTPUT]).unwrap());
}

#[test]
fn reports_different_protected_outputs() {
    let computation = computation(
        &[(A, 0), (OUTPUT, 0), (OTHER_OUTPUT, 0)],
        vec![
            definition(OUTPUT, &[], vec![term(1, &[], vec![factor(A, &[])])]),
            definition(OTHER_OUTPUT, &[], vec![term(1, &[], vec![factor(A, &[])])]),
        ],
    );
    let lhs = State::new(computation.clone(), vec![OUTPUT]).unwrap();
    let rhs = State::new(computation, vec![OTHER_OUTPUT]).unwrap();

    assert_eq!(
        equivalent_states(&lhs, &rhs),
        Err(VerifyError::ProtectedOutputsMismatch {
            lhs: vec![OUTPUT],
            rhs: vec![OTHER_OUTPUT],
        })
    );
}

#[test]
fn rejects_different_leaf_tensor_metadata() {
    let lhs = computation(
        &[(A, 1), (OUTPUT, 1)],
        vec![definition(
            OUTPUT,
            &[index(0)],
            vec![term(1, &[], vec![factor(A, &[0])])],
        )],
    );
    let mut rhs = lhs.clone();
    rhs.tensors.get_mut(&A).unwrap().symmetry = vec![SymmetryGenerator {
        perm: vec![0],
        action: SymmetryAction::Identity,
    }];

    assert_eq!(
        equivalent_computations(&lhs, &rhs, &[OUTPUT]),
        Err(VerifyError::TensorMetadataMismatch { tensor: A })
    );
}

#[test]
fn ignores_metadata_of_eliminated_intermediates() {
    let lhs = computation(
        &[(A, 0), (TEMP, 0), (OUTPUT, 0)],
        vec![
            definition(TEMP, &[], Vec::new()),
            definition(OUTPUT, &[], vec![term(1, &[], vec![factor(A, &[])])]),
        ],
    );
    let rhs = computation(
        &[(A, 0), (TEMP, 1), (OUTPUT, 0)],
        vec![definition(
            OUTPUT,
            &[],
            vec![term(1, &[], vec![factor(A, &[])])],
        )],
    );

    assert!(equivalent_computations(&lhs, &rhs, &[OUTPUT]).unwrap());
}

fn computation(tensors: &[(TensorId, usize)], definitions: Vec<TensorDef>) -> Computation {
    Computation {
        ranges: BTreeSet::from([RANGE]),
        tensors: tensors
            .iter()
            .map(|&(tensor, rank)| {
                (
                    tensor,
                    TensorInfo {
                        rank,
                        symmetry: Vec::new(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>(),
        definitions,
    }
}

fn definition(base: TensorId, exts: &[Index], rhs: Vec<Term>) -> TensorDef {
    TensorDef {
        base,
        exts: exts.to_vec(),
        rhs,
    }
}

fn term(coeff: i64, sums: &[Index], factors: Vec<TensorRef>) -> Term {
    Term {
        sums: sums.to_vec(),
        coeff: Coefficient::from_integer(coeff.into()),
        factors,
    }
}

fn factor(tensor: TensorId, indices: &[u32]) -> TensorRef {
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
