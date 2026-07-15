use gristmill_rewrites::canon::{CanonError, canon_expr, canon_term};
use gristmill_rewrites::{
    Coefficient, Computation, Index, IndexId, RangeId, SymmetryAction, SymmetryGenerator, TensorId,
    TensorInfo, TensorRef, Term,
};
use std::collections::{BTreeMap, BTreeSet};

const R0: RangeId = RangeId(0);
const R1: RangeId = RangeId(1);
const A: TensorId = TensorId(0);
const B: TensorId = TensorId(1);

fn index(id: u32, range: RangeId) -> Index {
    Index {
        id: IndexId(id),
        range,
    }
}

fn tensor(rank: usize) -> TensorInfo {
    TensorInfo {
        rank,
        symmetry: Vec::new(),
    }
}

fn tensor_with_symmetry(rank: usize, perm: &[usize], action: SymmetryAction) -> TensorInfo {
    TensorInfo {
        rank,
        symmetry: vec![SymmetryGenerator {
            perm: perm.to_vec(),
            action,
        }],
    }
}

fn make_computation(tensors: impl IntoIterator<Item = (TensorId, TensorInfo)>) -> Computation {
    Computation {
        ranges: BTreeSet::from([R0, R1]),
        tensors: BTreeMap::from_iter(tensors),
        definitions: Vec::new(),
    }
}

fn rational(numer: i64, denom: i64) -> Coefficient {
    Coefficient::new(numer.into(), denom.into())
}

fn factor(tensor: TensorId, indices: &[u32]) -> TensorRef {
    TensorRef {
        tensor,
        indices: indices.iter().copied().map(IndexId).collect(),
    }
}

#[test]
fn canonicalizes_factor_order_and_summed_index_ids() {
    let computation = make_computation([(A, tensor(2)), (B, tensor(2))]);
    let first = Term {
        sums: vec![index(7, R0)],
        coeff: rational(2, 3),
        factors: vec![factor(B, &[7, 1]), factor(A, &[0, 7])],
    };
    let renamed_and_reordered = Term {
        sums: vec![index(12, R0)],
        coeff: rational(2, 3),
        factors: vec![factor(A, &[0, 12]), factor(B, &[12, 1])],
    };

    let first = canon_term(&computation, &[index(0, R0), index(1, R0)], &first).unwrap();
    let renamed = canon_term(
        &computation,
        &[index(0, R0), index(1, R0)],
        &renamed_and_reordered,
    )
    .unwrap();

    let expected = Some(Term {
        sums: vec![index(2, R0)],
        coeff: rational(2, 3),
        factors: vec![factor(A, &[0, 2]), factor(B, &[2, 1])],
    });
    assert_eq!(first, expected);
    assert_eq!(renamed, expected);
}

#[test]
fn canonicalizes_a_linear_expression() {
    let computation = make_computation([(A, tensor(2)), (B, tensor(2))]);
    let terms = vec![
        Term {
            sums: Vec::new(),
            coeff: rational(4, 1),
            factors: vec![factor(B, &[0, 1])],
        },
        Term {
            sums: Vec::new(),
            coeff: rational(2, 1),
            factors: vec![factor(A, &[0, 1])],
        },
        Term {
            sums: Vec::new(),
            coeff: rational(3, 1),
            factors: vec![factor(A, &[0, 1])],
        },
        Term {
            sums: Vec::new(),
            coeff: rational(1, 1),
            factors: vec![factor(B, &[1, 0])],
        },
        Term {
            sums: Vec::new(),
            coeff: rational(-1, 1),
            factors: vec![factor(B, &[1, 0])],
        },
    ];

    assert_eq!(
        canon_expr(&computation, &[index(0, R0), index(1, R0)], &terms),
        Ok(vec![
            Term {
                sums: Vec::new(),
                coeff: rational(5, 1),
                factors: vec![factor(A, &[0, 1])],
            },
            Term {
                sums: Vec::new(),
                coeff: rational(4, 1),
                factors: vec![factor(B, &[0, 1])],
            },
        ])
    );
}

#[test]
fn canonicalizes_symmetric_and_antisymmetric_factors() {
    let symmetric = make_computation([(
        A,
        tensor_with_symmetry(2, &[1, 0], SymmetryAction::Identity),
    )]);
    let antisymmetric =
        make_computation([(A, tensor_with_symmetry(2, &[1, 0], SymmetryAction::Negate))]);
    let boundary = [index(0, R0), index(1, R0)];
    let direct = Term {
        sums: Vec::new(),
        coeff: rational(3, 2),
        factors: vec![factor(A, &[0, 1])],
    };
    let symmetric_swap = Term {
        sums: Vec::new(),
        coeff: rational(3, 2),
        factors: vec![factor(A, &[1, 0])],
    };
    let antisymmetric_swap = Term {
        sums: Vec::new(),
        coeff: rational(-3, 2),
        factors: vec![factor(A, &[1, 0])],
    };

    assert_eq!(
        canon_term(&symmetric, &boundary, &direct),
        canon_term(&symmetric, &boundary, &symmetric_swap)
    );
    assert_eq!(
        canon_term(&antisymmetric, &boundary, &direct),
        canon_term(&antisymmetric, &boundary, &antisymmetric_swap)
    );
}

#[test]
fn closes_the_declared_symmetry_generators() {
    let computation = make_computation([(
        A,
        tensor_with_symmetry(3, &[1, 2, 0], SymmetryAction::Identity),
    )]);
    let boundary = [index(0, R0), index(1, R0), index(2, R0)];
    let original = Term {
        sums: Vec::new(),
        coeff: rational(1, 1),
        factors: vec![factor(A, &[0, 1, 2])],
    };
    let twice_permuted = Term {
        sums: Vec::new(),
        coeff: rational(1, 1),
        factors: vec![factor(A, &[2, 0, 1])],
    };

    assert_eq!(
        canon_term(&computation, &boundary, &original),
        canon_term(&computation, &boundary, &twice_permuted)
    );
}

#[test]
fn recognizes_zero_from_coefficients_and_antisymmetry() {
    let computation = make_computation([
        (A, tensor_with_symmetry(2, &[1, 0], SymmetryAction::Negate)),
        (
            B,
            tensor_with_symmetry(2, &[1, 0], SymmetryAction::Identity),
        ),
    ]);
    let diagonal = Term {
        sums: Vec::new(),
        coeff: rational(5, 1),
        factors: vec![factor(A, &[0, 0])],
    };
    let zero = Term {
        sums: Vec::new(),
        coeff: rational(0, 1),
        factors: vec![factor(A, &[0, 1])],
    };
    let antisymmetric_symmetric_contraction = Term {
        sums: vec![index(0, R0), index(1, R0)],
        coeff: rational(1, 1),
        factors: vec![factor(A, &[0, 1]), factor(B, &[0, 1])],
    };

    assert_eq!(
        canon_term(&computation, &[index(0, R0)], &diagonal),
        Ok(None)
    );
    assert_eq!(
        canon_term(&computation, &[index(0, R0), index(1, R0)], &zero),
        Ok(None)
    );
    assert_eq!(
        canon_term(&computation, &[], &antisymmetric_symmetric_contraction),
        Ok(None)
    );
}

#[test]
fn distinguishes_different_contractions_and_ranges() {
    let computation = make_computation([(A, tensor(2)), (B, tensor(2))]);
    let boundary = [index(0, R0), index(1, R0)];
    let matrix_product = Term {
        sums: vec![index(2, R0)],
        coeff: rational(1, 1),
        factors: vec![factor(A, &[0, 2]), factor(B, &[2, 1])],
    };
    let transposed_left = Term {
        sums: vec![index(2, R0)],
        coeff: rational(1, 1),
        factors: vec![factor(A, &[2, 0]), factor(B, &[2, 1])],
    };
    let different_sum_range = Term {
        sums: vec![index(2, R1)],
        coeff: rational(1, 1),
        factors: vec![factor(A, &[0, 2]), factor(B, &[2, 1])],
    };

    let matrix_product = canon_term(&computation, &boundary, &matrix_product).unwrap();
    let transposed_left = canon_term(&computation, &boundary, &transposed_left).unwrap();
    let different_sum_range = canon_term(&computation, &boundary, &different_sum_range).unwrap();

    assert_ne!(matrix_product, transposed_left);
    assert_ne!(matrix_product, different_sum_range);
}

#[test]
fn canonicalizes_multiple_same_range_dummies_with_repeated_factors() {
    let computation = make_computation([(A, tensor(2))]);
    let fixed = [index(0, R0), index(1, R0)];
    let first = Term {
        sums: vec![index(7, R0), index(8, R0)],
        coeff: rational(1, 1),
        factors: vec![factor(A, &[0, 7]), factor(A, &[7, 8]), factor(A, &[8, 1])],
    };
    let renamed_and_reordered = Term {
        sums: vec![index(10, R0), index(20, R0)],
        coeff: rational(1, 1),
        factors: vec![
            factor(A, &[10, 1]),
            factor(A, &[20, 10]),
            factor(A, &[0, 20]),
        ],
    };

    assert_eq!(
        canon_term(&computation, &fixed, &first),
        canon_term(&computation, &fixed, &renamed_and_reordered)
    );
}

#[test]
fn preserves_fixed_index_ids() {
    let computation = make_computation([(A, tensor(2))]);
    let term = Term {
        sums: Vec::new(),
        coeff: rational(1, 1),
        factors: vec![factor(A, &[10, 20])],
    };

    assert_eq!(
        canon_term(&computation, &[index(10, R0), index(20, R0)], &term),
        Ok(Some(term))
    );
}

#[test]
fn validates_the_local_term_context() {
    let computation = make_computation([(A, tensor(2))]);
    let unknown_index = Term {
        sums: Vec::new(),
        coeff: rational(1, 1),
        factors: vec![factor(A, &[0, 9])],
    };
    assert_eq!(
        canon_term(&computation, &[index(0, R0)], &unknown_index),
        Err(CanonError::UnknownIndex { index: IndexId(9) })
    );

    let duplicate_sum = Term {
        sums: vec![index(2, R0), index(2, R0)],
        coeff: rational(1, 1),
        factors: vec![factor(A, &[0, 2])],
    };
    assert_eq!(
        canon_term(&computation, &[index(0, R0)], &duplicate_sum),
        Err(CanonError::DuplicateSummedIndex { index: IndexId(2) })
    );

    let invalid_symmetry = make_computation([(
        A,
        tensor_with_symmetry(2, &[0, 0], SymmetryAction::Identity),
    )]);
    let valid_shape = Term {
        sums: Vec::new(),
        coeff: rational(1, 1),
        factors: vec![factor(A, &[0, 1])],
    };
    assert_eq!(
        canon_term(
            &invalid_symmetry,
            &[index(0, R0), index(1, R0)],
            &valid_shape
        ),
        Err(CanonError::InvalidSymmetryPermutation {
            tensor: A,
            perm: vec![0, 0],
        })
    );
}
