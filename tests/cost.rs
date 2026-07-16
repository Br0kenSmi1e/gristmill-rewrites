use gristmill_rewrites::{
    Coefficient, Computation, CostError, Index, IndexId, RangeId, TensorDef, TensorId, TensorInfo,
    TensorRef, Term, log_flops,
};
use std::collections::{BTreeMap, BTreeSet};

const RANGE: RangeId = RangeId(0);
const UNIT_RANGE: RangeId = RangeId(1);
const A: TensorId = TensorId(0);
const B: TensorId = TensorId(1);
const OUTPUT: TensorId = TensorId(2);

#[test]
fn follows_python_gristmill_for_a_contraction() {
    let computation = computation(
        BTreeSet::from([RANGE]),
        vec![TensorDef {
            base: OUTPUT,
            exts: vec![index(0, RANGE), index(1, RANGE)],
            rhs: vec![Term {
                sums: vec![index(2, RANGE)],
                coeff: integer(1),
                factors: vec![tensor(A, &[0, 2]), tensor(B, &[2, 1])],
            }],
        }],
    );
    let log_sizes = BTreeMap::from([(RANGE, 10.0_f64.ln())]);

    assert_close(
        log_flops(&computation, &log_sizes).unwrap(),
        2000.0_f64.ln(),
    );
}

#[test]
fn counts_addition_between_definition_terms() {
    let computation = computation(
        BTreeSet::from([RANGE]),
        vec![TensorDef {
            base: OUTPUT,
            exts: vec![index(0, RANGE), index(1, RANGE)],
            rhs: vec![single(1, A, &[0, 1]), single(1, B, &[0, 1])],
        }],
    );
    let log_sizes = BTreeMap::from([(RANGE, 10.0_f64.ln())]);

    assert_close(log_flops(&computation, &log_sizes).unwrap(), 100.0_f64.ln());
}

#[test]
fn treats_size_one_sums_as_having_no_addition() {
    let computation = computation(
        BTreeSet::from([RANGE, UNIT_RANGE]),
        vec![TensorDef {
            base: OUTPUT,
            exts: vec![index(0, RANGE), index(1, RANGE)],
            rhs: vec![Term {
                sums: vec![index(2, UNIT_RANGE)],
                coeff: integer(1),
                factors: vec![tensor(A, &[0, 2]), tensor(B, &[2, 1])],
            }],
        }],
    );
    let log_sizes = BTreeMap::from([(RANGE, 10.0_f64.ln()), (UNIT_RANGE, 0.0)]);

    assert_close(log_flops(&computation, &log_sizes).unwrap(), 100.0_f64.ln());
}

#[test]
fn ignores_numeric_coefficients_and_copies() {
    let computation = computation(
        BTreeSet::from([RANGE]),
        vec![TensorDef {
            base: OUTPUT,
            exts: vec![index(0, RANGE), index(1, RANGE)],
            rhs: vec![single(2, A, &[0, 1])],
        }],
    );
    let log_sizes = BTreeMap::from([(RANGE, 10.0_f64.ln())]);

    assert_eq!(
        log_flops(&computation, &log_sizes),
        Err(CostError::ZeroTotalFlops)
    );
}

#[test]
fn adds_costs_across_definitions_in_log_space() {
    let mut computation = computation(
        BTreeSet::from([RANGE]),
        vec![TensorDef {
            base: OUTPUT,
            exts: vec![index(0, RANGE), index(1, RANGE)],
            rhs: vec![Term {
                sums: vec![index(2, RANGE)],
                coeff: integer(1),
                factors: vec![tensor(A, &[0, 2]), tensor(B, &[2, 1])],
            }],
        }],
    );
    let second_output = TensorId(3);
    computation.tensors.insert(second_output, tensor_info(2));
    let mut second = computation.definitions[0].clone();
    second.base = second_output;
    computation.definitions.push(second);
    let log_sizes = BTreeMap::from([(RANGE, 10.0_f64.ln())]);

    assert_close(
        log_flops(&computation, &log_sizes).unwrap(),
        4000.0_f64.ln(),
    );
}

#[test]
fn reports_missing_and_invalid_log_sizes() {
    let computation = computation(
        BTreeSet::from([RANGE]),
        vec![TensorDef {
            base: OUTPUT,
            exts: vec![index(0, RANGE), index(1, RANGE)],
            rhs: vec![single(1, A, &[0, 1])],
        }],
    );

    assert_eq!(
        log_flops(&computation, &BTreeMap::new()),
        Err(CostError::MissingLogSize { range: RANGE })
    );
    assert_eq!(
        log_flops(&computation, &BTreeMap::from([(RANGE, f64::NAN)])),
        Err(CostError::InvalidLogSize { range: RANGE })
    );
    assert_eq!(
        log_flops(&computation, &BTreeMap::from([(RANGE, -1.0)])),
        Err(CostError::InvalidLogSize { range: RANGE })
    );
}

fn computation(ranges: BTreeSet<RangeId>, definitions: Vec<TensorDef>) -> Computation {
    Computation {
        ranges,
        tensors: BTreeMap::from([
            (A, tensor_info(2)),
            (B, tensor_info(2)),
            (OUTPUT, tensor_info(2)),
        ]),
        definitions,
    }
}

fn tensor_info(rank: usize) -> TensorInfo {
    TensorInfo {
        rank,
        symmetry: Vec::new(),
    }
}

fn index(id: u32, range: RangeId) -> Index {
    Index {
        id: IndexId(id),
        range,
    }
}

fn single(coeff: i64, tensor_id: TensorId, indices: &[u32]) -> Term {
    Term {
        sums: Vec::new(),
        coeff: integer(coeff),
        factors: vec![tensor(tensor_id, indices)],
    }
}

fn tensor(tensor: TensorId, indices: &[u32]) -> TensorRef {
    TensorRef {
        tensor,
        indices: indices.iter().copied().map(IndexId).collect(),
    }
}

fn integer(value: i64) -> Coefficient {
    Coefficient::from_integer(value.into())
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "actual {actual}, expected {expected}"
    );
}
