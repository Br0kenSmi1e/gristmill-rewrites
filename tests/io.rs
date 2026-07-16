use gristmill_rewrites::io::{IoJsonError, from_json, read_json, to_json, write_json};
use gristmill_rewrites::{
    Coefficient, Computation, Index, IndexId, RangeId, State, StateError, SymmetryAction,
    SymmetryGenerator, TensorDef, TensorId, TensorInfo, TensorRef, Term,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const RANGE: RangeId = RangeId(0);
const INPUT: TensorId = TensorId(1);
const OUTPUT: TensorId = TensorId(4);

#[test]
fn pretty_json_round_trips_exact_coefficients() {
    let state = sample_state();

    let encoded = to_json(&state).unwrap();
    let decoded = from_json(&encoded).unwrap();

    assert_eq!(decoded, state);
    assert!(encoded.contains('\n'));
    assert!(encoded.contains("\"coeff\": \"-3/2\""));
    assert!(encoded.contains("\"protected_outputs\""));
}

#[test]
fn json_file_round_trip_preserves_the_state() {
    let path = unique_temp_path("round-trip");
    let state = sample_state();

    write_json(&path, &state).unwrap();
    let decoded = read_json(&path).unwrap();

    assert_eq!(decoded, state);
    fs::remove_file(path).ok();
}

#[test]
fn distinguishes_json_and_filesystem_errors() {
    let malformed = unique_temp_path("malformed");
    fs::write(&malformed, "{ not json").unwrap();
    assert!(matches!(read_json(&malformed), Err(IoJsonError::Json(_))));
    fs::remove_file(malformed).ok();

    let missing = unique_temp_path("missing");
    fs::remove_file(&missing).ok();
    assert!(matches!(read_json(missing), Err(IoJsonError::Io(_))));
}

#[test]
fn rejects_an_invalid_exact_coefficient() {
    let encoded = to_json(&sample_state()).unwrap();
    let invalid = encoded.replace("-3/2", "3/0");

    assert!(matches!(from_json(&invalid), Err(IoJsonError::Json(_))));
}

#[test]
fn rejects_json_that_does_not_form_a_valid_state() {
    let encoded = to_json(&sample_state()).unwrap();
    let invalid = encoded.replace(
        "\"protected_outputs\": [\n    4\n  ]",
        "\"protected_outputs\": [\n    1\n  ]",
    );

    match from_json(&invalid) {
        Err(IoJsonError::State(error)) => assert_eq!(
            error,
            StateError::MissingProtectedOutputDefinition { tensor: INPUT }
        ),
        result => panic!("expected invalid-state error, got {result:?}"),
    }
}

fn sample_state() -> State {
    let computation = Computation {
        ranges: BTreeSet::from([RANGE]),
        tensors: BTreeMap::from([
            (
                INPUT,
                TensorInfo {
                    rank: 2,
                    symmetry: vec![SymmetryGenerator {
                        perm: vec![1, 0],
                        action: SymmetryAction::Negate,
                    }],
                },
            ),
            (
                OUTPUT,
                TensorInfo {
                    rank: 1,
                    symmetry: Vec::new(),
                },
            ),
        ]),
        definitions: vec![TensorDef {
            base: OUTPUT,
            exts: vec![index(7)],
            rhs: vec![Term {
                sums: vec![index(9)],
                coeff: Coefficient::new((-3).into(), 2.into()),
                factors: vec![TensorRef {
                    tensor: INPUT,
                    indices: vec![IndexId(7), IndexId(9)],
                }],
            }],
        }],
    };
    State::new(computation, vec![OUTPUT]).unwrap()
}

fn index(id: u32) -> Index {
    Index {
        id: IndexId(id),
        range: RANGE,
    }
}

fn unique_temp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "gristmill-rewrites-{name}-{}-{nanos}.json",
        std::process::id()
    ))
}
