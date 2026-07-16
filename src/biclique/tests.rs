//! Unit tests for biclique normalization, graph construction, and search.

use super::{
    graph::Edge,
    normalize::{canon_split, enumerate_splits},
    *,
};
use crate::{
    parenthesize::bipartition_term,
    repr::{Coefficient, Computation, IndexId, RangeId, TensorId, TensorInfo, TensorRef},
};

const RANGE: RangeId = RangeId(0);
const A: TensorId = TensorId(0);
const B: TensorId = TensorId(1);
const C: TensorId = TensorId(2);
const D: TensorId = TensorId(3);

#[test]
fn enumerates_each_unordered_factor_partition_once() {
    let term = Term {
        sums: Vec::new(),
        coeff: one(),
        factors: vec![scalar(A), scalar(B), scalar(C)],
    };

    let splits = enumerate_splits(&[], &term);

    assert_eq!(splits.len(), 3);
    assert_eq!(tensors(&splits[0].left), vec![A]);
    assert_eq!(tensors(&splits[0].right), vec![B, C]);
    assert_eq!(tensors(&splits[1].left), vec![A, B]);
    assert_eq!(tensors(&splits[1].right), vec![C]);
    assert_eq!(tensors(&splits[2].left), vec![A, C]);
    assert_eq!(tensors(&splits[2].right), vec![B]);
}

#[test]
fn includes_the_split_of_a_binary_term() {
    let term = Term {
        sums: Vec::new(),
        coeff: one(),
        factors: vec![scalar(A), scalar(B)],
    };

    let splits = enumerate_splits(&[], &term);

    assert_eq!(splits.len(), 1);
    assert_eq!(tensors(&splits[0].left), vec![A]);
    assert_eq!(tensors(&splits[0].right), vec![B]);
}

#[test]
fn separates_child_local_and_contracted_sums() {
    let term = Term {
        sums: vec![index(0), index(1)],
        coeff: Coefficient::from_integer(6.into()),
        factors: vec![tensor(A, &[2, 0]), tensor(B, &[0, 1]), tensor(C, &[1, 3])],
    };

    let splits = enumerate_splits(&[index(2), index(3)], &term);
    let split = &splits[1];

    assert_eq!(split.left.sums, vec![index(0)]);
    assert!(split.right.sums.is_empty());
    assert_eq!(split.left_exts, vec![index(2), index(1)]);
    assert_eq!(split.right_exts, vec![index(3), index(1)]);
    assert_eq!(split.contracted, vec![index(1)]);
    assert_eq!(split.left.coeff, one());
    assert_eq!(split.right.coeff, one());
}

#[test]
fn canonicalizes_both_owner_forms_when_they_differ() {
    let computation = tensor_computation(&[(A, 2), (B, 2)]);
    let term = Term {
        sums: vec![index(10), index(11)],
        coeff: one(),
        factors: vec![tensor(A, &[10, 11]), tensor(B, &[11, 10])],
    };
    let split = bipartition_term(&[], &term, &[true, false]);

    let (left_owned, right_owned) = canon_split(&computation, &[], split).unwrap().unwrap();

    assert_eq!(
        left_owned.left.factors[0].indices,
        vec![IndexId(0), IndexId(1)]
    );
    assert_eq!(
        left_owned.right.factors[0].indices,
        vec![IndexId(1), IndexId(0)]
    );
    assert_eq!(
        right_owned.left.factors[0].indices,
        vec![IndexId(1), IndexId(0)]
    );
    assert_eq!(
        right_owned.right.factors[0].indices,
        vec![IndexId(0), IndexId(1)]
    );
    assert_eq!(left_owned.contracted, vec![index(0), index(1)]);
    assert_eq!(right_owned.contracted, vec![index(0), index(1)]);
}

#[test]
fn returns_both_owner_forms_when_they_are_the_same() {
    let computation = tensor_computation(&[(A, 2), (B, 2)]);
    let term = Term {
        sums: vec![index(10), index(11)],
        coeff: one(),
        factors: vec![tensor(A, &[10, 11]), tensor(B, &[10, 11])],
    };
    let split = bipartition_term(&[], &term, &[true, false]);

    let (left_owned, right_owned) = canon_split(&computation, &[], split).unwrap().unwrap();

    assert_eq!(left_owned, right_owned);
    assert_eq!(
        left_owned.left.factors[0].indices,
        vec![IndexId(0), IndexId(1)]
    );
    assert_eq!(
        left_owned.right.factors[0].indices,
        vec![IndexId(0), IndexId(1)]
    );
}

#[test]
fn aligns_local_sums_after_definition_and_contracted_indices() {
    let computation = tensor_computation(&[(A, 3), (B, 1), (C, 1)]);
    let term = Term {
        sums: vec![index(11), index(12)],
        coeff: one(),
        factors: vec![tensor(A, &[10, 11, 12]), tensor(B, &[11]), tensor(C, &[12])],
    };
    let definition_exts = [index(10)];
    let split = bipartition_term(&definition_exts, &term, &[true, true, false]);

    let (canonical, _) = canon_split(&computation, &definition_exts, split)
        .unwrap()
        .unwrap();

    assert_eq!(canonical.left.sums, vec![index(2)]);
    assert_eq!(canonical.left_exts, vec![index(0), index(1)]);
    assert_eq!(canonical.right.sums, Vec::new());
    assert_eq!(canonical.right_exts, vec![index(1)]);
    assert_eq!(canonical.contracted, vec![index(1)]);
}

#[test]
fn builds_owner_graphs_and_mirrors_equal_interfaces() {
    let computation = tensor_computation(&[(A, 0), (B, 0), (C, 0)]);
    let definition = TensorDef {
        base: D,
        exts: Vec::new(),
        rhs: vec![
            Term {
                sums: Vec::new(),
                coeff: integer(2),
                factors: vec![scalar(A), scalar(B)],
            },
            Term {
                sums: Vec::new(),
                coeff: integer(3),
                factors: vec![scalar(A), scalar(C)],
            },
        ],
    };

    let graphs = build_graphs(&computation, &definition).unwrap();

    assert_eq!(graphs.len(), 2);
    for graph in graphs {
        assert!(graph.left_exts.is_empty());
        assert!(graph.right_exts.is_empty());
        assert!(graph.contracted.is_empty());
        assert_eq!(graph.edges.len(), 4);

        let left_a = scalar_node(&graph.left_nodes, A);
        let left_b = scalar_node(&graph.left_nodes, B);
        let left_c = scalar_node(&graph.left_nodes, C);
        let right_a = scalar_node(&graph.right_nodes, A);
        let right_b = scalar_node(&graph.right_nodes, B);
        let right_c = scalar_node(&graph.right_nodes, C);

        assert_eq!(graph.edges.get(&(left_a, right_b)), Some(&edge(2, &[0])));
        assert_eq!(graph.edges.get(&(left_b, right_a)), Some(&edge(2, &[0])));
        assert_eq!(graph.edges.get(&(left_a, right_c)), Some(&edge(3, &[1])));
        assert_eq!(graph.edges.get(&(left_c, right_a)), Some(&edge(3, &[1])));

        let bicliques = enumerate_bicliques(&graph);
        assert_eq!(bicliques.len(), 2);
        assert!(bicliques.iter().any(|biclique| {
            biclique.left == vec![(left_a, one())]
                && biclique.right == vec![(right_b, integer(2)), (right_c, integer(3))]
        }));
    }
}

#[test]
fn enumerates_one_maximal_rank_one_rectangle() {
    let graph = test_graph(&[
        (0, 0, 2, &[0]),
        (0, 1, 3, &[1]),
        (1, 0, 4, &[2]),
        (1, 1, 6, &[3]),
    ]);

    let bicliques = enumerate_bicliques(&graph);

    assert_eq!(bicliques.len(), 1);
    assert_eq!(bicliques[0].left, vec![(0, one()), (1, integer(2))]);
    assert_eq!(bicliques[0].right, vec![(0, integer(2)), (1, integer(3))]);
    assert_eq!(bicliques[0].terms, BTreeSet::from([0, 1, 2, 3]));
}

#[test]
fn rejects_a_non_rank_one_rectangle() {
    let graph = test_graph(&[
        (0, 0, 1, &[0]),
        (0, 1, 1, &[1]),
        (1, 0, 1, &[2]),
        (1, 1, 2, &[3]),
    ]);

    let bicliques = enumerate_bicliques(&graph);

    assert_eq!(bicliques.len(), 4);
    assert!(
        bicliques
            .iter()
            .all(|biclique| biclique.left.len() == 1 || biclique.right.len() == 1)
    );
}

#[test]
fn rejects_edges_that_reuse_a_source_term() {
    let graph = test_graph(&[(0, 0, 1, &[0]), (0, 1, 1, &[0])]);

    let bicliques = enumerate_bicliques(&graph);

    assert!(bicliques.is_empty());
}

fn tensor_computation(tensors: &[(TensorId, usize)]) -> Computation {
    let mut computation = Computation::default();
    computation.ranges.insert(RANGE);
    for &(tensor, rank) in tensors {
        computation.tensors.insert(
            tensor,
            TensorInfo {
                rank,
                symmetry: Vec::new(),
            },
        );
    }
    computation
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

fn tensors(term: &Term) -> Vec<TensorId> {
    term.factors.iter().map(|factor| factor.tensor).collect()
}

fn scalar_node(nodes: &[Term], tensor: TensorId) -> usize {
    nodes
        .iter()
        .position(|term| term.factors == vec![scalar(tensor)])
        .unwrap()
}

fn test_graph(edges: &[(usize, usize, i64, &[usize])]) -> Graph {
    let node_count = edges
        .iter()
        .map(|(left, right, _, _)| (*left).max(*right))
        .max()
        .unwrap()
        + 1;
    Graph {
        left_exts: Vec::new(),
        right_exts: Vec::new(),
        contracted: Vec::new(),
        left_nodes: (0..node_count)
            .map(|node| unit_term(TensorId(node as u32)))
            .collect(),
        right_nodes: (0..node_count)
            .map(|node| unit_term(TensorId(node as u32)))
            .collect(),
        edges: edges
            .iter()
            .map(|&(left, right, coeff, terms)| ((left, right), edge(coeff, terms)))
            .collect(),
    }
}

fn unit_term(tensor: TensorId) -> Term {
    Term {
        sums: Vec::new(),
        coeff: one(),
        factors: vec![scalar(tensor)],
    }
}

fn edge(coeff: i64, terms: &[usize]) -> Edge {
    Edge {
        coeff: integer(coeff),
        terms: terms.iter().copied().collect(),
    }
}

fn index(id: u32) -> Index {
    Index {
        id: IndexId(id),
        range: RANGE,
    }
}

fn one() -> Coefficient {
    Coefficient::from_integer(1.into())
}

fn integer(value: i64) -> Coefficient {
    Coefficient::from_integer(value.into())
}
