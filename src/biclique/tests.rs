//! Unit tests for biclique normalization, graph construction, and search.

use super::{
    graph::{Edge, Edges, build},
    normalize::{
        align_bipartition, canon_bipartition, enumerate_bipartitions, normalize_definition,
    },
    search::{Delta, SearchNode, sift},
    *,
};
use crate::{
    parenthesize::{TermBipartition, bipartition_term},
    repr::{Coefficient, Computation, IndexId, RangeId, TensorId, TensorInfo, TensorRef},
};
use std::collections::BTreeMap;

const RANGE: RangeId = RangeId(0);
const A: TensorId = TensorId(0);
const B: TensorId = TensorId(1);
const C: TensorId = TensorId(2);
const D: TensorId = TensorId(3);

#[test]
fn enumerates_each_oriented_factor_bipartition() {
    let term = Term {
        sums: Vec::new(),
        coeff: one(),
        factors: vec![scalar(A), scalar(B), scalar(C)],
    };

    let bipartitions = enumerate_bipartitions(&[], &term);

    assert_eq!(bipartitions.len(), 6);
    assert_eq!(tensors(&bipartitions[0].left), vec![A]);
    assert_eq!(tensors(&bipartitions[0].right), vec![B, C]);
    assert_eq!(tensors(&bipartitions[1].left), vec![B]);
    assert_eq!(tensors(&bipartitions[1].right), vec![A, C]);
    assert_eq!(tensors(&bipartitions[2].left), vec![A, B]);
    assert_eq!(tensors(&bipartitions[2].right), vec![C]);
    assert_eq!(tensors(&bipartitions[3].left), vec![C]);
    assert_eq!(tensors(&bipartitions[3].right), vec![A, B]);
    assert_eq!(tensors(&bipartitions[4].left), vec![A, C]);
    assert_eq!(tensors(&bipartitions[4].right), vec![B]);
    assert_eq!(tensors(&bipartitions[5].left), vec![B, C]);
    assert_eq!(tensors(&bipartitions[5].right), vec![A]);
}

#[test]
fn includes_the_bipartition_of_a_binary_term() {
    let term = Term {
        sums: Vec::new(),
        coeff: one(),
        factors: vec![scalar(A), scalar(B)],
    };

    let bipartitions = enumerate_bipartitions(&[], &term);

    assert_eq!(bipartitions.len(), 2);
    assert_eq!(tensors(&bipartitions[0].left), vec![A]);
    assert_eq!(tensors(&bipartitions[0].right), vec![B]);
    assert_eq!(tensors(&bipartitions[1].left), vec![B]);
    assert_eq!(tensors(&bipartitions[1].right), vec![A]);
}

#[test]
fn separates_child_local_and_contracted_sums() {
    let term = Term {
        sums: vec![index(0), index(1)],
        coeff: Coefficient::from_integer(6.into()),
        factors: vec![tensor(A, &[2, 0]), tensor(B, &[0, 1]), tensor(C, &[1, 3])],
    };

    let bipartitions = enumerate_bipartitions(&[index(2), index(3)], &term);
    let bipartition = &bipartitions[2];

    assert_eq!(bipartition.left.sums, vec![index(0)]);
    assert!(bipartition.right.sums.is_empty());
    assert_eq!(bipartition.left_exts, vec![index(2), index(1)]);
    assert_eq!(bipartition.right_exts, vec![index(3), index(1)]);
    assert_eq!(bipartition.contracted, vec![index(1)]);
    assert_eq!(bipartition.left.coeff, one());
    assert_eq!(bipartition.right.coeff, one());
}

#[test]
fn canonicalizes_each_orientation_by_its_left_side() {
    let computation = tensor_computation(&[(A, 2), (B, 2)]);
    let term = Term {
        sums: vec![index(10), index(11)],
        coeff: one(),
        factors: vec![tensor(A, &[10, 11]), tensor(B, &[11, 10])],
    };
    let aligned = align_bipartition(&[], bipartition_term(&[], &term, &[true, false])).unwrap();
    let reversed = align_bipartition(&[], bipartition_term(&[], &term, &[false, true])).unwrap();

    let left_first = canon_bipartition(&computation, &[], aligned)
        .unwrap()
        .unwrap();
    let reversed_left_first = canon_bipartition(&computation, &[], reversed)
        .unwrap()
        .unwrap();

    assert_eq!(
        left_first.left.factors[0].indices,
        vec![IndexId(0), IndexId(1)]
    );
    assert_eq!(
        left_first.right.factors[0].indices,
        vec![IndexId(1), IndexId(0)]
    );
    assert_eq!(
        reversed_left_first.left.factors[0].indices,
        vec![IndexId(0), IndexId(1)]
    );
    assert_eq!(
        reversed_left_first.right.factors[0].indices,
        vec![IndexId(1), IndexId(0)]
    );
    assert_eq!(left_first.contracted, vec![index(0), index(1)]);
    assert_eq!(reversed_left_first.contracted, vec![index(0), index(1)]);
}

#[test]
fn returns_the_left_first_form() {
    let computation = tensor_computation(&[(A, 2), (B, 2)]);
    let term = Term {
        sums: vec![index(10), index(11)],
        coeff: one(),
        factors: vec![tensor(A, &[10, 11]), tensor(B, &[10, 11])],
    };
    let bipartition = bipartition_term(&[], &term, &[true, false]);
    let aligned = align_bipartition(&[], bipartition).unwrap();

    let canonical = canon_bipartition(&computation, &[], aligned)
        .unwrap()
        .unwrap();

    assert_eq!(
        canonical.left.factors[0].indices,
        vec![IndexId(0), IndexId(1)]
    );
    assert_eq!(
        canonical.right.factors[0].indices,
        vec![IndexId(0), IndexId(1)]
    );
}

#[test]
fn moves_child_coefficients_to_the_canonical_bipartition() {
    let computation = tensor_computation(&[(A, 0), (B, 0)]);
    let aligned = TermBipartition {
        coeff: integer(5),
        left: Term {
            sums: Vec::new(),
            coeff: integer(2),
            factors: vec![scalar(A)],
        },
        left_exts: Vec::new(),
        right: Term {
            sums: Vec::new(),
            coeff: integer(3),
            factors: vec![scalar(B)],
        },
        right_exts: Vec::new(),
        contracted: Vec::new(),
    };

    let candidate = canon_bipartition(&computation, &[], aligned)
        .unwrap()
        .unwrap();

    assert_eq!(candidate.coeff, integer(30));
    assert_eq!(candidate.left.coeff, one());
    assert_eq!(candidate.right.coeff, one());
}

#[test]
fn aligns_new_indices_without_renaming_definition_externals() {
    let term = Term {
        sums: vec![index(11), index_in(12, RangeId(1)), index(10)],
        coeff: one(),
        factors: vec![
            tensor(A, &[0, 2, 11, 12, 10]),
            tensor(B, &[11]),
            tensor(C, &[12, 10]),
        ],
    };
    let definition_exts = [index(0), index(2)];
    let bipartition = bipartition_term(&definition_exts, &term, &[true, true, false]);

    let aligned = align_bipartition(&definition_exts, bipartition).unwrap();

    assert_eq!(aligned.left.sums, vec![index(4)]);
    assert_eq!(aligned.left.factors[0].indices, ids(&[0, 2, 4, 3, 1]));
    assert_eq!(aligned.left.factors[1].indices, ids(&[4]));
    assert_eq!(
        aligned.left_exts,
        vec![index(0), index(1), index(2), index_in(3, RangeId(1))]
    );
    assert_eq!(aligned.right.sums, Vec::new());
    assert_eq!(aligned.right.factors[0].indices, ids(&[3, 1]));
    assert_eq!(aligned.right_exts, vec![index(1), index_in(3, RangeId(1))]);
    assert_eq!(aligned.contracted, vec![index(1), index_in(3, RangeId(1))]);
}

#[test]
fn contracted_permutations_preserve_ranges() {
    let computation = tensor_computation(&[(A, 2), (B, 2)]);
    let contracted = vec![index_in(0, RangeId(1)), index_in(1, RANGE)];
    let aligned = TermBipartition {
        coeff: one(),
        left: Term {
            sums: Vec::new(),
            coeff: one(),
            factors: vec![tensor(A, &[1, 0])],
        },
        left_exts: contracted.clone(),
        right: Term {
            sums: Vec::new(),
            coeff: one(),
            factors: vec![tensor(B, &[0, 1])],
        },
        right_exts: contracted.clone(),
        contracted,
    };

    let canonical = canon_bipartition(&computation, &[], aligned.clone())
        .unwrap()
        .unwrap();

    assert_eq!(canonical, aligned);
}

#[test]
fn builds_a_graph_from_both_bipartition_orientations() {
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

    let canonical = normalize_definition(&computation, &definition).unwrap();
    let graphs = build(canonical);

    assert_eq!(graphs.len(), 1);
    for (key, graph) in graphs {
        assert!(key.left_exts.is_empty());
        assert!(key.right_exts.is_empty());
        assert!(key.contracted.is_empty());
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

        let bicliques = enumerate_bicliques(&graph.edges);
        assert_eq!(bicliques.len(), 2);
        assert!(bicliques.iter().any(|biclique| {
            biclique.left == vec![(left_a, one())]
                && biclique.right == vec![(right_b, integer(2)), (right_c, integer(3))]
        }));
    }
}

#[test]
fn enumerates_one_maximal_rank_one_rectangle() {
    let edges = test_edges(&[
        (0, 0, 2, &[0]),
        (0, 1, 3, &[1]),
        (1, 0, 4, &[2]),
        (1, 1, 6, &[3]),
    ]);

    let bicliques = enumerate_bicliques(&edges);

    assert_eq!(bicliques.len(), 1);
    assert_eq!(bicliques[0].left, vec![(0, one()), (1, integer(2))]);
    assert_eq!(bicliques[0].right, vec![(0, integer(2)), (1, integer(3))]);
    assert_eq!(bicliques[0].terms, BTreeSet::from([0, 1, 2, 3]));
}

#[test]
fn emits_a_dense_maximal_rectangle_once() {
    let edges = test_edges(&[
        (0, 0, 2, &[0]),
        (0, 1, 3, &[1]),
        (0, 2, 5, &[2]),
        (1, 0, 4, &[3]),
        (1, 1, 6, &[4]),
        (1, 2, 10, &[5]),
    ]);

    let bicliques = enumerate_bicliques(&edges);

    assert_eq!(bicliques.len(), 1);
    assert_eq!(bicliques[0].left, vec![(0, one()), (1, integer(2))]);
    assert_eq!(
        bicliques[0].right,
        vec![(0, integer(2)), (1, integer(3)), (2, integer(5))]
    );
    assert_eq!(bicliques[0].terms, BTreeSet::from([0, 1, 2, 3, 4, 5]));
}

#[test]
fn sift_branches_outside_the_best_pivot_frontier() {
    fn delta() -> Delta {
        Delta {
            coeff: one(),
            terms: BTreeSet::from([1]),
        }
    }

    let biclique = Biclique {
        left: vec![(0, one())],
        right: vec![(0, one())],
        terms: BTreeSet::from([0]),
    };
    let candidates = vec![
        SearchNode::Left(1),
        SearchNode::Right(1),
        SearchNode::Left(2),
    ];
    let frontier = candidates
        .iter()
        .copied()
        .map(|node| (node, delta()))
        .collect();
    let child_frontiers = BTreeMap::from([
        (
            SearchNode::Left(1),
            BTreeMap::from([
                (SearchNode::Right(1), delta()),
                (SearchNode::Left(2), delta()),
            ]),
        ),
        (
            SearchNode::Right(1),
            BTreeMap::from([(SearchNode::Left(2), delta())]),
        ),
        (SearchNode::Left(2), BTreeMap::new()),
    ]);

    let current = sift(&biclique, &candidates, &frontier, &child_frontiers);

    assert_eq!(current, vec![SearchNode::Left(1)]);
}

#[test]
fn rejects_a_non_rank_one_rectangle() {
    let edges = test_edges(&[
        (0, 0, 1, &[0]),
        (0, 1, 1, &[1]),
        (1, 0, 1, &[2]),
        (1, 1, 2, &[3]),
    ]);

    let bicliques = enumerate_bicliques(&edges);

    assert_eq!(bicliques.len(), 4);
    assert!(
        bicliques
            .iter()
            .all(|biclique| biclique.left.len() == 1 || biclique.right.len() == 1)
    );
}

#[test]
fn rejects_edges_that_reuse_a_source_term() {
    let edges = test_edges(&[(0, 0, 1, &[0]), (0, 1, 1, &[0])]);

    let bicliques = enumerate_bicliques(&edges);

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
        indices: ids(indices),
    }
}

fn ids(indices: &[u32]) -> Vec<IndexId> {
    indices.iter().copied().map(IndexId).collect()
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

fn test_edges(edges: &[(usize, usize, i64, &[usize])]) -> Edges {
    edges
        .iter()
        .map(|&(left, right, coeff, terms)| ((left, right), edge(coeff, terms)))
        .collect()
}

fn edge(coeff: i64, terms: &[usize]) -> Edge {
    Edge {
        coeff: integer(coeff),
        terms: terms.iter().copied().collect(),
    }
}

fn index(id: u32) -> Index {
    index_in(id, RANGE)
}

fn index_in(id: u32, range: RangeId) -> Index {
    Index {
        id: IndexId(id),
        range,
    }
}

fn one() -> Coefficient {
    Coefficient::from_integer(1.into())
}

fn integer(value: i64) -> Coefficient {
    Coefficient::from_integer(value.into())
}
