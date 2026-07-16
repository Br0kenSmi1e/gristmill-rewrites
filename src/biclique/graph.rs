//! Construction of weighted bipartite graphs from normalized term bipartitions.

use crate::{
    parenthesize::TermBipartition,
    repr::{Coefficient, Index, Term},
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Edge {
    pub(super) coeff: Coefficient,
    pub(super) terms: BTreeSet<usize>,
}

pub(super) type Edges = BTreeMap<(usize, usize), Edge>;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct GraphKey {
    pub(super) left_exts: Vec<Index>,
    pub(super) right_exts: Vec<Index>,
    pub(super) contracted: Vec<Index>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct GraphData {
    pub(super) left_nodes: Vec<Term>,
    pub(super) right_nodes: Vec<Term>,
    pub(super) edges: Edges,
}

pub(super) type Graph = (GraphKey, GraphData);

pub(super) fn build(bipartitions: Vec<(usize, TermBipartition)>) -> Vec<Graph> {
    let mut graphs = BTreeMap::<GraphKey, GraphData>::new();

    for (source_term, bipartition) in bipartitions {
        insert_bipartition(&mut graphs, source_term, bipartition);
    }

    for graph in graphs.values_mut() {
        graph.edges.retain(|_, edge| edge.coeff != zero());
    }
    graphs
        .into_iter()
        .filter(|(_, graph)| !graph.edges.is_empty())
        .collect()
}

fn insert_bipartition(
    graphs: &mut BTreeMap<GraphKey, GraphData>,
    source_term: usize,
    bipartition: TermBipartition,
) {
    let TermBipartition {
        coeff,
        left,
        left_exts,
        right,
        right_exts,
        contracted,
    } = bipartition;

    let key = GraphKey {
        left_exts,
        right_exts,
        contracted,
    };
    let graph = graphs.entry(key).or_default();
    let left = intern_node(&mut graph.left_nodes, left);
    let right = intern_node(&mut graph.right_nodes, right);
    let edge = graph.edges.entry((left, right)).or_insert_with(|| Edge {
        coeff: zero(),
        terms: BTreeSet::new(),
    });

    if edge.terms.insert(source_term) {
        edge.coeff += coeff;
    }
}

fn intern_node(nodes: &mut Vec<Term>, term: Term) -> usize {
    if let Some(position) = nodes.iter().position(|node| node == &term) {
        position
    } else {
        let position = nodes.len();
        nodes.push(term);
        position
    }
}

fn zero() -> Coefficient {
    Coefficient::from_integer(0.into())
}
