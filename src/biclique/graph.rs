//! Construction of weighted bipartite graphs from normalized term splits.

use super::normalize::{Owner, canon_split, enumerate_splits};
use crate::{
    canon::CanonError,
    parenthesize::TermBipartition,
    repr::{Coefficient, Computation, Index, TensorDef, Term},
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Edge {
    pub(super) coeff: Coefficient,
    pub(super) terms: BTreeSet<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Graph {
    pub(super) left_exts: Vec<Index>,
    pub(super) right_exts: Vec<Index>,
    pub(super) contracted: Vec<Index>,
    pub(super) left_nodes: Vec<Term>,
    pub(super) right_nodes: Vec<Term>,
    pub(super) edges: BTreeMap<(usize, usize), Edge>,
}

type GraphKey = (Owner, Vec<Index>, Vec<Index>, Vec<Index>);

pub(super) fn build_graphs(
    computation: &Computation,
    definition: &TensorDef,
) -> Result<Vec<Graph>, CanonError> {
    let mut graphs = BTreeMap::<GraphKey, Graph>::new();

    for (term_position, term) in definition.rhs.iter().enumerate() {
        for split in enumerate_splits(&definition.exts, term) {
            let Some((left_owned, right_owned)) =
                canon_split(computation, &definition.exts, split)?
            else {
                continue;
            };

            insert_split(
                &mut graphs,
                Owner::Left,
                definition.exts.len(),
                term_position,
                left_owned,
            );
            insert_split(
                &mut graphs,
                Owner::Right,
                definition.exts.len(),
                term_position,
                right_owned,
            );
        }
    }

    for graph in graphs.values_mut() {
        graph.edges.retain(|_, edge| edge.coeff != zero());
    }
    Ok(graphs
        .into_values()
        .filter(|graph| !graph.edges.is_empty())
        .collect())
}

fn insert_split(
    graphs: &mut BTreeMap<GraphKey, Graph>,
    owner: Owner,
    definition_ext_count: usize,
    term_position: usize,
    split: TermBipartition,
) {
    let TermBipartition {
        coeff,
        mut left,
        left_exts,
        mut right,
        right_exts,
        contracted,
    } = split;
    let coeff = &coeff * &left.coeff * &right.coeff;
    left.coeff = one();
    right.coeff = one();

    let left_external = left_exts
        .iter()
        .filter(|index| (index.id.0 as usize) < definition_ext_count)
        .copied()
        .collect::<Vec<_>>();
    let right_external = right_exts
        .iter()
        .filter(|index| (index.id.0 as usize) < definition_ext_count)
        .copied()
        .collect::<Vec<_>>();

    match left_external.cmp(&right_external) {
        Ordering::Less => insert_edge(
            graphs,
            owner,
            left_exts,
            right_exts,
            contracted,
            left,
            right,
            coeff,
            term_position,
        ),
        Ordering::Greater => insert_edge(
            graphs,
            owner.opposite(),
            right_exts,
            left_exts,
            contracted,
            right,
            left,
            coeff,
            term_position,
        ),
        Ordering::Equal => {
            insert_edge(
                graphs,
                owner,
                left_exts.clone(),
                right_exts.clone(),
                contracted.clone(),
                left.clone(),
                right.clone(),
                coeff.clone(),
                term_position,
            );
            insert_edge(
                graphs,
                owner.opposite(),
                right_exts,
                left_exts,
                contracted,
                right,
                left,
                coeff,
                term_position,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn insert_edge(
    graphs: &mut BTreeMap<GraphKey, Graph>,
    owner: Owner,
    left_exts: Vec<Index>,
    right_exts: Vec<Index>,
    contracted: Vec<Index>,
    left: Term,
    right: Term,
    coeff: Coefficient,
    term_position: usize,
) {
    let key = (
        owner,
        left_exts.clone(),
        right_exts.clone(),
        contracted.clone(),
    );
    let graph = graphs.entry(key).or_insert_with(|| Graph {
        left_exts,
        right_exts,
        contracted,
        left_nodes: Vec::new(),
        right_nodes: Vec::new(),
        edges: BTreeMap::new(),
    });
    let left = intern_node(&mut graph.left_nodes, left);
    let right = intern_node(&mut graph.right_nodes, right);
    let edge = graph.edges.entry((left, right)).or_insert_with(|| Edge {
        coeff: zero(),
        terms: BTreeSet::new(),
    });

    if edge.terms.insert(term_position) {
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

fn one() -> Coefficient {
    Coefficient::from_integer(1.into())
}
