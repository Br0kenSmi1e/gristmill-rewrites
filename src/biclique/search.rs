//! Enumeration of maximal weighted rank-one bicliques.

use super::graph::Graph;
use crate::repr::Coefficient;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Biclique {
    pub(super) left: Vec<(usize, Coefficient)>,
    pub(super) right: Vec<(usize, Coefficient)>,
    pub(super) terms: BTreeSet<usize>,
}

pub(super) fn enumerate_bicliques(graph: &Graph) -> Vec<Biclique> {
    let mut results = Vec::new();
    let mut seen = BTreeSet::new();

    for (&(left, right), edge) in &graph.edges {
        let biclique = Biclique {
            left: vec![(left, one())],
            right: vec![(right, edge.coeff.clone())],
            terms: edge.terms.clone(),
        };
        search_bicliques(graph, biclique, 0, &mut seen, &mut results);
    }

    results
}

fn search_bicliques(
    graph: &Graph,
    biclique: Biclique,
    position: usize,
    seen: &mut BTreeSet<(Vec<usize>, Vec<usize>)>,
    results: &mut Vec<Biclique>,
) {
    let node_count = graph.left_nodes.len() + graph.right_nodes.len();
    if position == node_count {
        if (biclique.left.len() > 1 || biclique.right.len() > 1) && is_maximal(graph, &biclique) {
            let biclique = normalize_biclique(biclique);
            let key = (
                biclique.left.iter().map(|(node, _)| *node).collect(),
                biclique.right.iter().map(|(node, _)| *node).collect(),
            );
            if seen.insert(key) {
                results.push(biclique);
            }
        }
        return;
    }

    if position < graph.left_nodes.len() {
        let node = position;
        if biclique.left.iter().any(|(selected, _)| *selected == node) {
            search_bicliques(graph, biclique, position + 1, seen, results);
            return;
        }
        if let Some(next) = add_left(graph, &biclique, node) {
            search_bicliques(graph, next, position + 1, seen, results);
        }
    } else {
        let node = position - graph.left_nodes.len();
        if biclique.right.iter().any(|(selected, _)| *selected == node) {
            search_bicliques(graph, biclique, position + 1, seen, results);
            return;
        }
        if let Some(next) = add_right(graph, &biclique, node) {
            search_bicliques(graph, next, position + 1, seen, results);
        }
    }

    search_bicliques(graph, biclique, position + 1, seen, results);
}

fn add_left(graph: &Graph, biclique: &Biclique, node: usize) -> Option<Biclique> {
    let first_right = biclique.right.first()?;
    let first_edge = graph.edges.get(&(node, first_right.0))?;
    let coeff = &first_edge.coeff / &first_right.1;
    let mut added_terms = BTreeSet::new();

    for (right, right_coeff) in &biclique.right {
        let edge = graph.edges.get(&(node, *right))?;
        if edge.coeff != &coeff * right_coeff
            || !edge.terms.is_disjoint(&biclique.terms)
            || !edge.terms.is_disjoint(&added_terms)
        {
            return None;
        }
        added_terms.extend(&edge.terms);
    }

    let mut result = biclique.clone();
    result.left.push((node, coeff));
    result.terms.extend(added_terms);
    Some(result)
}

fn add_right(graph: &Graph, biclique: &Biclique, node: usize) -> Option<Biclique> {
    let first_left = biclique.left.first()?;
    let first_edge = graph.edges.get(&(first_left.0, node))?;
    let coeff = &first_edge.coeff / &first_left.1;
    let mut added_terms = BTreeSet::new();

    for (left, left_coeff) in &biclique.left {
        let edge = graph.edges.get(&(*left, node))?;
        if edge.coeff != left_coeff * &coeff
            || !edge.terms.is_disjoint(&biclique.terms)
            || !edge.terms.is_disjoint(&added_terms)
        {
            return None;
        }
        added_terms.extend(&edge.terms);
    }

    let mut result = biclique.clone();
    result.right.push((node, coeff));
    result.terms.extend(added_terms);
    Some(result)
}

fn is_maximal(graph: &Graph, biclique: &Biclique) -> bool {
    let left_maximal = (0..graph.left_nodes.len()).all(|node| {
        biclique.left.iter().any(|(selected, _)| *selected == node)
            || add_left(graph, biclique, node).is_none()
    });
    let right_maximal = (0..graph.right_nodes.len()).all(|node| {
        biclique.right.iter().any(|(selected, _)| *selected == node)
            || add_right(graph, biclique, node).is_none()
    });
    left_maximal && right_maximal
}

fn normalize_biclique(mut biclique: Biclique) -> Biclique {
    biclique.left.sort_by_key(|(node, _)| *node);
    biclique.right.sort_by_key(|(node, _)| *node);

    let scale = biclique.left[0].1.clone();
    for (_, coeff) in &mut biclique.left {
        *coeff /= &scale;
    }
    for (_, coeff) in &mut biclique.right {
        *coeff *= &scale;
    }

    biclique
}

fn one() -> Coefficient {
    Coefficient::from_integer(1.into())
}
