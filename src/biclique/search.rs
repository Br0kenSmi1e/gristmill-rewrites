//! Enumeration of maximal weighted rank-one bicliques.

use super::graph::Edges;
use crate::repr::Coefficient;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum SearchNode {
    Left(usize),
    Right(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Delta {
    pub(super) coeff: Coefficient,
    pub(super) terms: BTreeSet<usize>,
}

pub(super) type Frontier = BTreeMap<SearchNode, Delta>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Biclique {
    pub(super) left: Vec<(usize, Coefficient)>,
    pub(super) right: Vec<(usize, Coefficient)>,
    pub(super) terms: BTreeSet<usize>,
}

pub(super) fn enumerate_bicliques(edges: &Edges) -> Vec<Biclique> {
    if edges.len() < 2 {
        return Vec::new();
    }

    let mut biclique = Biclique {
        left: Vec::new(),
        right: Vec::new(),
        terms: BTreeSet::new(),
    };
    let mut candidates = all_candidates(edges);
    let frontier = candidates
        .iter()
        .copied()
        .map(|node| {
            (
                node,
                Delta {
                    coeff: one(),
                    terms: BTreeSet::new(),
                },
            )
        })
        .collect();
    let mut results = Vec::new();

    expand(
        edges,
        &mut biclique,
        &frontier,
        &mut candidates,
        &mut results,
    );
    results
}

fn all_candidates(edges: &Edges) -> Vec<SearchNode> {
    edges
        .keys()
        .map(|(left, _)| *left)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(SearchNode::Left)
        .chain(
            edges
                .keys()
                .map(|(_, right)| *right)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(SearchNode::Right),
        )
        .collect()
}

fn expand(
    edges: &Edges,
    biclique: &mut Biclique,
    frontier: &Frontier,
    candidates: &mut Vec<SearchNode>,
    results: &mut Vec<Biclique>,
) {
    if has_sharing(biclique) && frontier.is_empty() {
        results.push(normalize_biclique(biclique.clone()));
        return;
    }

    let child_frontiers = build_child_frontiers(edges, biclique, frontier);
    let current = sift(biclique, candidates, frontier, &child_frontiers);

    for node in current {
        let Some(delta) = frontier.get(&node) else {
            continue;
        };
        let Some(position) = candidates.iter().position(|candidate| *candidate == node) else {
            continue;
        };

        let removed = candidates.remove(position);
        let child_frontier = &child_frontiers[&removed];
        let mut child_candidates = candidates
            .iter()
            .copied()
            .filter(|candidate| child_frontier.contains_key(candidate))
            .collect::<Vec<_>>();

        push(biclique, removed, delta);
        expand(
            edges,
            biclique,
            child_frontier,
            &mut child_candidates,
            results,
        );
        pop(biclique, removed, delta);
    }
}

pub(super) fn sift(
    biclique: &Biclique,
    candidates: &[SearchNode],
    frontier: &Frontier,
    child_frontiers: &BTreeMap<SearchNode, Frontier>,
) -> Vec<SearchNode> {
    if biclique.left.is_empty() && biclique.right.is_empty() {
        return candidates
            .iter()
            .filter(|node| matches!(node, SearchNode::Left(_)))
            .copied()
            .collect();
    }

    if biclique.left.len() == 1 && biclique.right.is_empty() {
        return candidates
            .iter()
            .filter(|node| matches!(node, SearchNode::Right(_)))
            .filter(|node| matches!(frontier.get(node), Some(delta) if !delta.terms.is_empty()))
            .copied()
            .collect();
    }

    let candidate_set = candidates.iter().copied().collect::<BTreeSet<_>>();
    let mut pivot_neighbors = None;
    let mut best_score = 0;
    for node in candidates {
        let neighbors = &child_frontiers[node];
        let score = neighbors
            .keys()
            .filter(|candidate| candidate_set.contains(candidate))
            .count();
        if score > best_score {
            best_score = score;
            pivot_neighbors = Some(neighbors);
        }
    }

    candidates
        .iter()
        .filter(|node| !pivot_neighbors.is_some_and(|neighbors| neighbors.contains_key(node)))
        .copied()
        .collect()
}

fn build_child_frontiers(
    edges: &Edges,
    biclique: &Biclique,
    frontier: &Frontier,
) -> BTreeMap<SearchNode, Frontier> {
    frontier
        .iter()
        .map(|(&chosen, chosen_delta)| {
            let child = frontier
                .iter()
                .filter_map(|(&candidate, candidate_delta)| {
                    if chosen == candidate {
                        return None;
                    }
                    update_delta(
                        edges,
                        biclique,
                        chosen,
                        chosen_delta,
                        candidate,
                        candidate_delta,
                    )
                    .map(|delta| (candidate, delta))
                })
                .collect();
            (chosen, child)
        })
        .collect()
}

fn update_delta(
    edges: &Edges,
    biclique: &Biclique,
    chosen: SearchNode,
    chosen_delta: &Delta,
    candidate: SearchNode,
    candidate_delta: &Delta,
) -> Option<Delta> {
    if matches!(
        (chosen, candidate),
        (SearchNode::Left(_), SearchNode::Left(_)) | (SearchNode::Right(_), SearchNode::Right(_))
    ) {
        return chosen_delta
            .terms
            .is_disjoint(&candidate_delta.terms)
            .then(|| candidate_delta.clone());
    }

    let (left, right) = match (chosen, candidate) {
        (SearchNode::Left(left), SearchNode::Right(right))
        | (SearchNode::Right(right), SearchNode::Left(left)) => (left, right),
        _ => unreachable!(),
    };
    let edge = edges.get(&(left, right))?;

    if !chosen_delta.terms.is_disjoint(&candidate_delta.terms)
        || !biclique.terms.is_disjoint(&edge.terms)
        || !chosen_delta.terms.is_disjoint(&edge.terms)
        || !candidate_delta.terms.is_disjoint(&edge.terms)
    {
        return None;
    }

    let expected = &edge.coeff / &chosen_delta.coeff;
    let mut next = candidate_delta.clone();
    if candidate_delta.terms.is_empty() {
        next.coeff = expected;
    } else if candidate_delta.coeff != expected {
        return None;
    }
    next.terms.extend(&edge.terms);
    Some(next)
}

fn has_sharing(biclique: &Biclique) -> bool {
    biclique.left.len() >= 2 || biclique.right.len() >= 2
}

fn push(biclique: &mut Biclique, node: SearchNode, delta: &Delta) {
    debug_assert!(biclique.terms.is_disjoint(&delta.terms));
    biclique.terms.extend(&delta.terms);

    match node {
        SearchNode::Left(node) => biclique.left.push((node, delta.coeff.clone())),
        SearchNode::Right(node) => biclique.right.push((node, delta.coeff.clone())),
    }
}

fn pop(biclique: &mut Biclique, node: SearchNode, delta: &Delta) {
    for term in &delta.terms {
        let removed = biclique.terms.remove(term);
        debug_assert!(removed);
    }

    match node {
        SearchNode::Left(_) => {
            biclique.left.pop();
        }
        SearchNode::Right(_) => {
            biclique.right.pop();
        }
    }
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
