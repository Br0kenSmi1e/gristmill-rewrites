//! Biclique factorization queries and policy choices.

mod graph;
mod normalize;
mod search;

#[cfg(test)]
mod tests;

use self::{
    graph::{Graph, build},
    normalize::normalize_definition,
    search::{Biclique, enumerate_bicliques},
};
use crate::{
    action::{Action, DefinitionPosition, QueryError},
    repr::{Coefficient, Index, TensorDef, TensorId, Term},
    state::{State, StateError},
};
use std::collections::BTreeSet;

/// The policy interface for one biclique factorization query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BicliqueSpace {
    target: DefinitionPosition,
    base: TensorId,
    graphs: Vec<Graph>,
    candidates: Vec<(usize, Biclique)>,
}

impl BicliqueSpace {
    pub fn target(&self) -> DefinitionPosition {
        self.target
    }

    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    pub fn shape(&self, candidate: usize) -> Option<(usize, usize)> {
        self.candidates
            .get(candidate)
            .map(|(_, biclique)| (biclique.left.len(), biclique.right.len()))
    }

    pub fn select(
        &self,
        candidate: usize,
        left: &[bool],
        right: &[bool],
    ) -> Result<Action, BicliqueChoiceError> {
        let Some((graph, biclique)) = self
            .candidates
            .get(candidate)
            .map(|(graph, biclique)| (&self.graphs[*graph], biclique))
        else {
            return Err(BicliqueChoiceError::CandidateOutOfBounds {
                index: candidate,
                len: self.candidates.len(),
            });
        };

        if left.len() != biclique.left.len() {
            return Err(BicliqueChoiceError::WrongLeftMaskLength {
                expected: biclique.left.len(),
                got: left.len(),
            });
        }
        if right.len() != biclique.right.len() {
            return Err(BicliqueChoiceError::WrongRightMaskLength {
                expected: biclique.right.len(),
                got: right.len(),
            });
        }
        if left.iter().all(|selected| !selected) || right.iter().all(|selected| !selected) {
            return Err(BicliqueChoiceError::EmptySide);
        }

        Ok(Action::Biclique(make_action(
            self.target,
            self.base,
            graph,
            biclique,
            left,
            right,
        )))
    }
}

/// An invalid biclique-factorization choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BicliqueChoiceError {
    CandidateOutOfBounds { index: usize, len: usize },
    WrongLeftMaskLength { expected: usize, got: usize },
    WrongRightMaskLength { expected: usize, got: usize },
    EmptySide,
}

/// One validated biclique factorization choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BicliqueAction {
    target: DefinitionPosition,
    terms: Vec<usize>,
    children: (TensorDef, TensorDef),
    contracted: Vec<Index>,
}

impl BicliqueAction {
    pub fn target(&self) -> DefinitionPosition {
        self.target
    }
}

fn make_action(
    target: DefinitionPosition,
    base: TensorId,
    graph: &Graph,
    biclique: &Biclique,
    left_mask: &[bool],
    right_mask: &[bool],
) -> BicliqueAction {
    let left = biclique
        .left
        .iter()
        .zip(left_mask)
        .filter_map(|(node, &selected)| selected.then_some(node.clone()))
        .collect::<Vec<_>>();
    let right = biclique
        .right
        .iter()
        .zip(right_mask)
        .filter_map(|(node, &selected)| selected.then_some(node.clone()))
        .collect::<Vec<_>>();

    let mut terms = BTreeSet::new();
    for (left_node, _) in &left {
        for (right_node, _) in &right {
            let edge = &graph.edges[&(*left_node, *right_node)];
            debug_assert!(terms.is_disjoint(&edge.terms));
            terms.extend(&edge.terms);
        }
    }

    BicliqueAction {
        target,
        terms: terms.into_iter().collect(),
        children: (
            side_definition(base, &graph.left_exts, &graph.left_nodes, &left),
            side_definition(base, &graph.right_exts, &graph.right_nodes, &right),
        ),
        contracted: graph.contracted.clone(),
    }
}

fn side_definition(
    base: TensorId,
    exts: &[Index],
    nodes: &[Term],
    selected: &[(usize, Coefficient)],
) -> TensorDef {
    TensorDef {
        base,
        exts: exts.to_vec(),
        rhs: selected
            .iter()
            .map(|(node, coeff)| {
                let mut term = nodes[*node].clone();
                term.coeff *= coeff;
                term
            })
            .collect(),
    }
}

fn same_rewrite(left: &BicliqueAction, right: &BicliqueAction) -> bool {
    if left.target != right.target
        || left.terms != right.terms
        || left.contracted != right.contracted
    {
        return false;
    }

    let left_children = normalize_children(left.children.clone());
    left_children == normalize_children(right.children.clone())
        || left_children == normalize_children((right.children.1.clone(), right.children.0.clone()))
}

fn normalize_children((mut left, mut right): (TensorDef, TensorDef)) -> (TensorDef, TensorDef) {
    let scale = left.rhs[0].coeff.clone();
    for term in &mut left.rhs {
        term.coeff /= &scale;
    }
    for term in &mut right.rhs {
        term.coeff *= &scale;
    }
    (left, right)
}

pub(crate) fn query(
    state: &State,
    target: DefinitionPosition,
) -> Result<BicliqueSpace, QueryError> {
    let definition = state
        .computation()
        .definitions
        .get(target.0)
        .ok_or(QueryError::DefinitionOutOfBounds { position: target })?;
    let canonical = normalize_definition(state.computation(), definition)
        .map_err(QueryError::Canonicalization)?;
    let graphs = build(canonical);
    let mut candidates = Vec::new();
    let mut rewrites = Vec::new();

    for (graph_position, graph) in graphs.iter().enumerate() {
        for biclique in enumerate_bicliques(graph) {
            let rewrite = make_action(
                target,
                definition.base,
                graph,
                &biclique,
                &vec![true; biclique.left.len()],
                &vec![true; biclique.right.len()],
            );
            if rewrites
                .iter()
                .any(|existing| same_rewrite(existing, &rewrite))
            {
                continue;
            }
            rewrites.push(rewrite);
            candidates.push((graph_position, biclique));
        }
    }

    Ok(BicliqueSpace {
        target,
        base: definition.base,
        graphs,
        candidates,
    })
}

pub(crate) fn apply(state: &mut State, action: BicliqueAction) -> Result<(), StateError> {
    let (left, right) = action.children;
    let (left_coeff, left_ref) = state.add_intermediate(left)?;
    let (right_coeff, right_ref) = state.add_intermediate(right)?;
    let replacement = Term {
        sums: action.contracted,
        coeff: left_coeff * right_coeff,
        factors: vec![left_ref, right_ref],
    };

    state.replace_terms(action.target.0, &action.terms, vec![replacement])
}
