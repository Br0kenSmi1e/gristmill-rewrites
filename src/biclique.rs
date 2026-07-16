//! Biclique factorization queries and policy choices.

use crate::{
    action::{Action, DefinitionPosition, QueryError},
    canon::{CanonError, canon_term},
    parenthesize::{TermBipartition, bipartition_term},
    repr::{Coefficient, Computation, Index, IndexId, TensorDef, TensorId, Term},
    state::{State, StateError},
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

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

fn enumerate_splits(exts: &[Index], term: &Term) -> Vec<TermBipartition> {
    if term.factors.len() < 2 {
        return Vec::new();
    }

    let split_count = 1_usize
        .checked_shl((term.factors.len() - 1) as u32)
        .and_then(|count| count.checked_sub(1))
        .expect("the split count must fit in usize");
    let mut splits = Vec::with_capacity(split_count);

    for choice in 0..split_count {
        let mut left = vec![true];
        left.extend((0..term.factors.len() - 1).map(|position| choice & (1 << position) != 0));
        splits.push(bipartition_term(exts, term, &left));
    }

    splits
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Owner {
    Left,
    Right,
}

impl Owner {
    fn opposite(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

fn canon_split(
    computation: &Computation,
    definition_exts: &[Index],
    split: TermBipartition,
) -> Result<Option<(TermBipartition, TermBipartition)>, CanonError> {
    let TermBipartition {
        coeff,
        left,
        left_exts,
        right,
        right_exts,
        contracted,
    } = split;
    let mut candidates = Vec::new();
    let mut zero = false;

    for order in contracted_orders(&contracted) {
        let scope = definition_exts
            .iter()
            .chain(&order)
            .copied()
            .collect::<Vec<_>>();
        let (fixed, aligned_left) = align_term(&scope, &left)?;
        let (_, aligned_right) = align_term(&scope, &right)?;
        let Some(mut canonical_left) = canon_term(computation, &fixed, &aligned_left)? else {
            zero = true;
            continue;
        };
        let Some(mut canonical_right) = canon_term(computation, &fixed, &aligned_right)? else {
            zero = true;
            continue;
        };

        canonical_left.coeff *= canonical_right.coeff.clone();
        canonical_right.coeff = Coefficient::from_integer(1.into());

        candidates.push(TermBipartition {
            coeff: coeff.clone(),
            left: canonical_left,
            left_exts: align_exts(&scope, &fixed, &left_exts)?,
            right: canonical_right,
            right_exts: align_exts(&scope, &fixed, &right_exts)?,
            contracted: fixed[definition_exts.len()..].to_vec(),
        });
    }

    if zero || candidates.is_empty() {
        return Ok(None);
    }

    let Some(left_owned) = choose_candidate(&candidates, Owner::Left) else {
        return Ok(None);
    };
    let Some(right_owned) = choose_candidate(&candidates, Owner::Right) else {
        return Ok(None);
    };

    Ok(Some((left_owned, right_owned)))
}

fn contracted_orders(contracted: &[Index]) -> Vec<Vec<Index>> {
    fn generate(
        contracted: &[Index],
        ranges: &[crate::repr::RangeId],
        position: usize,
        used: &mut [bool],
        current: &mut Vec<Index>,
        result: &mut Vec<Vec<Index>>,
    ) {
        if position == contracted.len() {
            result.push(current.clone());
            return;
        }

        for index in 0..contracted.len() {
            if used[index] || contracted[index].range != ranges[position] {
                continue;
            }
            used[index] = true;
            current.push(contracted[index]);
            generate(contracted, ranges, position + 1, used, current, result);
            current.pop();
            used[index] = false;
        }
    }

    let mut contracted = contracted.to_vec();
    contracted.sort_by_key(|index| (index.range, index.id));
    let ranges = contracted
        .iter()
        .map(|index| index.range)
        .collect::<Vec<_>>();
    let mut result = Vec::new();
    generate(
        &contracted,
        &ranges,
        0,
        &mut vec![false; contracted.len()],
        &mut Vec::with_capacity(contracted.len()),
        &mut result,
    );
    result
}

fn align_term(scope: &[Index], term: &Term) -> Result<(Vec<Index>, Term), CanonError> {
    let mut ids = BTreeMap::new();
    let mut fixed = Vec::with_capacity(scope.len());
    for (position, index) in scope.iter().enumerate() {
        let id = index_id(position)?;
        if ids.insert(index.id, id).is_some() {
            return Err(CanonError::DuplicateFixedIndex { index: index.id });
        }
        fixed.push(Index {
            id,
            range: index.range,
        });
    }

    let mut sums = Vec::with_capacity(term.sums.len());
    let mut sum_ids = BTreeSet::new();
    for (position, sum) in term.sums.iter().enumerate() {
        if !sum_ids.insert(sum.id) {
            return Err(CanonError::DuplicateSummedIndex { index: sum.id });
        }
        if ids.contains_key(&sum.id) {
            return Err(CanonError::FixedAndSummedIndexOverlap { index: sum.id });
        }
        let position = scope
            .len()
            .checked_add(position)
            .ok_or(CanonError::ExhaustedIndexIds)?;
        let id = index_id(position)?;
        ids.insert(sum.id, id);
        sums.push(Index {
            id,
            range: sum.range,
        });
    }

    let factors = term
        .factors
        .iter()
        .map(|factor| {
            let indices = factor
                .indices
                .iter()
                .map(|index| {
                    ids.get(index)
                        .copied()
                        .ok_or(CanonError::UnknownIndex { index: *index })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(crate::repr::TensorRef {
                tensor: factor.tensor,
                indices,
            })
        })
        .collect::<Result<Vec<_>, CanonError>>()?;

    Ok((
        fixed,
        Term {
            sums,
            coeff: term.coeff.clone(),
            factors,
        },
    ))
}

fn align_exts(
    scope: &[Index],
    fixed: &[Index],
    selected: &[Index],
) -> Result<Vec<Index>, CanonError> {
    if let Some(index) = selected
        .iter()
        .find(|selected| !scope.iter().any(|index| index.id == selected.id))
    {
        return Err(CanonError::UnknownIndex { index: index.id });
    }

    let selected = selected
        .iter()
        .map(|index| index.id)
        .collect::<BTreeSet<_>>();
    Ok(scope
        .iter()
        .zip(fixed)
        .filter_map(|(original, aligned)| selected.contains(&original.id).then_some(*aligned))
        .collect())
}

fn index_id(position: usize) -> Result<IndexId, CanonError> {
    u32::try_from(position)
        .map(IndexId)
        .map_err(|_| CanonError::ExhaustedIndexIds)
}

fn choose_candidate(candidates: &[TermBipartition], owner: Owner) -> Option<TermBipartition> {
    let mut best = 0;
    for candidate in 1..candidates.len() {
        if compare_candidate(&candidates[candidate], &candidates[best], owner) == Ordering::Less {
            best = candidate;
        }
    }

    if candidates.iter().any(|candidate| {
        compare_candidate(candidate, &candidates[best], owner) == Ordering::Equal
            && candidate.left.coeff != candidates[best].left.coeff
    }) {
        None
    } else {
        Some(candidates[best].clone())
    }
}

fn compare_candidate(left: &TermBipartition, right: &TermBipartition, owner: Owner) -> Ordering {
    match owner {
        Owner::Left => compare_term(&left.left, &right.left)
            .then_with(|| compare_term(&left.right, &right.right)),
        Owner::Right => compare_term(&left.right, &right.right)
            .then_with(|| compare_term(&left.left, &right.left)),
    }
}

fn compare_term(left: &Term, right: &Term) -> Ordering {
    left.sums
        .cmp(&right.sums)
        .then_with(|| left.factors.cmp(&right.factors))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Edge {
    coeff: Coefficient,
    terms: BTreeSet<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Graph {
    left_exts: Vec<Index>,
    right_exts: Vec<Index>,
    contracted: Vec<Index>,
    left_nodes: Vec<Term>,
    right_nodes: Vec<Term>,
    edges: BTreeMap<(usize, usize), Edge>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Biclique {
    left: Vec<(usize, Coefficient)>,
    right: Vec<(usize, Coefficient)>,
    terms: BTreeSet<usize>,
}

type GraphKey = (Owner, Vec<Index>, Vec<Index>, Vec<Index>);

fn build_graphs(
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

fn enumerate_bicliques(graph: &Graph) -> Vec<Biclique> {
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

fn zero() -> Coefficient {
    Coefficient::from_integer(0.into())
}

fn one() -> Coefficient {
    Coefficient::from_integer(1.into())
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
    let graphs =
        build_graphs(state.computation(), definition).map_err(QueryError::Canonicalization)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::{Coefficient, IndexId, RangeId, TensorId, TensorInfo, TensorRef};

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
}
