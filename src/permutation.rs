//! Permutation factorization queries and policy choices.

use crate::{
    action::{Action, DefinitionPosition, QueryError},
    repr::{Coefficient, Index, IndexId, TensorDef, TensorId, TensorRef, Term},
    state::{State, StateError},
};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, btree_map::Entry},
};

/// One owned semantic permutation-factorization family.
pub type PermutationSnapshot = (Vec<Index>, Vec<Term>, Vec<(Vec<usize>, Coefficient)>);

/// The policy interface for one permutation factorization query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermutationSpace {
    target: DefinitionPosition,
    base: TensorId,
    families: Vec<PermutationFamily>,
}

impl PermutationSpace {
    pub fn target(&self) -> DefinitionPosition {
        self.target
    }

    /// Return owned semantic descriptions of the maximal factorization families.
    ///
    /// Each tuple contains the intermediate external indices, its possible root
    /// terms, and the relative permutation uses with their normalized
    /// coefficients. Root and use order matches the masks accepted by
    /// [`Self::select`].
    pub fn snapshot(&self) -> Vec<PermutationSnapshot> {
        self.families
            .iter()
            .map(|family| {
                (
                    family.exts.clone(),
                    family.roots.clone(),
                    family.uses.clone(),
                )
            })
            .collect()
    }

    pub fn candidate_count(&self) -> usize {
        self.families.len()
    }

    pub fn shape(&self, candidate: usize) -> Option<(usize, usize)> {
        self.families
            .get(candidate)
            .map(|family| (family.roots.len(), family.uses.len()))
    }

    /// Select a subfamily of one maximal permutation factorization.
    pub fn select(
        &self,
        candidate: usize,
        roots: &[bool],
        uses: &[bool],
    ) -> Result<Action, PermutationChoiceError> {
        let Some(family) = self.families.get(candidate) else {
            return Err(PermutationChoiceError::CandidateOutOfBounds {
                index: candidate,
                len: self.families.len(),
            });
        };

        if roots.len() != family.roots.len() {
            return Err(PermutationChoiceError::WrongRootMaskLength {
                expected: family.roots.len(),
                got: roots.len(),
            });
        }
        if uses.len() != family.uses.len() {
            return Err(PermutationChoiceError::WrongUseMaskLength {
                expected: family.uses.len(),
                got: uses.len(),
            });
        }
        if roots.iter().filter(|&&selected| selected).count() < 2 {
            return Err(PermutationChoiceError::TooFewRoots);
        }
        if uses.iter().filter(|&&selected| selected).count() < 2 {
            return Err(PermutationChoiceError::TooFewUses);
        }

        Ok(Action::Permutation(make_action(
            self.target,
            self.base,
            family,
            roots,
            uses,
        )))
    }
}

/// An invalid permutation-factorization choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermutationChoiceError {
    CandidateOutOfBounds { index: usize, len: usize },
    WrongRootMaskLength { expected: usize, got: usize },
    WrongUseMaskLength { expected: usize, got: usize },
    TooFewRoots,
    TooFewUses,
}

/// One validated permutation factorization choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermutationAction {
    target: DefinitionPosition,
    terms: Vec<usize>,
    intermediate: TensorDef,
    uses: Vec<(Vec<usize>, Coefficient)>,
}

impl PermutationAction {
    pub fn target(&self) -> DefinitionPosition {
        self.target
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PermutationFamily {
    exts: Vec<Index>,
    roots: Vec<Term>,
    uses: Vec<(Vec<usize>, Coefficient)>,
    /// Original term positions indexed first by root and then by use.
    source_terms: Vec<Vec<Vec<usize>>>,
}

struct Occurrence {
    coeff: Coefficient,
    terms: Vec<usize>,
}

struct RootedTerm {
    root: Term,
    /// Original term positions keyed by their permutation and coefficient
    /// relative to `root`.
    occurrences: BTreeMap<(Vec<usize>, Coefficient), Vec<usize>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SearchNode {
    Root(usize),
    Use(usize),
}

pub(crate) fn query(
    state: &State,
    target: DefinitionPosition,
) -> Result<PermutationSpace, QueryError> {
    let definition = state
        .computation()
        .definitions
        .get(target.0)
        .ok_or(QueryError::DefinitionOutOfBounds { position: target })?;
    let external = definition
        .exts
        .iter()
        .map(|index| index.id)
        .collect::<BTreeSet<_>>();
    let mut groups =
        BTreeMap::<Vec<Index>, BTreeMap<TensorId, BTreeMap<Vec<IndexId>, Occurrence>>>::new();

    for (position, term) in definition.rhs.iter().enumerate() {
        let [factor] = term.factors.as_slice() else {
            continue;
        };
        let used = factor.indices.iter().copied().collect::<BTreeSet<_>>();
        if !term.sums.is_empty()
            || used.len() != factor.indices.len()
            || used.len() < 2
            || !used.is_subset(&external)
        {
            continue;
        }

        let exts = definition
            .exts
            .iter()
            .filter(|index| used.contains(&index.id))
            .copied()
            .collect::<Vec<_>>();
        let occurrence = groups
            .entry(exts)
            .or_default()
            .entry(factor.tensor)
            .or_default()
            .entry(factor.indices.clone())
            .or_insert_with(|| Occurrence {
                coeff: Coefficient::from_integer(0.into()),
                terms: Vec::new(),
            });
        occurrence.coeff += &term.coeff;
        occurrence.terms.push(position);
    }

    let mut families = BTreeMap::<Vec<usize>, PermutationFamily>::new();
    for (exts, mut rows) in groups {
        for occurrences in rows.values_mut() {
            occurrences
                .retain(|_, occurrence| occurrence.coeff != Coefficient::from_integer(0.into()));
        }

        let rooted = rooted_terms(&exts, &rows);
        for (covered, family) in maximal_families(exts.clone(), &rooted) {
            match families.entry(covered) {
                Entry::Vacant(entry) => {
                    entry.insert(family);
                }
                Entry::Occupied(mut entry)
                    if compare_families(&family, entry.get()) == Ordering::Less =>
                {
                    entry.insert(family);
                }
                Entry::Occupied(_) => {}
            }
        }
    }

    Ok(PermutationSpace {
        target,
        base: definition.base,
        families: families.into_values().collect(),
    })
}

fn rooted_terms(
    exts: &[Index],
    rows: &BTreeMap<TensorId, BTreeMap<Vec<IndexId>, Occurrence>>,
) -> Vec<RootedTerm> {
    let mut rooted = Vec::new();

    for (&tensor, occurrences) in rows {
        for (pivot, root) in occurrences {
            let mut relative = BTreeMap::new();
            for (indices, occurrence) in occurrences {
                let Some(permutation) = relative_permutation(exts, pivot, indices) else {
                    continue;
                };
                let previous = relative.insert(
                    (permutation, &occurrence.coeff / &root.coeff),
                    occurrence.terms.clone(),
                );
                debug_assert!(previous.is_none());
            }

            if relative.len() < 2 {
                continue;
            }
            rooted.push(RootedTerm {
                root: Term {
                    sums: Vec::new(),
                    coeff: root.coeff.clone(),
                    factors: vec![TensorRef {
                        tensor,
                        indices: pivot.clone(),
                    }],
                },
                occurrences: relative,
            });
        }
    }

    rooted
}

fn relative_permutation(
    exts: &[Index],
    pivot: &[IndexId],
    occurrence: &[IndexId],
) -> Option<Vec<usize>> {
    if pivot.len() != exts.len() || occurrence.len() != exts.len() {
        return None;
    }

    let positions = exts
        .iter()
        .enumerate()
        .map(|(position, index)| (index.id, position))
        .collect::<BTreeMap<_, _>>();
    let mut permutation = vec![usize::MAX; exts.len()];
    let mut images = BTreeSet::new();

    for (&source, &target) in pivot.iter().zip(occurrence) {
        let (&source, &target) = (positions.get(&source)?, positions.get(&target)?);
        if permutation[source] != usize::MAX || exts[source].range != exts[target].range {
            return None;
        }
        permutation[source] = target;
        images.insert(target);
    }

    (images.len() == exts.len()).then_some(permutation)
}

fn maximal_families(
    exts: Vec<Index>,
    rooted: &[RootedTerm],
) -> Vec<(Vec<usize>, PermutationFamily)> {
    if rooted.len() < 2 {
        return Vec::new();
    }

    let uses = rooted
        .iter()
        .flat_map(|root| root.occurrences.keys().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if uses.len() < 2 {
        return Vec::new();
    }

    // A clique contains mutually compatible root terms and permutation uses.
    // Root-root compatibility keeps one orientation per tensor, use-use
    // compatibility is unconditional, and every root-use edge witnesses one
    // original expression term with the required coefficient.
    let nodes = (0..rooted.len())
        .map(SearchNode::Root)
        .chain((0..uses.len()).map(SearchNode::Use))
        .collect::<Vec<_>>();
    let mut neighbors = nodes
        .iter()
        .copied()
        .map(|node| (node, BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();

    for (position, &left) in nodes.iter().enumerate() {
        for &right in &nodes[position + 1..] {
            if compatible(left, right, rooted, &uses) {
                neighbors.get_mut(&left).unwrap().insert(right);
                neighbors.get_mut(&right).unwrap().insert(left);
            }
        }
    }

    maximal_cliques(&neighbors)
        .into_iter()
        .filter_map(|clique| make_family(&exts, rooted, &uses, &clique))
        .collect()
}

fn compatible(
    left: SearchNode,
    right: SearchNode,
    rooted: &[RootedTerm],
    uses: &[(Vec<usize>, Coefficient)],
) -> bool {
    match (left, right) {
        (SearchNode::Root(left), SearchNode::Root(right)) => {
            rooted[left].root.factors[0].tensor != rooted[right].root.factors[0].tensor
        }
        (SearchNode::Use(_), SearchNode::Use(_)) => true,
        (SearchNode::Root(root), SearchNode::Use(use_))
        | (SearchNode::Use(use_), SearchNode::Root(root)) => {
            rooted[root].occurrences.contains_key(&uses[use_])
        }
    }
}

fn maximal_cliques(neighbors: &BTreeMap<SearchNode, BTreeSet<SearchNode>>) -> Vec<Vec<SearchNode>> {
    let mut clique = Vec::new();
    let candidates = neighbors.keys().copied().collect();
    let mut results = Vec::new();
    expand(
        neighbors,
        &mut clique,
        candidates,
        BTreeSet::new(),
        &mut results,
    );
    results
}

fn expand(
    neighbors: &BTreeMap<SearchNode, BTreeSet<SearchNode>>,
    clique: &mut Vec<SearchNode>,
    mut candidates: BTreeSet<SearchNode>,
    mut excluded: BTreeSet<SearchNode>,
    results: &mut Vec<Vec<SearchNode>>,
) {
    if candidates.is_empty() && excluded.is_empty() {
        results.push(clique.clone());
        return;
    }

    let pivot = candidates
        .union(&excluded)
        .copied()
        .max_by_key(|node| candidates.intersection(&neighbors[node]).count());
    let current = pivot.map_or_else(
        || candidates.iter().copied().collect::<Vec<_>>(),
        |pivot| candidates.difference(&neighbors[&pivot]).copied().collect(),
    );

    for node in current {
        let adjacent = &neighbors[&node];
        clique.push(node);
        expand(
            neighbors,
            clique,
            candidates.intersection(adjacent).copied().collect(),
            excluded.intersection(adjacent).copied().collect(),
            results,
        );
        clique.pop();
        candidates.remove(&node);
        excluded.insert(node);
    }
}

fn make_family(
    exts: &[Index],
    rooted: &[RootedTerm],
    uses: &[(Vec<usize>, Coefficient)],
    clique: &[SearchNode],
) -> Option<(Vec<usize>, PermutationFamily)> {
    let mut roots = clique
        .iter()
        .filter_map(|node| match node {
            SearchNode::Root(root) => Some(*root),
            SearchNode::Use(_) => None,
        })
        .collect::<Vec<_>>();
    let mut selected_uses = clique
        .iter()
        .filter_map(|node| match node {
            SearchNode::Root(_) => None,
            SearchNode::Use(use_) => Some(*use_),
        })
        .collect::<Vec<_>>();
    roots.sort_unstable();
    selected_uses.sort_unstable();
    if roots.len() < 2 || selected_uses.len() < 2 {
        return None;
    }

    let mut covered = BTreeSet::new();
    let mut source_terms = Vec::new();
    for &root in &roots {
        let mut row = Vec::new();
        for &use_ in &selected_uses {
            let positions = rooted[root].occurrences.get(&uses[use_])?.clone();
            for &position in &positions {
                if !covered.insert(position) {
                    return None;
                }
            }
            row.push(positions);
        }
        source_terms.push(row);
    }

    Some((
        covered.into_iter().collect(),
        PermutationFamily {
            exts: exts.to_vec(),
            roots: roots
                .into_iter()
                .map(|root| rooted[root].root.clone())
                .collect(),
            uses: selected_uses
                .into_iter()
                .map(|use_| uses[use_].clone())
                .collect(),
            source_terms,
        },
    ))
}

fn compare_families(left: &PermutationFamily, right: &PermutationFamily) -> Ordering {
    for (left, right) in left.roots.iter().zip(&right.roots) {
        let left_factor = &left.factors[0];
        let right_factor = &right.factors[0];
        let ordering = left_factor
            .tensor
            .cmp(&right_factor.tensor)
            .then_with(|| left_factor.indices.cmp(&right_factor.indices))
            .then_with(|| left.coeff.cmp(&right.coeff));
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.uses.cmp(&right.uses)
}

fn make_action(
    target: DefinitionPosition,
    base: TensorId,
    family: &PermutationFamily,
    roots: &[bool],
    uses: &[bool],
) -> PermutationAction {
    let mut terms = BTreeSet::new();
    for (root, &selected_root) in family.source_terms.iter().zip(roots) {
        if !selected_root {
            continue;
        }
        for (positions, &selected_use) in root.iter().zip(uses) {
            if selected_use {
                debug_assert!(positions.iter().all(|position| !terms.contains(position)));
                terms.extend(positions);
            }
        }
    }

    PermutationAction {
        target,
        terms: terms.into_iter().collect(),
        intermediate: TensorDef {
            base,
            exts: family.exts.clone(),
            rhs: family
                .roots
                .iter()
                .zip(roots)
                .filter(|(_, selected)| **selected)
                .map(|(root, _)| root.clone())
                .collect(),
        },
        uses: family
            .uses
            .iter()
            .zip(uses)
            .filter(|(_, selected)| **selected)
            .map(|(use_, _)| use_.clone())
            .collect(),
    }
}

pub(crate) fn apply(state: &mut State, action: PermutationAction) -> Result<(), StateError> {
    let exts = action
        .intermediate
        .exts
        .iter()
        .map(|index| index.id)
        .collect::<Vec<_>>();
    let (intermediate_coeff, intermediate_ref) = state.add_intermediate(action.intermediate)?;
    let replacements = action
        .uses
        .into_iter()
        .map(|(permutation, coefficient)| {
            let substitution = exts
                .iter()
                .copied()
                .enumerate()
                .map(|(position, index)| (index, exts[permutation[position]]))
                .collect::<BTreeMap<_, _>>();
            Term {
                sums: Vec::new(),
                coeff: &intermediate_coeff * coefficient,
                factors: vec![TensorRef {
                    tensor: intermediate_ref.tensor,
                    indices: intermediate_ref
                        .indices
                        .iter()
                        .map(|index| substitution[index])
                        .collect(),
                }],
            }
        })
        .collect();

    state.replace_terms(action.target.0, &action.terms, replacements)
}
