//! Biclique factorization queries and policy choices.

use crate::{
    action::{ActionQuery, DefinitionPosition, QueryError},
    canon::{CanonError, canon_term},
    parenthesize,
    repr::{Coefficient, Computation, Index, IndexId, Term},
    state::State,
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// The policy interface for one biclique factorization query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BicliqueSpace {
    target: DefinitionPosition,
}

impl BicliqueSpace {
    pub fn target(&self) -> DefinitionPosition {
        self.target
    }
}

/// One validated biclique factorization choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BicliqueAction {
    target: DefinitionPosition,
}

impl BicliqueAction {
    pub fn target(&self) -> DefinitionPosition {
        self.target
    }
}

#[allow(dead_code)]
#[allow(clippy::type_complexity)]
fn enumerate_splits(
    exts: &[Index],
    term: &Term,
) -> Vec<(Term, Vec<Index>, Term, Vec<Index>, Vec<Index>)> {
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
        splits.push(parenthesize::select(exts, term, &left));
    }

    splits
}

#[derive(Clone, Copy)]
enum Owner {
    Left,
    Right,
}

#[allow(dead_code)]
#[allow(clippy::type_complexity)]
fn canon_split(
    computation: &Computation,
    definition_exts: &[Index],
    split: (Term, Vec<Index>, Term, Vec<Index>, Vec<Index>),
) -> Result<
    Option<(
        (Term, Vec<Index>, Term, Vec<Index>, Vec<Index>),
        (Term, Vec<Index>, Term, Vec<Index>, Vec<Index>),
    )>,
    CanonError,
> {
    let (left, left_exts, right, right_exts, contracted) = split;
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

        candidates.push((
            canonical_left,
            align_exts(&scope, &fixed, &left_exts)?,
            canonical_right,
            align_exts(&scope, &fixed, &right_exts)?,
            fixed[definition_exts.len()..].to_vec(),
        ));
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

#[allow(clippy::type_complexity)]
fn choose_candidate(
    candidates: &[(Term, Vec<Index>, Term, Vec<Index>, Vec<Index>)],
    owner: Owner,
) -> Option<(Term, Vec<Index>, Term, Vec<Index>, Vec<Index>)> {
    let mut best = 0;
    for candidate in 1..candidates.len() {
        if compare_candidate(&candidates[candidate], &candidates[best], owner) == Ordering::Less {
            best = candidate;
        }
    }

    if candidates.iter().any(|candidate| {
        compare_candidate(candidate, &candidates[best], owner) == Ordering::Equal
            && candidate.0.coeff != candidates[best].0.coeff
    }) {
        None
    } else {
        Some(candidates[best].clone())
    }
}

#[allow(clippy::type_complexity)]
fn compare_candidate(
    left: &(Term, Vec<Index>, Term, Vec<Index>, Vec<Index>),
    right: &(Term, Vec<Index>, Term, Vec<Index>, Vec<Index>),
    owner: Owner,
) -> Ordering {
    match owner {
        Owner::Left => {
            compare_term(&left.0, &right.0).then_with(|| compare_term(&left.2, &right.2))
        }
        Owner::Right => {
            compare_term(&left.2, &right.2).then_with(|| compare_term(&left.0, &right.0))
        }
    }
}

fn compare_term(left: &Term, right: &Term) -> Ordering {
    left.sums
        .cmp(&right.sums)
        .then_with(|| left.factors.cmp(&right.factors))
}

pub(crate) fn query(
    _state: &State,
    target: DefinitionPosition,
) -> Result<BicliqueSpace, QueryError> {
    Err(QueryError::Unsupported {
        query: ActionQuery::BicliqueFactor(target),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::{Coefficient, IndexId, RangeId, TensorId, TensorInfo, TensorRef};

    const RANGE: RangeId = RangeId(0);
    const A: TensorId = TensorId(0);
    const B: TensorId = TensorId(1);
    const C: TensorId = TensorId(2);

    #[test]
    fn enumerates_each_unordered_factor_partition_once() {
        let term = Term {
            sums: Vec::new(),
            coeff: one(),
            factors: vec![scalar(A), scalar(B), scalar(C)],
        };

        let splits = enumerate_splits(&[], &term);

        assert_eq!(splits.len(), 3);
        assert_eq!(tensors(&splits[0].0), vec![A]);
        assert_eq!(tensors(&splits[0].2), vec![B, C]);
        assert_eq!(tensors(&splits[1].0), vec![A, B]);
        assert_eq!(tensors(&splits[1].2), vec![C]);
        assert_eq!(tensors(&splits[2].0), vec![A, C]);
        assert_eq!(tensors(&splits[2].2), vec![B]);
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
        assert_eq!(tensors(&splits[0].0), vec![A]);
        assert_eq!(tensors(&splits[0].2), vec![B]);
    }

    #[test]
    fn separates_child_local_and_contracted_sums() {
        let term = Term {
            sums: vec![index(0), index(1)],
            coeff: Coefficient::from_integer(6.into()),
            factors: vec![tensor(A, &[2, 0]), tensor(B, &[0, 1]), tensor(C, &[1, 3])],
        };

        let splits = enumerate_splits(&[index(2), index(3)], &term);
        let (left, left_exts, right, right_exts, contracted) = &splits[1];

        assert_eq!(left.sums, vec![index(0)]);
        assert!(right.sums.is_empty());
        assert_eq!(left_exts, &vec![index(2), index(1)]);
        assert_eq!(right_exts, &vec![index(3), index(1)]);
        assert_eq!(contracted, &vec![index(1)]);
        assert_eq!(left.coeff, one());
        assert_eq!(right.coeff, one());
    }

    #[test]
    fn canonicalizes_both_owner_forms_when_they_differ() {
        let computation = tensor_computation(&[(A, 2), (B, 2)]);
        let term = Term {
            sums: vec![index(10), index(11)],
            coeff: one(),
            factors: vec![tensor(A, &[10, 11]), tensor(B, &[11, 10])],
        };
        let split = parenthesize::select(&[], &term, &[true, false]);

        let (left_owned, right_owned) = canon_split(&computation, &[], split).unwrap().unwrap();

        assert_eq!(
            left_owned.0.factors[0].indices,
            vec![IndexId(0), IndexId(1)]
        );
        assert_eq!(
            left_owned.2.factors[0].indices,
            vec![IndexId(1), IndexId(0)]
        );
        assert_eq!(
            right_owned.0.factors[0].indices,
            vec![IndexId(1), IndexId(0)]
        );
        assert_eq!(
            right_owned.2.factors[0].indices,
            vec![IndexId(0), IndexId(1)]
        );
        assert_eq!(left_owned.4, vec![index(0), index(1)]);
        assert_eq!(right_owned.4, vec![index(0), index(1)]);
    }

    #[test]
    fn returns_both_owner_forms_when_they_are_the_same() {
        let computation = tensor_computation(&[(A, 2), (B, 2)]);
        let term = Term {
            sums: vec![index(10), index(11)],
            coeff: one(),
            factors: vec![tensor(A, &[10, 11]), tensor(B, &[10, 11])],
        };
        let split = parenthesize::select(&[], &term, &[true, false]);

        let (left_owned, right_owned) = canon_split(&computation, &[], split).unwrap().unwrap();

        assert_eq!(left_owned, right_owned);
        assert_eq!(
            left_owned.0.factors[0].indices,
            vec![IndexId(0), IndexId(1)]
        );
        assert_eq!(
            left_owned.2.factors[0].indices,
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
        let split = parenthesize::select(&definition_exts, &term, &[true, true, false]);

        let (canonical, _) = canon_split(&computation, &definition_exts, split)
            .unwrap()
            .unwrap();

        let (left, left_exts, right, right_exts, contracted) = &canonical;
        assert_eq!(left.sums, vec![index(2)]);
        assert_eq!(left_exts, &vec![index(0), index(1)]);
        assert_eq!(right.sums, Vec::new());
        assert_eq!(right_exts, &vec![index(1)]);
        assert_eq!(contracted, &vec![index(1)]);
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

    fn index(id: u32) -> Index {
        Index {
            id: IndexId(id),
            range: RANGE,
        }
    }

    fn one() -> Coefficient {
        Coefficient::from_integer(1.into())
    }
}
