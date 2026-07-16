//! Canonical normalization of term bipartitions.

use crate::{
    canon::{CanonError, canon_term},
    parenthesize::{TermBipartition, bipartition_term},
    repr::{Coefficient, Computation, Index, IndexId, Term},
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn enumerate_splits(exts: &[Index], term: &Term) -> Vec<TermBipartition> {
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
pub(super) enum Owner {
    Left,
    Right,
}

impl Owner {
    pub(super) fn opposite(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

pub(super) fn canon_split(
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
