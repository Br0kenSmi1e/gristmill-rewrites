//! Canonical normalization of term bipartitions.

use crate::{
    canon::{CanonError, canon_term, compare_term_bodies},
    parenthesize::{TermBipartition, bipartition_term},
    repr::{Coefficient, Computation, Index, IndexId, TensorDef, Term},
};
use std::cmp::Ordering;
use std::collections::BTreeMap;

pub(super) fn normalize_definition(
    computation: &Computation,
    definition: &TensorDef,
) -> Result<Vec<(usize, TermBipartition)>, CanonError> {
    let mut canonical = Vec::new();

    for (source_term, term) in definition.rhs.iter().enumerate() {
        for bipartition in enumerate_bipartitions(&definition.exts, term) {
            let aligned = align_bipartition(&definition.exts, bipartition)?;
            let Some(bipartition) = canon_bipartition(computation, &definition.exts, aligned)?
            else {
                continue;
            };

            canonical.push((source_term, bipartition));
        }
    }

    Ok(canonical)
}

pub(super) fn enumerate_bipartitions(exts: &[Index], term: &Term) -> Vec<TermBipartition> {
    if term.factors.len() < 2 {
        return Vec::new();
    }

    let orientation_count = 1_usize
        .checked_shl(term.factors.len() as u32)
        .expect("the bipartition count must fit in usize");
    let mut bipartitions = Vec::with_capacity(orientation_count - 2);

    for choice in 1..orientation_count - 1 {
        let left = (0..term.factors.len())
            .map(|position| choice & (1 << position) != 0)
            .collect::<Vec<_>>();
        bipartitions.push(bipartition_term(exts, term, &left));
    }

    bipartitions
}

pub(super) fn align_bipartition(
    definition_exts: &[Index],
    bipartition: TermBipartition,
) -> Result<TermBipartition, CanonError> {
    let external_ids = definition_exts
        .iter()
        .map(|index| index.id)
        .collect::<Vec<_>>();
    let contracted_ids = canonical_index_ids(&external_ids, bipartition.contracted.len())?;
    let contracted_renames = bipartition
        .contracted
        .iter()
        .zip(&contracted_ids)
        .map(|(old, new)| (old.id, *new))
        .collect::<BTreeMap<_, _>>();
    let fixed_ids = external_ids
        .iter()
        .chain(&contracted_ids)
        .copied()
        .collect::<Vec<_>>();

    let left_sum_ids = canonical_index_ids(&fixed_ids, bipartition.left.sums.len())?;
    let mut left_renames = contracted_renames.clone();
    left_renames.extend(
        bipartition
            .left
            .sums
            .iter()
            .zip(left_sum_ids)
            .map(|(old, new)| (old.id, new)),
    );

    let right_sum_ids = canonical_index_ids(&fixed_ids, bipartition.right.sums.len())?;
    let mut right_renames = contracted_renames.clone();
    right_renames.extend(
        bipartition
            .right
            .sums
            .iter()
            .zip(right_sum_ids)
            .map(|(old, new)| (old.id, new)),
    );

    let TermBipartition {
        coeff,
        left,
        left_exts,
        right,
        right_exts,
        contracted,
    } = bipartition;

    Ok(TermBipartition {
        coeff,
        left: rename_term(left, &left_renames),
        left_exts: rename_indices(left_exts, &contracted_renames),
        right: rename_term(right, &right_renames),
        right_exts: rename_indices(right_exts, &contracted_renames),
        contracted: rename_indices(contracted, &contracted_renames),
    })
}

fn canonical_index_ids(reserved: &[IndexId], count: usize) -> Result<Vec<IndexId>, CanonError> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let mut ids = Vec::with_capacity(count);
    for value in 0..=u32::MAX {
        let id = IndexId(value);
        if !reserved.contains(&id) {
            ids.push(id);
            if ids.len() == count {
                return Ok(ids);
            }
        }
    }

    Err(CanonError::ExhaustedIndexIds)
}

fn rename_term(mut term: Term, renames: &BTreeMap<IndexId, IndexId>) -> Term {
    term.sums = rename_indices(term.sums, renames);

    for factor in &mut term.factors {
        for index in &mut factor.indices {
            if let Some(&renamed) = renames.get(index) {
                *index = renamed;
            }
        }
    }

    term
}

fn rename_indices(mut indices: Vec<Index>, renames: &BTreeMap<IndexId, IndexId>) -> Vec<Index> {
    for index in &mut indices {
        if let Some(&renamed) = renames.get(&index.id) {
            index.id = renamed;
        }
    }

    indices
}

pub(super) fn canon_bipartition(
    computation: &Computation,
    definition_exts: &[Index],
    aligned: TermBipartition,
) -> Result<Option<TermBipartition>, CanonError> {
    let fixed = definition_exts
        .iter()
        .chain(&aligned.contracted)
        .copied()
        .collect::<Vec<_>>();
    let mut best: Option<TermBipartition> = None;
    let mut coeff_conflict = false;

    for permutation in contracted_permutations(&aligned.contracted) {
        let renames = permutation
            .iter()
            .zip(&aligned.contracted)
            .map(|(&source, target)| (aligned.contracted[source].id, target.id))
            .collect::<BTreeMap<_, _>>();
        let left = rename_term(aligned.left.clone(), &renames);
        let right = rename_term(aligned.right.clone(), &renames);
        let Some(mut left) = canon_term(computation, &fixed, &left)? else {
            return Ok(None);
        };
        let Some(mut right) = canon_term(computation, &fixed, &right)? else {
            return Ok(None);
        };

        let coeff = &aligned.coeff * &left.coeff * &right.coeff;
        left.coeff = Coefficient::from_integer(1.into());
        right.coeff = Coefficient::from_integer(1.into());
        let candidate = TermBipartition {
            coeff,
            left,
            right,
            ..aligned.clone()
        };

        match best.as_ref().map(|best| {
            (
                compare_term_bodies(&candidate.left, &best.left)
                    .then_with(|| compare_term_bodies(&candidate.right, &best.right)),
                candidate.coeff != best.coeff,
            )
        }) {
            None | Some((Ordering::Less, _)) => {
                best = Some(candidate);
                coeff_conflict = false;
            }
            Some((Ordering::Equal, conflict)) => coeff_conflict |= conflict,
            Some((Ordering::Greater, _)) => {}
        }
    }

    if coeff_conflict {
        return Ok(None);
    }

    Ok(Some(best.expect(
        "the identity contracted permutation is always generated",
    )))
}

fn contracted_permutations(contracted: &[Index]) -> Vec<Vec<usize>> {
    fn generate(
        contracted: &[Index],
        position: usize,
        current: &mut [usize],
        result: &mut Vec<Vec<usize>>,
    ) {
        if position == current.len() {
            result.push(current.to_vec());
            return;
        }

        for next in position..current.len() {
            let source = current[next];
            if contracted[source].range != contracted[position].range {
                continue;
            }

            current.swap(position, next);
            generate(contracted, position + 1, current, result);
            current.swap(position, next);
        }
    }

    let mut current = (0..contracted.len()).collect::<Vec<_>>();
    let mut result = Vec::new();
    generate(contracted, 0, &mut current, &mut result);
    result
}
