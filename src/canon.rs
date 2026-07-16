//! Canonicalization of individual tensor product terms.

use crate::repr::{
    Coefficient, Computation, Index, IndexId, SymmetryAction, SymmetryGenerator, TensorId,
    TensorRef, Term,
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
mod tests;

/// A failure to align or allocate index identities during canonicalization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonError {
    DuplicateFixedIndex { index: IndexId },
    DuplicateSummedIndex { index: IndexId },
    FixedAndSummedIndexOverlap { index: IndexId },
    UnknownIndex { index: IndexId },
    ExhaustedIndexIds,
}

/// Canonicalize one tensor product term while preserving its fixed indices.
///
/// Fixed index IDs are left unchanged. Factors are ordered first by tensor
/// identity, all orders among equal tensors and all declared symmetry variants
/// are explored, and summed indices are renamed by first occurrence. The
/// lexicographically smallest candidate is retained.
///
/// `None` denotes a zero term, including one fixed by a sign-negating
/// symmetry.
pub(crate) fn canon_term(
    computation: &Computation,
    fixed_indices: &[Index],
    term: &Term,
) -> Result<Option<Term>, CanonError> {
    let fixed_ids = fixed_indices
        .iter()
        .map(|index| index.id)
        .collect::<BTreeSet<_>>();
    if is_zero(&term.coeff) {
        return Ok(None);
    }

    let mut best = None;
    let mut saw_identity = false;
    let mut saw_negate = false;
    let variants = symmetry_variants(computation, &term.factors);
    let tensor_ids = term
        .factors
        .iter()
        .map(|factor| factor.tensor)
        .collect::<Vec<_>>();
    let factor_orders = best_factor_orders(&tensor_ids);
    for (variant_factors, action) in variants {
        for factor_order in &factor_orders {
            let factors = factor_order
                .iter()
                .map(|&position| variant_factors[position].clone())
                .collect();
            let candidate = rename_dummies(
                Term {
                    sums: term.sums.clone(),
                    coeff: term.coeff.clone(),
                    factors,
                },
                &fixed_ids,
            )?;

            match best
                .as_ref()
                .map(|best| compare_term_bodies(&candidate, best))
            {
                None | Some(Ordering::Less) => {
                    best = Some(candidate);
                    saw_identity = action == SymmetryAction::Identity;
                    saw_negate = action == SymmetryAction::Negate;
                }
                Some(Ordering::Equal) => match action {
                    SymmetryAction::Identity => saw_identity = true,
                    SymmetryAction::Negate => saw_negate = true,
                },
                Some(Ordering::Greater) => {}
            }
        }
    }

    if saw_identity && saw_negate {
        return Ok(None);
    }

    let mut best = best.expect("the identity symmetry and factor ordering always form a candidate");
    if saw_negate {
        best.coeff = -best.coeff;
    }

    Ok(Some(best))
}

/// Canonicalize a linear expression while preserving its fixed indices.
///
/// Terms are canonicalized with [`canon_term`], equal term bodies are merged,
/// zero terms are removed, and the result is sorted deterministically.
pub(crate) fn canon_expr(
    computation: &Computation,
    fixed_indices: &[Index],
    terms: &[Term],
) -> Result<Vec<Term>, CanonError> {
    let mut canonical = Vec::<Term>::new();

    for term in terms {
        let Some(term) = canon_term(computation, fixed_indices, term)? else {
            continue;
        };

        if let Some(existing) = canonical
            .iter_mut()
            .find(|existing| existing.sums == term.sums && existing.factors == term.factors)
        {
            existing.coeff += term.coeff;
        } else {
            canonical.push(term);
        }
    }

    canonical.retain(|term| !is_zero(&term.coeff));
    canonical.sort_by(|left, right| {
        left.sums
            .cmp(&right.sums)
            .then_with(|| left.factors.cmp(&right.factors))
    });

    Ok(canonical)
}

fn factor_variants(
    factor: &TensorRef,
    rank: usize,
    generators: &[SymmetryGenerator],
) -> Vec<(TensorRef, SymmetryAction)> {
    let mut group = vec![((0..rank).collect::<Vec<_>>(), SymmetryAction::Identity)];
    let mut position = 0;
    while position < group.len() {
        let (perm, action) = group[position].clone();
        for generator in generators {
            let next = (
                compose(&perm, &generator.perm),
                combine(action, generator.action),
            );
            if !group.contains(&next) {
                group.push(next);
            }
        }
        position += 1;
    }

    let mut variants = Vec::new();
    for (perm, action) in group {
        let variant = (
            TensorRef {
                tensor: factor.tensor,
                indices: perm
                    .iter()
                    .map(|&position| factor.indices[position])
                    .collect(),
            },
            action,
        );
        if !variants.contains(&variant) {
            variants.push(variant);
        }
    }
    variants
}

/// Compose two preimage permutations, applying `left` and then `right`.
fn compose(left: &[usize], right: &[usize]) -> Vec<usize> {
    right.iter().map(|&position| left[position]).collect()
}

fn combine(left: SymmetryAction, right: SymmetryAction) -> SymmetryAction {
    match (left, right) {
        (SymmetryAction::Identity, action) | (action, SymmetryAction::Identity) => action,
        (SymmetryAction::Negate, SymmetryAction::Negate) => SymmetryAction::Identity,
    }
}

fn best_factor_orders(tensors: &[TensorId]) -> Vec<Vec<usize>> {
    fn generate(
        tensors: &[TensorId],
        position: usize,
        current: &mut [usize],
        result: &mut Vec<Vec<usize>>,
    ) {
        if position == current.len() {
            result.push(current.to_vec());
            return;
        }

        let tensor = tensors[current[position]];
        for next in position..current.len() {
            if tensors[current[next]] != tensor {
                continue;
            }
            current.swap(position, next);
            generate(tensors, position + 1, current, result);
            current.swap(position, next);
        }
    }

    let mut current = (0..tensors.len()).collect::<Vec<_>>();
    current.sort_by_key(|&position| tensors[position]);
    let mut result = Vec::new();
    generate(tensors, 0, &mut current, &mut result);
    result
}

fn symmetry_variants(
    computation: &Computation,
    factors: &[TensorRef],
) -> Vec<(Vec<TensorRef>, SymmetryAction)> {
    let variants_by_factor = factors
        .iter()
        .map(|factor| {
            let tensor = &computation.tensors[&factor.tensor];
            factor_variants(factor, tensor.rank, &tensor.symmetry)
        })
        .collect::<Vec<_>>();

    let mut product = vec![(Vec::new(), SymmetryAction::Identity)];
    for variants in variants_by_factor {
        let mut next = Vec::new();
        for (factors, action) in product {
            for (factor, variant_action) in &variants {
                let mut factors = factors.clone();
                factors.push(factor.clone());
                next.push((factors, combine(action, *variant_action)));
            }
        }
        product = next;
    }

    product
}

fn rename_dummies(mut term: Term, fixed_ids: &BTreeSet<IndexId>) -> Result<Term, CanonError> {
    let sum_ranges = term
        .sums
        .iter()
        .map(|index| (index.id, index.range))
        .collect::<BTreeMap<_, _>>();
    let canonical_ids = canonical_sum_ids(fixed_ids, sum_ranges.len())?;
    let mut renamed = BTreeMap::new();
    let mut sums = Vec::with_capacity(sum_ranges.len());

    for factor in &mut term.factors {
        for index in &mut factor.indices {
            let original = *index;
            let Some(&range) = sum_ranges.get(&original) else {
                continue;
            };
            if let Some(&id) = renamed.get(&original) {
                *index = id;
                continue;
            }

            let id = canonical_ids[renamed.len()];
            renamed.insert(original, id);
            sums.push(Index { id, range });
            *index = id;
        }
    }

    debug_assert_eq!(renamed.len(), sum_ranges.len());
    term.sums = sums;
    Ok(term)
}

fn compare_term_bodies(left: &Term, right: &Term) -> Ordering {
    left.factors
        .iter()
        .map(|factor| factor.tensor)
        .cmp(right.factors.iter().map(|factor| factor.tensor))
        .then_with(|| {
            left.factors
                .iter()
                .flat_map(|factor| factor.indices.iter().copied())
                .cmp(
                    right
                        .factors
                        .iter()
                        .flat_map(|factor| factor.indices.iter().copied()),
                )
        })
        .then_with(|| left.sums.cmp(&right.sums))
}

fn canonical_sum_ids(
    fixed_ids: &BTreeSet<IndexId>,
    count: usize,
) -> Result<Vec<IndexId>, CanonError> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let mut ids = Vec::with_capacity(count);

    for value in 0..=u32::MAX {
        let id = IndexId(value);
        if !fixed_ids.contains(&id) {
            ids.push(id);
            if ids.len() == count {
                return Ok(ids);
            }
        }
    }

    Err(CanonError::ExhaustedIndexIds)
}

fn is_zero(coeff: &Coefficient) -> bool {
    coeff == &Coefficient::from_integer(0.into())
}
