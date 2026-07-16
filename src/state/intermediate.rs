//! Recognition, reuse, and creation of symbolic intermediates.

use super::{State, StateError, canonicalize::canonicalize_definition};
use crate::repr::{Coefficient, Computation, TensorDef, TensorId, TensorInfo, TensorRef, Term};
use std::collections::BTreeSet;

impl State {
    /// Return a reference to an equivalent intermediate, adding one if needed.
    ///
    /// `candidate.base` is ignored. The returned coefficient and tensor
    /// reference are expressed in the candidate's index scope. Protected
    /// outputs are not eligible for reuse.
    pub(crate) fn add_intermediate(
        &mut self,
        mut candidate: TensorDef,
    ) -> Result<(Coefficient, TensorRef), StateError> {
        if let Some(alias) = trivial_alias(&candidate) {
            return Ok(alias);
        }

        let exts = candidate.exts.clone();
        let mut insertion = None;

        for permutation in permutations(exts.len()) {
            candidate.exts = permutation.iter().map(|&position| exts[position]).collect();
            let canonical = canonicalize_definition(&self.computation, &candidate)
                .map_err(StateError::Canonicalization)?;

            if permutation.iter().copied().eq(0..exts.len()) {
                insertion = Some(canonical.clone());
            }

            for existing in &self.computation.definitions {
                if self.protected_outputs.contains(&existing.base) {
                    continue;
                }
                if canonical.exts != existing.exts {
                    continue;
                }
                if let Some(coeff) = proportional_rhs(&canonical.rhs, &existing.rhs) {
                    return Ok((
                        coeff,
                        TensorRef {
                            tensor: existing.base,
                            indices: candidate.exts.iter().map(|index| index.id).collect(),
                        },
                    ));
                }
            }
        }

        let mut definition = insertion.expect("the identity permutation is always generated");
        let Some(first) = definition.rhs.first() else {
            return Err(StateError::ZeroIntermediate);
        };
        let coeff = first.coeff.clone();
        for term in &mut definition.rhs {
            term.coeff = term.coeff.clone() / coeff.clone();
        }

        let tensor = fresh_tensor_id(&self.computation)?;
        definition.base = tensor;
        self.computation.tensors.insert(
            tensor,
            TensorInfo {
                rank: definition.exts.len(),
                symmetry: Vec::new(),
            },
        );
        self.computation.definitions.push(definition);

        Ok((
            coeff,
            TensorRef {
                tensor,
                indices: exts.iter().map(|index| index.id).collect(),
            },
        ))
    }
}

fn trivial_alias(candidate: &TensorDef) -> Option<(Coefficient, TensorRef)> {
    let [term] = candidate.rhs.as_slice() else {
        return None;
    };
    let [factor] = term.factors.as_slice() else {
        return None;
    };
    if !term.sums.is_empty() || term.coeff == Coefficient::from_integer(0.into()) {
        return None;
    }

    let exts = candidate
        .exts
        .iter()
        .map(|index| index.id)
        .collect::<BTreeSet<_>>();
    let indices = factor.indices.iter().copied().collect::<BTreeSet<_>>();
    if exts.len() != candidate.exts.len()
        || indices.len() != factor.indices.len()
        || exts != indices
    {
        return None;
    }

    Some((term.coeff.clone(), factor.clone()))
}

fn permutations(len: usize) -> Vec<Vec<usize>> {
    fn generate(position: usize, current: &mut [usize], result: &mut Vec<Vec<usize>>) {
        if position == current.len() {
            result.push(current.to_vec());
            return;
        }
        for next in position..current.len() {
            current.swap(position, next);
            generate(position + 1, current, result);
            current.swap(position, next);
        }
    }

    let mut current = (0..len).collect::<Vec<_>>();
    let mut result = Vec::new();
    generate(0, &mut current, &mut result);
    result
}

fn proportional_rhs(candidate: &[Term], existing: &[Term]) -> Option<Coefficient> {
    let (Some(candidate_first), Some(existing_first)) = (candidate.first(), existing.first())
    else {
        return None;
    };
    if candidate.len() != existing.len() {
        return None;
    }

    let coeff = candidate_first.coeff.clone() / existing_first.coeff.clone();
    candidate
        .iter()
        .zip(existing)
        .all(|(candidate, existing)| {
            candidate.sums == existing.sums
                && candidate.factors == existing.factors
                && candidate.coeff == existing.coeff.clone() * coeff.clone()
        })
        .then_some(coeff)
}

fn fresh_tensor_id(computation: &Computation) -> Result<TensorId, StateError> {
    let mut candidate = 0;
    for tensor in computation.tensors.keys() {
        if tensor.0 > candidate {
            break;
        }
        candidate = candidate
            .checked_add(1)
            .ok_or(StateError::ExhaustedTensorIds)?;
    }
    Ok(TensorId(candidate))
}
