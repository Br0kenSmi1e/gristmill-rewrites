//! Definition-level canonicalization for validated states.

use super::State;
use crate::canon::{CanonError, canon_expr};
use crate::repr::{Computation, Index, IndexId, TensorDef, TensorRef, Term};
use std::collections::BTreeMap;

impl State {
    pub(super) fn canonicalize_definitions(&mut self) -> Result<(), CanonError> {
        for position in 0..self.computation.definitions.len() {
            let canonical = {
                let definition = &self.computation.definitions[position];
                canonicalize_definition(&self.computation, definition)?
            };
            self.computation.definitions[position] = canonical;
        }

        Ok(())
    }
}

pub(super) fn canonicalize_definition(
    computation: &Computation,
    definition: &TensorDef,
) -> Result<TensorDef, CanonError> {
    let mut external_ids = BTreeMap::new();
    let mut exts = Vec::with_capacity(definition.exts.len());
    for (position, external) in definition.exts.iter().enumerate() {
        let id = index_id(position)?;
        external_ids.insert(external.id, id);
        exts.push(Index {
            id,
            range: external.range,
        });
    }

    let mut rhs = Vec::with_capacity(definition.rhs.len());
    for term in &definition.rhs {
        let mut index_ids = external_ids.clone();
        let mut sums = Vec::with_capacity(term.sums.len());
        for (position, sum) in term.sums.iter().enumerate() {
            let position = definition
                .exts
                .len()
                .checked_add(position)
                .ok_or(CanonError::ExhaustedIndexIds)?;
            let id = index_id(position)?;
            index_ids.insert(sum.id, id);
            sums.push(Index {
                id,
                range: sum.range,
            });
        }

        let factors = term
            .factors
            .iter()
            .map(|factor| TensorRef {
                tensor: factor.tensor,
                indices: factor
                    .indices
                    .iter()
                    .map(|index| index_ids[index])
                    .collect(),
            })
            .collect();
        rhs.push(Term {
            sums,
            coeff: term.coeff.clone(),
            factors,
        });
    }

    Ok(TensorDef {
        base: definition.base,
        rhs: canon_expr(computation, &exts, &rhs)?,
        exts,
    })
}

fn index_id(position: usize) -> Result<IndexId, CanonError> {
    u32::try_from(position)
        .map(IndexId)
        .map_err(|_| CanonError::ExhaustedIndexIds)
}
