//! Exact equivalence checking for symbolic computations and rewrite states.

use crate::{
    canon::{CanonError, canon_expr},
    repr::{
        Coefficient, Computation, Index, IndexId, RangeId, TensorDef, TensorId, TensorRef, Term,
    },
    state::{State, StateError},
};
use std::collections::{BTreeMap, BTreeSet};

/// A structural or canonicalization failure during equivalence checking.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    InvalidComputation {
        side: &'static str,
        error: StateError,
    },
    ProtectedOutputsMismatch {
        lhs: Vec<TensorId>,
        rhs: Vec<TensorId>,
    },
    TensorMetadataMismatch {
        tensor: TensorId,
    },
    OutputArityMismatch {
        tensor: TensorId,
        lhs: usize,
        rhs: usize,
    },
    OutputRangeMismatch {
        tensor: TensorId,
        position: usize,
        lhs: RangeId,
        rhs: RangeId,
    },
    SourceArityMismatch {
        tensor: TensorId,
        expected: usize,
        got: usize,
    },
    ExhaustedIndexIds,
    Canonicalization(CanonError),
}

impl From<CanonError> for VerifyError {
    fn from(error: CanonError) -> Self {
        Self::Canonicalization(error)
    }
}

/// Validate two computations and compare the requested output tensors exactly.
pub fn equivalent_computations(
    lhs: &Computation,
    rhs: &Computation,
    outputs: &[TensorId],
) -> Result<bool, VerifyError> {
    let lhs = State::new(lhs.clone(), outputs.to_vec())
        .map_err(|error| VerifyError::InvalidComputation { side: "lhs", error })?;
    let rhs = State::new(rhs.clone(), outputs.to_vec())
        .map_err(|error| VerifyError::InvalidComputation { side: "rhs", error })?;
    equivalent_states(&lhs, &rhs)
}

/// Compare the protected outputs of two validated rewrite states exactly.
pub fn equivalent_states(lhs: &State, rhs: &State) -> Result<bool, VerifyError> {
    let lhs_outputs = lhs
        .protected_outputs()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let rhs_outputs = rhs
        .protected_outputs()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if lhs_outputs != rhs_outputs {
        return Err(VerifyError::ProtectedOutputsMismatch {
            lhs: lhs_outputs.into_iter().collect(),
            rhs: rhs_outputs.into_iter().collect(),
        });
    }

    let first_fresh = first_fresh_index(lhs.computation(), rhs.computation());
    let mut lhs_fresh = first_fresh;
    let mut rhs_fresh = first_fresh;
    let lhs_definitions = inline_intermediates(lhs.computation(), &lhs_outputs, &mut lhs_fresh)?;
    let rhs_definitions = inline_intermediates(rhs.computation(), &rhs_outputs, &mut rhs_fresh)?;

    for &output in &lhs_outputs {
        let lhs = lhs_definitions
            .iter()
            .find(|definition| definition.base == output)
            .expect("validated protected outputs always have one definition");
        let rhs = rhs_definitions
            .iter()
            .find(|definition| definition.base == output)
            .expect("validated protected outputs always have one definition");
        compare_output_interfaces(lhs, rhs)?;
    }

    let context = merged_context(
        lhs.computation(),
        rhs.computation(),
        &lhs_definitions,
        &rhs_definitions,
    )?;
    for output in lhs_outputs {
        let lhs = lhs_definitions
            .iter()
            .find(|definition| definition.base == output)
            .expect("validated protected outputs always have one definition");
        let rhs = rhs_definitions
            .iter()
            .find(|definition| definition.base == output)
            .expect("validated protected outputs always have one definition");

        let mut difference = lhs.rhs.clone();
        difference.extend(rhs.rhs.iter().cloned().map(|mut term| {
            term.coeff = -term.coeff;
            term
        }));
        if !canon_expr(&context, &lhs.exts, &difference)?.is_empty() {
            return Ok(false);
        }
    }

    Ok(true)
}

fn merged_context(
    lhs: &Computation,
    rhs: &Computation,
    lhs_definitions: &[TensorDef],
    rhs_definitions: &[TensorDef],
) -> Result<Computation, VerifyError> {
    let lhs_tensors = factor_tensors(lhs_definitions);
    let rhs_tensors = factor_tensors(rhs_definitions);
    let mut tensors = BTreeMap::new();

    for &tensor in lhs_tensors.union(&rhs_tensors) {
        let lhs_info = lhs_tensors.contains(&tensor).then(|| &lhs.tensors[&tensor]);
        let rhs_info = rhs_tensors.contains(&tensor).then(|| &rhs.tensors[&tensor]);
        let info = match (lhs_info, rhs_info) {
            (Some(lhs), Some(rhs)) if lhs != rhs => {
                return Err(VerifyError::TensorMetadataMismatch { tensor });
            }
            (Some(info), _) | (_, Some(info)) => info,
            (None, None) => unreachable!(),
        };
        tensors.insert(tensor, info.clone());
    }

    Ok(Computation {
        ranges: lhs.ranges.union(&rhs.ranges).copied().collect(),
        tensors,
        definitions: Vec::new(),
    })
}

fn factor_tensors(definitions: &[TensorDef]) -> BTreeSet<TensorId> {
    definitions
        .iter()
        .flat_map(|definition| &definition.rhs)
        .flat_map(|term| &term.factors)
        .map(|factor| factor.tensor)
        .collect()
}

fn inline_intermediates(
    computation: &Computation,
    outputs: &BTreeSet<TensorId>,
    fresh: &mut u64,
) -> Result<Vec<TensorDef>, VerifyError> {
    let mut definitions = computation.definitions.clone();

    while let Some(position) = definitions
        .iter()
        .rposition(|definition| !outputs.contains(&definition.base))
    {
        let source = definitions.remove(position);
        for target in &mut definitions {
            *target = inline_source(target, &source, fresh)?;
        }
    }

    Ok(definitions)
}

fn inline_source(
    target: &TensorDef,
    source: &TensorDef,
    fresh: &mut u64,
) -> Result<TensorDef, VerifyError> {
    let mut rhs = Vec::new();
    for term in &target.rhs {
        rhs.extend(inline_source_in_term(term, source, fresh)?);
    }
    Ok(TensorDef {
        base: target.base,
        exts: target.exts.clone(),
        rhs,
    })
}

fn inline_source_in_term(
    term: &Term,
    source: &TensorDef,
    fresh: &mut u64,
) -> Result<Vec<Term>, VerifyError> {
    let mut products = vec![Term {
        sums: term.sums.clone(),
        coeff: term.coeff.clone(),
        factors: Vec::new(),
    }];

    for factor in &term.factors {
        let expansion = if factor.tensor == source.base {
            instantiate_source(factor, source, fresh)?
        } else {
            vec![Term {
                sums: Vec::new(),
                coeff: Coefficient::from_integer(1.into()),
                factors: vec![factor.clone()],
            }]
        };

        let mut next = Vec::with_capacity(products.len() * expansion.len());
        for product in products {
            for part in &expansion {
                let mut product = product.clone();
                product.coeff *= &part.coeff;
                product.sums.extend(&part.sums);
                product.factors.extend(part.factors.iter().cloned());
                next.push(product);
            }
        }
        products = next;
    }

    Ok(products)
}

fn instantiate_source(
    factor: &TensorRef,
    source: &TensorDef,
    fresh: &mut u64,
) -> Result<Vec<Term>, VerifyError> {
    if factor.indices.len() != source.exts.len() {
        return Err(VerifyError::SourceArityMismatch {
            tensor: source.base,
            expected: source.exts.len(),
            got: factor.indices.len(),
        });
    }

    let externals = source
        .exts
        .iter()
        .zip(&factor.indices)
        .map(|(source, target)| (source.id, *target))
        .collect::<BTreeMap<_, _>>();
    let mut expansion = Vec::with_capacity(source.rhs.len());

    for source_term in &source.rhs {
        let mut sums = Vec::with_capacity(source_term.sums.len());
        let mut dummies = BTreeMap::new();
        for sum in &source_term.sums {
            let id = fresh_index(fresh)?;
            dummies.insert(sum.id, id);
            sums.push(Index {
                id,
                range: sum.range,
            });
        }

        let factors = source_term
            .factors
            .iter()
            .map(|factor| TensorRef {
                tensor: factor.tensor,
                indices: factor
                    .indices
                    .iter()
                    .map(|index| {
                        externals
                            .get(index)
                            .or_else(|| dummies.get(index))
                            .copied()
                            .expect("validated source indices are external or summed")
                    })
                    .collect(),
            })
            .collect();
        expansion.push(Term {
            sums,
            coeff: source_term.coeff.clone(),
            factors,
        });
    }

    Ok(expansion)
}

fn compare_output_interfaces(lhs: &TensorDef, rhs: &TensorDef) -> Result<(), VerifyError> {
    if lhs.exts.len() != rhs.exts.len() {
        return Err(VerifyError::OutputArityMismatch {
            tensor: lhs.base,
            lhs: lhs.exts.len(),
            rhs: rhs.exts.len(),
        });
    }
    for (position, (lhs_index, rhs_index)) in lhs.exts.iter().zip(&rhs.exts).enumerate() {
        if lhs_index.range != rhs_index.range {
            return Err(VerifyError::OutputRangeMismatch {
                tensor: lhs.base,
                position,
                lhs: lhs_index.range,
                rhs: rhs_index.range,
            });
        }
    }
    Ok(())
}

fn first_fresh_index(lhs: &Computation, rhs: &Computation) -> u64 {
    lhs.definitions
        .iter()
        .chain(&rhs.definitions)
        .flat_map(|definition| {
            definition
                .exts
                .iter()
                .map(|index| index.id)
                .chain(definition.rhs.iter().flat_map(|term| {
                    term.sums.iter().map(|index| index.id).chain(
                        term.factors
                            .iter()
                            .flat_map(|factor| factor.indices.iter().copied()),
                    )
                }))
        })
        .map(|index| u64::from(index.0) + 1)
        .max()
        .unwrap_or(0)
}

fn fresh_index(next: &mut u64) -> Result<IndexId, VerifyError> {
    let id = u32::try_from(*next)
        .map(IndexId)
        .map_err(|_| VerifyError::ExhaustedIndexIds)?;
    *next += 1;
    Ok(id)
}
