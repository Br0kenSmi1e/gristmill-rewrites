//! Floating-point operation counts for complete computations.

use crate::repr::{Computation, Index, RangeId};
use std::collections::BTreeMap;

/// Failure to calculate a logarithmic FLOP count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostError {
    MissingLogSize { range: RangeId },
    InvalidLogSize { range: RangeId },
    ZeroTotalFlops,
}

#[derive(Clone, Copy)]
struct LogProduct {
    value: f64,
    is_one: bool,
}

/// Calculate the natural logarithm of the total FLOP count.
///
/// This follows Python Gristmill's default `get_flop_cost` behavior: tensor
/// multiplications and additions are counted, while multiplication by numeric
/// coefficients, copying, and initialization are ignored.
pub fn log_flops(
    computation: &Computation,
    log_sizes: &BTreeMap<RangeId, f64>,
) -> Result<f64, CostError> {
    let mut total = None;

    for definition in &computation.definitions {
        let external = log_product(&definition.exts, log_sizes)?;

        for term in &definition.rhs {
            let summed = log_product(&term.sums, log_sizes)?;
            let multiplications = term.factors.len().saturating_sub(1);
            let additions = usize::from(!summed.is_one);
            let operations = multiplications + additions;

            if operations > 0 {
                add_log_cost(
                    &mut total,
                    (operations as f64).ln() + external.value + summed.value,
                );
            }
        }

        if definition.rhs.len() > 1 {
            add_log_cost(
                &mut total,
                ((definition.rhs.len() - 1) as f64).ln() + external.value,
            );
        }
    }

    total.ok_or(CostError::ZeroTotalFlops)
}

fn log_product(
    indices: &[Index],
    log_sizes: &BTreeMap<RangeId, f64>,
) -> Result<LogProduct, CostError> {
    let mut value = 0.0;
    let mut is_one = true;

    for index in indices {
        let log_size = *log_sizes
            .get(&index.range)
            .ok_or(CostError::MissingLogSize { range: index.range })?;
        if !log_size.is_finite() || log_size < 0.0 {
            return Err(CostError::InvalidLogSize { range: index.range });
        }
        value += log_size;
        is_one &= log_size == 0.0;
    }

    Ok(LogProduct { value, is_one })
}

fn add_log_cost(total: &mut Option<f64>, cost: f64) {
    *total = Some(match *total {
        Some(current) => logaddexp(current, cost),
        None => cost,
    });
}

fn logaddexp(left: f64, right: f64) -> f64 {
    let maximum = left.max(right);
    maximum + (-(left - right).abs()).exp().ln_1p()
}
