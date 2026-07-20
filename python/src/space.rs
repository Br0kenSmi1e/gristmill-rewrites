use crate::{
    error,
    repr::{PyTerm, coefficient, index_tuple, term_tuple},
};
use gristmill_rewrites::{
    Action as RustAction, BicliqueSpace as RustBicliqueSpace,
    ParenthesizeSpace as RustParenthesizeSpace, PermutationSpace as RustPermutationSpace, State,
};
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyAny, PyBool, PyTuple},
};
use std::sync::Arc;

#[pyclass(name = "Action", frozen)]
pub(crate) struct PyAction {
    pub(crate) source: Arc<State>,
    pub(crate) inner: RustAction,
}

pub(crate) enum SpaceKind {
    Parenthesize(RustParenthesizeSpace),
    Biclique(RustBicliqueSpace),
    Permutation(RustPermutationSpace),
}

#[pyclass(name = "Space", frozen)]
pub(crate) struct PySpace {
    pub(crate) source: Arc<State>,
    pub(crate) inner: SpaceKind,
}

#[pymethods]
impl PySpace {
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        match &self.inner {
            SpaceKind::Parenthesize(space) => {
                let (exts, term) = space.snapshot();
                (index_tuple(py, &exts)?, Py::new(py, PyTerm::from(term))?).into_pyobject(py)
            }
            SpaceKind::Biclique(space) => {
                let candidates = space
                    .snapshot()
                    .into_iter()
                    .map(|(left_exts, left_terms, right_exts, right_terms)| {
                        (
                            index_tuple(py, &left_exts)?,
                            term_tuple(py, &left_terms)?,
                            index_tuple(py, &right_exts)?,
                            term_tuple(py, &right_terms)?,
                        )
                            .into_pyobject(py)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                PyTuple::new(py, candidates)
            }
            SpaceKind::Permutation(space) => {
                let candidates = space
                    .snapshot()
                    .into_iter()
                    .map(|(exts, roots, uses)| {
                        let uses = uses
                            .into_iter()
                            .map(|(permutation, weight)| {
                                (PyTuple::new(py, permutation)?, coefficient(py, &weight)?)
                                    .into_pyobject(py)
                            })
                            .collect::<PyResult<Vec<_>>>()?;
                        (
                            index_tuple(py, &exts)?,
                            term_tuple(py, &roots)?,
                            PyTuple::new(py, uses)?,
                        )
                            .into_pyobject(py)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                PyTuple::new(py, candidates)
            }
        }
    }

    #[pyo3(signature = (*args))]
    fn select(&self, args: &Bound<'_, PyTuple>) -> PyResult<PyAction> {
        let inner = match &self.inner {
            SpaceKind::Parenthesize(space) => {
                require_arity(args, 1, "parenthesization")?;
                let left = bool_mask(&args.get_item(0)?, "left")?;
                space.select(&left).map_err(error::debug)?
            }
            SpaceKind::Biclique(space) => {
                require_arity(args, 3, "biclique")?;
                let candidate = nonnegative_usize(&args.get_item(0)?, "candidate")?;
                let left = bool_mask(&args.get_item(1)?, "left")?;
                let right = bool_mask(&args.get_item(2)?, "right")?;
                space
                    .select(candidate, &left, &right)
                    .map_err(error::debug)?
            }
            SpaceKind::Permutation(space) => {
                require_arity(args, 3, "permutation")?;
                let candidate = nonnegative_usize(&args.get_item(0)?, "candidate")?;
                let roots = bool_mask(&args.get_item(1)?, "roots")?;
                let uses = bool_mask(&args.get_item(2)?, "uses")?;
                space
                    .select(candidate, &roots, &uses)
                    .map_err(error::debug)?
            }
        };

        Ok(PyAction {
            source: self.source.clone(),
            inner,
        })
    }
}

fn require_arity(args: &Bound<'_, PyTuple>, expected: usize, kind: &str) -> PyResult<()> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(PyTypeError::new_err(format!(
            "{kind} select expects {expected} argument(s), got {}",
            args.len()
        )))
    }
}

fn bool_mask(value: &Bound<'_, PyAny>, field: &str) -> PyResult<Vec<bool>> {
    let mask = value
        .extract::<Vec<Bound<'_, PyAny>>>()
        .map_err(|_| PyTypeError::new_err(format!("{field} must be a sequence of bool values")))?;
    mask.into_iter()
        .enumerate()
        .map(|(position, value)| {
            if !value.is_exact_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(format!(
                    "{field}[{position}] must be bool"
                )));
            }
            value.extract::<bool>()
        })
        .collect()
}

pub(crate) fn nonnegative_usize(value: &Bound<'_, PyAny>, field: &str) -> PyResult<usize> {
    if value.is_exact_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "{field} must be a non-negative integer, not bool"
        )));
    }
    value
        .extract::<usize>()
        .map_err(|_| PyTypeError::new_err(format!("{field} must be a non-negative integer")))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAction>()?;
    module.add_class::<PySpace>()?;
    Ok(())
}
