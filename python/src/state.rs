use crate::{
    error::{self, GristmillError},
    repr::PyComputation,
    space::{PyAction, PySpace, SpaceKind, nonnegative_usize},
};
use gristmill_rewrites::{
    self as rust, ActionQuery, ActionSpace, DefinitionPosition, State, TermPosition,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyTuple,
};
use std::{path::PathBuf, sync::Arc};

#[pyclass(name = "State", frozen)]
pub(crate) struct PyState {
    pub(crate) inner: Arc<State>,
}

#[pymethods]
impl PyState {
    #[getter]
    fn computation(&self) -> PyComputation {
        PyComputation {
            inner: self.inner.computation().clone(),
        }
    }

    #[getter]
    fn protected_outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.inner.protected_outputs().iter().map(|tensor| tensor.0),
        )
    }

    #[pyo3(signature = (kind, *target))]
    fn query(&self, kind: &str, target: &Bound<'_, PyTuple>) -> PyResult<PySpace> {
        let inner = match kind {
            "parenthesize" => {
                require_target_arity(target, 2, kind)?;
                let definition = nonnegative_usize(&target.get_item(0)?, "definition")?;
                let term = nonnegative_usize(&target.get_item(1)?, "term")?;
                let query = ActionQuery::Parenthesize(TermPosition {
                    definition: DefinitionPosition(definition),
                    term,
                });
                let ActionSpace::Parenthesize(space) =
                    rust::query(&self.inner, query).map_err(error::debug)?
                else {
                    unreachable!("a parenthesization query returns its typed space")
                };
                SpaceKind::Parenthesize(space)
            }
            "biclique" => {
                require_target_arity(target, 1, kind)?;
                let definition = nonnegative_usize(&target.get_item(0)?, "definition")?;
                let query = ActionQuery::BicliqueFactor(DefinitionPosition(definition));
                let ActionSpace::Biclique(space) =
                    rust::query(&self.inner, query).map_err(error::debug)?
                else {
                    unreachable!("a biclique query returns its typed space")
                };
                SpaceKind::Biclique(space)
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown query kind {kind:?}; expected 'parenthesize' or 'biclique'"
                )));
            }
        };

        Ok(PySpace {
            source: self.inner.clone(),
            inner,
        })
    }

    fn apply(&self, action: &PyAction) -> PyResult<Self> {
        if !Arc::ptr_eq(&self.inner, &action.source) {
            return Err(GristmillError::new_err(
                "action was selected from a different state",
            ));
        }

        let state = rust::apply(&self.inner, action.inner.clone()).map_err(error::debug)?;
        Ok(Self {
            inner: Arc::new(state),
        })
    }
}

fn require_target_arity(target: &Bound<'_, PyTuple>, expected: usize, kind: &str) -> PyResult<()> {
    if target.len() == expected {
        Ok(())
    } else {
        Err(PyTypeError::new_err(format!(
            "{kind} query expects {expected} target argument(s), got {}",
            target.len()
        )))
    }
}

#[pyfunction]
pub(crate) fn from_json(text: &str) -> PyResult<PyState> {
    let state = rust::io::from_json(text).map_err(error::display)?;
    Ok(PyState {
        inner: Arc::new(state),
    })
}

#[pyfunction]
pub(crate) fn read_json(path: PathBuf) -> PyResult<PyState> {
    let state = rust::io::read_json(path).map_err(error::display)?;
    Ok(PyState {
        inner: Arc::new(state),
    })
}

#[pyfunction]
pub(crate) fn to_json(state: &PyState) -> PyResult<String> {
    rust::io::to_json(&state.inner).map_err(error::display)
}

#[pyfunction]
pub(crate) fn write_json(state: &PyState, path: PathBuf) -> PyResult<()> {
    rust::io::write_json(path, &state.inner).map_err(error::display)
}

#[pyfunction]
pub(crate) fn equivalent(lhs: &PyState, rhs: &PyState) -> PyResult<bool> {
    rust::equivalent_states(&lhs.inner, &rhs.inner).map_err(error::debug)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyState>()?;
    module.add_function(wrap_pyfunction!(equivalent, module)?)?;
    module.add_function(wrap_pyfunction!(from_json, module)?)?;
    module.add_function(wrap_pyfunction!(read_json, module)?)?;
    module.add_function(wrap_pyfunction!(to_json, module)?)?;
    module.add_function(wrap_pyfunction!(write_json, module)?)?;
    Ok(())
}
