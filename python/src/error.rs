use pyo3::{PyErr, create_exception, exceptions::PyException};
use std::fmt;

create_exception!(
    gristmill_rewrites,
    GristmillError,
    PyException,
    "A gristmill rewrite, validation, or serialization failure."
);

pub(crate) fn debug(error: impl fmt::Debug) -> PyErr {
    GristmillError::new_err(format!("{error:?}"))
}

pub(crate) fn display(error: impl fmt::Display) -> PyErr {
    GristmillError::new_err(error.to_string())
}
