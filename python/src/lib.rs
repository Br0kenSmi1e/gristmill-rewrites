mod error;
mod repr;
mod space;
mod state;

use crate::error::GristmillError;
use pyo3::prelude::*;

#[pymodule]
fn _core(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    repr::register(module)?;
    space::register(module)?;
    state::register(module)?;
    module.add("GristmillError", py.get_type::<GristmillError>())?;
    Ok(())
}
