use gristmill_rewrites::{
    Coefficient, Computation, Index, SymmetryAction, SymmetryGenerator, TensorDef, TensorInfo,
    TensorRef, Term,
};
use pyo3::{
    prelude::*,
    types::{PyAny, PyDict, PyFrozenSet, PyModule, PyTuple},
};

#[pyclass(name = "Index", frozen, eq, skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PyIndex {
    inner: Index,
}

impl From<Index> for PyIndex {
    fn from(inner: Index) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyIndex {
    #[getter]
    fn id(&self) -> u32 {
        self.inner.id.0
    }

    #[getter]
    fn range(&self) -> u32 {
        self.inner.range.0
    }

    fn __repr__(&self) -> String {
        format!(
            "Index(id={}, range={})",
            self.inner.id.0, self.inner.range.0
        )
    }
}

#[pyclass(name = "SymmetryGenerator", frozen, eq, skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PySymmetryGenerator {
    inner: SymmetryGenerator,
}

#[pymethods]
impl PySymmetryGenerator {
    #[getter]
    fn perm<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.perm.iter().copied())
    }

    #[getter]
    fn action(&self) -> &'static str {
        match self.inner.action {
            SymmetryAction::Identity => "Identity",
            SymmetryAction::Negate => "Negate",
        }
    }

    fn __repr__(&self) -> String {
        format!("SymmetryGenerator({:?})", self.inner)
    }
}

#[pyclass(name = "TensorInfo", frozen, eq, skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PyTensorInfo {
    inner: TensorInfo,
}

#[pymethods]
impl PyTensorInfo {
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank
    }

    #[getter]
    fn symmetry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let generators = self
            .inner
            .symmetry
            .iter()
            .cloned()
            .map(|inner| Py::new(py, PySymmetryGenerator { inner }))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, generators)
    }

    fn __repr__(&self) -> String {
        format!("TensorInfo({:?})", self.inner)
    }
}

#[pyclass(name = "TensorRef", frozen, eq, skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PyTensorRef {
    inner: TensorRef,
}

#[pymethods]
impl PyTensorRef {
    #[getter]
    fn tensor(&self) -> u32 {
        self.inner.tensor.0
    }

    #[getter]
    fn indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.indices.iter().map(|index| index.0))
    }

    fn __repr__(&self) -> String {
        format!("TensorRef({:?})", self.inner)
    }
}

#[pyclass(name = "Term", frozen, eq, skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PyTerm {
    pub(crate) inner: Term,
}

impl From<Term> for PyTerm {
    fn from(inner: Term) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTerm {
    #[getter]
    fn sums<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        index_tuple(py, &self.inner.sums)
    }

    #[getter]
    fn coeff<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        coefficient(py, &self.inner.coeff)
    }

    #[getter]
    fn factors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let factors = self
            .inner
            .factors
            .iter()
            .cloned()
            .map(|inner| Py::new(py, PyTensorRef { inner }))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, factors)
    }

    fn __repr__(&self) -> String {
        format!("Term({:?})", self.inner)
    }
}

#[pyclass(name = "TensorDef", frozen, eq, skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PyTensorDef {
    inner: TensorDef,
}

#[pymethods]
impl PyTensorDef {
    #[getter]
    fn base(&self) -> u32 {
        self.inner.base.0
    }

    #[getter]
    fn exts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        index_tuple(py, &self.inner.exts)
    }

    #[getter]
    fn rhs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        term_tuple(py, &self.inner.rhs)
    }

    fn __repr__(&self) -> String {
        format!("TensorDef({:?})", self.inner)
    }
}

#[pyclass(name = "Computation", frozen, eq, skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PyComputation {
    pub(crate) inner: Computation,
}

#[pymethods]
impl PyComputation {
    #[getter]
    fn ranges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        PyFrozenSet::new(py, self.inner.ranges.iter().map(|range| range.0))
    }

    #[getter]
    fn tensors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let tensors = PyDict::new(py);
        for (tensor, info) in &self.inner.tensors {
            tensors.set_item(
                tensor.0,
                Py::new(
                    py,
                    PyTensorInfo {
                        inner: info.clone(),
                    },
                )?,
            )?;
        }
        PyModule::import(py, "types")?
            .getattr("MappingProxyType")?
            .call1((tensors,))
    }

    #[getter]
    fn definitions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let definitions = self
            .inner
            .definitions
            .iter()
            .cloned()
            .map(|inner| Py::new(py, PyTensorDef { inner }))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, definitions)
    }

    fn __repr__(&self) -> String {
        format!(
            "Computation(ranges={}, tensors={}, definitions={})",
            self.inner.ranges.len(),
            self.inner.tensors.len(),
            self.inner.definitions.len()
        )
    }
}

pub(crate) fn index_tuple<'py>(
    py: Python<'py>,
    indices: &[Index],
) -> PyResult<Bound<'py, PyTuple>> {
    let indices = indices
        .iter()
        .copied()
        .map(|inner| Py::new(py, PyIndex { inner }))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, indices)
}

pub(crate) fn term_tuple<'py>(py: Python<'py>, terms: &[Term]) -> PyResult<Bound<'py, PyTuple>> {
    let terms = terms
        .iter()
        .cloned()
        .map(|inner| Py::new(py, PyTerm { inner }))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, terms)
}

pub(crate) fn coefficient<'py>(
    py: Python<'py>,
    coefficient: &Coefficient,
) -> PyResult<Bound<'py, PyAny>> {
    PyModule::import(py, "fractions")?
        .getattr("Fraction")?
        .call1((coefficient.to_string(),))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyIndex>()?;
    module.add_class::<PySymmetryGenerator>()?;
    module.add_class::<PyTensorInfo>()?;
    module.add_class::<PyTensorRef>()?;
    module.add_class::<PyTerm>()?;
    module.add_class::<PyTensorDef>()?;
    module.add_class::<PyComputation>()?;
    Ok(())
}
