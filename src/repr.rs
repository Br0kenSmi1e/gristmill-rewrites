//! Symbolic tensor representation.

use num_rational::BigRational;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// An exact scalar coefficient.
pub type Coefficient = BigRational;

/// The integer identity of an index range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RangeId(pub u32);

/// The integer identity of an index within its local scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IndexId(pub u32);

/// The integer identity of a tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TensorId(pub u32);

/// The scalar action associated with a tensor-index permutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymmetryAction {
    Identity,
    Negate,
}

/// One generator of a tensor's permutation symmetry.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymmetryGenerator {
    pub perm: Vec<usize>,
    pub action: SymmetryAction,
}

/// Rank and symmetry metadata for one tensor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInfo {
    pub rank: usize,
    pub symmetry: Vec<SymmetryGenerator>,
}

/// An index declaration in a definition or term.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Index {
    pub id: IndexId,
    pub range: RangeId,
}

/// One indexed occurrence of a tensor.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TensorRef {
    pub tensor: TensorId,
    pub indices: Vec<IndexId>,
}

/// One explicitly summed product term.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Term {
    /// Indices bound by this term. Their identities are local to the term.
    pub sums: Vec<Index>,
    #[serde(with = "crate::io::coefficient")]
    pub coeff: Coefficient,
    pub factors: Vec<TensorRef>,
}

/// A tensor definition with definition-local external indices.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDef {
    pub base: TensorId,
    pub exts: Vec<Index>,
    pub rhs: Vec<Term>,
}

/// An unvalidated symbolic tensor computation.
///
/// Tensor and range labels remain outside this core representation. Adapters
/// are responsible for mapping external names to these integer identities.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Computation {
    pub ranges: BTreeSet<RangeId>,
    pub tensors: BTreeMap<TensorId, TensorInfo>,
    pub definitions: Vec<TensorDef>,
}
