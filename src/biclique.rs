//! Biclique factorization queries and policy choices.

use crate::{
    action::{ActionQuery, DefinitionPosition, QueryError},
    parenthesize,
    repr::{Index, Term},
    state::State,
};

/// The policy interface for one biclique factorization query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BicliqueSpace {
    target: DefinitionPosition,
}

impl BicliqueSpace {
    pub fn target(&self) -> DefinitionPosition {
        self.target
    }
}

/// One validated biclique factorization choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BicliqueAction {
    target: DefinitionPosition,
}

impl BicliqueAction {
    pub fn target(&self) -> DefinitionPosition {
        self.target
    }
}

#[allow(dead_code)]
#[allow(clippy::type_complexity)]
fn enumerate_splits(
    exts: &[Index],
    term: &Term,
) -> Vec<(Term, Vec<Index>, Term, Vec<Index>, Vec<Index>)> {
    if term.factors.len() < 2 {
        return Vec::new();
    }

    let split_count = 1_usize
        .checked_shl((term.factors.len() - 1) as u32)
        .and_then(|count| count.checked_sub(1))
        .expect("the split count must fit in usize");
    let mut splits = Vec::with_capacity(split_count);

    for choice in 0..split_count {
        let mut left = vec![true];
        left.extend((0..term.factors.len() - 1).map(|position| choice & (1 << position) != 0));
        splits.push(parenthesize::select(exts, term, &left));
    }

    splits
}

pub(crate) fn query(
    _state: &State,
    target: DefinitionPosition,
) -> Result<BicliqueSpace, QueryError> {
    Err(QueryError::Unsupported {
        query: ActionQuery::BicliqueFactor(target),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::{Coefficient, IndexId, RangeId, TensorId, TensorRef};

    const RANGE: RangeId = RangeId(0);
    const A: TensorId = TensorId(0);
    const B: TensorId = TensorId(1);
    const C: TensorId = TensorId(2);

    #[test]
    fn enumerates_each_unordered_factor_partition_once() {
        let term = Term {
            sums: Vec::new(),
            coeff: one(),
            factors: vec![scalar(A), scalar(B), scalar(C)],
        };

        let splits = enumerate_splits(&[], &term);

        assert_eq!(splits.len(), 3);
        assert_eq!(tensors(&splits[0].0), vec![A]);
        assert_eq!(tensors(&splits[0].2), vec![B, C]);
        assert_eq!(tensors(&splits[1].0), vec![A, B]);
        assert_eq!(tensors(&splits[1].2), vec![C]);
        assert_eq!(tensors(&splits[2].0), vec![A, C]);
        assert_eq!(tensors(&splits[2].2), vec![B]);
    }

    #[test]
    fn includes_the_split_of_a_binary_term() {
        let term = Term {
            sums: Vec::new(),
            coeff: one(),
            factors: vec![scalar(A), scalar(B)],
        };

        let splits = enumerate_splits(&[], &term);

        assert_eq!(splits.len(), 1);
        assert_eq!(tensors(&splits[0].0), vec![A]);
        assert_eq!(tensors(&splits[0].2), vec![B]);
    }

    #[test]
    fn separates_child_local_and_contracted_sums() {
        let term = Term {
            sums: vec![index(0), index(1)],
            coeff: Coefficient::from_integer(6.into()),
            factors: vec![tensor(A, &[2, 0]), tensor(B, &[0, 1]), tensor(C, &[1, 3])],
        };

        let splits = enumerate_splits(&[index(2), index(3)], &term);
        let (left, left_exts, right, right_exts, contracted) = &splits[1];

        assert_eq!(left.sums, vec![index(0)]);
        assert!(right.sums.is_empty());
        assert_eq!(left_exts, &vec![index(2), index(1)]);
        assert_eq!(right_exts, &vec![index(3), index(1)]);
        assert_eq!(contracted, &vec![index(1)]);
        assert_eq!(left.coeff, one());
        assert_eq!(right.coeff, one());
    }

    fn scalar(tensor: TensorId) -> TensorRef {
        TensorRef {
            tensor,
            indices: Vec::new(),
        }
    }

    fn tensor(tensor: TensorId, indices: &[u32]) -> TensorRef {
        TensorRef {
            tensor,
            indices: indices.iter().copied().map(IndexId).collect(),
        }
    }

    fn tensors(term: &Term) -> Vec<TensorId> {
        term.factors.iter().map(|factor| factor.tensor).collect()
    }

    fn index(id: u32) -> Index {
        Index {
            id: IndexId(id),
            range: RANGE,
        }
    }

    fn one() -> Coefficient {
        Coefficient::from_integer(1.into())
    }
}
