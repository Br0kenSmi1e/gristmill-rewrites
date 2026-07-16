from collections.abc import Mapping, Sequence
from fractions import Fraction
from os import PathLike
from typing import Literal, TypeAlias, final, overload

@final
class Index:
    @property
    def id(self) -> int: ...
    @property
    def range(self) -> int: ...

@final
class SymmetryGenerator:
    @property
    def perm(self) -> tuple[int, ...]: ...
    @property
    def action(self) -> Literal["Identity", "Negate"]: ...

@final
class TensorInfo:
    @property
    def rank(self) -> int: ...
    @property
    def symmetry(self) -> tuple[SymmetryGenerator, ...]: ...

@final
class TensorRef:
    @property
    def tensor(self) -> int: ...
    @property
    def indices(self) -> tuple[int, ...]: ...

@final
class Term:
    @property
    def sums(self) -> tuple[Index, ...]: ...
    @property
    def coeff(self) -> Fraction: ...
    @property
    def factors(self) -> tuple[TensorRef, ...]: ...

@final
class TensorDef:
    @property
    def base(self) -> int: ...
    @property
    def exts(self) -> tuple[Index, ...]: ...
    @property
    def rhs(self) -> tuple[Term, ...]: ...

@final
class Computation:
    @property
    def ranges(self) -> frozenset[int]: ...
    @property
    def tensors(self) -> Mapping[int, TensorInfo]: ...
    @property
    def definitions(self) -> tuple[TensorDef, ...]: ...

@final
class Action: ...

ParenthesizeSnapshot: TypeAlias = tuple[tuple[Index, ...], Term]
BicliqueSnapshot: TypeAlias = tuple[
    tuple[Index, ...],
    tuple[Term, ...],
    tuple[Index, ...],
    tuple[Term, ...],
]

@final
class Space:
    def snapshot(
        self,
    ) -> ParenthesizeSnapshot | tuple[BicliqueSnapshot, ...]: ...
    @overload
    def select(self, left: Sequence[bool]) -> Action: ...
    @overload
    def select(
        self,
        candidate: int,
        left: Sequence[bool],
        right: Sequence[bool],
    ) -> Action: ...

class GristmillError(Exception): ...

@final
class State:
    @property
    def computation(self) -> Computation: ...
    @property
    def protected_outputs(self) -> tuple[int, ...]: ...
    @overload
    def query(
        self,
        kind: Literal["parenthesize"],
        definition: int,
        term: int,
    ) -> Space: ...
    @overload
    def query(self, kind: Literal["biclique"], definition: int) -> Space: ...
    def apply(self, action: Action) -> State: ...

def equivalent(lhs: State, rhs: State) -> bool: ...
def from_json(text: str) -> State: ...
def read_json(path: str | PathLike[str]) -> State: ...
def to_json(state: State) -> str: ...
def write_json(state: State, path: str | PathLike[str]) -> None: ...
