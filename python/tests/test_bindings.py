import json
from fractions import Fraction
from types import MappingProxyType

import pytest

import gristmill_rewrites as gr


def scalar_tensor(tensor: int) -> dict:
    return {"tensor": tensor, "indices": []}


def term(coeff: int, *tensors: int) -> dict:
    return {
        "sums": [],
        "coeff": str(coeff),
        "factors": [scalar_tensor(tensor) for tensor in tensors],
    }


def state_json(definition: dict, tensor_count: int) -> str:
    return json.dumps(
        {
            "computation": {
                "ranges": [],
                "tensors": {
                    str(tensor): {"rank": 0, "symmetry": []}
                    for tensor in range(tensor_count)
                },
                "definitions": [definition],
            },
            "protected_outputs": [definition["base"]],
        }
    )


def parenthesization_json() -> str:
    return state_json(
        {
            "base": 3,
            "exts": [],
            "rhs": [term(5, 0, 1, 2)],
        },
        4,
    )


def biclique_json() -> str:
    return state_json(
        {
            "base": 4,
            "exts": [],
            "rhs": [
                term(2, 0, 2),
                term(3, 0, 3),
                term(4, 1, 2),
                term(6, 1, 3),
            ],
        },
        5,
    )


def test_exports_only_the_initial_rewrite_surface():
    assert gr.__all__ == (
        "Action",
        "Computation",
        "GristmillError",
        "Index",
        "Space",
        "State",
        "SymmetryGenerator",
        "TensorDef",
        "TensorInfo",
        "TensorRef",
        "Term",
        "from_json",
        "read_json",
        "to_json",
        "write_json",
    )
    assert not hasattr(gr, "PermutationSpace")
    assert not hasattr(gr, "ParenthesizeSpace")
    assert not hasattr(gr, "BicliqueSpace")


def test_state_is_transparent_and_read_only():
    state = gr.from_json(parenthesization_json())

    assert state.protected_outputs == (3,)
    assert isinstance(state.computation, gr.Computation)
    assert state.computation.ranges == frozenset()
    assert isinstance(state.computation.tensors, MappingProxyType)
    assert tuple(state.computation.tensors) == (0, 1, 2, 3)

    definition = state.computation.definitions[0]
    assert definition.base == 3
    assert definition.exts == ()
    assert len(definition.rhs) == 1

    product = definition.rhs[0]
    assert product.sums == ()
    assert product.coeff == Fraction(5, 1)
    assert tuple(factor.tensor for factor in product.factors) == (0, 1, 2)
    assert all(factor.indices == () for factor in product.factors)

    with pytest.raises(AttributeError):
        product.coeff = Fraction(1, 1)
    with pytest.raises(TypeError):
        state.computation.tensors[4] = state.computation.tensors[0]


def test_json_round_trip_preserves_state(tmp_path):
    state = gr.from_json(parenthesization_json())
    path = tmp_path / "state.json"

    gr.write_json(state, path)
    loaded = gr.read_json(path)

    assert gr.to_json(loaded) == gr.to_json(state)


def test_parenthesization_space_shows_the_target_term_and_applies_an_action():
    state = gr.from_json(parenthesization_json())
    space = state.query("parenthesize", 0, 0)
    exts, term = space.snapshot()

    assert isinstance(space, gr.Space)
    assert exts == ()
    assert tuple(factor.tensor for factor in term.factors) == (0, 1, 2)

    action = space.select((True, True, False))
    rewritten = state.apply(action)

    assert isinstance(action, gr.Action)
    assert len(state.computation.definitions) == 1
    assert len(rewritten.computation.definitions) == 2

    independently_loaded = gr.from_json(parenthesization_json())
    with pytest.raises(gr.GristmillError, match="different state"):
        independently_loaded.apply(action)


def test_biclique_space_shows_weighted_terms_in_mask_order():
    state = gr.from_json(biclique_json())
    space = state.query("biclique", 0)
    candidates = space.snapshot()

    assert isinstance(space, gr.Space)
    assert len(candidates) == 1

    left_exts, left_terms, right_exts, right_terms = candidates[0]
    assert left_exts == ()
    assert right_exts == ()
    assert tuple(term.coeff for term in left_terms) == (Fraction(1), Fraction(2))
    assert tuple(term.coeff for term in right_terms) == (Fraction(2), Fraction(3))
    assert tuple(term.factors[0].tensor for term in left_terms) == (0, 1)
    assert tuple(term.factors[0].tensor for term in right_terms) == (2, 3)

    action = space.select(0, (True, True), (True, True))
    rewritten = state.apply(action)

    assert len(rewritten.computation.definitions) == 3


def test_biclique_choice_errors_are_python_exceptions():
    space = gr.from_json(biclique_json()).query("biclique", 0)

    with pytest.raises(gr.GristmillError, match="CandidateOutOfBounds"):
        space.select(1, (True,), (True,))
    with pytest.raises(gr.GristmillError, match="WrongLeftMaskLength"):
        space.select(0, (True,), (True, True))


def test_only_the_rewrite_protocol_methods_are_public():
    state = gr.from_json(parenthesization_json())
    space = state.query("parenthesize", 0, 0)
    action = space.select((True, True, False))

    assert public_methods(state) == {"apply", "query"}
    assert public_methods(space) == {"select", "snapshot"}
    assert public_methods(action) == set()


def public_methods(value) -> set[str]:
    return {
        name
        for name in dir(value)
        if not name.startswith("_") and callable(getattr(value, name))
    }
