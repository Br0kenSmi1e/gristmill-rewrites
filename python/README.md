# gristmill-rewrites

Python bindings for the immutable symbolic rewrite state in the
`gristmill-rewrites` Rust crate.

```python
from gristmill_rewrites import read_json, write_json

state = read_json("input.json")

# Parenthesize definition 0, term 0. The mask follows term.factors.
parenthesize = state.query("parenthesize", 0, 0)
exts, term = parenthesize.snapshot()
print(exts)
print(term)

action = parenthesize.select((True, True, False))
parenthesized = state.apply(action)

# Factor a biclique in definition 0. The masks follow the displayed terms.
biclique = state.query("biclique", 0)
candidates = biclique.snapshot()
left_exts, left_terms, right_exts, right_terms = candidates[0]
print(left_exts, left_terms)
print(right_exts, right_terms)

action = biclique.select(
    0,
    (True,) * len(left_terms),
    (True,) * len(right_terms),
)
factored = state.apply(action)

write_json(factored, "output.json")
```

The parenthesization snapshot is `(exts, term)`. A biclique snapshot is a
tuple of candidates, each shaped as
`(left_exts, left_terms, right_exts, right_terms)`.

Exact equivalence checking is available as a module-level function:

```python
from gristmill_rewrites import equivalent, read_json

working = read_json("working_eqn.json")
optimized = read_json("gristmill_optimized.json")
assert equivalent(working, optimized)
```

`State` and its symbolic representation are exposed through immutable Python
objects. The rewrite protocol consists only of `State.query`,
`Space.snapshot`, `Space.select`, and `State.apply`. Actions are opaque and can
only be created by selecting a valid choice from a space. Permutation
factorization is not part of the initial Python interface.
