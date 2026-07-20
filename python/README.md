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

# Factor a shared pattern under external-index permutations.
permutation = state.query("permutation", 0)
families = permutation.snapshot()
exts, roots, uses = families[0]
print(exts, roots)
print(uses)  # ((slot permutation, Fraction weight), ...)

action = permutation.select(
    0,
    (True,) * len(roots),
    (True,) * len(uses),
)
permuted = state.apply(action)

write_json(factored, "output.json")
```

The parenthesization snapshot is `(exts, term)`. A biclique snapshot is a
tuple of candidates, each shaped as
`(left_exts, left_terms, right_exts, right_terms)`.
A permutation snapshot is a tuple of maximal families, each shaped as
`(exts, roots, uses)`. Each use is `(permutation, coefficient)`, where the
permutation contains external-slot positions and the coefficient is an exact
`Fraction`. Selection takes a candidate index followed by masks for the roots
and uses; each mask must select at least two entries.

Exact equivalence checking is available as a module-level function:

```python
from gristmill_rewrites import equivalent, read_json

working = read_json("working_eqn.json")
optimized = read_json("gristmill_optimized.json")
assert equivalent(working, optimized)
```

The logarithmic FLOP cost is also a module-level function. Its mapping values
are natural logarithms of the corresponding range sizes:

```python
import math

from gristmill_rewrites import log_flops

cost = log_flops(state, {0: math.log(1000), 1: math.log(5000)})
```

`State` and its symbolic representation are exposed through immutable Python
objects. The rewrite protocol consists only of `State.query`,
`Space.snapshot`, `Space.select`, and `State.apply`. Actions are opaque and can
only be created by selecting a valid choice from a space.
