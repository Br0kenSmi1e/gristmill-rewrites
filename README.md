# gristmill-rewrites

`gristmill-rewrites` is a small, trusted symbolic-rewrite environment for
Gristmill-style tensor computations. Its Rust kernel owns valid
transformations, immutable state transitions, exact equivalence checking, and
authoritative FLOP-cost evaluation; optimization policies and research agents
remain outside the kernel and decide which target and action to select. The
repository provides the rewrite environment, not a built-in optimizer or
learned policy.

## Background

The problem setting comes from the
[DrudgeCAS](https://github.com/DrudgeCAS/drudge) ecosystem. Drudge is a
computer algebra system for tensorial and noncommutative algebra, motivated in
part by the lengthy symbolic derivations that arise in quantum chemistry and
many-body theory. Methods such as coupled-cluster theory produce large sums of
tensor contractions; deriving the equations symbolically is only the first
step, because the resulting expressions still need an efficient numerical
evaluation scheme.

The [original Gristmill](https://github.com/DrudgeCAS/gristmill) was built on
Drudge to optimize those tensor computations and generate numerical code. It
uses transformations such as contraction parenthesization and factorization to
reduce FLOP counts, while applying the same ideas to tensor computations beyond
quantum chemistry. `gristmill-rewrites` focuses on the rewrite-and-evaluation
part of that setting and presents it as an environment for external
optimization policies.

A tensor computation can usually be expressed by many algebraically equivalent
programs. Changing the parenthesization of a contraction or factoring common
structure from a sum preserves its symbolic result, but can change its
floating-point operation count substantially.

This makes optimization a sequential decision problem. At each step, a policy
observes the current computation, chooses a rewrite target, chooses one valid
action for that target, and receives a new state. An external experiment driver
can then assess a final state by exact correctness, symbolic cost, measured
runtime, and the recorded action trace.

## Intended use

The project is intended for researchers and developers building optimization
policies outside the symbolic implementation. Examples include:

- hand-written rewrite heuristics;
- heuristic learning and reinforcement-learning policies;
- transformer-based action selection;
- LLM-driven or autoresearch-style heuristic development; and
- reproducible comparison of symbolic optimization strategies.

A driver might conceptually separate policy decisions into functions such as
`select_target(observation, targets)` and
`select_action(observation, target, actions)`. These are examples of an
experiment architecture, not functions provided by this package. The driver
owns iteration, stopping conditions, logging, and policy execution; the kernel
owns symbolic meaning.

## Design principles

- **Kernel-owned semantics.** Policies choose among transformations supplied by
  the kernel instead of constructing arbitrary symbolic mutations.
- **State-bound actions.** An action is opaque and can only be obtained by
  selecting a valid choice from a `Space`. It can only be applied to the
  `State` from which that space was queried.
- **Immutable transitions.** Applying an action returns a new validated state;
  the source state is unchanged and remains available for comparison or
  branching.
- **Inspectable observations.** State data is exposed through read-only Python
  objects, while `Space.snapshot()` returns owned, plain symbolic values that a
  policy can inspect.
- **Replaceable policies.** Hand-written, learned, or generated policy code can
  change without changing rewrite semantics. Saved results can be reloaded and
  checked with the same kernel.

This separation is an architectural and experimental boundary. It reduces the
symbolic surface available to policy code, but it is not a complete security
sandbox for running untrusted code.

## Rewrite model

The Python interface follows one state-transition protocol:

```text
State.query(...) → Space
Space.snapshot() → plain inspectable choices
Space.select(...) → opaque Action
State.apply(Action) → new State
```

`State.computation` and `State.protected_outputs` provide the read-only
observation from which a driver can identify possible targets. `query` binds a
rewrite kind and target to the current state. `snapshot` exposes the semantic
choice data, and `select` validates a policy's masks or candidate index before
creating an action. `apply` performs the symbolic transformation and returns a
new canonical state without mutating the original.

Keeping the driver in control of calls to this protocol is useful when a policy
is learned or generated: the policy proposes decisions, while kernel objects
remain authoritative for the transition, cost, and equivalence result.

## Available rewrites

The current Python bindings expose three rewrite kinds:

- **Parenthesization** targets one term in one definition with
  `state.query("parenthesize", definition, term)`. Its snapshot is
  `(exts, term)`. A boolean mask passed to `space.select(left)` partitions the
  term's factor occurrences into two nonempty groups. The rewrite forms the two
  child expressions and replaces the original product with their contraction,
  adding or reusing intermediates as applicable. A selectable term must contain
  more than two factors.
- **Biclique factorization** targets a definition with
  `state.query("biclique", definition)`. Its snapshot is a tuple of candidates,
  each shaped as
  `(left_exts, left_terms, right_exts, right_terms)`. A candidate represents
  factorable Cartesian-product structure among terms. Selection uses
  `space.select(candidate, left_mask, right_mask)`, where the masks follow the
  displayed term order and each selected side must be nonempty. A snapshot may
  contain no candidates.
- **Permutation factorization** targets a definition with
  `state.query("permutation", definition)`. Its snapshot is a tuple of maximal
  families shaped as `(exts, roots, uses)`. Each use contains an external-slot
  permutation and its exact coefficient. Selection uses
  `space.select(candidate, root_mask, use_mask)`; each mask must select at
  least two entries. The current search considers sum-free terms containing
  one tensor occurrence with distinct external indices, and only
  range-preserving permutations.

See the
[Python package guide](python/README.md) for detailed snapshot and selection
examples.

## Python example

The Python package is a PyO3 extension backed by the Rust kernel and requires
Python 3.11 or newer. For local development, install it from the repository
root with:

```bash
uv sync --project python
```

The following experiment assumes that definition `0`, term `0` in
`input.json` contains at least three factors:

```python
import math

from gristmill_rewrites import equivalent, log_flops, read_json, write_json

log_sizes = {0: math.log(1000), 1: math.log(5000)}
state = read_json("input.json")
initial_cost = log_flops(state, log_sizes)

space = state.query("parenthesize", 0, 0)
exts, term = space.snapshot()
print("target external indices:", exts)
print("target term:", term)

factor_count = len(term.factors)
if factor_count < 3:
    raise RuntimeError("the selected term cannot be parenthesized")

cut = factor_count // 2
left = tuple(position < cut for position in range(factor_count))
action = space.select(left)
rewritten = state.apply(action)

assert equivalent(state, rewritten)
rewritten_cost = log_flops(rewritten, log_sizes)
print("log FLOPs:", initial_cost, "->", rewritten_cost)

# Optional experiment artifact.
write_json(rewritten, "output.json")
```

When saved as `experiment.py` in the repository root, run it in the local
environment with `uv run --project python python experiment.py`.

## Cost evaluation

`log_flops(state, log_sizes)` returns the natural logarithm of the total FLOP
count. Each key in `log_sizes` is a range ID and each value is the **natural
logarithm of that range's size**, not the size itself:

```python
score = log_flops(state, {0: math.log(1000), 1: math.log(5000)})
```

Lower values mean fewer FLOPs. The implementation counts tensor-factor
multiplications, additions associated with summation, and additions between
terms on a definition's right-hand side. It ignores multiplication by numeric
coefficients, copying, and initialization. Every range referenced by a
definition's external or summed indices must have a finite, nonnegative log
size; a computation with no counted operations has no logarithmic FLOP value
and produces an error.

This is the kernel's symbolic cost model. Measured runtime, memory traffic, and
hardware-specific effects remain the responsibility of the external
experiment.

## Exact equivalence checking

`equivalent(lhs, rhs)` is the package's authoritative exact-equivalence check
for two rewrite states. It compares the protected outputs after recursively
inlining non-protected intermediates and canonically reducing their exact
symbolic differences. It returns `True` when those outputs agree and `False`
when a symbolic difference remains; incompatible protected outputs, tensor
metadata, or interfaces are reported as errors.

The guarantee is scoped to computations represented and validated by this
kernel. It is not a claim of general equivalence for arbitrary mathematical
programs.

States can be loaded and saved with `read_json` and `write_json`, or converted
to and from strings with `from_json` and `to_json`. JSON import validates and
canonicalizes the state. The format stores `computation` and
`protected_outputs` at the top level, with exact coefficients represented as
rational strings. A typical experiment loads a baseline, applies and records a
sequence of actions, writes the final state, and uses `equivalent` to recheck
the saved result against the baseline.

## Repository layout

- `src/` contains the Rust symbolic representation, canonicalization,
  transitions, rewrite implementations, JSON I/O, cost model, and verifier.
- `tests/` contains Rust integration tests for state validation, rewrites,
  serialization, cost, and equivalence.
- `python/src/` contains the PyO3 binding layer.
- `python/gristmill_rewrites/` contains the Python package and type stubs.
- `python/tests/` checks the public binding surface and end-to-end rewrite
  behavior.
- [`python/README.md`](python/README.md) is the Python-specific usage guide.

## Development and testing

The Rust crate is the implementation kernel; Maturin builds it into the Python
extension. From the repository root, check the Rust code with:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

From the `python/` directory, create the development environment, build the
extension, and run the Python tests with:

```bash
uv sync
uv run maturin develop
uv run pytest tests
```

## Current scope and non-goals

`gristmill-rewrites` is responsible for symbolic transitions and evaluation.
It is not currently intended to be:

- a complete optimizing compiler;
- an exhaustive rewrite planner;
- a reinforcement-learning framework;
- a model-training system;
- a benchmark-results repository; or
- a home for problem-specific heuristic policies.

Those components can be built around the environment while keeping the
symbolic transition and evaluation rules fixed and independently recheckable.
