# ACTS IPM equivalence checker

`acts_equiv.py` checks whether two ACTS input parameter models describe exactly the
same set of valid complete tests.

It parses each model, creates typed SMT variables for its parameters, adds the finite
parameter domains and the ACTS constraints, and asks Z3 whether the exclusive-or of
the two resulting formulas is satisfiable. `unsat` means equivalent; `sat` produces a
concrete assignment accepted by exactly one model.

## Setup and use

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 acts_equiv.py model1.txt model2.txt
```

Exit status is `0` for equivalent, `1` for non-equivalent, and `2` for an input/tool
error. Add `--json` for structured output.

## Supported ACTS input

- `[Parameter]` declarations with `enum`, `bool`/`boolean`, and
  `int`/`integer`/`number`/`range` types. An omitted type means `enum`.
- `[Constraint]` expressions using parentheses, `&&`, `||`, `=>`, `!`, `=`, `==`,
  `!=`, `<`, `>`, `<=`, `>=`, `+`, `-`, `*`, `/`, and `%`.
- Boolean parameters may appear directly as propositions.
- `--` comments, including inline comments outside quoted strings.

`[System]`, `[Relation]`, `[Test Set]`, and other sections do not affect the valid
configuration set and are ignored. Enum literals in constraints must be double-quoted,
as required by ACTS. Parameter declarations and constraints must each fit on one line.

The models must use the same parameter names and compatible types. Parameter order,
domain order, constraint order, system name, relations, and test sets are irrelevant.
Declared domains may differ: Z3 will decide whether the difference is observable after
constraints are applied.

Integer `/` and `%` use Z3's mathematical integer semantics. This agrees with ordinary
ACTS models whose operands are non-negative; models relying on negative-operand Java
division/remainder should be normalized before comparison.
