#!/usr/bin/env python3
"""Semantic equivalence checker for ACTS input parameter models."""

from __future__ import annotations

import argparse
import ast as py_ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import z3
except ImportError:  # pragma: no cover - exercised by the CLI in an uninstalled env
    z3 = None


class ModelError(ValueError):
    """An ACTS model cannot be parsed or translated."""


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: str
    values: Tuple[Any, ...]
    line: int


@dataclass(frozen=True)
class Node:
    tag: str
    value: Any = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None


@dataclass(frozen=True)
class Constraint:
    text: str
    line: int
    expression: Node


@dataclass(frozen=True)
class ActsModel:
    source: str
    parameters: Mapping[str, Parameter]
    constraints: Tuple[Constraint, ...]


_SECTION = re.compile(r"^\s*\[\s*([^]]+?)\s*]\s*$")
_PARAMETER = re.compile(
    r"^\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*"
    r"(?:\(\s*([A-Za-z]+)\s*\))?\s*:\s*(.*?)\s*$"
)
_KIND = {
    None: "enum",
    "enum": "enum",
    "bool": "bool",
    "boolean": "bool",
    "int": "int",
    "integer": "int",
    "number": "int",
    "range": "int",
}


def _strip_comment(line: str) -> str:
    """Remove an ACTS -- comment, preserving -- inside quoted strings."""
    quote = False
    escaped = False
    for i, char in enumerate(line):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = False
        elif char == '"':
            quote = True
        elif char == "-" and i + 1 < len(line) and line[i + 1] == "-":
            return line[:i]
    return line


def _split_values(text: str, source: str, line: int) -> List[str]:
    result: List[str] = []
    start = 0
    quote = False
    escaped = False
    for i, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = False
        elif char == '"':
            quote = True
        elif char == ",":
            result.append(text[start:i].strip())
            start = i + 1
    if quote:
        raise ModelError(f"{source}:{line}: unterminated quoted parameter value")
    result.append(text[start:].strip())
    if any(value == "" for value in result):
        raise ModelError(f"{source}:{line}: empty parameter value")
    return result


def _decode_string(token: str, source: str, line: int) -> str:
    if not (len(token) >= 2 and token[0] == token[-1] == '"'):
        return token
    try:
        value = py_ast.literal_eval(token)
    except (ValueError, SyntaxError) as exc:
        raise ModelError(f"{source}:{line}: invalid quoted string {token!r}") from exc
    if not isinstance(value, str):
        raise ModelError(f"{source}:{line}: invalid string value {token!r}")
    return value


@dataclass(frozen=True)
class Token:
    kind: str
    value: Any
    column: int


_TOKEN = re.compile(
    r"\s*(?:"
    r"(?P<STRING>\"(?:\\.|[^\"\\])*\")|"
    r"(?P<INT>\d+)|"
    r"(?P<ID>[A-Za-z_$][A-Za-z0-9_$]*)|"
    r"(?P<OP>=>|&&|\|\||!=|>=|<=|==|[()=<>+*/%!-])"
    r")"
)


def _tokenize(text: str, source: str, line: int) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match:
            raise ModelError(
                f"{source}:{line}:{pos + 1}: unexpected character {text[pos]!r}"
            )
        kind = match.lastgroup or ""
        raw = match.group(kind)
        value: Any = raw
        if kind == "INT":
            value = int(raw)
        elif kind == "STRING":
            value = _decode_string(raw, source, line)
        tokens.append(Token(kind, value, match.start(kind) + 1))
        pos = match.end()
    tokens.append(Token("EOF", "", len(text) + 1))
    return tokens


class _ExpressionParser:
    _BP = {
        "=>": (10, 10),  # right associative
        "||": (20, 21),
        "&&": (30, 31),
        "=": (40, 41),
        "==": (40, 41),
        "!=": (40, 41),
        ">": (40, 41),
        "<": (40, 41),
        ">=": (40, 41),
        "<=": (40, 41),
        "+": (50, 51),
        "-": (50, 51),
        "*": (60, 61),
        "/": (60, 61),
        "%": (60, 61),
    }

    def __init__(self, text: str, source: str, line: int):
        self.source = source
        self.line = line
        self.tokens = _tokenize(text, source, line)
        self.index = 0

    def current(self) -> Token:
        return self.tokens[self.index]

    def take(self) -> Token:
        token = self.current()
        self.index += 1
        return token

    def error(self, message: str, token: Optional[Token] = None) -> ModelError:
        at = token or self.current()
        return ModelError(f"{self.source}:{self.line}:{at.column}: {message}")

    def parse(self) -> Node:
        node = self.expression(0)
        if self.current().kind != "EOF":
            raise self.error(f"unexpected token {self.current().value!r}")
        return node

    def expression(self, min_bp: int) -> Node:
        token = self.take()
        if token.kind == "INT":
            left = Node("int", token.value)
        elif token.kind == "STRING":
            left = Node("string", token.value)
        elif token.kind == "ID":
            lowered = token.value.lower()
            if lowered in ("true", "false"):
                left = Node("bool", lowered == "true")
            else:
                left = Node("name", token.value)
        elif token.kind == "OP" and token.value == "(":
            left = self.expression(0)
            closing = self.take()
            if closing.kind != "OP" or closing.value != ")":
                raise self.error("expected ')'", closing)
        elif token.kind == "OP" and token.value in ("!", "-", "+"):
            left = Node("unary", token.value, right=self.expression(70))
        else:
            raise self.error(f"expected expression, got {token.value!r}", token)

        while True:
            token = self.current()
            if token.kind != "OP" or token.value not in self._BP:
                break
            left_bp, right_bp = self._BP[token.value]
            if left_bp < min_bp:
                break
            self.take()
            right = self.expression(right_bp)
            op = "=" if token.value == "==" else token.value
            left = Node("binary", op, left, right)
        return left


def parse_model(path: Path) -> ActsModel:
    source = str(path)
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ModelError(f"cannot read {source}: {exc}") from exc

    section = ""
    parameters: Dict[str, Parameter] = {}
    constraints: List[Constraint] = []
    for line_number, original in enumerate(lines, 1):
        text = _strip_comment(original).strip()
        if not text:
            continue
        heading = _SECTION.match(text)
        if heading:
            section = heading.group(1).strip().lower()
            continue
        if section == "parameter":
            match = _PARAMETER.match(text)
            if not match:
                raise ModelError(f"{source}:{line_number}: invalid parameter declaration")
            name, raw_kind, values_text = match.groups()
            kind_key = raw_kind.lower() if raw_kind else None
            if kind_key not in _KIND:
                raise ModelError(
                    f"{source}:{line_number}: unsupported parameter type {raw_kind!r}"
                )
            if name in parameters:
                raise ModelError(f"{source}:{line_number}: duplicate parameter {name!r}")
            kind = _KIND[kind_key]
            raw_values = _split_values(values_text, source, line_number)
            values: List[Any] = []
            for raw in raw_values:
                if kind == "int":
                    try:
                        value: Any = int(raw)
                    except ValueError as exc:
                        raise ModelError(
                            f"{source}:{line_number}: {raw!r} is not an integer"
                        ) from exc
                elif kind == "bool":
                    if raw.lower() not in ("true", "false"):
                        raise ModelError(
                            f"{source}:{line_number}: {raw!r} is not a Boolean"
                        )
                    value = raw.lower() == "true"
                else:
                    value = _decode_string(raw, source, line_number)
                if value in values:
                    raise ModelError(
                        f"{source}:{line_number}: duplicate value {raw!r} for {name!r}"
                    )
                values.append(value)
            parameters[name] = Parameter(name, kind, tuple(values), line_number)
        elif section == "constraint":
            expression = _ExpressionParser(text, source, line_number).parse()
            constraints.append(Constraint(text, line_number, expression))

    if not parameters:
        raise ModelError(f"{source}: no parameters found in [Parameter]")
    return ActsModel(source, parameters, tuple(constraints))


class SMTModel:
    """One ACTS model translated into a caller-provided Z3 context."""

    def __init__(self, model: ActsModel, context: Any):
        if z3 is None:
            raise RuntimeError("z3-solver is required; install it with: pip install -r requirements.txt")
        self.model = model
        self.context = context
        self.variables: Dict[str, Any] = {}
        for parameter in model.parameters.values():
            if parameter.kind == "int":
                variable = z3.Int(parameter.name, ctx=context)
            elif parameter.kind == "bool":
                variable = z3.Bool(parameter.name, ctx=context)
            else:
                variable = z3.String(parameter.name, ctx=context)
            self.variables[parameter.name] = variable

        domain_formulas = [self._domain(p) for p in model.parameters.values()]
        constraint_formulas = [self._compile(c.expression, c.line)[0] for c in model.constraints]
        for formula, constraint in zip(constraint_formulas, model.constraints):
            if not z3.is_bool(formula):
                raise ModelError(
                    f"{model.source}:{constraint.line}: constraint is not a Boolean expression"
                )
        self.domain_formula = self._and(domain_formulas)
        self.constraint_formula = self._and(constraint_formulas)
        self.formula = z3.And(self.domain_formula, self.constraint_formula)

    def _and(self, formulas: Sequence[Any]) -> Any:
        if not formulas:
            return z3.BoolVal(True, ctx=self.context)
        return z3.And(*formulas)

    def _constant(self, value: Any, kind: str) -> Any:
        if kind == "int":
            return z3.IntVal(value, ctx=self.context)
        if kind == "bool":
            return z3.BoolVal(value, ctx=self.context)
        return z3.StringVal(value, ctx=self.context)

    def _domain(self, parameter: Parameter) -> Any:
        variable = self.variables[parameter.name]
        return z3.Or(
            *[variable == self._constant(value, parameter.kind) for value in parameter.values]
        )

    def _compile(self, node: Node, line: int) -> Tuple[Any, str]:
        if node.tag in ("int", "bool", "string"):
            kind = {"string": "enum"}.get(node.tag, node.tag)
            return self._constant(node.value, kind), kind
        if node.tag == "name":
            if node.value not in self.variables:
                raise ModelError(
                    f"{self.model.source}:{line}: unknown parameter {node.value!r}; "
                    "enum values in constraints must be double-quoted"
                )
            parameter = self.model.parameters[node.value]
            return self.variables[node.value], parameter.kind
        if node.tag == "unary":
            value, kind = self._compile(node.right, line)  # type: ignore[arg-type]
            if node.value == "!":
                self._require(kind == "bool", line, "'!' requires a Boolean operand")
                return z3.Not(value), "bool"
            self._require(kind == "int", line, f"unary {node.value!r} requires an integer")
            return (-value if node.value == "-" else value), "int"

        left, left_kind = self._compile(node.left, line)  # type: ignore[arg-type]
        right, right_kind = self._compile(node.right, line)  # type: ignore[arg-type]
        op = node.value
        if op in ("&&", "||", "=>"):
            self._require(
                left_kind == right_kind == "bool", line, f"{op!r} requires Boolean operands"
            )
            if op == "&&":
                return z3.And(left, right), "bool"
            if op == "||":
                return z3.Or(left, right), "bool"
            return z3.Implies(left, right), "bool"
        if op in ("=", "!="):
            self._require(
                left_kind == right_kind,
                line,
                f"cannot compare {left_kind} with {right_kind}",
            )
            equality = left == right
            return (equality if op == "=" else z3.Not(equality)), "bool"
        if op in (">", "<", ">=", "<="):
            self._require(
                left_kind == right_kind == "int", line, f"{op!r} requires integer operands"
            )
            return {">": left > right, "<": left < right, ">=": left >= right, "<=": left <= right}[op], "bool"
        if op in ("+", "-", "*", "/", "%"):
            self._require(
                left_kind == right_kind == "int", line, f"{op!r} requires integer operands"
            )
            result = {
                "+": lambda: left + right,
                "-": lambda: left - right,
                "*": lambda: left * right,
                "/": lambda: left / right,
                "%": lambda: left % right,
            }[op]()
            return result, "int"
        raise AssertionError(f"unhandled AST operator {op!r}")

    def _require(self, condition: bool, line: int, message: str) -> None:
        if not condition:
            raise ModelError(f"{self.model.source}:{line}: {message}")


@dataclass(frozen=True)
class Comparison:
    equivalent: bool
    reason: str
    witness: Optional[Mapping[str, Any]] = None
    valid_in_first: Optional[bool] = None
    valid_in_second: Optional[bool] = None

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"equivalent": self.equivalent, "reason": self.reason}
        if self.witness is not None:
            result["witness"] = dict(self.witness)
            result["valid_in_first"] = self.valid_in_first
            result["valid_in_second"] = self.valid_in_second
        return result


def compare_models(first: ActsModel, second: ActsModel) -> Comparison:
    if z3 is None:
        raise RuntimeError("z3-solver is required; install it with: pip install -r requirements.txt")
    first_names, second_names = set(first.parameters), set(second.parameters)
    if first_names != second_names:
        missing = sorted(first_names - second_names)
        extra = sorted(second_names - first_names)
        details = []
        if missing:
            details.append("only in first: " + ", ".join(missing))
        if extra:
            details.append("only in second: " + ", ".join(extra))
        return Comparison(False, "different parameter names (" + "; ".join(details) + ")")
    for name in sorted(first_names):
        kind1, kind2 = first.parameters[name].kind, second.parameters[name].kind
        if kind1 != kind2:
            return Comparison(False, f"parameter {name!r} has different types: {kind1} vs {kind2}")

    # Building these independent translations also catches each model's own type errors.
    SMTModel(first, z3.Context())
    SMTModel(second, z3.Context())

    comparison_context = z3.Context()
    smt_first = SMTModel(first, comparison_context)
    smt_second = SMTModel(second, comparison_context)
    solver = z3.Solver(ctx=comparison_context)
    solver.add(z3.Xor(smt_first.formula, smt_second.formula))
    status = solver.check()
    if status == z3.unsat:
        return Comparison(True, "no assignment satisfies exactly one model")
    if status == z3.unknown:
        return Comparison(False, "SMT solver returned unknown: " + solver.reason_unknown())

    solution = solver.model()
    witness: Dict[str, Any] = {}
    for name in sorted(first_names):
        parameter = first.parameters[name]
        value = solution.eval(smt_first.variables[name], model_completion=True)
        if parameter.kind == "int":
            witness[name] = value.as_long()
        elif parameter.kind == "bool":
            witness[name] = z3.is_true(value)
        else:
            witness[name] = value.as_string()
    first_valid = z3.is_true(solution.eval(smt_first.formula, model_completion=True))
    second_valid = z3.is_true(solution.eval(smt_second.formula, model_completion=True))
    return Comparison(
        False,
        "a counterexample assignment satisfies exactly one model",
        witness,
        first_valid,
        second_valid,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether two ACTS input parameter models admit the same valid tests."
    )
    parser.add_argument("first", type=Path, help="first ACTS .txt model")
    parser.add_argument("second", type=Path, help="second ACTS .txt model")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if z3 is None:
            raise RuntimeError(
                "z3-solver is not installed; run: python3 -m pip install -r requirements.txt"
            )
        result = compare_models(parse_model(args.first), parse_model(args.second))
    except (ModelError, RuntimeError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    elif result.equivalent:
        print("EQUIVALENT")
        print(result.reason)
    else:
        print("NOT EQUIVALENT")
        print(result.reason)
        if result.witness is not None:
            print("counterexample:")
            for name, value in result.witness.items():
                print(f"  {name} = {json.dumps(value)}")
            print(f"valid in first:  {str(result.valid_in_first).lower()}")
            print(f"valid in second: {str(result.valid_in_second).lower()}")
    return 0 if result.equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
