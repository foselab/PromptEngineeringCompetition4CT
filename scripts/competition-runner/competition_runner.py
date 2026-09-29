#!/usr/bin/env python3
"""Run IPM generation and scoring for the prompt-engineering competition."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
GENERATOR_PATH = SCRIPTS_DIR / "acts-ipm-runner" / "generate_ipm.py"
CHECKER_PATH = SCRIPTS_DIR / "acts-ipm-equivalence-checker" / "acts_equiv.py"


class CompetitionError(Exception):
    """A fatal competition-runner error."""


@dataclass(frozen=True)
class Weights:
    parameter_count: float = 5.0
    parameter_type: float = 4.0
    parameter_name: float = 3.0
    level: float = 2.0
    constraint: float = 1.0

    def validate(self) -> None:
        values = (
            self.parameter_count,
            self.parameter_type,
            self.parameter_name,
            self.level,
            self.constraint,
        )
        if any(value <= 0 for value in values):
            raise CompetitionError("all score weights must be greater than zero")
        if not all(left > right for left, right in zip(values, values[1:])):
            raise CompetitionError("weights must satisfy w1 > w2 > w3 > w4 > w5")


@dataclass
class ErrorDetail:
    category: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    benchmark: str
    specification: str
    reference: str
    generated: str
    score: float
    raw_score: float
    maximum_raw_score: float
    components: Dict[str, Any]
    errors: List[ErrorDetail]

    def as_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["errors"] = [asdict(error) for error in self.errors]
        return result


def load_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise CompetitionError(f"required script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CompetitionError(f"could not load script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalized_count_score(reference_count: int, predicted_count: int) -> float:
    """Proposal formula t-|t-p|/t, clamped to the score range [0, 1]."""
    if reference_count <= 0:
        return 1.0 if predicted_count == 0 else 0.0
    return max(0.0, 1.0 - abs(reference_count - predicted_count) / reference_count)


def display_value(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def maximum_raw_score(reference_parameter_count: int, weights: Weights) -> float:
    return (
        weights.parameter_count
        + reference_parameter_count
        * (weights.parameter_type + weights.parameter_name + 2 * weights.level)
        + weights.constraint
    )


def embedding_endpoint(host: str) -> str:
    base = host.rstrip("/")
    if base.endswith("/api/embed"):
        return base
    if base.endswith("/api"):
        return base + "/embed"
    return base + "/api/embed"


class NameSimilarity:
    """Exact or embedding-based semantic similarity with an explicit cutoff."""

    def __init__(
        self,
        *,
        mode: str,
        threshold: float,
        model: str,
        host: str,
        api_key: Optional[str],
        timeout: float,
    ):
        if not 0 <= threshold < 1:
            raise CompetitionError("--similarity-threshold must be at least 0 and less than 1")
        self.mode = mode
        self.threshold = threshold
        self.model = model
        self.host = host
        self.api_key = api_key
        self.timeout = timeout
        self.vectors: Dict[str, Tuple[float, ...]] = {}

    def prepare(self, texts: Iterable[str]) -> None:
        if self.mode == "exact":
            return
        missing = sorted({text for text in texts if text not in self.vectors})
        if not missing:
            return
        payload = json.dumps({"model": self.model, "input": missing}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            embedding_endpoint(self.host), data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise CompetitionError(
                f"embedding API returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise CompetitionError(f"could not call embedding API: {exc}") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CompetitionError("embedding API returned invalid JSON") from exc
        embeddings = result.get("embeddings") if isinstance(result, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(missing):
            raise CompetitionError("embedding API returned an unexpected number of vectors")
        for text, vector in zip(missing, embeddings):
            if not isinstance(vector, list) or not vector:
                raise CompetitionError("embedding API returned an invalid vector")
            try:
                self.vectors[text] = tuple(float(value) for value in vector)
            except (TypeError, ValueError) as exc:
                raise CompetitionError("embedding API returned a non-numeric vector") from exc

    def score(self, first: str, second: str) -> float:
        if first == second:
            return 1.0
        if self.mode == "exact":
            return 0.0
        left, right = self.vectors[first], self.vectors[second]
        if len(left) != len(right):
            raise CompetitionError("embedding vectors have inconsistent dimensions")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        cosine = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
        cosine = max(-1.0, min(1.0, cosine))
        if cosine <= self.threshold:
            return 0.0
        return min(1.0, (cosine - self.threshold) / (1.0 - self.threshold))


def optimal_alignment(
    reference: Sequence[str], predicted: Sequence[str], similarity: NameSimilarity
) -> List[Tuple[int, int, float]]:
    """Maximum-weight one-to-one alignment using the Hungarian algorithm."""
    size = max(len(reference), len(predicted))
    if size == 0:
        return []
    scores = [[0.0 for _ in range(size)] for _ in range(size)]
    for i, left in enumerate(reference):
        for j, right in enumerate(predicted):
            scores[i][j] = similarity.score(left, right)

    # Hungarian minimization over costs 1-score, with square zero-score padding.
    costs = [[1.0 - scores[i][j] for j in range(size)] for i in range(size)]
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        minv = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_to_column = [-1] * size
    for column in range(1, size + 1):
        if p[column]:
            row_to_column[p[column] - 1] = column - 1
    return [
        (i, j, scores[i][j])
        for i, j in enumerate(row_to_column[: len(reference)])
        if 0 <= j < len(predicted) and scores[i][j] > 0
    ]


def required_sections(text: str) -> List[str]:
    headings = {
        match.group(1).strip().lower()
        for match in re.finditer(r"(?m)^\s*\[\s*([^]]+?)\s*]\s*$", text)
    }
    return [name for name in ("system", "parameter") if name not in headings]


def score_models(
    reference: Any,
    predicted: Any,
    checker: Any,
    similarity: NameSimilarity,
    weights: Weights,
) -> Tuple[float, float, float, Dict[str, Any], List[ErrorDetail]]:
    ref_parameters = list(reference.parameters.values())
    pred_parameters = list(predicted.parameters.values())
    comparison_texts: List[str] = [p.name for p in ref_parameters + pred_parameters]
    for parameter in ref_parameters + pred_parameters:
        comparison_texts.extend(display_value(value) for value in parameter.values)
    similarity.prepare(comparison_texts)

    parameter_pairs = optimal_alignment(
        [p.name for p in ref_parameters], [p.name for p in pred_parameters], similarity
    )
    matched_ref = {i for i, _, _ in parameter_pairs}
    matched_pred = {j for _, j, _ in parameter_pairs}
    errors: List[ErrorDetail] = []

    parameter_count = normalized_count_score(len(ref_parameters), len(pred_parameters))
    if len(ref_parameters) != len(pred_parameters):
        errors.append(
            ErrorDetail(
                "parameter_count",
                f"expected {len(ref_parameters)} parameters, generated {len(pred_parameters)}",
            )
        )

    type_sum = 0.0
    name_sum = 0.0
    level_count_sum = 0.0
    level_name_sum = 0.0
    parameter_details: List[Dict[str, Any]] = []

    for ref_index, pred_index, name_score in parameter_pairs:
        expected = ref_parameters[ref_index]
        actual = pred_parameters[pred_index]
        type_score = 1.0 if expected.kind == actual.kind else 0.0
        value_count_score = normalized_count_score(len(expected.values), len(actual.values))
        ref_values = [display_value(value) for value in expected.values]
        pred_values = [display_value(value) for value in actual.values]
        value_pairs = optimal_alignment(ref_values, pred_values, similarity)
        value_name_score = (
            sum(score for _, _, score in value_pairs) / len(ref_values) if ref_values else 1.0
        )
        type_sum += type_score
        name_sum += name_score
        level_count_sum += value_count_score
        level_name_sum += value_name_score

        if name_score < 1.0:
            errors.append(
                ErrorDetail(
                    "parameter_name",
                    f"parameter {actual.name!r} aligned with expected {expected.name!r}",
                    {"similarity": round(name_score, 6)},
                )
            )
        if not type_score:
            errors.append(
                ErrorDetail(
                    "parameter_type",
                    f"parameter {expected.name!r}: expected {expected.kind}, generated {actual.kind}",
                )
            )
        if len(expected.values) != len(actual.values):
            errors.append(
                ErrorDetail(
                    "value_count",
                    f"parameter {expected.name!r}: expected {len(expected.values)} values, generated {len(actual.values)}",
                )
            )
        matched_ref_values = {i for i, _, _ in value_pairs}
        matched_pred_values = {j for _, j, _ in value_pairs}
        for i, j, value_score in value_pairs:
            if value_score < 1.0:
                errors.append(
                    ErrorDetail(
                        "value_name",
                        f"parameter {expected.name!r}: value {pred_values[j]!r} aligned with expected {ref_values[i]!r}",
                        {"similarity": round(value_score, 6)},
                    )
                )
        missing_values = [ref_values[i] for i in range(len(ref_values)) if i not in matched_ref_values]
        extra_values = [pred_values[j] for j in range(len(pred_values)) if j not in matched_pred_values]
        if missing_values:
            errors.append(
                ErrorDetail(
                    "missing_values",
                    f"parameter {expected.name!r} is missing values: {', '.join(missing_values)}",
                    {"values": missing_values},
                )
            )
        if extra_values:
            errors.append(
                ErrorDetail(
                    "spurious_values",
                    f"parameter {expected.name!r} has spurious values: {', '.join(extra_values)}",
                    {"values": extra_values},
                )
            )
        parameter_details.append(
            {
                "expected": expected.name,
                "generated": actual.name,
                "name_score": name_score,
                "type_score": type_score,
                "level_count_score": value_count_score,
                "level_name_score": value_name_score,
            }
        )

    for index, parameter in enumerate(ref_parameters):
        if index not in matched_ref:
            errors.append(ErrorDetail("missing_parameter", f"missing parameter {parameter.name!r}"))
    for index, parameter in enumerate(pred_parameters):
        if index not in matched_pred:
            errors.append(ErrorDetail("spurious_parameter", f"spurious parameter {parameter.name!r}"))

    constraint_score = 0.0
    constraint_details: Dict[str, Any]
    same_schema = (
        set(reference.parameters) == set(predicted.parameters)
        and all(
            reference.parameters[name].kind == predicted.parameters[name].kind
            for name in reference.parameters.keys() & predicted.parameters.keys()
        )
    )
    if not same_schema:
        constraint_details = {"equivalent": False, "reason": "parameter schemas differ"}
        errors.append(
            ErrorDetail(
                "constraint_not_comparable",
                "constraints could not be compared because parameter names or types differ",
            )
        )
    else:
        comparison = checker.compare_models(reference, predicted)
        constraint_details = comparison.as_dict()
        constraint_score = 1.0 if comparison.equivalent else 0.0
        if not comparison.equivalent:
            details = comparison.as_dict()
            errors.append(
                ErrorDetail(
                    "constraint_equivalence",
                    f"generated constraints are not semantically equivalent: {comparison.reason}",
                    details,
                )
            )

    reference_count = len(ref_parameters)
    raw_score = (
        weights.parameter_count * parameter_count
        + weights.parameter_type * type_sum
        + weights.parameter_name * name_sum
        + weights.level * level_count_sum
        + weights.level * level_name_sum
        + weights.constraint * constraint_score
    )
    maximum = maximum_raw_score(reference_count, weights)
    percentage = 100.0 * raw_score / maximum if maximum else 0.0
    components = {
        "parameter_count": parameter_count,
        "parameter_type_sum": type_sum,
        "parameter_name_sum": name_sum,
        "level_count_sum": level_count_sum,
        "level_name_sum": level_name_sum,
        "constraint": constraint_score,
        "parameter_alignment": parameter_details,
        "constraint_comparison": constraint_details,
    }
    return percentage, raw_score, maximum, components, errors


def find_benchmarks(specifications_dir: Path, references_dir: Path, spec_glob: str) -> List[Tuple[Path, Path]]:
    if not specifications_dir.is_dir():
        raise CompetitionError(f"specifications folder not found: {specifications_dir}")
    if not references_dir.is_dir():
        raise CompetitionError(f"reference IPM folder not found: {references_dir}")
    specifications = sorted(path for path in specifications_dir.glob(spec_glob) if path.is_file())
    if not specifications:
        raise CompetitionError(
            f"no specifications matching {spec_glob!r} found in {specifications_dir}"
        )
    pairs: List[Tuple[Path, Path]] = []
    for specification in specifications:
        candidates = [
            references_dir / f"{specification.stem}.acts",
            references_dir / f"{specification.stem}.txt",
        ]
        existing = list(dict.fromkeys(path for path in candidates if path.is_file()))
        if not existing:
            raise CompetitionError(
                f"no reference IPM found for {specification.name}; expected "
                + " or ".join(path.name for path in candidates)
            )
        if len(existing) > 1:
            raise CompetitionError(
                f"ambiguous reference IPMs for {specification.name}: "
                + ", ".join(path.name for path in existing)
            )
        pairs.append((specification, existing[0]))
    return pairs


def model_directory(index: int, model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-") or "model"
    return f"{index + 1:02d}-{slug}"


def write_text_report(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "IPM Prompt Engineering Competition Report",
        "=" * 41,
        f"Final score: {report['final_score']:.4f} / {report['maximum_final_score']:.4f}",
        f"Normalized final score: {report['normalized_final_score']:.4f} / 100.0000",
        f"Benchmarks: {report['benchmark_count']}",
        "",
        "Error summary:",
    ]
    if report["error_summary"]:
        for category, count in report["error_summary"].items():
            lines.append(f"  {category}: {count}")
    else:
        lines.append("  No errors detected.")
    lines.append("")
    for model in report["models"]:
        lines.extend(
            [
                f"Model: {model['model']}",
                f"Score: {model['score']:.4f} / {model['maximum_score']:.4f}",
                f"Normalized score: {model['normalized_score']:.4f} / 100.0000",
                "-" * 60,
            ]
        )
        for result in model["benchmarks"]:
            lines.append(
                f"{result['benchmark']}: {result['raw_score']:.4f} / "
                f"{result['maximum_raw_score']:.4f} "
                f"({result['score']:.4f} / 100.0000 normalized)"
            )
            errors = result["errors"]
            if not errors:
                lines.append("  No errors detected.")
            else:
                for error in errors:
                    lines.append(f"  [{error['category']}] {error['message']}")
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and score ACTS IPMs for every competition benchmark."
    )
    parser.add_argument("--specifications-dir", type=Path, required=True)
    parser.add_argument("--references-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Ollama generation model; repeat for known and transfer LLMs",
    )
    parser.add_argument("--provider", choices=("local", "cloud"), default="local")
    parser.add_argument("--host", help="override the Ollama host")
    parser.add_argument("--api-key-env", default="OLLAMA_API_KEY")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--num-predict", type=int, default=8192)
    parser.add_argument("--spec-glob", default="*.txt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--name-similarity", choices=("embedding", "exact"), default="embedding")
    parser.add_argument("--embedding-model", default="embeddinggemma")
    parser.add_argument("--embedding-host", help="override the embedding API host")
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--w1", type=float, default=5.0, help="parameter-count weight")
    parser.add_argument("--w2", type=float, default=4.0, help="parameter-type weight")
    parser.add_argument("--w3", type=float, default=3.0, help="parameter-name weight")
    parser.add_argument("--w4", type=float, default=2.0, help="level count/name weight")
    parser.add_argument("--w5", type=float, default=1.0, help="constraint weight")
    return parser


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        weights = Weights(args.w1, args.w2, args.w3, args.w4, args.w5)
        weights.validate()
        if args.timeout <= 0:
            raise CompetitionError("--timeout must be greater than zero")
        if args.num_predict <= 0:
            raise CompetitionError("--num-predict must be greater than zero")
        if not args.prompt_file.is_file():
            raise CompetitionError(f"prompt file not found: {args.prompt_file}")
        report_json = args.output_dir / "report.json"
        report_text = args.output_dir / "report.txt"
        if not args.overwrite and (report_json.exists() or report_text.exists()):
            raise CompetitionError(
                f"a report already exists in {args.output_dir}; use --overwrite to replace it"
            )

        generator = load_module("competition_generate_ipm", GENERATOR_PATH)
        checker = load_module("competition_acts_equiv", CHECKER_PATH)
        if checker.z3 is None:
            raise CompetitionError(
                "z3-solver is not installed; run: python3 -m pip install -r "
                "scripts/competition-runner/requirements.txt"
            )
        template = generator.read_utf8(args.prompt_file, "prompt template")
        if generator.PLACEHOLDER not in template:
            raise CompetitionError(
                f"prompt must contain the literal placeholder {generator.PLACEHOLDER}"
            )
        benchmark_files = find_benchmarks(
            args.specifications_dir, args.references_dir, args.spec_glob
        )

        references: Dict[str, Any] = {}
        for specification_path, reference_path in benchmark_files:
            try:
                reference_model = checker.parse_model(reference_path)
                checker.SMTModel(reference_model, checker.z3.Context())
                references[specification_path.stem] = reference_model
            except (checker.ModelError, RuntimeError) as exc:
                raise CompetitionError(f"invalid reference IPM: {exc}") from exc

        if args.provider == "cloud":
            host = args.host or generator.DEFAULT_CLOUD_HOST
            api_key = os.environ.get(args.api_key_env)
            if not api_key:
                raise CompetitionError(
                    f"cloud mode requires an API key in {args.api_key_env}"
                )
        else:
            host = args.host or os.environ.get("OLLAMA_HOST") or generator.DEFAULT_LOCAL_HOST
            api_key = None
        embedding_host = args.embedding_host or host
        similarity = NameSimilarity(
            mode=args.name_similarity,
            threshold=args.similarity_threshold,
            model=args.embedding_model,
            host=embedding_host,
            api_key=api_key,
            timeout=args.timeout,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        model_reports: List[Dict[str, Any]] = []
        for model_index, model in enumerate(args.model):
            generated_dir = args.output_dir / "generated" / model_directory(model_index, model)
            generated_dir.mkdir(parents=True, exist_ok=True)
            benchmark_results: List[BenchmarkResult] = []
            for specification_path, reference_path in benchmark_files:
                output_path = generated_dir / f"{specification_path.stem}.acts"
                errors: List[ErrorDetail] = []
                failed_maximum = maximum_raw_score(
                    len(references[specification_path.stem].parameters), weights
                )
                if output_path.exists() and not args.overwrite:
                    errors.append(
                        ErrorDetail(
                            "generation",
                            f"generated output already exists: {output_path}; use --overwrite to replace it",
                        )
                    )
                    benchmark_results.append(
                        BenchmarkResult(
                            specification_path.stem,
                            str(specification_path.resolve()),
                            str(reference_path.resolve()),
                            str(output_path.resolve()),
                            0.0,
                            0.0,
                            failed_maximum,
                            {},
                            errors,
                        )
                    )
                    continue
                try:
                    specification = generator.read_utf8(specification_path, "specification")
                    prompt = generator.render_prompt(template, specification)
                    generated, _ = generator.call_ollama(
                        host=host,
                        model=model,
                        prompt=prompt,
                        api_key=api_key,
                        timeout=args.timeout,
                        temperature=0.0,
                        seed=0,
                        num_predict=args.num_predict,
                    )
                    acts_text = generator.extract_acts(generated)
                    output_path.write_text(acts_text, encoding="utf-8")
                except (generator.GenerationError, OSError) as exc:
                    errors.append(ErrorDetail("generation", str(exc)))
                    benchmark_results.append(
                        BenchmarkResult(
                            specification_path.stem,
                            str(specification_path.resolve()),
                            str(reference_path.resolve()),
                            str(output_path.resolve()),
                            0.0,
                            0.0,
                            failed_maximum,
                            {},
                            errors,
                        )
                    )
                    continue

                missing_sections = required_sections(acts_text)
                if missing_sections:
                    errors.append(
                        ErrorDetail(
                            "invalid_format",
                            "missing required section(s): " + ", ".join(f"[{s.title()}]" for s in missing_sections),
                        )
                    )
                try:
                    predicted = checker.parse_model(output_path)
                except checker.ModelError as exc:
                    errors.append(ErrorDetail("invalid_syntax", str(exc)))
                    predicted = None
                if predicted is not None:
                    try:
                        checker.SMTModel(predicted, checker.z3.Context())
                    except (checker.ModelError, RuntimeError) as exc:
                        errors.append(ErrorDetail("invalid_constraint", str(exc)))
                        predicted = None

                if errors or predicted is None:
                    benchmark_results.append(
                        BenchmarkResult(
                            specification_path.stem,
                            str(specification_path.resolve()),
                            str(reference_path.resolve()),
                            str(output_path.resolve()),
                            0.0,
                            0.0,
                            failed_maximum,
                            {},
                            errors,
                        )
                    )
                    continue

                score, raw, maximum, components, score_errors = score_models(
                    references[specification_path.stem], predicted, checker, similarity, weights
                )
                benchmark_results.append(
                    BenchmarkResult(
                        specification_path.stem,
                        str(specification_path.resolve()),
                        str(reference_path.resolve()),
                        str(output_path.resolve()),
                        score,
                        raw,
                        maximum,
                        components,
                        score_errors,
                    )
                )

            model_score = sum(result.raw_score for result in benchmark_results)
            model_maximum = sum(result.maximum_raw_score for result in benchmark_results)
            model_reports.append(
                {
                    "model": model,
                    "score": model_score,
                    "maximum_score": model_maximum,
                    "normalized_score": (
                        100.0 * model_score / model_maximum if model_maximum else 0.0
                    ),
                    "benchmarks": [result.as_dict() for result in benchmark_results],
                }
            )

        final_score = sum(model["score"] for model in model_reports)
        maximum_final_score = sum(model["maximum_score"] for model in model_reports)
        normalized_final_score = (
            100.0 * final_score / maximum_final_score if maximum_final_score else 0.0
        )
        error_counts: Dict[str, int] = {}
        for model_report in model_reports:
            for benchmark in model_report["benchmarks"]:
                for error in benchmark["errors"]:
                    category = error["category"]
                    error_counts[category] = error_counts.get(category, 0) + 1
        report: Dict[str, Any] = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt_file": str(args.prompt_file.resolve()),
            "specifications_dir": str(args.specifications_dir.resolve()),
            "references_dir": str(args.references_dir.resolve()),
            "benchmark_count": len(benchmark_files),
            "scoring": {
                "weights": asdict(weights),
                "count_formula": "max(0, 1 - abs(reference - predicted) / reference)",
                "name_similarity": args.name_similarity,
                "embedding_model": args.embedding_model if args.name_similarity == "embedding" else None,
                "similarity_threshold": args.similarity_threshold,
                "normalized_benchmark_score_range": [0, 100],
                "final_score_rule": "sum of raw weighted benchmark scores for all models",
            },
            "models": model_reports,
            "final_score": final_score,
            "maximum_final_score": maximum_final_score,
            "normalized_final_score": normalized_final_score,
            "error_summary": dict(sorted(error_counts.items())),
        }
        report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_text_report(report, report_text)
        print(f"Final score: {final_score:.4f} / {maximum_final_score:.4f}")
        print(f"Normalized final score: {normalized_final_score:.4f} / 100.0000")
        print(f"JSON report: {report_json.resolve()}")
        print(f"Text report: {report_text.resolve()}")
        return 0
    except (CompetitionError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
