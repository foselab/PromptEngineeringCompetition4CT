import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock


RUNNER_PATH = Path(__file__).resolve().parents[1] / "competition_runner.py"
SPEC = importlib.util.spec_from_file_location("test_competition_runner_module", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: str
    values: tuple


@dataclass(frozen=True)
class Model:
    parameters: dict


class ExactSimilarity:
    def prepare(self, texts):
        pass

    def score(self, first, second):
        return 1.0 if first == second else 0.0


class SemanticSimilarity(ExactSimilarity):
    equivalents = {frozenset(("OS", "OperatingSystem")): 0.8}

    def score(self, first, second):
        if first == second:
            return 1.0
        return self.equivalents.get(frozenset((first, second)), 0.0)


class Comparison:
    equivalent = True
    reason = "same space"

    def as_dict(self):
        return {"equivalent": True, "reason": self.reason}


class Checker:
    @staticmethod
    def compare_models(first, second):
        return Comparison()


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class CompetitionRunnerTests(unittest.TestCase):
    def test_normalized_count_score_is_clamped(self):
        self.assertEqual(runner.normalized_count_score(4, 4), 1.0)
        self.assertEqual(runner.normalized_count_score(4, 2), 0.5)
        self.assertEqual(runner.normalized_count_score(4, 12), 0.0)

    def test_optimal_alignment_uses_semantic_similarity(self):
        pairs = runner.optimal_alignment(
            ["OS", "Browser"], ["Browser", "OperatingSystem"], SemanticSimilarity()
        )
        self.assertEqual(pairs, [(0, 1, 0.8), (1, 0, 1.0)])

    def test_perfect_model_scores_100(self):
        parameters = {
            "OS": Parameter("OS", "enum", ("Windows", "Linux")),
            "Enabled": Parameter("Enabled", "bool", (True, False)),
        }
        model = Model(parameters)
        score, raw, maximum, components, errors = runner.score_models(
            model, model, Checker(), ExactSimilarity(), runner.Weights()
        )
        self.assertEqual(score, 100.0)
        self.assertEqual(raw, maximum)
        self.assertEqual(components["constraint"], 1.0)
        self.assertEqual(errors, [])

    def test_reports_missing_parameter_and_values(self):
        reference = Model(
            {
                "OS": Parameter("OS", "enum", ("Windows", "Linux")),
                "Browser": Parameter("Browser", "enum", ("Edge", "Firefox")),
            }
        )
        predicted = Model(
            {"OS": Parameter("OS", "enum", ("Windows", "macOS"))}
        )
        score, _, _, _, errors = runner.score_models(
            reference, predicted, Checker(), ExactSimilarity(), runner.Weights()
        )
        categories = {error.category for error in errors}
        self.assertLess(score, 100.0)
        self.assertIn("parameter_count", categories)
        self.assertIn("missing_parameter", categories)
        self.assertIn("missing_values", categories)
        self.assertIn("spurious_values", categories)
        self.assertIn("constraint_not_comparable", categories)

    def test_weight_order_is_enforced(self):
        with self.assertRaises(runner.CompetitionError):
            runner.Weights(5, 4, 3, 2, 2).validate()

    @unittest.skipUnless(importlib.util.find_spec("z3"), "z3-solver is not installed")
    def test_full_run_with_real_checker_and_mocked_ollama(self):
        generated = """[System]
Name: Browser
[Parameter]
OS (enum): Windows, Linux
Browser (enum): Edge, Firefox
[Constraint]
OS = \"Linux\" => Browser != \"Edge\"
"""
        response = json.dumps({"response": generated, "done": True}).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specifications = root / "specifications"
            references = root / "IPM"
            specifications.mkdir()
            references.mkdir()
            (specifications / "browser.txt").write_text("A browser specification")
            (references / "browser.acts").write_text(generated)
            (specifications / "browser_copy.txt").write_text("Another browser specification")
            (references / "browser_copy.acts").write_text(generated)
            prompt = root / "prompt.txt"
            prompt.write_text("Generate ACTS for {SPECIFICATION}")
            output = root / "results"
            with mock.patch(
                "urllib.request.urlopen", return_value=FakeResponse(response)
            ):
                exit_code = runner.run(
                    [
                        "--specifications-dir", str(specifications),
                        "--references-dir", str(references),
                        "--prompt-file", str(prompt),
                        "--output-dir", str(output),
                        "--model", "test-model",
                        "--name-similarity", "exact",
                    ]
                )
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(exit_code, 0)
            self.assertEqual(report["final_score"], 56.0)
            self.assertEqual(report["maximum_final_score"], 56.0)
            self.assertEqual(report["normalized_final_score"], 100.0)
            self.assertEqual(report["models"][0]["normalized_score"], 100.0)
            self.assertEqual(report["models"][0]["benchmarks"][0]["errors"], [])


if __name__ == "__main__":
    unittest.main()
