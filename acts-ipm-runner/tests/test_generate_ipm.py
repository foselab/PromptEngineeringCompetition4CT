import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import generate_ipm


VALID_ACTS = """[System]
Name: Example

[Parameter]
Browser (enum): Chrome, Firefox
LoggedIn (boolean): TRUE, FALSE

[Constraint]
"""


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class GeneratorTests(unittest.TestCase):
    def test_prompt_requires_placeholder(self):
        with self.assertRaises(generate_ipm.GenerationError):
            generate_ipm.render_prompt("No placeholder", "spec")

    def test_extracts_fenced_model(self):
        self.assertEqual(generate_ipm.extract_acts(f"Here:\n```acts\n{VALID_ACTS}```"), VALID_ACTS)

    def test_end_to_end_against_mock_ollama(self):
        response = json.dumps(
            {"response": f"```acts\n{VALID_ACTS}```", "done": True}
        ).encode()
        with mock.patch(
            "generate_ipm.urllib.request.urlopen", return_value=FakeResponse(response)
        ) as urlopen:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prompt = root / "prompt.txt"
                spec = root / "case.txt"
                prompt.write_text("Build this:\n{SPECIFICATION}", encoding="utf-8")
                spec.write_text("A browser form", encoding="utf-8")

                result = generate_ipm.run(
                    [
                        "--model", "test-model",
                        "--prompt-file", str(prompt),
                        "--specification-file", str(spec),
                        "--output-dir", str(root / "results"),
                        "--host", "http://ollama.test:11434",
                    ]
                )

                self.assertEqual(result, 0)
                self.assertEqual((root / "results" / "case.acts").read_text(), VALID_ACTS)
                request = urlopen.call_args.args[0]
                request_body = json.loads(request.data)
                self.assertEqual(request.full_url, "http://ollama.test:11434/api/generate")
                self.assertEqual(request_body["model"], "test-model")
                self.assertEqual(request_body["prompt"], "Build this:\nA browser form")
                self.assertFalse(request_body["stream"])


if __name__ == "__main__":
    unittest.main()
