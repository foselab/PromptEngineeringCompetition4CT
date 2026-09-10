import tempfile
import unittest
from pathlib import Path

from acts_equiv import ModelError, compare_models, parse_model


def model(text: str):
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    with handle:
        handle.write(text)
    return parse_model(Path(handle.name))


class EquivalenceTests(unittest.TestCase):
    def test_reordered_and_logically_equivalent(self):
        first = model("""
[Parameter]
OS (enum): Windows, Linux
Debug (boolean): true, false
[Constraint]
OS = "Windows" => Debug = false
""")
        second = model("""
[Parameter]
Debug (bool): false, true
OS (enum): Linux, Windows
[Constraint]
OS != "Windows" || !Debug
""")
        self.assertTrue(compare_models(first, second).equivalent)

    def test_counterexample(self):
        first = model("""
[Parameter]
P (int): 0, 1, 2
[Constraint]
P >= 1
""")
        second = model("""
[Parameter]
P (int): 0, 1, 2
[Constraint]
P > 1
""")
        result = compare_models(first, second)
        self.assertFalse(result.equivalent)
        self.assertEqual(result.witness, {"P": 1})
        self.assertTrue(result.valid_in_first)
        self.assertFalse(result.valid_in_second)

    def test_domain_difference_can_be_unobservable(self):
        first = model("""
[Parameter]
P (int): 0, 1, 2
[Constraint]
P != 2
""")
        second = model("""
[Parameter]
P (int): 0, 1
""")
        self.assertTrue(compare_models(first, second).equivalent)

    def test_arithmetic_and_bare_boolean(self):
        first = model("""
[Parameter]
X (int): 0, 1, 2
B (boolean): true, false
[Constraint]
B => X + 1 >= 2
""")
        second = model("""
[Parameter]
X (int): 0, 1, 2
B (boolean): true, false
[Constraint]
!B || X >= 1
""")
        self.assertTrue(compare_models(first, second).equivalent)

    def test_unknown_unquoted_enum_value_is_rejected(self):
        bad = model("""
[Parameter]
P (enum): red, blue
[Constraint]
P = red
""")
        with self.assertRaises(ModelError):
            compare_models(bad, bad)

    def test_parameter_name_mismatch(self):
        first = model("[Parameter]\nA (int): 0\n")
        second = model("[Parameter]\nB (int): 0\n")
        result = compare_models(first, second)
        self.assertFalse(result.equivalent)
        self.assertIn("different parameter names", result.reason)


if __name__ == "__main__":
    unittest.main()
