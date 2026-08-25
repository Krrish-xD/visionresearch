"""Unit tests for deterministic natural-language answer parsing."""

import unittest
from src.formalization.parser import (
    parse_count,
    parse_yes_no,
    parse_choice_letter,
    parse_vlm_answer_to_claim
)

class TestParser(unittest.TestCase):

    def test_parse_count_digits(self):
        val, norm = parse_count("There are 4 chairs in the room.")
        self.assertEqual(val, 4)
        self.assertEqual(norm, "4")

    def test_parse_count_words(self):
        val, norm = parse_count("three")
        self.assertEqual(val, 3)
        self.assertEqual(norm, "3")

    def test_parse_yes_no(self):
        val, norm = parse_yes_no("Yes, definitely.")
        self.assertEqual(val, True)
        self.assertEqual(norm, "yes")

        val, norm = parse_yes_no("No.")
        self.assertEqual(val, False)
        self.assertEqual(norm, "no")

    def test_parse_choice(self):
        self.assertEqual(parse_choice_letter("(a) open"), "a")
        self.assertEqual(parse_choice_letter("The answer is (B)"), "b")
        self.assertEqual(parse_choice_letter("A"), "a")

    def test_parse_vlm_answer_to_claim(self):
        claim, norm, status = parse_vlm_answer_to_claim(
            raw_answer="4",
            answer_type="count",
            question="How many chair legs are visible?",
            gold_facts=[{"predicate": "count", "subject": "chair_leg", "value": 3}]
        )
        self.assertEqual(status, "success")
        self.assertEqual(claim["predicate"], "count")
        self.assertEqual(claim["subject"], "chair_leg")
        self.assertEqual(claim["value"], 4)

if __name__ == "__main__":
    unittest.main()
