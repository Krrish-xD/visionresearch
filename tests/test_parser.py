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

    def test_parse_options_map_and_text_matching(self):
        from src.formalization.parser import parse_options_map, match_option_letter_or_text
        opts = "(a) Open (b) Closed"
        opt_map = parse_options_map(opts)
        self.assertEqual(opt_map["a"], "open")
        self.assertEqual(opt_map["b"], "closed")

        # Direct text matching
        self.assertEqual(match_option_letter_or_text("open", options=opts), "a")
        self.assertEqual(match_option_letter_or_text("Closed.", options=opts), "b")
        self.assertEqual(match_option_letter_or_text("(A)", options=opts), "a")

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

    def test_parse_vlm_answer_to_claim_with_option_text(self):
        claim, norm, status = parse_vlm_answer_to_claim(
            raw_answer="Open",
            answer_type="relation",
            question="Are the butterfly's wings closer to being open or closed?",
            gold_facts=[{"predicate": "relation", "subject": "item_1", "value": "(a)", "attribute_type": "option"}],
            options="(a) Open (b) Closed"
        )
        self.assertEqual(status, "success")
        self.assertEqual(norm, "(a)")
        self.assertEqual(claim["value"], "(a)")

if __name__ == "__main__":
    unittest.main()
