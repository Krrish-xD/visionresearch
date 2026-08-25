"""Tests for dataset loading, categorization, and schema compliance."""

import pytest
import os
import tempfile
import json
from src.datasets.mmvp import map_mmvp_category, determine_answer_type, MMVP_CATEGORIES
from src.datasets.clevr import generate_clevr_subset
from src.datasets.gqa import generate_gqa_subset
from src.formalization.validators import validate_item

class TestDatasetUtilities:
    def test_map_mmvp_category(self):
        # Index 1 should map to the first category
        assert map_mmvp_category(1) == MMVP_CATEGORIES[0]
        # Index 35 should map to the second category
        assert map_mmvp_category(35) == MMVP_CATEGORIES[1]
        # Out-of-range index should be capped to the last category
        assert map_mmvp_category(999) == MMVP_CATEGORIES[-1]

    def test_determine_answer_type(self):
        assert determine_answer_type("How many cats are there?", "") == "count"
        assert determine_answer_type("Is there a blue cube?", "") == "yes_no"
        assert determine_answer_type("What color is the cylinder?", "") == "attribute"
        assert determine_answer_type("Which object is to the left of the table?", "") == "relation"
        assert determine_answer_type("Choose the correct perspective", "(a) Front (b) Back") == "choice"

    def test_generate_clevr_subset_schema_validity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "clevr_test.jsonl")
            items = generate_clevr_subset(output_path=out_file, subset_size=10, seed=123)
            assert len(items) == 10
            assert os.path.exists(out_file)
            
            for it in items:
                valid, msg = validate_item(it)
                assert valid, f"Item validation failed: {msg}"
                assert it["dataset"] == "clevr"
                assert "item_id" in it
                assert len(it["gold_facts"]) >= 1

    def test_generate_gqa_subset_schema_validity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "gqa_test.jsonl")
            items = generate_gqa_subset(output_path=out_file, subset_size=10, seed=123)
            assert len(items) == 10
            assert os.path.exists(out_file)

            for it in items:
                valid, msg = validate_item(it)
                assert valid, f"Item validation failed: {msg}"
                assert it["dataset"] == "gqa"
                assert "item_id" in it
                assert len(it["gold_facts"]) >= 1
