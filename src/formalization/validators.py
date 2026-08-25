"""Validation routines for data schemas and execution contracts."""

import jsonschema
from typing import Dict, Any, Tuple
from src.formalization.schema import ITEM_SCHEMA, PREDICTION_SCHEMA, SOLVER_RESULT_SCHEMA

def validate_item(item: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a dataset item against ITEM_SCHEMA."""
    try:
        jsonschema.validate(instance=item, schema=ITEM_SCHEMA)
        return True, "valid"
    except jsonschema.ValidationError as e:
        return False, str(e.message)

def validate_prediction(pred: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a VLM prediction record against PREDICTION_SCHEMA."""
    try:
        jsonschema.validate(instance=pred, schema=PREDICTION_SCHEMA)
        return True, "valid"
    except jsonschema.ValidationError as e:
        return False, str(e.message)

def validate_solver_result(result: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a solver output record against SOLVER_RESULT_SCHEMA."""
    try:
        jsonschema.validate(instance=result, schema=SOLVER_RESULT_SCHEMA)
        return True, "valid"
    except jsonschema.ValidationError as e:
        return False, str(e.message)
