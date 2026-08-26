"""JSON Schema definitions for dataset items, symbolic claims, and VLM predictions."""

from typing import Dict, Any

ITEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["item_id", "dataset", "question", "answer_type", "gold_answer", "gold_facts", "category"],
    "properties": {
        "item_id": {"type": "string"},
        "dataset": {"type": "string", "enum": ["mmvp", "clevr", "gqa"]},
        "image_path": {"type": "string"},
        "image_bytes_base64": {"type": "string"},
        "question": {"type": "string"},
        "options": {"type": "string"},
        "answer_type": {"type": "string", "enum": ["count", "yes_no", "attribute", "relation", "choice"]},
        "gold_answer": {"type": "string"},
        "gold_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["predicate"],
                "properties": {
                    "predicate": {"type": "string", "enum": ["exists", "count", "attribute", "relation", "choice"]},
                    "subject": {"type": "string"},
                    "attribute_type": {"type": "string"},
                    "value": {},
                    "object": {"type": "string"},
                    "relation_type": {"type": "string"}
                }
            }
        },
        "category": {"type": "string"}
    }
}

PREDICTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["item_id", "model", "prompt_id", "raw_answer", "normalized_answer", "raw_confidence", "is_correct"],
    "properties": {
        "item_id": {"type": "string"},
        "model": {"type": "string"},
        "prompt_id": {"type": "string"},
        "raw_answer": {"type": "string"},
        "normalized_answer": {"type": "string"},
        "claim": {
            "type": ["object", "null"],
            "properties": {
                "predicate": {"type": "string"},
                "subject": {"type": "string"},
                "attribute_type": {"type": "string"},
                "value": {},
                "object": {"type": "string"},
                "relation_type": {"type": "string"}
            }
        },
        "raw_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "token_logprobs": {
            "type": "array",
            "items": {"type": "number"}
        },
        "answer_logits": {
            "type": "array",
            "items": {"type": "number"}
        },
        "is_correct": {"type": "boolean"},
        "parse_status": {"type": "string", "enum": ["success", "failed", "unsupported"]}
    }
}

SOLVER_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["item_id", "condition", "solver_status", "is_satisfied", "is_contradicted"],
    "properties": {
        "item_id": {"type": "string"},
        "condition": {"type": "string"},
        "weight": {"type": "number"},
        "solver_status": {"type": "string", "enum": ["sat", "unsat", "unknown", "timeout", "error"]},
        "is_satisfied": {"type": "boolean"},
        "is_contradicted": {"type": "boolean"},
        "solve_time_ms": {"type": "number"}
    }
}
