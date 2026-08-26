"""Z3 symbolic encoding for hard scene ground-truth facts and soft VLM claims."""

import z3
from typing import Dict, Any, List, Tuple

def sanitize_name(name: str) -> str:
    """Sanitize identifier for Z3 symbols."""
    return str(name).replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("'", "")

def encode_fact_to_z3(fact: Dict[str, Any], prefix: str = "sym") -> Tuple[z3.ExprRef, str]:
    """
    Encode a structured fact dictionary into a Z3 boolean expression.
    
    Supported predicates:
      - count(subject) = integer
      - exists(subject) = boolean
      - attribute(subject, attribute_type) = value (mapped to integer or enum representation)
      - relation(subject, relation_type, object) = boolean
      - choice(subject) = choice_id
    """
    predicate = fact.get("predicate", "")
    subject = sanitize_name(fact.get("subject", "obj"))
    value = fact.get("value")

    if predicate == "count":
        var = z3.Int(f"count_{subject}")
        try:
            int_val = int(value) if value is not None else 0
        except (ValueError, TypeError):
            int_val = abs(hash(str(value).lower())) % 100000
        return (var == int_val), f"count_{subject}"

    elif predicate == "exists":
        var = z3.Bool(f"exists_{subject}")
        bool_val = bool(value) if value is not None else True
        return (var == bool_val), f"exists_{subject}"

    elif predicate == "attribute":
        attr_type = sanitize_name(fact.get("attribute_type", "prop"))
        val_str = str(value).lower()
        val_id = abs(hash(val_str)) % 100000
        var = z3.Int(f"attr_{subject}_{attr_type}")
        return (var == val_id), f"attr_{subject}_{attr_type}"

    elif predicate == "relation":
        rel_type = sanitize_name(fact.get("relation_type", "related"))
        obj = sanitize_name(fact.get("object", "target"))
        var = z3.Bool(f"rel_{subject}_{rel_type}_{obj}")
        bool_val = bool(value) if value is not None else True
        return (var == bool_val), f"rel_{subject}_{rel_type}_{obj}"

    elif predicate == "choice":
        val_str = str(value).lower()
        val_id = abs(hash(val_str)) % 100000
        var = z3.Int(f"choice_{subject}")
        return (var == val_id), f"choice_{subject}"

    # Default generic boolean symbol
    var = z3.Bool(f"sym_{subject}_{predicate}")
    return (var == True), f"sym_{subject}_{predicate}"
