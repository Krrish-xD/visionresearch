"""Deterministic parser mapping natural-language / constrained VLM answers to symbolic claims."""

import re
from typing import Dict, Any, Optional, Tuple

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15
}

def normalize_text(text: str) -> str:
    """Strip whitespace, markdown, and lower-case text."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = re.sub(r"[\*\_`]", "", text)
    return text

def parse_choice_letter(raw_answer: str) -> Optional[str]:
    """Extract choice letter (e.g. 'a', 'b', 'c', 'd') from raw response."""
    text = normalize_text(raw_answer)
    m = re.search(r"\b\(([a-d])\)", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([a-d])\b", text)
    if m:
        return m.group(1)
    if text.startswith("a") or text.startswith("option a"):
        return "a"
    if text.startswith("b") or text.startswith("option b"):
        return "b"
    return None

def parse_count(raw_answer: str) -> Tuple[Optional[int], str]:
    """Extract integer count from answer string."""
    text = normalize_text(raw_answer)
    # Check for direct integer
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return int(m.group(1)), str(int(m.group(1)))
    # Check for number words
    for word, val in WORD_TO_NUM.items():
        if re.search(rf"\b{word}\b", text):
            return val, str(val)
    return None, text

def parse_yes_no(raw_answer: str) -> Tuple[Optional[bool], str]:
    """Extract boolean value from answer string."""
    text = normalize_text(raw_answer)
    if re.search(r"\b(yes|true|correct|present)\b", text):
        return True, "yes"
    if re.search(r"\b(no|false|incorrect|absent)\b", text):
        return False, "no"
    return None, text

def extract_subject_from_question(question: str) -> str:
    """Extract default subject from question text."""
    q = normalize_text(question)
    q = re.sub(r"[^\w\s]", "", q)
    # e.g., 'how many chair legs are visible' -> 'chair_legs'
    m = re.search(r"how many\s+([a-z\s]+?)\s+(are|is|can|do)", q)
    if m:
        subj = m.group(1).strip().replace(" ", "_")
        return subj
    m = re.search(r"is there (a|an)?\s*([a-z\s]+)", q)
    if m:
        subj = m.group(2).strip().replace(" ", "_")
        return subj
    tokens = [t for t in q.split() if t not in {"is", "are", "the", "a", "an", "of", "in", "on", "how", "many", "what", "which", "there", "visible"}]
    return "_".join(tokens[:2]) if tokens else "object"

def parse_vlm_answer_to_claim(
    raw_answer: str,
    answer_type: str,
    question: str = "",
    gold_facts: Optional[list] = None,
    options: str = ""
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """
    Parse a raw VLM output into a structured symbolic claim.
    
    Returns:
        (claim_dict, normalized_answer, parse_status)
    """
    norm_text = normalize_text(raw_answer)
    subject = "target_object"
    if gold_facts and len(gold_facts) > 0 and isinstance(gold_facts[0], dict):
        ref_fact = gold_facts[0]
        subject = ref_fact.get("subject", subject)
    elif question:
        subject = extract_subject_from_question(question)

    if answer_type == "count":
        val, norm = parse_count(raw_answer)
        if val is not None:
            return {"predicate": "count", "subject": subject, "value": val}, norm, "success"
        return None, norm_text, "failed"

    elif answer_type == "yes_no":
        val, norm = parse_yes_no(raw_answer)
        if val is not None:
            return {"predicate": "exists", "subject": subject, "value": val}, norm, "success"
        return None, norm_text, "failed"

    elif answer_type == "attribute":
        # Extract attribute value
        tokens = norm_text.split()
        if tokens:
            attr_val = tokens[0]
            attr_type = "property"
            if gold_facts and len(gold_facts) > 0 and isinstance(gold_facts[0], dict) and "attribute_type" in gold_facts[0]:
                attr_type = gold_facts[0]["attribute_type"]
            return {"predicate": "attribute", "subject": subject, "attribute_type": attr_type, "value": attr_val}, attr_val, "success"
        return None, norm_text, "failed"

    elif answer_type == "choice" or (options and parse_choice_letter(raw_answer) is not None):
        letter = parse_choice_letter(raw_answer)
        if letter is not None:
            # Map choice letter to claim if gold facts are structured
            if gold_facts and len(gold_facts) > 0 and isinstance(gold_facts[0], dict):
                gold_fact = gold_facts[0]
                # If letter matches gold or differs, determine value
                return {
                    "predicate": gold_fact.get("predicate", "choice"),
                    "subject": gold_fact.get("subject", subject),
                    "value": f"({letter})",
                    "attribute_type": gold_fact.get("attribute_type", "option")
                }, f"({letter})", "success"
            return {"predicate": "choice", "subject": subject, "value": f"({letter})"}, f"({letter})", "success"
        return None, norm_text, "failed"

    elif answer_type == "relation":
        # Check standard spatial relations
        for rel in ["left", "right", "above", "below", "front", "behind", "next_to", "inside"]:
            if rel in norm_text:
                obj_b = "reference_object"
                if gold_facts and len(gold_facts) > 0 and isinstance(gold_facts[0], dict):
                    obj_b = gold_facts[0].get("object", obj_b)
                return {"predicate": "relation", "subject": subject, "relation_type": rel, "object": obj_b}, rel, "success"
        return None, norm_text, "failed"

    return None, norm_text, "unsupported"
