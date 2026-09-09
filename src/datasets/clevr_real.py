"""Real CLEVR v1.0 dataset loader.

Converts the official CLEVR v1.0 release (questions + scene graphs + images) into
the unified ``data/processed/clevr.jsonl`` ITEM_SCHEMA format used by the
evaluation pipeline, the same way ``src/datasets/mmvp.py`` standardizes MMVP.

Unlike ``src/datasets/clevr.py`` (which only synthesizes placeholder items),
this loader reads real ground-truth scene graphs and question answers so that
``gold_facts`` encode the true label and ``image_path`` points to real CLEVR
renderings.

The fact *value* is the dataset's ground-truth answer (the authoritative label
for this real image); predicate/subject/attribute_type are parsed from the
question text so the downstream Z3 verifier can detect contradictions between
VLM claims and ground truth.
"""

import os
import re
import json
import random
from typing import List, Dict, Any, Tuple

CLEVR_ROOT = os.path.join("data", "raw", "clevr", "CLEVR_v1.0")

CLEVR_COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
CLEVR_SHAPES = ["cube", "sphere", "cylinder"]
CLEVR_MATERIALS = ["rubber", "metal"]
CLEVR_SIZES = ["small", "large"]

REL_WORDS = ["left", "right", "above", "below", "in front of", "behind"]


def _normalize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text.strip().lower()


def _find_in(text: str, vocab: List[str]) -> List[str]:
    t = _normalize(text)
    return [v for v in vocab if re.search(rf"\b{re.escape(v)}\b", t)]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_scenes(scenes_path: str) -> Dict[int, Dict[str, Any]]:
    data = _load_json(scenes_path)
    scenes = data.get("scenes", data if isinstance(data, list) else [])
    by_idx: Dict[int, Dict[str, Any]] = {}
    for sc in scenes:
        idx = int(sc.get("image_index", sc.get("imageId", -1)))
        by_idx[idx] = sc
    return by_idx


def _load_questions(questions_path: str) -> List[Dict[str, Any]]:
    data = _load_json(questions_path)
    return data.get("questions", data if isinstance(data, list) else [])


def _strip_plural(word: str) -> str:
    return re.sub(r"s\b$", "", word) if word.endswith("s") else word


def _build_subject(colors, shapes, materials, sizes) -> str:
    parts = []
    if sizes:
        parts.append(sizes[0])
    if colors:
        parts.append(colors[0])
    if materials:
        parts.append(materials[0])
    if shapes:
        parts.append(_strip_plural(shapes[0]))
    return "_".join(parts) if parts else "target_object"


def _derive_fact(question: str, answer: str, question_type: str,
                 answer_type: str) -> Tuple[str, str, Dict[str, Any]]:
    """Derive (schema_answer_type, category, gold_fact) for a CLEVR question."""
    q = _normalize(question)
    colors = _find_in(q, CLEVR_COLORS)
    shapes = _find_in(q, CLEVR_SHAPES)
    materials = _find_in(q, CLEVR_MATERIALS)
    sizes = _find_in(q, CLEVR_SIZES)
    at = _normalize(answer_type)
    qt = _normalize(question_type)
    ans = _normalize(answer)

    if at == "count" or q.startswith("how many"):
        schema_at, category = "count", "counting"
    elif at == "yes/no":
        schema_at = "yes_no"
        category = "existence" if "exist" in qt else "spatial_relation"
    else:
        schema_at, category = "attribute", "attribute"

    subject = _build_subject(colors, shapes, materials, sizes)

    if schema_at == "count":
        try:
            value = int(ans)
        except (ValueError, TypeError):
            m = re.search(r"\b(\d+)\b", ans)
            value = int(m.group(1)) if m else 0
        fact = {"predicate": "count", "subject": subject, "value": value}
    elif schema_at == "yes_no":
        fact = {"predicate": "exists", "subject": subject, "value": ans == "yes"}
    else:
        attr_type = "property"
        for cand in ("color", "shape", "material", "size"):
            if f"query_{cand}" in qt or f"what {cand}" in q or f"how {cand}" in q:
                attr_type = cand
                break
        fact = {"predicate": "attribute", "subject": subject,
                "attribute_type": attr_type, "value": ans}
    return schema_at, category, fact


def _image_path(split: str, image_index: int) -> str:
    fname = f"CLEVR_{split}_{int(image_index):06d}.png"
    return f"data/raw/clevr/CLEVR_v1.0/images/{split}/{fname}"


def prepare_clevr_dataset(
    output_path: str = "data/processed/clevr.jsonl",
    clevr_root: str = CLEVR_ROOT,
    limit: int = 1000,
    seed: int = 42,
    splits: Tuple[str, ...] = ("train", "val"),
) -> List[Dict[str, Any]]:
    """Build a real CLEVR grounding subset from the official v1.0 release.

    Reads questions + scene graphs, derives one symbolic gold fact per question
    (value = ground-truth answer), and writes the standardized JSONL consumed by
    the VLM evaluation pipeline. Output is limited to `limit` items sampled
    deterministically, stratified by category.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    random.seed(seed)

    raw_items: List[Dict[str, Any]] = []
    counter = 0
    for split in splits:
        q_path = os.path.join(clevr_root, "questions", f"CLEVR_{split}_questions.json")
        s_path = os.path.join(clevr_root, "scenes", f"CLEVR_{split}_scenes.json")
        if not (os.path.exists(q_path) and os.path.exists(s_path)):
            continue
        scenes = _load_scenes(s_path)
        questions = _load_questions(q_path)
        for q in questions:
            answer = _normalize(q.get("answer", ""))
            qtext = _normalize(q.get("question", ""))
            at = _normalize(q.get("answer_type", ""))
            qt = _normalize(q.get("question_type", ""))
            img_idx = int(q.get("image_index", q.get("imageId", -1)))

            schema_at, category, fact = _derive_fact(qtext, answer, qt, at)
            counter += 1
            raw_items.append({
                "item_id": f"clevr_{split}_{img_idx:06d}_{counter}",
                "dataset": "clevr",
                "image_path": _image_path(split, img_idx),
                "question": qtext,
                "options": "",
                "answer_type": schema_at,
                "gold_answer": answer,
                "gold_facts": [fact],
                "category": category,
                "split": split,
            })

    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for it in raw_items:
        by_cat.setdefault(it["category"], []).append(it)
    for cat in by_cat:
        random.shuffle(by_cat[cat])

    items: List[Dict[str, Any]] = []
    if by_cat:
        cats = list(by_cat.keys())
        while len(items) < limit and any(by_cat[c] for c in cats):
            for c in cats:
                if by_cat[c]:
                    items.append(by_cat[c].pop())
                if len(items) >= limit:
                    break

    with open(output_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    print(f"Successfully prepared {len(items)} real CLEVR items -> {output_path}")
    return items


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare real CLEVR dataset JSONL")
    parser.add_argument("--output", default="data/processed/clevr.jsonl")
    parser.add_argument("--root", default=CLEVR_ROOT)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    prepare_clevr_dataset(output_path=args.output, clevr_root=args.root, limit=args.limit)
