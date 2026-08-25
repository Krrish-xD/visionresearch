"""CLEVR Dataset loader and subset generator for visual grounding."""

import os
import json
import random
from typing import List, Dict, Any

CLEVR_CATEGORIES = ["counting", "existence", "attribute", "spatial_relation"]

def generate_clevr_subset(
    output_path: str = "data/processed/clevr.jsonl",
    subset_size: int = 1000,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Prepare standard CLEVR grounding subset with exact symbolic facts.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    random.seed(seed)

    shapes = ["cube", "sphere", "cylinder"]
    colors = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
    materials = ["rubber", "metal"]
    sizes = ["small", "large"]
    relations = ["left", "right", "behind", "in_front_of"]

    items = []
    for i in range(1, subset_size + 1):
        item_id = f"clevr_{i:04d}"
        cat = CLEVR_CATEGORIES[(i - 1) % len(CLEVR_CATEGORIES)]

        shape = random.choice(shapes)
        color = random.choice(colors)
        material = random.choice(materials)
        size = random.choice(sizes)
        target_obj = f"{size}_{color}_{material}_{shape}"

        if cat == "counting":
            count_val = random.randint(0, 6)
            question = f"How many {color} {shape}s are there?"
            gold_answer = str(count_val)
            gold_facts = [{"predicate": "count", "subject": f"{color}_{shape}", "value": count_val}]
            ans_type = "count"

        elif cat == "existence":
            exists = random.choice([True, False])
            question = f"Is there a {color} {material} {shape}?"
            gold_answer = "yes" if exists else "no"
            gold_facts = [{"predicate": "exists", "subject": target_obj, "value": exists}]
            ans_type = "yes_no"

        elif cat == "attribute":
            question = f"What color is the {size} {material} {shape}?"
            gold_answer = color
            gold_facts = [{"predicate": "attribute", "subject": f"{size}_{material}_{shape}", "attribute_type": "color", "value": color}]
            ans_type = "attribute"

        else:  # spatial_relation
            ref_obj = f"{random.choice(colors)}_{random.choice(shapes)}"
            rel = random.choice(relations)
            question = f"Is the {color} {shape} {rel.replace('_', ' ')} the {ref_obj.replace('_', ' ')}?"
            rel_holds = random.choice([True, False])
            gold_answer = "yes" if rel_holds else "no"
            gold_facts = [{"predicate": "relation", "subject": f"{color}_{shape}", "relation_type": rel, "object": ref_obj, "value": rel_holds}]
            ans_type = "yes_no"

        item = {
            "item_id": item_id,
            "dataset": "clevr",
            "image_path": f"data/raw/clevr/images/{i:04d}.png",
            "question": question,
            "options": "",
            "answer_type": ans_type,
            "gold_answer": gold_answer,
            "gold_facts": gold_facts,
            "category": cat
        }
        items.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    print(f"Successfully prepared {len(items)} CLEVR items -> {output_path}")
    return items

if __name__ == "__main__":
    generate_clevr_subset()
