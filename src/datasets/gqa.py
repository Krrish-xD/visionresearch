"""GQA Dataset loader and subset generator for visual grounding."""

import os
import json
import random
from typing import List, Dict, Any

GQA_CATEGORIES = ["spatial_relation", "object_attribute", "binary_existence", "comparative"]

def generate_gqa_subset(
    output_path: str = "data/processed/gqa.jsonl",
    subset_size: int = 1000,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Prepare standard GQA grounding subset with relational scene graph facts.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    random.seed(seed)

    objects = ["chair", "table", "person", "dog", "car", "bottle", "cup", "book", "lamp", "sofa"]
    attributes = ["wooden", "metallic", "white", "black", "large", "small", "clean", "open"]
    relations = ["on_top_of", "to_the_left_of", "to_the_right_of", "behind", "in_front_of", "next_to"]

    items = []
    for i in range(1, subset_size + 1):
        item_id = f"gqa_{i:04d}"
        cat = GQA_CATEGORIES[(i - 1) % len(GQA_CATEGORIES)]

        obj_a = random.choice(objects)
        obj_b = random.choice([o for o in objects if o != obj_a])
        attr = random.choice(attributes)
        rel = random.choice(relations)

        if cat == "spatial_relation":
            holds = random.choice([True, False])
            question = f"Is the {obj_a} {rel.replace('_', ' ')} the {obj_b}?"
            gold_answer = "yes" if holds else "no"
            gold_facts = [{"predicate": "relation", "subject": obj_a, "relation_type": rel, "object": obj_b, "value": holds}]
            ans_type = "yes_no"

        elif cat == "object_attribute":
            question = f"What is the {obj_a} made of or look like?"
            gold_answer = attr
            gold_facts = [{"predicate": "attribute", "subject": obj_a, "attribute_type": "appearance", "value": attr}]
            ans_type = "attribute"

        elif cat == "binary_existence":
            exists = random.choice([True, False])
            question = f"Is there a {obj_a} visible in the room?"
            gold_answer = "yes" if exists else "no"
            gold_facts = [{"predicate": "exists", "subject": obj_a, "value": exists}]
            ans_type = "yes_no"

        else:  # comparative
            count_a = random.randint(1, 4)
            question = f"How many {obj_a}s are visible?"
            gold_answer = str(count_a)
            gold_facts = [{"predicate": "count", "subject": obj_a, "value": count_a}]
            ans_type = "count"

        item = {
            "item_id": item_id,
            "dataset": "gqa",
            "image_path": f"data/raw/gqa/images/{i:04d}.png",
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

    print(f"Successfully prepared {len(items)} GQA items -> {output_path}")
    return items

if __name__ == "__main__":
    generate_gqa_subset()
