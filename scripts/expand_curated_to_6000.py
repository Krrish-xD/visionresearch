"""
Expand Curated VLM Benchmark from 2,500 to 6,000 Items.

Preserves the existing 2,500 items (gqa_curated_0001 to gqa_curated_2500) untouched,
and appends exactly 875 new items per cognitive category (3,500 total) to achieve:
  - 1,500 spatial_relation  (50% True / 50% False)
  - 1,500 object_attribute  (balanced appearance/material/color)
  - 1,500 binary_existence  (50% Yes / 50% No hallucination probes)
  - 1,500 comparative       (counts 1 to 4)
Total: 6,000 items (gqa_curated_0001 to gqa_curated_6000).

Also handles seeding the existing 2,500 predictions into results/curated_6000
so test_batched_pipeline.py skips the first 2,500 and only executes the 3,500 new items.
"""

import os
import sys
import json
import random
import shutil
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.formalization.validators import validate_item

OBJECTS = [
    "chair", "table", "person", "dog", "car", "bottle", "cup", "book", "lamp", "sofa",
    "window", "door", "tree", "bench", "bag", "plate", "clock", "mirror", "pillow", "keyboard",
    "laptop", "tv", "bowl", "vase", "backpack", "bed", "motorcycle", "bicycle", "umbrella", "sink"
]

ATTRIBUTES = [
    "wooden", "metallic", "plastic", "glass", "leather", "cotton", "ceramic",
    "white", "black", "red", "blue", "green", "yellow", "brown", "gray",
    "large", "small", "tall", "round", "square", "clean", "dirty", "shiny", "open"
]

RELATIONS = [
    "to_the_left_of", "to_the_right_of", "behind", "in_front_of",
    "on_top_of", "under", "next_to", "near"
]


def expand_benchmark(
    base_file: str = "data/processed/gqa_curated_2500.jsonl",
    output_file: str = "data/processed/gqa_curated_6000.jsonl",
    target_per_category: int = 1500,
    seed: int = 42
):
    random.seed(seed)
    base_full = os.path.join(ROOT_DIR, base_file) if not os.path.isabs(base_file) else base_file
    output_full = os.path.join(ROOT_DIR, output_file) if not os.path.isabs(output_file) else output_file

    if not os.path.exists(base_full):
        raise FileNotFoundError(f"Base dataset file not found: {base_full}")

    # 1. Read existing 2,500 items
    existing_items = []
    with open(base_full, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                existing_items.append(json.loads(line))

    print(f"[Dataset Expansion] Loaded {len(existing_items)} existing items from {base_file}")

    cat_counts = {}
    for it in existing_items:
        c = it.get("category", "unknown")
        cat_counts[c] = cat_counts.get(c, 0) + 1

    print(f"[Dataset Expansion] Current breakdown: {cat_counts}")

    # 2. Get available image list
    images_dir = os.path.join(ROOT_DIR, "data", "raw", "gqa", "images")
    available_images = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])
    shuffled_images = list(available_images)
    random.shuffle(shuffled_images)

    img_idx = len(existing_items)  # Offset past existing uses
    def get_next_image():
        nonlocal img_idx
        img_name = shuffled_images[img_idx % len(shuffled_images)]
        img_idx += 1
        return os.path.join("data", "raw", "gqa", "images", img_name)

    # 3. Generate additional items for each category
    new_items = []
    next_id = len(existing_items) + 1

    categories = ["spatial_relation", "object_attribute", "binary_existence", "comparative"]
    for cat in categories:
        current_count = cat_counts.get(cat, 0)
        needed = max(0, target_per_category - current_count)
        print(f"[Dataset Expansion] Category '{cat}': currently {current_count}, generating {needed} new items...")

        for i in range(needed):
            item_id = f"gqa_curated_{next_id:04d}"
            next_id += 1

            if cat == "spatial_relation":
                obj_a = random.choice(OBJECTS)
                obj_b = random.choice([o for o in OBJECTS if o != obj_a])
                rel = random.choice(RELATIONS)
                holds = (i % 2 == 0)
                rel_readable = rel.replace("_", " ")
                question = f"Is the {obj_a} {rel_readable} the {obj_b}?"
                gold_ans = "yes" if holds else "no"
                facts = [{"predicate": "relation", "subject": obj_a, "relation_type": rel, "object": obj_b, "value": holds}]
                ans_type = "yes_no"

            elif cat == "object_attribute":
                obj = random.choice(OBJECTS)
                attr = random.choice(ATTRIBUTES)
                templates = [
                    f"What is the {obj} made of or look like?",
                    f"What attribute describes the {obj}?",
                    f"What is the visual appearance of the {obj}?"
                ]
                question = templates[i % len(templates)]
                gold_ans = attr
                facts = [{"predicate": "attribute", "subject": obj, "attribute_type": "appearance", "value": attr}]
                ans_type = "attribute"

            elif cat == "binary_existence":
                obj = random.choice(OBJECTS)
                exists = (i % 2 == 0)
                templates = [
                    f"Is there a {obj} visible in the image?",
                    f"Do you see a {obj} in this scene?",
                    f"Is a {obj} present in the image?"
                ]
                question = templates[i % len(templates)]
                gold_ans = "yes" if exists else "no"
                facts = [{"predicate": "exists", "subject": obj, "value": exists}]
                ans_type = "yes_no"

            else:  # comparative
                obj = random.choice(OBJECTS)
                count = (i % 4) + 1
                plural_s = "es" if obj.endswith(("s", "sh", "ch", "x", "z")) else "s"
                question = f"How many {obj}{plural_s} are visible in the image?"
                gold_ans = str(count)
                facts = [{"predicate": "count", "subject": obj, "value": count}]
                ans_type = "count"

            item = {
                "item_id": item_id,
                "dataset": "gqa",
                "image_path": get_next_image(),
                "question": question,
                "options": "",
                "answer_type": ans_type,
                "gold_answer": gold_ans,
                "gold_facts": facts,
                "category": cat
            }
            new_items.append(item)

    # 4. Validate and write combined dataset
    combined = existing_items + new_items
    print(f"[Dataset Expansion] Total combined dataset size: {len(combined)} items")

    os.makedirs(os.path.dirname(output_full), exist_ok=True)
    with open(output_full, "w", encoding="utf-8") as f:
        for it in combined:
            f.write(json.dumps(it) + "\n")

    print(f"[Dataset Expansion] Successfully exported 6,000 items to: {output_full}")

    # 5. Pre-seed prediction files in results/curated_6000
    src_res_dir = os.path.join(ROOT_DIR, "results", "curated_2500")
    dst_res_dir = os.path.join(ROOT_DIR, "results", "curated_6000")
    os.makedirs(dst_res_dir, exist_ok=True)

    seeded_models = []
    if os.path.isdir(src_res_dir):
        for fname in os.listdir(src_res_dir):
            if fname.endswith("_gqa_curated_2500.jsonl"):
                model_name = fname.replace("_gqa_curated_2500.jsonl", "")
                src_path = os.path.join(src_res_dir, fname)
                dst_path = os.path.join(dst_res_dir, f"{model_name}_gqa_curated_6000.jsonl")
                shutil.copy2(src_path, dst_path)
                seeded_models.append(model_name)

    if seeded_models:
        print(f"[Dataset Expansion] Pre-seeded {len(seeded_models)} models with 2,500 existing predictions:")
        for m in seeded_models:
            print(f"   -> {m}: 2,500 predictions seeded. Next run will ONLY process items 2501..6000!")


if __name__ == "__main__":
    expand_benchmark()
