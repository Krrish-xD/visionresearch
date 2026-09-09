import json
from collections import Counter

with open('data/raw/clevr/CLEVR_v1.0/questions/CLEVR_val_questions.json') as f:
    data = json.load(f)
qs = data['questions']

answer_types = Counter()
for q in qs:
    ans = q['answer']
    if ans in ('yes', 'no'):
        answer_types['yes_no'] += 1
    elif ans.isdigit():
        answer_types['count'] += 1
    else:
        answer_types['attribute'] += 1
print('Answer types:', answer_types)

last_fns = Counter()
for q in qs:
    prog = q.get('program', [])
    if prog:
        last_fns[prog[-1]['function']] += 1
print('Last functions:', last_fns)

for target_fn in ['count', 'exist', 'query_color', 'query_shape', 'query_material', 'query_size', 'equal_int', 'equal_material', 'equal_size', 'equal_color', 'equal_shape']:
    for q in qs:
        prog = q.get('program', [])
        if prog and prog[-1]['function'] == target_fn:
            print(f"\n[{target_fn}] Q: {q['question']}  A: {q['answer']}")
            print(f"  Program functions: {[s['function'] for s in prog]}")
            print(f"  Value inputs: {[s.get('value_inputs', []) for s in prog]}")
            break

# Also check scene graph structure for image_index 0
with open('data/raw/clevr/CLEVR_v1.0/scenes/CLEVR_val_scenes.json') as f:
    scenes = json.load(f)
scene = scenes['scenes'][0]
print("\n\nScene 0:")
print("Objects:")
for i, obj in enumerate(scene['objects']):
    print(f"  [{i}] {obj['color']} {obj['size']} {obj['material']} {obj['shape']} at {obj['pixel_coords']}")
print("Relationships:")
for k, v in scene['relationships'].items():
    print(f"  {k}: {v}")
print("Directions:", scene.get('directions'))
