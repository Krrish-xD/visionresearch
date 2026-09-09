import json

with open('data/raw/gqa/val_sceneGraphs.json') as f:
    sg = json.load(f)

keys = list(sg.keys())
print(f'Number of scene graphs: {len(keys)}')
print(f'First key: {keys[0]}')
sample = sg[keys[0]]
print(f'Sample scene keys: {list(sample.keys())}')
print(f'Image ID: {sample.get("image_id")}')
print(f'Width: {sample.get("width")}, Height: {sample.get("height")}')
print(f'Num objects: {len(sample.get("objects", {}))}')
for obj_id, obj in list(sample['objects'].items())[:5]:
    print(f'  Object {obj_id}: {obj}')
print(f'Relationships sample:', list(sample.get('relationships', {}).values())[:3])
for rel in list(sample.get('objects', {}).values())[0].get('relations', []):
    print(f'  Relation: {rel}')
