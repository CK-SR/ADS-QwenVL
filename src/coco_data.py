from __future__ import annotations
import json, os, random
from collections import defaultdict
from typing import Dict, List


def load_coco_samples(coco_root: str, split: str = "val2017", limit: int | None = None, seed: int = 42, shuffle: bool = False, skip_empty: bool = True) -> List[Dict]:
    assert split in {"train2017", "val2017"}
    ann = os.path.join(coco_root, "annotations", f"instances_{split}.json")
    with open(ann, "r", encoding="utf-8") as f:
        data = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    image_to_objects = defaultdict(set)
    for a in data["annotations"]:
        image_to_objects[a["image_id"]].add(cat_id_to_name[a["category_id"]])

    samples = []
    for im in data["images"]:
        objs = sorted(image_to_objects.get(im["id"], set()))
        if skip_empty and not objs:
            continue
        samples.append({
            "image_id": im["id"],
            "file_name": im["file_name"],
            "image_path": os.path.join(coco_root, split, im["file_name"]),
            "gt_objects": objs,
        })
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(samples)
    if limit:
        samples = samples[:limit]
    return samples
