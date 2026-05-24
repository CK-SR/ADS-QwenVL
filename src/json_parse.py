from __future__ import annotations
import json, re
from .coco_vocab import normalize_object_name, validate_object_name


def extract_first_json(text: str) -> str | None:
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0) if m else None


def parse_objects_response(raw_response: str):
    invalid_json = False
    invalid_objects = []
    parsed_objects = []
    js = extract_first_json(raw_response)
    if js is None:
        return {"invalid_json": True, "invalid_objects": [], "parsed_objects": []}
    try:
        obj = json.loads(js)
    except Exception:
        return {"invalid_json": True, "invalid_objects": [], "parsed_objects": []}
    if "objects" not in obj or not isinstance(obj["objects"], list):
        return {"invalid_json": True, "invalid_objects": [], "parsed_objects": []}
    for x in obj["objects"]:
        n = normalize_object_name(str(x))
        if validate_object_name(n):
            parsed_objects.append(n)
        else:
            invalid_objects.append(str(x))
    dedup = []
    seen = set()
    for x in parsed_objects:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return {"invalid_json": invalid_json, "invalid_objects": invalid_objects, "parsed_objects": dedup}
