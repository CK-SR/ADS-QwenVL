from __future__ import annotations

COCO_80_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

ALLOWED_SET = set(COCO_80_CATEGORIES)
SYNONYMS = {
    "sofa": "couch",
    "television": "tv",
    "tv monitor": "tv",
    "phone": "cell phone",
    "mobile phone": "cell phone",
    "cellphone": "cell phone",
    "table": "dining table",
}


def normalize_object_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return SYNONYMS.get(s, s)


def validate_object_name(name: str) -> bool:
    return normalize_object_name(name) in ALLOWED_SET
