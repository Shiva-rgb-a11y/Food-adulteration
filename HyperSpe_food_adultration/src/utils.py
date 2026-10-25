from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: Any) -> Path:
    p = ensure_dir(Path(path).parent) / Path(path).name
    with p.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)
    return p


def load_class_mapping(model_path: str | Path) -> dict[str, int]:
    mapping_path = Path(model_path).resolve().parent.parent / 'class_mapping.json'
    if not mapping_path.is_file():
        raise FileNotFoundError(f'Class mapping not found beside model output: {mapping_path}')

    with mapping_path.open(encoding='utf-8') as f:
        mapping = json.load(f)

    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f'Invalid class mapping in {mapping_path}')
    normalized = {str(label): int(index) for label, index in mapping.items()}
    if sorted(normalized.values()) != list(range(len(normalized))):
        raise ValueError(f'Class indices in {mapping_path} must be contiguous from zero')
    return normalized


def class_to_percent(label: str | int) -> int:
    if isinstance(label, str):
        label = label.lower()
        if label.startswith('per'):
            return int(label[3:])
        if '%' in label:
            return int(label.replace('%', ''))
    return int(label)
