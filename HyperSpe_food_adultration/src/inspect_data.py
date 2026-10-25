from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def summarize_image_dataset(root: Path) -> dict:
    image_root = root / 'data' / 'adultrated_database'
    classes = {}
    if image_root.exists():
        for class_dir in sorted(image_root.iterdir()):
            if not class_dir.is_dir():
                continue
            files = sorted(class_dir.rglob('*.jpg'))
            classes[class_dir.name] = {
                'count': len(files),
                'sample_files': [str(p.relative_to(root)) for p in files[:3]],
                'image_shape': None,
            }
            if files:
                from PIL import Image
                img = Image.open(files[0])
                classes[class_dir.name]['image_shape'] = list(img.size)
    return {'image_dataset': classes}


def summarize_csv_datasets(root: Path) -> dict:
    result = {}
    for csv_path in sorted((root / 'data').rglob('*.csv')):
        if 'Multimodal Dataset for Turmeric Adulteration Detec' not in str(csv_path):
            continue
        df = pd.read_csv(csv_path)
        result[csv_path.name] = {
            'rows': int(df.shape[0]),
            'columns': list(df.columns),
            'target_counts': df.iloc[:, -1].value_counts(dropna=False).to_dict() if df.shape[1] > 0 else {},
            'sample_head': df.head(3).to_dict(orient='records'),
        }
    return {'csv_datasets': result}


def summarize_checkpoint(root: Path) -> dict:
    ckpt_path = root / 'best.pth'
    data = {'exists': ckpt_path.exists()}
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        data['type'] = type(ckpt).__name__
        data['keys'] = list(ckpt.keys()) if isinstance(ckpt, dict) else []
        if isinstance(ckpt, dict):
            data['epoch'] = ckpt.get('epoch')
            data['best_val_psnr'] = ckpt.get('best_val_psnr')
            data['args'] = ckpt.get('args')
            data['model_state_dict_keys_sample'] = list(ckpt.get('model_state_dict', {}).keys())[:15]
    return {'checkpoint': data}


def generate_report() -> str:
    image_summary = summarize_image_dataset(ROOT)
    csv_summary = summarize_csv_datasets(ROOT)
    checkpoint_summary = summarize_checkpoint(ROOT)

    report_path = ROOT / 'outputs' / 'dataset_exploration' / 'dataset_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    text = """# Dataset exploration report

## Project discovery

This project contains two principal data families:

1. `data/adultrated_database/` — a turmeric image dataset organized by adulteration percentage classes: `PER0`, `PER5`, `PER10`, and `PER15`.
2. `data/Multimodal Dataset for Turmeric Adulteration Detec/` — a tabular/multimodal turmeric dataset with color, FTIR, and image feature tables.

The checkpoint at `best.pth` was inspected and is not compatible with the percentage-classification task.

## 1) Image dataset summary

"""
    image_meta = image_summary['image_dataset']
    for name, meta in image_meta.items():
        text += f"- `{name}`: {meta['count']} images; sample: {meta['sample_files'][0] if meta['sample_files'] else 'n/a'}; image shape: {meta['image_shape']}\n"

    text += "\n## 2) Tabular dataset summary\n\n"
    for name, meta in csv_summary['csv_datasets'].items():
        text += f"### `{name}`\n"
        text += f"- shape: {meta['rows']} rows x {len(meta['columns'])} columns\n"
        text += f"- columns: {', '.join(meta['columns'][:8])} ...\n"
        if meta['target_counts']:
            text += "- target counts: " + ", ".join(f"{k}: {v}" for k, v in meta['target_counts'].items()) + "\n"
        text += "- sample rows:\n"
        for row in meta['sample_head'][:2]:
            text += '  - ' + json.dumps(row, ensure_ascii=False) + '\n'

    ckpt = checkpoint_summary['checkpoint']
    text += "\n## 3) Checkpoint inspection\n\n"
    text += f"- checkpoint exists: {ckpt['exists']}\n"
    if ckpt['exists']:
        text += f"- type: {ckpt['type']}\n"
        text += f"- keys: {ckpt['keys'][:10]}\n"
        text += f"- epoch: {ckpt['epoch']}\n"
        text += f"- best_val_psnr: {ckpt['best_val_psnr']}\n"
        if isinstance(ckpt.get('args'), dict):
            text += "- checkpoint args: " + json.dumps(ckpt['args'], ensure_ascii=False) + "\n"
        text += "- model_state_dict sample keys: " + ', '.join(ckpt['model_state_dict_keys_sample']) + "\n"
        text += "\nThis checkpoint is a dictionary from a different training run and contains `conv_in`, `body`, and transformer-style keys, which is not compatible with a simple image classifier for percentage classes. It should be treated as an unrelated legacy checkpoint, not as the pretrained model for this task.\n"

    text += "\n## 4) Task choice\n\n"
    text += "The most defensible task in this repository is a classification problem over the `data/adultrated_database` image folders: `PER0`, `PER5`, `PER10`, `PER15`. This maps directly to adulteration percentage labels, and the image counts are large enough for a train/validation/test split.\n"
    text += "The CSV dataset is useful as a secondary tabular benchmark or feature-analysis dataset, but it is not automatically compatible with the image checkpoint or the same preprocessing path unless a dedicated model is implemented for it.\n"
    text += "The pepper dataset is unrelated to the turmeric adulteration problem and should not be mixed with this task.\n"

    report_path.write_text(text, encoding='utf-8')

    json_data = {'image_dataset': image_summary['image_dataset'], 'csv_datasets': csv_summary['csv_datasets'], 'checkpoint': checkpoint_summary['checkpoint']}
    (ROOT / 'outputs' / 'dataset_exploration' / 'dataset_summary.json').write_text(json.dumps(json_data, indent=2), encoding='utf-8')

    return str(report_path)


if __name__ == '__main__':
    print(generate_report())
