from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .dataset import build_data_loaders
from .models import get_model


def load_best_model(model_path: str, n_classes: int, device: str):
    model = get_model(num_classes=n_classes, device=device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model.to(device)


def evaluate(model_path: str, data_root: str, output_dir: str, device: str = 'cpu'):
    _, _, test_loader, class_to_idx = build_data_loaders(data_root=data_root, batch_size=8)
    model = load_best_model(model_path, len(class_to_idx), device)
    model.eval()

    all_y, all_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1).cpu().tolist()
            labels = labels.cpu().tolist()
            all_y.extend(labels)
            all_pred.extend(pred)

    cm = confusion_matrix(all_y, all_pred, labels=list(range(len(class_to_idx))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_to_idx.keys()))
    disp.plot(cmap='Blues', xticks_rotation=45)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path / 'confusion_matrix.png', dpi=200)
    plt.close()

    metrics = {
        'labels': list(class_to_idx.keys()),
        'confusion_matrix': cm.tolist(),
    }
    (out_path / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', type=str, default='outputs/checkpoints/best_model.pth')
    p.add_argument('--data-root', type=str, default='data/adultrated_database')
    p.add_argument('--output-dir', type=str, default='outputs/metrics')
    p.add_argument('--device', type=str, default='cpu')
    return p.parse_args()


if __name__ == '__main__':
    evaluate(**vars(parse_args()))
