from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .models import get_model
from .preprocessing import build_transforms
from .utils import load_class_mapping


def predict_single_image(image_path: str, model_path: str, device: str = 'cpu'):
    class_name_map = load_class_mapping(model_path)
    reverse_map = {index: label for label, index in class_name_map.items()}
    image = Image.open(image_path).convert('RGB')
    _, eval_tfms = build_transforms(224)
    x = eval_tfms(image).unsqueeze(0)

    model = get_model(num_classes=len(class_name_map), device=device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1)
        prob, pred_idx = torch.max(probs, 1)
    label = reverse_map[int(pred_idx.item())]
    adulteration_pct = int(label.replace('PER', '')) if label.startswith('PER') else 0
    is_adulterated = adulteration_pct > 0
    return {
        'predicted_class': label,
        'adulteration_percent': adulteration_pct,
        'is_adulterated': is_adulterated,
        'confidence': float(prob.item()),
        'predicted_adulterant_type': 'pure turmeric' if adulteration_pct == 0 else 'adulterated turmeric',
    }


def parse_args():
    p = argparse.ArgumentParser(description='Predict whether a turmeric sample is adulterated and by what percent.')
    p.add_argument('--image', type=str, required=True)
    p.add_argument('--model-path', type=str, default='outputs/checkpoints/best_model.pth')
    p.add_argument('--device', type=str, default='cpu')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    result = predict_single_image(args.image, args.model_path, args.device)
    print(json.dumps(result, indent=2))
