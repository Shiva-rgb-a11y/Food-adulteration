from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import build_data_loaders
from .models import get_model
from .utils import ensure_dir, save_json


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {'accuracy': acc, 'balanced_accuracy': bal_acc, 'f1_macro': f1, 'precision_macro': precision, 'recall_macro': recall}


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / 'checkpoints')
    ensure_dir(out_dir / 'metrics')

    train_loader, val_loader, test_loader, class_to_idx = build_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
    )

    model = get_model(num_classes=len(class_to_idx), device=device.type)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state = None
    best_val_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_balanced_accuracy': val_metrics['balanced_accuracy'],
            'val_f1_macro': val_metrics['f1_macro'],
        })

        if val_metrics['f1_macro'] > best_val_f1 + 1e-4:
            best_val_f1 = val_metrics['f1_macro']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, out_dir / 'checkpoints' / 'best_model.pth')

        if epoch - history.index(max(history, key=lambda x: x['val_f1_macro'])) >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, device)
    save_json(out_dir / 'metrics' / 'final_test_metrics.json', test_metrics)
    save_json(out_dir / 'metrics' / 'training_history.json', history)
    save_json(out_dir / 'class_mapping.json', {k: int(v) for k, v in class_to_idx.items()})

    print('best_validation_f1_macro', best_val_f1)
    print('test_metrics', json.dumps(test_metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description='Train adulteration-percent image classifier.')
    parser.add_argument('--data-root', type=str, default='data/adultrated_database')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
