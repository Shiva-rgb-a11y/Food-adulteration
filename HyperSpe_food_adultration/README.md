# HyperSpe Food Adulteration Detection

HyperSpe Food Adulteration Detection is an image-classification project designed to identify turmeric adulteration levels from images. The system classifies turmeric samples into four adulteration percentage categories: `PER0`, `PER5`, `PER10`, and `PER15`.

The project includes a training and evaluation pipeline, validation monitoring across 24 epochs, and final classification metrics. The reported test accuracy is **86.02%**, with a macro F1-score of **86.62%**.

The integrated project design, methodology, implementation guide, and results are documented in [`PROJECT_REPORT.md`](PROJECT_REPORT.md).

## Project Overview

### Task Definition

The objective is to predict the adulteration percentage class of a turmeric image.

| Class | Adulteration Level |
|---|---:|
| `PER0` | 0% |
| `PER5` | 5% |
| `PER10` | 10% |
| `PER15` | 15% |

### Repository Contents

- `data/adultrated_database/` - Image dataset organized by turmeric adulteration percentage.
- `data/fake_and_real_peppers_ms/` - Separate pepper dataset, not used for turmeric classification.
- `data/Multimodal Dataset for Turmeric Adulteration Detec/` - Tabular dataset containing color, FTIR, and image descriptors.
- `outputs/checkpoints/best_model.pth` - Expected location of the trained classification checkpoint.
- `outputs/metrics/training_history.json` - Training and validation metrics recorded across epochs.
- `outputs/metrics/final_test_metrics.json` - Final evaluation metrics.
- `PROJECT_REPORT.md` - Detailed project documentation.

**Note:** `best.pth`, if present at the repository root, was identified as an unrelated checkpoint and should not be confused with the classification model.

## Methodology

The project follows a supervised image-classification workflow:

1. **Dataset preparation:** Load turmeric images organized by adulteration class.
2. **Data splitting:** Separate the data into training, validation, and test sets.
3. **Preprocessing:** Prepare images for model training. Preprocessing parameters should be fitted using training data only wherever applicable.
4. **Model training:** Train the classification model and monitor training loss and validation performance.
5. **Model selection:** Select a checkpoint using the validation results.
6. **Evaluation:** Measure classification performance on the held-out test set.
7. **Inference:** Predict the adulteration class for an individual turmeric image.

The precise model architecture, image transformations, optimizer, and data split ratio should be documented from the implementation and training configuration.

## Training Configuration

The supplied training history contains results for 24 epochs.

| Parameter | Value |
|---|---|
| Training epochs recorded | 24 |
| Initial training loss | 1.0295 |
| Final recorded training loss | 0.1913 |
| Best validation accuracy | 90.81% |
| Best validation macro F1-score | 91.12% |
| Best validation epoch by accuracy | 15 |

The training loss decreased substantially over the recorded epochs, indicating that the model learned useful patterns from the training data. Validation performance fluctuated during training, so generalization should be assessed using the held-out test set.

## Experimental Results

### Final Evaluation Metrics

The following metrics were provided for the final evaluation.

| Metric | Score |
|---|---:|
| Accuracy | 86.02% |
| Balanced Accuracy | 86.08% |
| Macro Precision | 87.68% |
| Macro Recall | 86.08% |
| Macro F1-score | 86.62% |

### Interpretation

- **Accuracy (86.02%):** Approximately 86 out of every 100 evaluated samples were classified correctly.
- **Balanced accuracy (86.08%):** The average recall across the classes was approximately 86.08%, giving each class equal weight.
- **Macro precision (87.68%):** The unweighted average of class-wise precision was approximately 87.68%.
- **Macro recall (86.08%):** The unweighted average of class-wise recall was approximately 86.08%.
- **Macro F1-score (86.62%):** The unweighted average of class-wise F1-scores was approximately 86.62%.

The close relationship between accuracy and balanced accuracy suggests that the overall and class-averaged recall results are similar. However, a confusion matrix and per-class metrics are needed to determine which adulteration levels are most frequently confused.

### Validation Performance

The highest validation accuracy occurred at epoch 15.

| Epoch | Training Loss | Validation Accuracy | Validation Macro F1 |
|---:|---:|---:|---:|
| 1 | 1.0295 | 74.59% | 72.79% |
| 4 | 0.4395 | 87.57% | 87.87% |
| 7 | 0.3645 | 89.73% | 89.98% |
| 12 | 0.2715 | 90.27% | 90.28% |
| 15 | 0.2334 | 90.81% | 91.12% |
| 18 | 0.2213 | 90.27% | 90.26% |
| 24 | 0.1913 | 90.27% | 90.47% |

These selected epochs show the progression of validation performance. The full history is available in the training metrics output, if retained in the repository.

**Important:** The best validation accuracy and final test accuracy are different measurements. Validation accuracy reached 90.81%, while final test accuracy was 86.02%. This difference may reflect generalization difficulty or differences between the validation and test samples. Further analysis is needed before drawing conclusions.

## How to Explore the Dataset

Run the dataset inspection script:

```bash
python src/inspect_data.py
```

Expected outputs:

```text
outputs/
└── dataset_exploration/
    ├── dataset_report.md
    └── dataset_summary.json
```

## How to Train the Model

Run the training pipeline:

```bash
python -m src.train \
    --data-root data/adultrated_database \
    --output-dir outputs \
    --epochs 24 \
    --batch-size 8 \
    --lr 1e-4 \
    --seed 42
```

This command uses 24 epochs and the example batch size, learning rate, and seed from the documented pipeline. Confirm these settings against the configuration used to produce the supplied metrics before claiming that the run reproduces those results.

Expected outputs:

```text
outputs/
├── checkpoints/
│   └── best_model.pth
└── metrics/
    ├── training_history.json
    └── final_test_metrics.json
```

## How to Evaluate the Model

```bash
python -m src.evaluate \
    --model-path outputs/checkpoints/best_model.pth \
    --data-root data/adultrated_database \
    --output-dir outputs/metrics
```

The evaluation script should use the held-out test split and report the metrics defined in the project configuration.

## How to Predict on a Single Image

```bash
python -m src.predict \
    --image path/to/sample.jpg \
    --model-path outputs/checkpoints/best_model.pth
```

Example output format:

```json
{
  "predicted_class": "PER10",
  "adulteration_percent": 10,
  "is_adulterated": true,
  "confidence": 0.81,
  "predicted_adulterant_type": "adulterated turmeric"
}
```

The values above illustrate the output format only. Actual predictions and confidence scores depend on the input image and trained checkpoint.

## Important Notes

- The task is four-class turmeric adulteration classification.
- The pepper dataset is separate and should not be mixed into the turmeric classification task without a specific experimental justification.
- The unrelated root-level `best.pth` checkpoint should not be used unless its architecture and task compatibility are verified.
- Training, validation, and test data must remain appropriately separated to avoid data leakage.
- The final test metrics reported above are the supplied experimental results, not independently reproduced results.
- The current metrics do not establish that the model outperforms other methods because comparable baseline results were not provided.
- Image-level classification identifies a predicted adulteration category; it does not, by itself, establish laboratory-grade chemical composition or food safety.

## Future Improvements

1. Generate a confusion matrix and class-wise precision, recall, and F1-scores.
2. Evaluate the model on additional independently collected turmeric samples.
3. Compare the proposed model against at least two baseline classifiers under identical data splits.
4. Investigate the validation-to-test performance gap.
5. Perform systematic hyperparameter tuning and document the experiments.
6. Evaluate robustness to changes in lighting, camera quality, image backgrounds, and turmeric sample preparation.
7. Document the model architecture, preprocessing pipeline, dataset size, split ratios, and computing environment for reproducibility.

## Conclusion

The HyperSpe Food Adulteration Detection project investigates supervised classification of turmeric images into four adulteration percentage categories. The supplied final evaluation achieved 86.02% accuracy, 86.08% balanced accuracy, and a macro F1-score of 86.62%. Validation accuracy peaked at 90.81% during epoch 15. These results provide a baseline for further evaluation, with per-class error analysis, controlled baseline comparisons, and testing on independent samples being important next steps.
