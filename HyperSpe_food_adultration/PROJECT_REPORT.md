# Turmeric Adulteration Detection from Images

## Project Report: Design, Implementation, and Evaluation

**Project:** HyperSpe Food Adulteration Detection  
**Task:** Classify turmeric images into four adulteration-percentage categories  
**Submission contents:** Project design, methods, implementation guide, recorded results, and limitations in one report; runnable source modules remain in `src/`.

## Abstract

This project implements an image-classification pipeline for estimating turmeric adulteration level from an image. It uses the image folders in `data/adultrated_database/`, whose labels are `PER0`, `PER5`, `PER10`, and `PER15`. A pretrained ResNet-18 backbone is fine-tuned as a four-class classifier. The pipeline includes dataset inspection, stratified train/validation/test splitting, image augmentation, model training, held-out evaluation, and single-image inference. The recorded test metrics are 86.02% accuracy and 0.8662 macro-F1. These are results on the repository's current image split, not a guarantee of performance on new acquisition conditions or independent producers.

## 1. Problem Definition and Scope

The system predicts one of four known adulteration-percentage categories from a turmeric image:

| Folder label | Meaning |
| --- | ---: |
| `PER0` | 0% adulteration (pure class) |
| `PER5` | 5% adulteration |
| `PER10` | 10% adulteration |
| `PER15` | 15% adulteration |

This is a **closed-set, four-class image classification task**. The system does not estimate arbitrary percentages, identify an adulterant's chemical identity, or replace laboratory testing. The pepper image dataset and the separate color/FTIR/feature CSV collection are not mixed into this model: they have different data modalities and label schemes and would require a separate, explicitly designed experiment.

### Dataset summary

The generated dataset exploration report records 1,236 JPEG images:

| Class | Images |
| --- | ---: |
| `PER0` | 300 |
| `PER5` | 308 |
| `PER10` | 319 |
| `PER15` | 309 |
| **Total** | **1,236** |

The class counts are relatively balanced. Images are reported as 1280 x 1024 pixels and are resized to 224 x 224 for the network. The class folders contain nested `T*` directories; the current splitting code stratifies by image label, not by `T*` group. This distinction matters for interpreting generalization and is discussed in Section 7.

## 2. Measurable Objectives

1. **Dataset integrity:** inspect the image classes and report per-class image counts before training; accept the four defined class folders and exclude unrelated folders from the classifier.
2. **Reproducible evaluation:** with seed 42, split the image samples into 70% training, 15% validation, and 15% test partitions using stratified sampling. Keep the test partition out of model selection and gradient updates.
3. **Training objective:** minimize four-class cross-entropy loss while fine-tuning a pretrained ResNet-18; select the saved model by validation macro-F1 rather than test performance.
4. **Performance reporting:** measure test accuracy, balanced accuracy, macro precision, macro recall, and macro-F1. Report the observed result without presenting it as a deployment guarantee.
5. **Usable inference:** accept one image and return its predicted folder label, corresponding percentage, adulterated/pure flag, and model softmax confidence.
6. **Reproducible hand-off:** save the best model, class-index mapping, training history, and evaluation metrics, and document commands to inspect, train, evaluate, and predict.

The repository's current recorded test results establish a baseline, not a pre-registered acceptance threshold. A future release should specify a target metric and validate it on a group-held-out dataset before claiming operational readiness.

## 3. Methodology and Algorithm Choices

### 3.1 Data preparation

`src/dataset.py` discovers JPEG files in the four supported class folders and creates label IDs from sorted folder names. It uses two stratified `train_test_split` calls with `random_state=42`: first, 30% is held out; then that portion is divided equally into validation and test. On 1,236 images this gives approximately 865 training, 185 validation, and 186 test images (exact counts follow the stratified split). Stratification preserves approximate class proportions in each partition.

Training images are resized to 224 x 224, randomly horizontally flipped and rotated, and given modest color jitter. Validation and test images receive resize only. All images are converted to tensors and normalized with the ImageNet channel means and standard deviations. Training-only augmentation makes the model less dependent on exact image orientation and modest acquisition/color variation while deterministic evaluation transforms keep comparisons consistent.

### 3.2 Classifier

The implementation in `src/models.py` uses **ResNet-18**, initialized with torchvision's default pretrained ImageNet weights. Its original fully connected head is replaced by dropout (0.2) and a linear layer with four outputs.

This is a practical choice for this dataset because:

- Transfer learning reuses visual features learned from a large image corpus, which is useful when the labeled project dataset is modest in size.
- ResNet-18 is substantially smaller than deeper common backbones, keeping fine-tuning and inference comparatively accessible.
- The residual architecture is a well-established baseline for image classification and is readily available in torchvision.
- A four-logit output and cross-entropy loss directly represent the four mutually exclusive folder classes.

These choices establish a defensible baseline; the repository does not contain a controlled model comparison proving ResNet-18 is optimal. The unrelated `best.pth` file contains transformer-style reconstruction checkpoint keys and is not used by this classifier.

### 3.3 Optimization and model selection

`src/train.py` uses AdamW, cross-entropy loss, a cosine annealing learning-rate schedule, gradient-norm clipping at 1.0, and seeded NumPy/PyTorch random generators. The command-line defaults are five epochs, batch size 8, learning rate $10^{-4}$, weight decay $10^{-4}$, image size 224, patience 3, and seed 42. These are the current defaults, not verified settings for the already-saved result: `training_history.json` contains 24 epoch records, and the artifact does not store the exact training arguments. The available history's maximum validation macro-F1 is 0.9112 at epoch 15.

The checkpoint is selected using validation macro-F1. Macro averaging gives each class equal weight, making it informative even if class frequencies differ. Test data is reserved for final reporting and is not used to select the checkpoint.

### 3.4 Evaluation measures

- **Accuracy:** fraction of test examples classified correctly.
- **Balanced accuracy:** mean per-class recall, reducing the influence of class frequency.
- **Macro precision, recall, and F1:** compute each measure per class and average classes equally.

The standalone evaluation command writes a confusion-matrix image and a JSON file containing the held-out test confusion matrix and labels. The training run writes aggregate test metrics to `outputs/metrics/final_test_metrics.json`.

## 4. Implementation and Module Demonstration

The runnable implementation is organized into these modules:

| Module | Responsibility / demonstrated behavior |
| --- | --- |
| `src/inspect_data.py` | Inspects image classes and counts, summarizes the separate CSV files, and records that the legacy root checkpoint is incompatible. Writes dataset exploration artifacts. |
| `src/preprocessing.py` | Defines training augmentation and deterministic evaluation preprocessing; loads images as RGB. |
| `src/dataset.py` | Builds stratified loaders and preserves the generated class-to-index mapping. |
| `src/models.py` | Defines the pretrained ResNet-18 classifier and device placement. |
| `src/train.py` | Seeds training, optimizes the network, selects by validation macro-F1, and saves weights, history, class mapping, and test metrics. |
| `src/evaluate.py` | Loads the classifier and evaluates only the held-out test split; writes a confusion matrix and label/confusion-matrix JSON. |
| `src/predict.py` | Loads one image and returns the predicted class, percentage, adulteration flag, and confidence. The mapping is read from the class-mapping artifact saved beside the checkpoint's output directory. |
| `src/utils.py` | Provides directory/JSON helpers, percentage conversion, and validated loading of the saved class mapping. |

The saved mapping is `PER0: 0`, `PER10: 1`, `PER15: 2`, `PER5: 3`, matching sorted folder order. Inference now reads this artifact instead of assuming a separate hard-coded order, preventing valid network outputs from being translated into the wrong percentage. The evaluator also reports test-only predictions rather than combining validation and test examples.

### Reproduction commands

Run these from the repository root after installing `requirements.txt` in an environment with a compatible PyTorch/torchvision installation. Pretrained weights may need to be downloaded on the first training run.

```bash
python -m pip install -r requirements.txt
python src/inspect_data.py
python -m src.train --data-root data/adultrated_database --output-dir outputs --epochs 5 --batch-size 8 --lr 1e-4 --seed 42
python -m src.evaluate --model-path outputs/checkpoints/best_model.pth --data-root data/adultrated_database --output-dir outputs/metrics
python -m src.predict --image data/adultrated_database/PER0/T1/MSI_0.jpg --model-path outputs/checkpoints/best_model.pth
```

Training writes `outputs/checkpoints/best_model.pth`, `outputs/class_mapping.json`, `outputs/metrics/training_history.json`, and `outputs/metrics/final_test_metrics.json`. Evaluation writes `outputs/metrics/confusion_matrix.png` and `outputs/metrics/metrics.json`. Prediction prints a JSON record; confidence is the model's maximum softmax probability and should not be interpreted as calibrated probability without a calibration study.

## 5. Recorded Results

The committed result artifact, `outputs/metrics/final_test_metrics.json`, contains:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.8602 (86.02%) |
| Balanced accuracy | 0.8608 (86.08%) |
| Macro precision | 0.8768 (87.68%) |
| Macro recall | 0.8608 (86.08%) |
| Macro-F1 | 0.8662 (86.62%) |

The recorded history has 24 epochs. Its highest validation macro-F1 is 0.9112 at epoch 15. Validation scores are model-selection evidence; the test metrics above are the final held-out estimate for the saved run. Exact historical training arguments are unavailable, so the defaults in the reproduction command should be treated as a documented rerun recipe, not asserted as the provenance of these metrics.

## 6. Outputs and Submission Structure

- `PROJECT_REPORT.md` — this integrated project design, implementation guide, and result report.
- `README.md` — short project overview and entry point to this report.
- `src/` — implementation modules described above.
- `outputs/dataset_exploration/` — generated dataset report and machine-readable summary.
- `outputs/checkpoints/best_model.pth` — selected classifier weights.
- `outputs/class_mapping.json` — class-to-output-index contract used by training and inference.
- `outputs/metrics/` — saved test metrics, training history, and generated evaluation outputs.

## 7. Limitations and Next Steps

1. **Image-level split risk:** nested `T*` folders may represent related captures or sample groups. Since the current split is by image, related images can appear in different partitions and inflate apparent generalization. The next evaluation should split by the appropriate physical sample/group identifier and compare results with this baseline.
2. **Limited external validity:** the dataset covers four percentages and one repository-specific acquisition setup. Performance on other adulterants, percentages, devices, lighting, sample preparation, or suppliers is unknown.
3. **No chemical confirmation:** visual appearance is an indirect signal. Predictions should be treated as screening output, not proof of composition or a replacement for validated laboratory analysis.
4. **Limited experiment record:** the saved history does not include the exact arguments, software versions, or hardware. Record these with future runs and pin dependencies for stronger reproducibility.
5. **No comparative study or calibration:** no baselines, confidence calibration, uncertainty rejection, or external test set are included. Compare a simple classical image-feature baseline and at least one alternative pretrained backbone under the same group-based split before selecting a production model.
6. **Multimodal data is out of scope:** the CSV dataset contains color, FTIR, and image-feature tables with a distinct label scheme; it should only be incorporated after matching samples and targets and defining a leakage-safe multimodal evaluation.

## 8. Conclusion

The repository implements an end-to-end four-class turmeric-image classification baseline, from inspection and preprocessing through training, held-out evaluation, and single-image prediction. The saved run achieves 86.02% test accuracy and 0.8662 macro-F1. The report makes the task and algorithm choices explicit and documents how to reproduce the workflow. The most important next scientific improvement is a group-held-out evaluation, followed by better run metadata and an external validation set.
