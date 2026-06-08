# MRI Classification Evaluation: Detecting Shortcut Learning in Brain Tumor CNNs

**Prathiyanka Arun, Ella Cao, Dhruti Vadlamudi**
Paul G. Allen School of Computer Science and Engineering, University of Washington, Seattle WA

<!-- IMAGE: Project poster (full) or a banner.
     Suggested: export the poster PDF to PNG → save as assets/poster.png -->
![Project Poster](assets/poster.png)

## Overview

CNN-based brain MRI tumor classifiers often achieve very high accuracy, but high accuracy does not guarantee the model is learning medically meaningful features. Models frequently rely on **shortcut learning** — basing predictions on misleading but easy-to-detect patterns (scanner noise, image borders, text markers, acquisition artifacts) rather than clinically relevant anatomy. This is a serious problem for clinical reliability: a model that looks accurate in testing can fail in deployment when those spurious features are absent or different.

This project studies whether a ResNet-18 brain tumor classifier focuses on clinically relevant brain regions or instead exploits non-biological artifacts. We use **Grad-CAM** to visualize the image regions driving predictions, and we test whether simple preprocessing and training interventions reduce shortcut learning.

## Key Findings

- The **baseline** ResNet-18 relied on a **location-based shortcut**, using tumor position relative to the skull boundary as a proxy for class identity. This was most evident in systematic glioma → meningioma misclassifications (meningiomas naturally appear near the skull).
- Grad-CAM showed correct predictions attending to internal brain tissue, while misclassifications concentrated attention on the skull boundary and brain edges.
- A preprocessing **intervention** (tight center crop + augmentation) **partially disrupted** the shortcut: misclassified attention shifted toward interior brain regions rather than the skull edge.
- Accuracy dropped from **96.1% → 93.5%**, suggesting the baseline was partially inflated by spurious spatial correlations rather than genuine tumor morphology recognition.

<!-- IMAGE: Side-by-side Grad-CAM comparison — baseline (skull-edge focus) vs intervention (interior focus).
     Save as assets/gradcam_comparison.png. This is your strongest "money shot" — keep it high up. -->
![Grad-CAM: Baseline vs Intervention](assets/gradcam_comparison.png)
*Baseline misclassifications focus on the skull boundary (location shortcut); intervention misclassifications shift toward interior brain regions.*

| Metric    | Baseline | Intervention |
|-----------|----------|--------------|
| Accuracy  | 0.961    | 0.935        |
| Precision | 0.960    | 0.938        |
| Recall    | 0.964    | 0.936        |

*Baseline ResNet-18 test set performance (macro-averaged).*

<!-- IMAGE: Confusion matrices, baseline and intervention side by side (Figure 1 from poster).
     Save as assets/confusion_matrices.png -->
![Confusion Matrices](assets/confusion_matrices.png)
*The primary error in both models is glioma–meningioma confusion, consistent with location-based shortcut learning.*

<!-- IMAGE: Training + validation accuracy curves over 10 epochs (Figure 2 from poster).
     Save as assets/training_curves.png -->
![Training Curves](assets/training_curves.png)
*Train and validation accuracy over 10 epochs: baseline (left) and intervention (right).*

## Dataset

- **Sartaj Brain Tumor MRI dataset** — 4 classes: glioma, meningioma, pituitary, no tumor.
- Place images under `data/` with one subfolder per class (loaded via `torchvision.datasets.ImageFolder`).
- Data is split 70% train / 15% validation / 15% test.

<!-- IMAGE: A grid of sample MRIs, one per class, with class labels.
     Save as assets/dataset_samples.png -->
![Dataset Samples](assets/dataset_samples.png)
*Representative scans from each class: glioma, meningioma, pituitary, no tumor.*

## Methods

**Preprocessing**
- *Baseline:* Load as RGB, resize to 224×224, convert to tensor, normalize with ImageNet mean/std.
- *Intervention:* Adds `CenterCrop(180)` to cut the skull boundary, plus `RandomHorizontalFlip`, `RandomRotation(15°)`, and `ColorJitter`.

<!-- IMAGE: Before/after of one scan showing the CenterCrop + augmentation pipeline.
     Save as assets/preprocessing.png -->
![Preprocessing Pipeline](assets/preprocessing.png)
*Left: baseline 224×224 resize. Right: intervention with center crop removing the skull boundary, plus augmentation.*


**Model**
- ResNet-18 pretrained on ImageNet, final layer swapped for the target output. Chosen for transfer learning, training stability, and easy Grad-CAM access at the final conv block.

**Training**
- Cross-Entropy Loss, Adam (lr = 1e-4), 10 epochs, batch size 32, on a T4 GPU (Colab). Train and validation accuracy tracked per epoch.

**Grad-CAM**
- Gradients of the predicted class score w.r.t. the final conv layer weight the activation maps, producing a heatmap of where the model looks, overlaid on the MRI.

<!-- IMAGE: Baseline Grad-CAM gallery (Figure 3). Rows 1-2 correct = internal focus;
     rows 3-4 misclassified = skull-boundary focus. Save as assets/gradcam_baseline.png -->
![Baseline Grad-CAM](assets/gradcam_baseline.png)
*Baseline ResNet-18: correct predictions attend inside the brain; misclassifications focus on the skull boundary.*

<!-- IMAGE: Intervention Grad-CAM gallery (Figure 4). Save as assets/gradcam_intervention.png -->
![Intervention Grad-CAM](assets/gradcam_intervention.png)
*Intervention ResNet-18: misclassifications show reduced skull-boundary attention, suggesting partial shortcut mitigation.*


## Repository Structure

```
.
├── data/                       # ImageFolder dataset (one subfolder per class)
├── assets/                     # Images used in this README (see placeholders below)
├── train_baseline.py           # Trains baseline ResNet-18 → baseline_resnet18.pth
├── train_intervention.py       # Trains intervention model → intervention_resnet18.pth
├── evaluate.py                 # Accuracy / precision / recall on the test split
├── gradcam.py                  # Grad-CAM heatmap for a single image
└── README.md
```

## Setup

```bash
pip install torch torchvision scikit-learn opencv-python pillow matplotlib numpy
```

## Usage

**Train the baseline model**
```bash
python train_baseline.py        # saves baseline_resnet18.pth
```

**Train the intervention model**
```bash
python train_intervention.py    # saves intervention_resnet18.pth
```

**Evaluate on the test set**
```bash
python evaluate.py              # prints accuracy, precision, recall
```

**Generate a Grad-CAM visualization**
```bash
python gradcam.py               # set img_path to a real image; saves gradcam_output.png
```

## Next Steps

1. Identify the specific non-medical features influencing predictions.
2. Apply preprocessing to remove artifacts, labels, and irrelevant background.
3. Use segmentation methods to isolate only brain or tumor regions.
4. Retrain and compare Grad-CAM visualizations before and after corrections.
5. Measure whether improvements increase both accuracy and interpretability.

## Limitations

- Grad-CAM is a post-hoc approximation of model attention.
- No radiologist-verified attention labels.
- Small dataset and a single architecture tested.

Simple preprocessing can partially mitigate but not fully resolve shortcut learning, underscoring the need for robust interpretability methods and careful dataset curation before clinical deployment.

## References

- Brown, A., Tomasev, N., Freyberg, J., et al. (2023). Detecting shortcut learning for fair medical AI using shortcut testing. *Nature Communications*, 14, 4314. https://doi.org/10.1038/s41467-023-39902-7
- Hill, B. G., Koback, F. L., & Schilling, P. L. (2024). The risk of shortcutting in deep learning algorithms for medical imaging research. *Scientific Reports*, 14, 29224. https://doi.org/10.1038/s41598-024-79838-6
- Ong Ly, C., Unnikrishnan, B., Tadic, T., et al. (2024). Shortcut learning in medical AI hinders generalization: method for estimating AI model generalization without external data. *npj Digital Medicine*, 7, 124. https://doi.org/10.1038/s41746-024-01118-4
