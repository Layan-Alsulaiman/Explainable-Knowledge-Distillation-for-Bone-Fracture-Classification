# 🦴 Explainable Knowledge Distillation for Bone Fracture Classification

An Explainable Knowledge Distillation (EKD) framework for binary bone fracture classification from X-ray images, developed as part of the **KAUST Academy Summer Program 2026**.

The project combines **Knowledge Distillation (KD)** with **Explainable Artificial Intelligence (XAI)** to train a lightweight student model that learns not only the predictive behavior of a larger teacher model, but also its class-discriminative spatial attention through **Grad-CAM-based attention alignment**.

---

## Overview

Deep learning models can achieve strong performance in medical image classification, but high-capacity architectures are often computationally expensive and difficult to interpret.

This project investigates whether knowledge from a large fracture-classification model can be transferred to a smaller model while also encouraging the student to focus on similar anatomically relevant regions.

The final framework uses:

* **Teacher:** EfficientNet-B6
* **Student:** MobileNetV2
* **Dataset:** FracAtlas
* **Task:** Binary Fracture / Non-Fracture classification
* **Explainability:** Grad-CAM
* **Distillation:** Prediction-based KD + Grad-CAM attention alignment

The EKD objective combines:

1. Weighted supervised classification loss
2. Temperature-scaled knowledge distillation loss
3. Teacher-student Grad-CAM attention-alignment loss

This allows the student to learn both **what to predict** and information about **where to focus**.

<img width="2048" height="1024" alt="General Framework" src="https://github.com/user-attachments/assets/887c7467-f015-4ce8-a466-bef4dbc07d7b" />

---

## Dataset

The experiments use the **FracAtlas** musculoskeletal X-ray dataset.

| Class        |    Images |
| ------------ | --------: |
| Non-Fracture |     3,366 |
| Fracture     |       717 |
| **Total**    | **4,083** |

The dataset was divided into fixed training, validation, and testing partitions:

| Split      | Total | Non-Fracture | Fracture |
| ---------- | ----: | -----------: | -------: |
| Training   | 3,266 |        2,692 |      574 |
| Validation |   485 |          403 |       82 |
| Testing    |   332 |          271 |       61 |

A fixed random seed of `42` was used for reproducibility, and the partitions were checked to ensure that no image appeared in more than one split.

---

## Preprocessing

The general preprocessing pipeline includes:

* X-ray image loading and RGB conversion
* Aspect-ratio-preserving resizing with padding
* Image normalization
* Class weighting to address class imbalance
* Mini-batch training
* Fixed train/validation/test partitions

Three training settings were investigated:

* **No Augmentation**
* **RandAugment**
* **AugMix**

RandAugment and AugMix were applied only to the training data.

---

## Baseline Model Selection

Before implementing EKD, several teacher and student architectures were evaluated.

### Teacher Candidates

* ResNet152V2
* DenseNet201
* EfficientNet-B6

### Student Candidates

* MobileNetV1
* MobileNetV2

The models were evaluated across No Augmentation, RandAugment, and AugMix using metrics including:

* Accuracy
* Precision
* Recall
* F1-score
* AUC
* Loss

Based on overall performance and generalization across the evaluated settings, **EfficientNet-B6** was selected as the teacher and **MobileNetV2** as the student.

---

## Explainable Knowledge Distillation

During EKD training, the EfficientNet-B6 teacher remains frozen.

For each input image:

1. The teacher produces class predictions.
2. The student produces its own predictions.
3. Temperature-scaled teacher predictions provide prediction-level knowledge.
4. Grad-CAM maps are generated from the teacher and student.
5. An attention-alignment loss encourages the student to learn class-discriminative spatial behavior similar to the teacher.
6. Only the student parameters are updated.

The final objective can be summarized as:

**EKD Loss = Supervised Classification Loss + Prediction Distillation Loss + Attention Alignment Loss**

This enables MobileNetV2 to learn simultaneously from:

* Ground-truth fracture labels
* EfficientNet-B6 predictions
* EfficientNet-B6 class-discriminative attention

---

## Results

EKD improved the validation accuracy and F1-score of MobileNetV2 under all three evaluated training settings.

| Training Setting | Baseline Accuracy | EKD Accuracy |  Δ Accuracy | Baseline F1 |     EKD F1 |        Δ F1 |
| ---------------- | ----------------: | -----------: | ----------: | ----------: | ---------: | ----------: |
| No Augmentation  |            0.8454 |   **0.8701** |     +0.0247 |      0.8521 | **0.8710** |     +0.0189 |
| RandAugment      |            0.8433 |   **0.8639** |     +0.0206 |      0.8519 | **0.8646** |     +0.0127 |
| AugMix           |            0.8247 |   **0.8742** | **+0.0495** |      0.8371 | **0.8751** | **+0.0380** |

### Best Configuration

**EKD + AugMix** produced the strongest validation performance:

* **Validation Accuracy:** 87.42%
* **Validation F1-score:** 87.51%
* **Validation AUC:** 86.64%

It also produced the largest improvement over its corresponding MobileNetV2 baseline.

---

## Grad-CAM Attention Transfer

In addition to classification performance, teacher and student Grad-CAM maps were compared qualitatively.

The EKD student did not reproduce the teacher's attention maps exactly and generally produced broader activation regions. However, several examples demonstrated partial spatial correspondence, with the student retaining attention over anatomically relevant regions also highlighted by the teacher.

These observations provide qualitative evidence that Grad-CAM alignment can encourage spatial knowledge transfer from a high-capacity teacher to a lightweight student.

---

## Repository Structure

```text
Explainable-Knowledge-Distillation-for-Bone-Fracture-Classification/
│
├── baseline/
│   ├── DenseNet201/
│   ├── EfficientNetB6/
│   ├── MobileNetV1/
│   ├── MobileNetV2/
│   └── ResNet152V2/
│
├── .gitignore
└── README.md
```

Additional EKD experiments, results, and project artifacts can be added to the repository as they are organized for release.

---

## Implementation

The project was developed primarily using:

* Python
* TensorFlow
* Keras / KerasCV
* NumPy
* Pandas
* OpenCV
* Scikit-learn
* Matplotlib

Baseline experiments were conducted using GPU-enabled environments, while the final EKD implementation was executed on the **KAUST IBEX high-performance computing cluster using an NVIDIA A100 GPU**.

---

## Limitations

The current study has several limitations:

* Evaluation is limited to the FracAtlas dataset.
* The current task is binary fracture/non-fracture classification.
* The validation and testing sets are relatively small and imbalanced.
* Attention-transfer evaluation is primarily qualitative.
* The final EKD experiments evaluate a single selected teacher-student pair.

Therefore, the results demonstrate the potential of explainability-guided knowledge distillation rather than establishing general effectiveness across all fracture-classification settings.

---

## Team

* **Samaher S. Alsharif**
* **Durar A. Alqahtani**
* **Layan F. Alsulaiman**
* **Somayah M. Alqahtani**
* **Yazan G. Rabaiah**
* **Yasser J. Alqanatish**

### Mentor

* **Dr. Salman Khan**

---

## Acknowledgments

This project was conducted as part of the **KAUST Academy Summer Program 2026**.
