# 5th Place Amini-Cocoa-Contamination-Challenge Solution By SPECIALZ🔥🔥🙌🏽

<img width="2048" height="640" alt="upscaled_leaf_2048x2048" src="https://github.com/user-attachments/assets/d53c3a78-b86e-4c0f-b30c-4704257e4b82" />

---

This project addresses the **Amini Cocoa Contamination Challenge**, which aims to detect multiple cocoa leaf diseases—such as **CSSVD** and **Anthracnose**—from images.

The goal is to build a real-time, smartphone-ready disease detection system using object detection models like **YOLOv11**, with emphasis on accuracy, inference efficiency, and explainability.

---

## 🌍 Objectives

- Build an end-to-end ML pipeline for disease detection.
- Leverage **YOLOv11** for object detection on cocoa leaves.
- Optimize for real-time inference on low-end smartphones.
- Use explainability methods and documentation to meet challenge requirements.

---

## 🧱 Architecture Overview
```csharp
┌─────────────────────────────┐
│        Image Dataset         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Preprocessing & Augmentation│
│  • Resize / Normalize       │
│  • Random flips / rotations │
│  • Color jitter / blur      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ YOLOv11 Training            │
│  • Custom config & anchors  │
│  • Multi-scale training     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Validation & Metrics Logging│
│  • mAP / Precision / Recall │
│  • Loss curves & checkpoints│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Inference                    │
│  • Test-Time Augmentation    │
│  • Weighted Box Fusion       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Post-processing & Submission│
│  • Format predictions       │
│  • Export submission file   │
└─────────────────────────────┘

```

---

## 🔄 ETL Process

### 🟢 Extract
- Data sourced from challenge dataset (COCO JSON + images).
- Images loaded via YOLOv11-compatible dataloader.

### 🟡 Transform
- Applied image augmentations: resize, crop, flip, HSV adjustments.
- Label formatting and normalization.
- Image size standardized to fit YOLO input requirements (e.g. 640x640).

### 🔵 Load
- Transformed data fed into the YOLOv11 training loop.
- Support for batch loading and caching for faster experimentation.

---

## 🧠 Modeling

### 🔍 Model
- Architecture: Two **YOLOv11** with custom training, one using YoloWeightedDataset and the other not.
- Pretrained weights used as a base.
- Tuned for optimal balance between accuracy and latency.

### 🛠 Feature Engineering
- Extensive augmentations (MixUp, Mosaic, HSV, flips).
- Confidence thresholds adjusted for better detection.
- WBF used to combine overlapping predictions.

### 🧪 Training
- Optimizers: AdamW.
- Custom learning rate schedulers.
- Best model observed around epoch 46 and 43 respectively.
- Patience is at 10

### 📏 Evaluation
- Metric: **mAP@0.5** on validation.
- Bounding box visualization to inspect predictions.

---

## 🤖 Inference Pipeline

- Uses **Test-Time Augmentation (TTA)** for robustness.
- Combines predictions with **Weighted Box Fusion (WBF)**.
- Outputs final results in COCO JSON or CSV format.

> Optimized for batch inference and smartphone compatibility.

---

## ⏱ Runtime Estimates

| Stage                   | Duration        |
|------------------------|-----------------|
| Preprocessing           | ~15–20 mins     |
| Model Training          | 8.507 hours     |
| Inference (TTA + WBF)   | 2 hours 40 mins |

---

## 📊 Performance Metrics

- **Validation mAP**: 0.821 and 0.818
- **Leaderboard Scores**:
  - Public: 0.809124261
  - Private: 0.823966249
- Other metrics: Precision, Recall (if applicable).

---

## 🧯 Error Handling & Logging

- Logging done with print statements during training/inference.
- Recommend adding `logging` module for production readiness.

---

## 🛠 Maintenance & Monitoring

- Checkpoints saved every few epochs for retraining.
- WBF + TTA pipeline is modular and can be toggled for faster inference.
- Easy conversion to **ONNX** or **TF-Lite** for mobile deployment.

---

## 📝 Notes for Hosts

- The model is deployable on Android devices.
- Code is modular—ETL, training, and inference are in separate blocks.
- Explainability components included to meet competition expectations.

---

## ✅ Final Thoughts

This project reflects an effort to bring cutting-edge AI to agriculture, empowering smallholder farmers with real-time tools for disease detection and crop monitoring.

---

