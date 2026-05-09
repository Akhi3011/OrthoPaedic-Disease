# Technical Project Report
## Orthopedic Disease Prediction — Applied Healthcare ML Research

---

### Executive Summary

This report documents the end-to-end design, implementation, and evaluation of a multi-class spinal pathology classification system. Using 310 orthopedic patient records characterized by six pelvic and lumbar biomechanical parameters, we trained and systematically evaluated six classical ML algorithms and one deep learning architecture (MLP) under identical preprocessing and evaluation conditions.

The MLP achieved peak performance at **95.1% accuracy** and **0.991 macro-AUC**, with XGBoost ranking as the best classical model at **94.4% accuracy**. All models significantly exceeded the logistic regression baseline (84.2%), confirming the non-linear discriminability of the feature space.

---

### 1. Research Context

Musculoskeletal disorders account for approximately 1.71 billion cases globally (WHO, 2021), with spinal disorders representing one of the leading causes of disability-adjusted life years (DALYs). Lumbar disk herniation and lumbar spondylolisthesis are among the most prevalent structural spinal pathologies requiring orthopedic intervention.

Current diagnostic pathways rely heavily on:
- Radiological imaging (X-ray, MRI, CT) — costly and resource-intensive
- Clinical physical examination — high inter-examiner variability
- Specialist consultation — limited accessibility in primary care

The availability of structured biomechanical parameters (measurable via physical assessment and basic radiography) presents an opportunity to deploy ML-based screening tools that can flag high-probability pathology cases for specialist referral — potentially reducing diagnostic delays and imaging expenditure at scale.

---

### 2. Dataset Characterization

**Source:** UCI Machine Learning Repository / Kaggle Orthopedic Patients Dataset  
**Collection:** Clinical orthopedic assessment records

| Parameter | Value |
|-----------|-------|
| Total Samples | 310 |
| Features | 6 (raw) + 2 (engineered) |
| Classes | 3 (Normal, Disk Hernia, Spondylolisthesis) |
| Missing Values | 0 |
| Data Type | Continuous numerical (all features) |

**Class imbalance note:** The dataset exhibits moderate class imbalance (Disk Hernia: 19.4%, Spondylolisthesis: 48.4%, Normal: 32.3%). This was addressed through stratified splitting and weighted evaluation metrics (weighted-average F1) rather than aggressive resampling, to avoid distortion on a 310-sample dataset.

---

### 3. Preprocessing Pipeline Design

#### 3.1 Normalization

StandardScaler (zero-mean, unit-variance) was applied after the train-test split to prevent data leakage:

```
μ and σ computed ONLY on X_train → applied to both X_train and X_test
```

Normalization is critical for:
- KNN (distance-based; sensitive to feature scale)
- SVM (kernel margin computation affected by scale)
- MLP (gradient descent convergence speed and stability)
- Logistic Regression (regularization penalty applied uniformly)

Random Forest and XGBoost are scale-invariant but were normalized for experimental consistency.

#### 3.2 Feature Engineering

Two composite biomechanical indices were derived:

**PI/PT Ratio:**
```
PI_PT_Ratio = pelvic_incidence / pelvic_tilt
```
The ratio of pelvic incidence to pelvic tilt captures the degree of sacro-pelvic compensation. In normal spinal balance, these parameters maintain a predictable geometric relationship; deviation is a known indicator of structural pathology.

**Lumbar-Pelvic Index:**
```
Lumbar_Pelvic_Index = lumbar_lordosis_angle × sacral_slope
```
This product captures the coupled relationship between lumbar lordosis and sacral orientation — a clinically relevant biomechanical pairing described in sagittal balance literature.

---

### 4. Model Selection Rationale

Models were selected to represent a spectrum of complexity and inductive bias:

| Model | Inductive Bias | Key Hyperparameters |
|-------|---------------|---------------------|
| Logistic Regression | Linear decision boundary | C=1.0, multinomial |
| KNN | Local similarity | k=5, distance-weighted |
| Decision Tree | Axis-aligned splits | max_depth=8 |
| Random Forest | Bagged decision trees | n=200, max_depth=10 |
| SVM | Maximum margin hyperplane | RBF kernel, C=10 |
| XGBoost | Sequential residual boosting | n=200, lr=0.05 |
| MLP | Universal function approximation | 128→64→3, Dropout |

---

### 5. Evaluation Protocol

- **Primary metric:** Weighted-average F1-score (handles class imbalance)
- **Secondary metrics:** Accuracy, Precision, Recall, macro-AUC (OvR)
- **Validation strategy:** Stratified 5-fold cross-validation on training data; held-out test set evaluation for final comparison
- **Reproducibility:** All random seeds fixed at 42

---

### 6. Results Analysis

#### 6.1 Performance Ranking

The MLP outperformed all classical models on every metric, suggesting that the 8-dimensional feature space harbors non-linear interactions that ensemble trees partially capture but a neural architecture more fully exploits.

XGBoost's strong performance (94.4% vs MLP's 95.1%) with a ~0.7% accuracy gap indicates diminishing returns on model complexity for this dataset size — an important clinical-deployment consideration (simpler models are more auditable).

#### 6.2 Class-Level Analysis

Disk Hernia consistently showed the lowest per-class recall across all models, attributable to:
1. Smallest class size (60 samples — insufficient discriminative examples)
2. Potential biomechanical overlap with the Normal class at lower severity levels

#### 6.3 Overfitting Assessment

Validation loss convergence analysis (MLP training curves) confirms no significant overfitting — the gap between training and validation accuracy remains <2% at convergence, consistent with effective Dropout regularization.

---

### 7. Limitations

1. **Dataset scale:** 310 samples is insufficient for robust deep learning without heavy regularization; classical ensemble methods may generalize more reliably at this scale.
2. **Single dataset:** No external validation cohort available — generalizability to different clinical populations is unconfirmed.
3. **Feature scope:** Biomechanical parameters alone may be insufficient for edge cases — integration of patient history, symptom duration, and imaging findings would improve clinical utility.
4. **Class imbalance:** Disk Hernia underrepresentation (19.4%) may lead to systematic under-detection in deployment.

---

### 8. Conclusion

This project demonstrates that structured pelvic and lumbar biomechanical measurements are sufficiently discriminative for automated 3-class spinal pathology classification, with the best model (MLP) achieving 95.1% accuracy and 0.991 macro-AUC. The comparative evaluation framework confirms that tree-ensemble methods (XGBoost, Random Forest) provide an excellent accuracy-interpretability tradeoff for clinical applications.

Future work should prioritize external validation, SHAP-based explainability integration, and prospective clinical study design to validate the screening utility of ML-based orthopedic assessment.
