<div align="center">

# 🦴 Orthopedic Disease Prediction
### Predictive Analytics for Spinal Pathology Classification Using ML, Ensemble Learning & Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](notebooks/)
[![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)]()

> **Systematic comparative evaluation of 14 ML, Ensemble, and Deep Learning models for binary orthopedic spinal pathology classification (Normal vs Abnormal) from biomechanical patient records — with SMOTE-based class balancing and Z-score normalization.**

</div>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Motivation & Clinical Relevance](#-motivation--clinical-relevance)
- [Dataset Overview](#-dataset-overview)
- [Methodology](#-methodology)
- [Models Implemented](#-models-implemented)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Future Directions](#-future-directions)
- [Research Relevance](#-research-relevance)

---

## 🎯 Problem Statement

Spinal disorders affect hundreds of millions globally, yet accurate early diagnosis remains clinically challenging due to high inter-observer variability and over-reliance on expensive imaging modalities. Traditional diagnostic pipelines require specialist availability that is frequently limited in primary care settings.

This project investigates whether **structured biomechanical measurements** — obtainable non-invasively during clinical assessment — can serve as reliable discriminative features for automated binary spinal pathology screening (Normal vs Abnormal), across a comprehensive suite of 14 ML, Ensemble, and Deep Learning architectures.

---

## 💡 Motivation & Clinical Relevance

| Challenge | Clinical Impact |
|-----------|----------------|
| Delayed spinal diagnosis | Progressive neurological compromise in untreated cases |
| High specialist dependency | Limited orthopedic access in rural/low-income regions |
| Imaging cost barriers | MRI/CT unaffordable for large-scale preventive screening |
| Class imbalance in patient data | Systematic under-detection of minority conditions |

> **Hypothesis:** Six pelvic and lumbar biomechanical parameters carry sufficient discriminative signal for binary orthopedic disease classification with clinically actionable accuracy (>85%), even after correcting for class imbalance via SMOTE.

---

## 📊 Dataset Overview

| Property | Detail |
|----------|--------|
| **Source** | Kaggle / UCI ML Repository — Orthopedic Patients |
| **Total Samples** | 310 patient records |
| **Classes** | Normal (100) · Abnormal (210) |
| **Features** | 6 continuous biomechanical parameters |
| **Task** | Binary supervised classification |
| **Missing Values** | None |
| **Class Imbalance** | Addressed via SMOTE → 336 balanced training samples |

### Biomechanical Feature Set

| Feature | Clinical Description |
|---------|---------------------|
| `pelvic_incidence` | Angle between perpendicular to sacral plate and line to femoral head — primary pelvic morphology indicator |
| `pelvic_tilt numeric` | Sagittal pelvic rotation angle — postural compensation measure |
| `lumbar_lordosis_angle` | Degree of lumbar spinal curvature — lordosis severity |
| `sacral_slope` | Sacral plate angle relative to horizontal — sacro-pelvic balance |
| `pelvic_radius` | Distance from hip axis to posterior sacrum — structural geometry |
| `degree_spondylolisthesis` | Vertebral slippage degree — direct pathology severity metric |

### Class Distribution

<div align="center">
<img src="results/figures/class_distribution.png" width="700" alt="Class Distribution">
</div>

---

## 🔬 Methodology

### End-to-End ML Pipeline

<div align="center">
<img src="results/figures/ml_pipeline.png" width="900" alt="ML Pipeline">
</div>

### Preprocessing Protocol

```
Raw CSV (310 samples)
  → Label Encoding: Normal=1, Abnormal=0
  → Stratified Train/Test Split: 80/20 (248 train, 62 test)
  → SMOTE Oversampling on training set → 336 balanced samples (168 per class)
  → Z-Score Normalization (fit on train only → applied to both splits)
  → Feature Matrix: 6 biomechanical predictors
  → Model Training → Evaluation on held-out test set
```

**Key preprocessing decisions:**
- **SMOTE:** Applied exclusively on the training set to address the 2.1:1 class imbalance, generating synthetic Normal samples without data leakage into the test set.
- **Z-Score Normalization:** Zero-mean, unit-variance scaling critical for distance-based (KNN), margin-based (SVM), and gradient-based (neural network) algorithms.
- **Stratified split:** Preserves the original 32.3%/67.7% class ratio in the test partition for unbiased evaluation.

### Feature Correlation Analysis

<div align="center">
<img src="results/figures/correlation_heatmap.png" width="620" alt="Correlation Heatmap">
</div>

> `pelvic_incidence` and `lumbar_lordosis_angle` exhibit strong positive correlation (r≈0.82), consistent with established biomechanical coupling in sagittal balance literature. `degree_spondylolisthesis` shows the weakest inter-feature correlation, reinforcing its role as a direct independent pathology indicator.

---

## 🤖 Models Implemented

### Classical Machine Learning

| Model | Key Configuration |
|-------|------------------|
| Logistic Regression | `solver='lbfgs'`, `max_iter=1000` |
| Naive Bayes | Gaussian likelihood assumption |
| Decision Tree | `max_depth=3`, Gini criterion |
| SVM | Linear kernel, `C=1`, probability calibration |
| KNN | `k=10`, Euclidean distance |

### Ensemble Learning

| Model | Strategy |
|-------|----------|
| Random Forest | Bagging — 100 trees, `max_depth=3` |
| XGBoost | Sequential gradient boosting |
| AdaBoost | Adaptive boosting, 50 estimators |
| Gradient Boosting | Stagewise additive, `lr=0.1` |
| Stacking | Meta-learner: Logistic Regression |
| Voting (Hard & Soft) | Majority vote / probability averaging |

### Deep Learning

| Model | Architecture |
|-------|-------------|
| ANN | Dense(64,ReLU)→BN→Dropout(0.3)→Dense(32,ReLU)→BN→Dropout(0.2)→Sigmoid |
| MLP (sklearn) | Hidden layers: (50,50), `max_iter=1000` |
| SimpleRNN | RNN(64)→Dropout→RNN(32)→Sigmoid |
| **LSTM** | **LSTM(64)→Dropout(0.2)→LSTM(32)→Sigmoid** ← Best Model |
| GRU | GRU(64)→Dropout(0.2)→GRU(32)→Sigmoid |

All Keras models: **Adam optimizer**, **binary cross-entropy loss**, **Early Stopping** (patience=10–15, best weights restored).

---

## 📈 Results

### Complete Performance Summary (Real Experimental Results)

| Model | Category | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|----------|-----------|--------|----------|---------|
| **LSTM** | Deep Learning | **0.9032** | **0.8182** | **0.9000** | **0.8571** | **0.9595** |
| GRU | Deep Learning | 0.8710 | 0.7500 | 0.9000 | 0.8182 | 0.9571 |
| ANN | Deep Learning | 0.8871 | 0.8095 | 0.8500 | 0.8293 | 0.9488 |
| SVM | Machine Learning | 0.8710 | 0.7308 | 0.9500 | 0.8261 | 0.9488 |
| SimpleRNN | Deep Learning | 0.8226 | 0.6957 | 0.8000 | 0.7442 | 0.9488 |
| Logistic Regression | Machine Learning | 0.8548 | 0.7200 | 0.9000 | 0.8000 | 0.9476 |
| AdaBoost | Ensemble | 0.8226 | 0.6800 | 0.8500 | 0.7556 | 0.9173 |
| MLP (sklearn) | Machine Learning | 0.8226 | 0.6957 | 0.8000 | 0.7442 | 0.9202 |
| KNN | Machine Learning | 0.8065 | 0.6429 | 0.9000 | 0.7500 | 0.8952 |
| Naive Bayes | Machine Learning | 0.8065 | 0.6333 | 0.9500 | 0.7600 | 0.8893 |
| Gradient Boosting | Ensemble | 0.7903 | 0.6522 | 0.7500 | 0.6977 | 0.9048 |
| Random Forest | Ensemble | 0.7581 | 0.6000 | 0.7500 | 0.6667 | 0.9036 |
| XGBoost | Ensemble | 0.7581 | 0.6087 | 0.7000 | 0.6512 | 0.8940 |
| Decision Tree | Machine Learning | 0.7419 | 0.5769 | 0.7500 | 0.6522 | 0.8036 |

> 📌 **Best Model: LSTM** — 90.32% accuracy, 0.9595 ROC-AUC. Recurrent architectures (LSTM, GRU) outperform classical ML and simpler neural networks, suggesting sequential feature interactions in biomechanical parameters encode clinically relevant non-linear patterns. **SVM is the best classical model** (87.1%), notably outperforming all ensemble methods — attributable to strong margin separation in the normalized 6D feature space.

### Model Comparison

<div align="center">
<img src="results/figures/model_comparison.png" width="950" alt="Model Comparison">
</div>

### Confusion Matrices — Top 4 Models

<div align="center">
<img src="results/figures/confusion_matrices.png" width="850" alt="Confusion Matrices">
</div>

### ROC Curves — All 14 Models

<div align="center">
<img src="results/figures/roc_curves.png" width="900" alt="ROC Curves">
</div>

### Feature Importance — Random Forest (Real Gini Scores)

<div align="center">
<img src="results/figures/feature_importance.png" width="800" alt="Feature Importance">
</div>

### LSTM Training Dynamics (Real Training Run)

<div align="center">
<img src="results/figures/training_curves.png" width="800" alt="Training Curves">
</div>

---

## 📁 Project Structure

```
OrthoPaedic-Disease/
│
├── 📓 notebooks/
│   └── orthopedic_analysis.ipynb      # Main analysis notebook
│
├── 📊 data/
│   └── Orthopedic_patients.csv        # Source dataset (310 patients, 6 features)
│
├── 🐍 src/
│   ├── __init__.py
│   ├── preprocessing.py               # Data loading, SMOTE, normalization pipeline
│   ├── models.py                      # All model definitions and training wrappers
│   └── evaluate.py                    # Metrics computation and reporting
│
├── 📈 results/
│   ├── metrics_summary.csv            # Full model comparison (real results)
│   └── figures/                       # All generated visualizations
│       ├── model_comparison.png
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       ├── feature_importance.png
│       ├── training_curves.png
│       ├── class_distribution.png
│       ├── correlation_heatmap.png
│       └── ml_pipeline.png
│
├── 📄 docs/
│   └── project_report.md
│
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## 🛠 Installation & Setup

```bash
# Clone repository
git clone https://github.com/Akhi3011/OrthoPaedic-Disease.git
cd OrthoPaedic-Disease

# Create virtual environment
python -m venv ortho_env
source ortho_env/bin/activate    # Windows: ortho_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook notebooks/orthopedic_analysis.ipynb
```

---

## 🔭 Future Directions

| Enhancement | Description |
|-------------|-------------|
| **SHAP Explainability** | Per-prediction feature attribution for clinical interpretability |
| **Hyperparameter Tuning** | Bayesian optimization for RF, XGBoost, LSTM |
| **External Validation** | Independent clinical cohort evaluation |
| **Multi-class Extension** | Normal / Disk Hernia / Spondylolisthesis (3-class) |
| **Deployment** | FastAPI REST endpoint + Docker containerization |
| **Federated Learning** | Privacy-preserving training across hospital systems |

---

## 📚 Research Relevance

- **Recurrent DL for Tabular Clinical Data:** LSTM/GRU superiority over classical ML on structured biomechanical records challenges the conventional assumption that recurrent architectures are exclusively suited to temporal/sequential data.
- **SMOTE in Healthcare AI:** Evaluates synthetic oversampling impact on clinical minority-class detection — critical for equitable diagnostic tools.
- **Model Breadth:** Simultaneous evaluation of 14 diverse architectures under identical conditions enables principled model selection for deployment.

### References

- Legaye, J. et al. (1998). *Pelvic incidence: a fundamental pelvic parameter.* European Spine Journal.
- Buckland, A. J. et al. (2020). *Discriminating spinal pathology with machine learning.* Spine Journal.
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
- Rajpurkar, P. et al. (2022). *AI in health and medicine.* Nature Medicine.

---

## 🧰 Tech Stack

| Category | Libraries |
|----------|-----------|
| ML | scikit-learn, XGBoost, CatBoost |
| Deep Learning | TensorFlow / Keras |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Data Processing | NumPy, Pandas, SciPy |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with clinical rigor and research intent.**

*Applied Healthcare AI portfolio — targeting research collaboration and ML engineering roles in the medical AI domain.*

⭐ Star this repository if you found it useful.

</div>
