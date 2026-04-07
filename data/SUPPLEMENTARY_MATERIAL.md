# Supplementary Material

**A Saturation-Aware Dual-Stage Framework for Intensity-Driven Adaptive PTW: Prioritizing Operational Safety and Site-Response Robustness in the Java-Sumatra Region**

Hanif Andi Nugraha, Dede Djuhana, Adhi Harmoko Saputro, Sigit Pramono

---

## Table S1. XGBoost Hyperparameters for All Pipeline Stages

### Stage 1: Intensity Gate Classifier

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `objective` | `multi:softprob` | Three-class probability output (Weak/Felt/Damaging) |
| `num_class` | 3 | Weak, Felt, Damaging |
| `n_estimators` | 1,200 | Extended boosting with early stopping |
| `max_depth` | 5 | Restricted to prevent overfitting on imbalanced classes |
| `learning_rate` | 0.03 | Conservative step size for stability |
| `min_child_weight` | 30 | Requires substantial sample support per leaf |
| `colsample_bytree` | 0.7 | Feature subsampling per tree |
| `reg_alpha` (L1) | 0.5 | Moderate L1 regularization |
| `reg_lambda` (L2) | 3.0 | Aggressive L2 penalization for Damaging class |
| `tree_method` | `hist` | Histogram-based split finding |
| `random_state` | 42 | Reproducibility seed |
| Class weighting | SMOTE + scale_pos_weight | Addresses 94:5.7:0.3 imbalance |

### Stage 1.5: Epicentral Distance Estimator

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `objective` | `reg:squarederror` | Continuous distance regression |
| `n_estimators` | 300 | Moderate complexity |
| `max_depth` | 7 | Deeper trees for continuous target |
| `learning_rate` | 0.05 | Standard step size |
| `subsample` | 0.7 | Row subsampling |
| `colsample_bytree` | 0.7 | Feature subsampling per tree |
| `min_child_weight` | 10 | Moderate leaf constraint |
| `reg_lambda` (L2) | 2.0 | Strong regularization |
| Features | 30 (Zhang et al., 2024) | Extracted at PTW = 2 s |

### Stage 2: Sa(T) Regressors (515 models)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `objective` | `reg:squarederror` | Log10 Sa(T) regression |
| `n_estimators` | 550 (max), median-selected | Median OOF iteration from 5-fold CV |
| `max_depth` | 9 | Deep trees for continuous spectral prediction |
| `learning_rate` | 0.01104 | Fine-grained (Optuna-tuned, 100 trials) |
| `subsample` | 0.6707 | Optuna-tuned |
| `colsample_bytree` | 0.6217 | Optuna-tuned |
| `min_child_weight` | 4 | Low constraint for fine regression |
| `gamma` | 0.0978 | Minimum loss reduction for split |
| `reg_alpha` (L1) | 1e-6 | Near-zero L1 |
| `reg_lambda` (L2) | 0.6772 | Moderate L2 (Optuna-tuned) |
| `tree_method` | `hist` | Histogram-based split finding |
| Features | 42 P-wave IMs | Extracted at assigned PTW (2, 3, 4, 6, or 8 s) |

**Optimization:** Stage 2 hyperparameters were optimized via 100-trial Optuna Bayesian search (Akiba et al., 2019) on a representative period subset, then applied to all 515 models. Stage 1 and Stage 1.5 parameters were manually tuned with emphasis on regularization strength to address class imbalance (Stage 1) and routing robustness (Stage 1.5).

---

## Table S2. Intensity-Distance PTW Lookup Table

The following table maps the predicted intensity class (via representative magnitude proxy) and predicted epicentral distance to the optimal P-wave time window (PTW) in seconds. PTW values are subject to the golden time constraint: PTW_selected <= t_golden - t_warn, where t_warn = 3 s.

**Intensity class to representative magnitude mapping:**
- *Weak* (PGA < 3.0 Gal): M_rep = 4.5
- *Felt* (3.0 <= PGA < 62.0 Gal): M_rep = 5.5
- *Damaging* (PGA >= 62.0 Gal): M_rep = 6.5

| M_rep \ Distance (km) | 0-25 | 25-50 | 50-100 | 100-200 | 200-400 | >400 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **< 4.5** (sub-Weak) | 2 | 2 | 2 | 3 | 3 | 3 |
| **4.5** (Weak) | 2 | 2 | 3 | 3 | 4 | 4 |
| **5.0** | 2 | 3 | 3 | 4 | 4 | 6 |
| **5.5** (Felt) | 2 | 3 | 4 | 4 | 6 | 6 |
| **6.0** | 2 | 3 | 4 | 6 | 8 | 8 |
| **6.5** (Damaging) | 2 | 3 | 4 | 6 | 8 | 10* |

*\*PTW = 10 s is truncated to 8 s in operational mode due to golden time constraints (ALL_PTW_VALUES = [2, 3, 4, 6, 8]).*

**Velocity model parameters:**
- V_P = 6.5 km/s (crustal P-wave velocity)
- V_S = 3.5 km/s (average S-wave velocity)
- t_warn = 3 s (minimum warning dissemination buffer)

**Golden time formula:**
t_golden(D) = D/V_S - D/V_P = D x (1/3.5 - 1/6.5)

**Default PTW routing (without distance correction):**
- Weak -> PTW = 3 s
- Felt -> PTW = 4 s
- Damaging -> PTW = 6 s

---

## Source Code

The complete pipeline source code is available at: https://github.com/hanif7108/adaptive-ptw-eews
