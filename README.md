# Credit Risk (Loan Default) Prediction

An end-to-end machine learning project that predicts whether a loan applicant will default. It compares 8 algorithms, tunes the best one (XGBoost), explains it with SHAP, and serves it through a FastAPI prediction service.

**Best model:** tuned XGBoost, **ROC-AUC 0.7615** on a held-out test set.

---

## Problem

A lender wants to know, before approving a loan, how likely the applicant is to default. The data is highly imbalanced: only **8.07%** of applicants defaulted, so plain accuracy is misleading (a model that always says "repaid" would be about 92% accurate and catch no defaulters). I compare models mainly on **ROC-AUC**, and report precision and recall as well.

## Dataset

[Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk)

I used only the main file, `application_train.csv`: 307,511 applicants and 122 columns. The dataset is not included in this repository. Download it from Kaggle and place `application_train.csv` in the project folder.

## Approach

1. **Cleaning:** dropped 17 columns with more than 60% missing values, then removed the ID column and separated the target (`TARGET`). This left 103 input columns.
2. **Split:** 80/20 train/test split, stratified so both parts keep the 8% default rate (246,008 train rows, 61,503 test rows).
3. **Missing values:** filled with the median (numbers) or the most common value (text), calculated from the **training data only** to avoid data leakage.
4. **Encoding:** one-hot encoding of the text columns, with test columns matched to the training columns (209 columns in total).
5. **Scaling:** `StandardScaler`, fitted on the training data only.
6. **Class imbalance:** compared two approaches, **SMOTE** (synthetic defaulters, training data only) and **class weights** (real data, defaulters weighted more).
7. **Models:** Logistic Regression, Decision Tree, Random Forest, XGBoost, Naive Bayes, KNN, and an ANN in PyTorch.
8. **Tuning:** `RandomizedSearchCV` with 3-fold cross-validation on the training data, for XGBoost.
9. **Extra analysis:** PCA and K-Means.
10. **Explainability:** SHAP.
11. **Deployment:** FastAPI service.

## Results

All scores are on the untouched test set (61,503 applicants), using a 0.5 decision threshold unless noted.

| Model | Imbalance handling | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| **XGBoost (tuned)** | scale_pos_weight | **0.7615** | 0.1710 | 0.6677 |
| XGBoost (default settings) | scale_pos_weight | 0.7584 | 0.1671 | 0.6810 |
| ANN (PyTorch) | weighted loss | 0.7452 | 0.1507 | 0.7110 |
| Logistic Regression | SMOTE | 0.7419 | 0.1623 | 0.6592 |
| Random Forest | class_weight | 0.7318 | 0.1641 | 0.6151 |
| Random Forest | SMOTE | 0.6738 | 0.1840 | 0.2354 |
| Decision Tree (depth 5) | SMOTE | 0.6536 | 0.1544 | 0.3700 |
| KNN (40k training sample)* | none | 0.6226 | 0.1050 | 0.6820 |
| Naive Bayes | none | 0.5159 | 0.0822 | 0.9823 |

\*KNN predicted no defaulters at the 0.5 threshold (precision and recall of 0). The values shown use a threshold of 0.08.

An SVM with an RBF kernel was skipped because it was too slow on 246,000 rows.

### Key findings

- **SMOTE helped Logistic Regression but hurt tree models.** Random Forest improved from 0.6738 to 0.7318 ROC-AUC (and recall from 0.24 to 0.62) when I replaced SMOTE with class weights. Synthetic rows from 209 mixed numeric and one-hot columns likely teach the trees patterns that real applicants do not follow.
- **XGBoost beat the ANN on this tabular data**, so I chose XGBoost.
- **Tuning gave a small, honest gain:** 0.7584 to 0.7615. The cross-validation score (0.7552) and the test score are close, so there is no sign of overfitting. Best settings: `n_estimators=500`, `learning_rate=0.03`, `max_depth=5`, `subsample=0.7`, `colsample_bytree=0.7`.
- **Naive Bayes performed at near-random level**, because its independence assumption does not hold for strongly correlated columns (for example, the `_AVG`, `_MODE`, `_MEDI` building features).
- **Precision stays low (about 0.17)** with a 0.5 threshold and class weighting, because the model flags many applicants to catch most defaulters. Random guessing would give about 0.08. The threshold is a business choice that trades precision against recall.
- **KNN** is slow on large data (it compares every test row with all stored training rows) and loses accuracy in many dimensions, so I trained it on a 40,000-row sample.

### Unsupervised analysis

- **PCA:** 140 components keep 90% of the variance, out of 209 columns.
- **K-Means (K=4)** on the PCA features found one larger, higher-risk segment (about 9.2% default rate, 171,328 training applicants) and three lower-risk segments (about 5.5% each). The segments are uneven in size, and K-Means never sees the target, so it is weaker than the supervised models.

## Explainability (SHAP)

![SHAP beeswarm plot](images/shap_beeswarm.png)

The three most influential features for the XGBoost model were:

1. `EXT_SOURCE_3`
2. `EXT_SOURCE_2`
3. `AMT_GOODS_PRICE`

`EXT_SOURCE_2` and `EXT_SOURCE_3` are normalized scores from external credit sources. SHAP shows what the model relies on, not what causes default.

## API

A FastAPI service loads the saved XGBoost model and repeats the same preparation as training (fill with training medians and modes, one-hot encode, match the 209 columns, scale) before predicting.

![API prediction](images/api_prediction.png)

### Run it

```bash
pip install -r requirements.txt
cd credit_api
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs, choose **POST /predict**, and send a body shaped like this (any of the 103 raw columns can be included; missing ones are filled with training values):

```json
{"features": {"NAME_CONTRACT_TYPE": "Cash loans", "AMT_INCOME_TOTAL": 202500.0, "AMT_CREDIT": 406597.5}}
```

Response:

```json
{"default_probability": 0.9, "risk": "high"}
```

(The example response above is from a real applicant in the dataset who defaulted. One applicant is a sanity check, not proof of model quality.)

## Project structure

```
.
├── notebook.ipynb          # full experiments, from cleaning to SHAP
├── requirements.txt
├── credit_api/
│   ├── app.py              # FastAPI service
│   └── model_artifacts.joblib
├── images/
│   ├── shap_beeswarm.png
│   └── api_prediction.png
└── README.md
```

## How to reproduce

1. Download `application_train.csv` from Kaggle and place it in the project folder.
2. Install the packages: `pip install -r requirements.txt` (on a computer without a GPU, install the CPU build of PyTorch from pytorch.org).
3. On Windows, keep `import torch` as the very first import in the notebook to avoid a DLL loading error.
4. Run `notebook.ipynb` from top to bottom.

## What I learned

- Data leakage is easy to cause: every fill value, encoder column list, and scaler must be learned from the training data only.
- On imbalanced data, accuracy is misleading, and ROC-AUC compares models fairly because it does not depend on one threshold.
- A popular technique (SMOTE) is not always better; testing it against class weights changed which model looked best.
- A deployed model needs its whole preparation pipeline, not just the model file.

## Possible improvements

- Add the other Home Credit tables (previous loans, credit bureau, payment history), which carry strong repayment signals.
- Choose the decision threshold using the cost of a missed defaulter versus a rejected good customer.
- Package the API in Docker.
