# Telco Customer Churn Prediction

Predicting which telecom customers are likely to cancel their subscription, with a
workflow built around the realities of the business problem rather than raw accuracy.

## Why this problem

Churn is expensive: acquiring a new customer costs far more than retaining an
existing one. But churn datasets are **imbalanced** — most customers stay — so a
model that predicts "everyone stays" scores ~73% accuracy while being useless.
This project treats the metric choice as the core analytical decision: we optimise
for **recall on the churn class** (catching customers before they leave) and report
the precision/recall trade-off explicitly.

## Approach

1. **Cleaning** — `TotalCharges` ships as a string with blanks for day-one
   customers; coerced to numeric and imputed. `customerID` dropped to avoid leakage.
2. **Feature engineering** — tenure buckets (churn risk is non-linear in tenure)
   and an average-monthly-spend feature to flag recent price changes.
3. **Modeling** — a logistic-regression baseline (interpretable, `class_weight`
   for imbalance) compared against a random forest. Preprocessing and estimator are
   wrapped in a single scikit-learn `Pipeline` to prevent train/test leakage.
4. **Evaluation** — classification report, confusion matrix, and ROC-AUC, read
   through a business lens (false negatives = missed churners = lost revenue).

## Results

| Model | Churn recall | Churn precision | ROC-AUC |
|-------|-------------|-----------------|---------|
| **Logistic Regression** | **0.79** | 0.50 | **0.842** |
| Random Forest | 0.47 | 0.61 | 0.824 |

The interpretable baseline **wins**: logistic regression catches 79% of churners
versus 47% for the random forest, at a higher ROC-AUC. On a retention problem the
cost of a missed churner (false negative) outweighs a false alarm, so recall is the
metric that matters — and the simpler model is the right call. A reminder that
model complexity is a means, not a goal.

## Key takeaway

The strongest churn drivers are **fiber-optic internet** and **month-to-month
contracts** (both raise churn odds sharply), followed by electronic-check payment.
The highest-risk segment is fiber customers on month-to-month plans paying by
electronic check — a concrete profile a retention team can target, not just a score.
The natural intervention: incentivise these customers onto annual contracts.

## Run it

```bash
pip install -r requirements.txt
# download the dataset (see src/churn_model.py docstring) into data/
python src/churn_model.py
```

## Data

IBM Telco Customer Churn (public) —
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

