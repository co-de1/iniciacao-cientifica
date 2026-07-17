import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score
)

from feature_extraction import (
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
)


# 1. TREINAR XGBOOST

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 2. ESCOLHER THRESHOLD

y_val_prob = model.predict_proba(X_val)[:, 1]

best_threshold = 0.5
best_score = 0

for threshold in np.arange(0.2, 0.81, 0.05):

    y_val_pred = (
        y_val_prob >= threshold
    ).astype(int)

    score = balanced_accuracy_score(
        y_val,
        y_val_pred
    )

    if score > best_score:
        best_score = score
        best_threshold = threshold

print(
    f"\nThreshold escolhido: "
    f"{best_threshold:.2f}"
)


# 3. TESTE FINAL

y_prob = model.predict_proba(X_test)[:, 1]

y_pred = (
    y_prob >= best_threshold
).astype(int)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

# 4. RESULTADOS

print("\nDetalhamento:")
print(f"TN - não-DAC correto: {tn}")
print(f"FP - falso DAC: {fp}")
print(f"FN - DAC não detectado: {fn}")
print(f"TP - DAC correto: {tp}")

print(
    "\nAccuracy:",
    f"{accuracy_score(y_test, y_pred):.4f}"
)

print(
    "Balanced Accuracy:",
    f"{balanced_accuracy_score(y_test, y_pred):.4f}"
)

print(
    "ROC AUC:",
    f"{roc_auc_score(y_test, y_prob):.4f}"
)

print("\nRelatório:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["não-DAC", "DAC"],
        zero_division=0
    )
)


# 5. MATRIZ DE CONFUSÃO

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["não-DAC", "DAC"],
    yticklabels=["não-DAC", "DAC"]
)

plt.xlabel("Predito")
plt.ylabel("Real")

plt.title(
    f"XGBoost\n"
    f"Threshold = {best_threshold:.2f}"
)

plt.tight_layout()
plt.show()