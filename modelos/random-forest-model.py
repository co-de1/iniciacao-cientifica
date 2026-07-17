import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
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

# TREINAR RANDOM FOREST

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ESCOLHER THRESHOLD NA VALIDAÇÃO

y_val_prob = model.predict_proba(X_val)[:, 1]

best_threshold = 0.5
best_balanced = -1

for threshold in np.arange(0.05, 0.96, 0.01):

    y_val_pred = (
        y_val_prob >= threshold
    ).astype(int)

    score = balanced_accuracy_score(
        y_val,
        y_val_pred
    )

    if score > best_balanced:
        best_balanced = score
        best_threshold = threshold

print(
    f"\nThreshold escolhido: "
    f"{best_threshold:.2f}"
)

print(
    f"Balanced Accuracy na validação: "
    f"{best_balanced:.4f}"
)

# TESTE FINAL

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
        labels=[0, 1],
        target_names=["não-DAC", "DAC"],
        zero_division=0
    )
)

# MATRIZ DE CONFUSÃO

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
    "Random Forest\n"
    f"Threshold = {best_threshold:.2f}"
)

plt.tight_layout()
plt.show()