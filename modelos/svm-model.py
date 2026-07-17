import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
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

# TREINAR SVM

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)

# ESCOLHER THRESHOLD

y_val_prob = model.predict_proba(X_val)[:, 1]

best_threshold = 0.5
best_score = 0

for t in np.arange(0.2, 0.81, 0.05):

    pred = (y_val_prob >= t).astype(int)

    score = balanced_accuracy_score(y_val, pred)

    if score > best_score:
        best_score = score
        best_threshold = t

print(f"\nThreshold escolhido: {best_threshold:.2f}")

# TESTE FINAL

y_prob = model.predict_proba(X_test)[:, 1]

y_pred = (y_prob >= best_threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\nDetalhamento:")

print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

print("\nAccuracy:",
      accuracy_score(y_test, y_pred))

print("Balanced Accuracy:",
      balanced_accuracy_score(y_test, y_pred))

print("ROC AUC:",
      roc_auc_score(y_test, y_prob))

print("\nRelatório:")

print(classification_report(
    y_test,
    y_pred,
    target_names=["não-DAC", "DAC"]
))

print("\nDistribuição do teste:")
print(y_test.value_counts())

print("\nDistribuição das predições:")
print(dict(zip(*np.unique(y_pred, return_counts=True))))

# MATRIZ DE CONFUSÃO

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["não-DAC","DAC"],
    yticklabels=["não-DAC","DAC"]
)

plt.xlabel("Predito")
plt.ylabel("Real")

plt.title(
    f"SVM RBF\nThreshold={best_threshold:.2f}"
)

plt.tight_layout()
plt.show()