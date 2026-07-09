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

from feature_extraction import *

# 1. TREINAR MODELO
model = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 2. PREVER PROBABILIDADES
y_prob = model.predict_proba(X_test)[:, 1]

# 3. DEFINIR THRESHOLD

threshold = 0.3

y_pred = (y_prob >= threshold).astype(int)

# 4. MATRIZ DE CONFUSÃO

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\nDetalhamento da Matriz de Confusão:")
print(f"TN - não-DAC corretamente classificado: {tn}")
print(f"FP - não-DAC classificado como DAC: {fp}")
print(f"FN - DAC classificado como não-DAC: {fn}")
print(f"TP - DAC corretamente classificado: {tp}")

# 5. MÉTRICAS PRINCIPAIS
print("\nMétricas usando threshold =", threshold)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

print("\nRelatório de Classificação:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["não-DAC", "DAC"]
))

# 6. DISTRIBUIÇÃO DAS CLASSES

print("\nDistribuição real no treino:")
print(y_train.value_counts())

print("\nDistribuição real no teste:")
print(y_test.value_counts())

print("\nDistribuição das predições:")
unique, counts = np.unique(y_pred, return_counts=True)
print(dict(zip(unique, counts)))

# 7. PLOTAR MATRIZ DE CONFUSÃO

labels = ['não-DAC', 'DAC']

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predito")
plt.ylabel("Real")
plt.title(f"Matriz de Confusão - Random Forest | Threshold = {threshold}")
plt.show()

# 8. TESTAR VÁRIOS THRESHOLDS
print("\nComparação entre diferentes thresholds:")

for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
    y_pred_t = (y_prob >= t).astype(int)

    cm_t = confusion_matrix(y_test, y_pred_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

    acc_t = accuracy_score(y_test, y_pred_t)
    bal_acc_t = balanced_accuracy_score(y_test, y_pred_t)

    recall_dac = tp_t / (tp_t + fn_t)
    precision_dac = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0

    print("\nThreshold:", t)
    print(f"TN={tn_t}, FP={fp_t}, FN={fn_t}, TP={tp_t}")
    print(f"Accuracy: {acc_t:.4f}")
    print(f"Balanced Accuracy: {bal_acc_t:.4f}")
    print(f"Precision DAC: {precision_dac:.4f}")
    print(f"Recall DAC: {recall_dac:.4f}")