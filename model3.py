from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# importa dados já processados
from feature_extraction import (
    X_train,
    X_test,
    y_train,
    y_test
)

# 1. CALCULAR PESO DAS CLASSES

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

scale_pos_weight = neg / pos

print("Scale pos weight:", scale_pos_weight)

# 2. MODELO XGBOOST

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# 3. TREINAR
model.fit(X_train, y_train)

# 4. PREDIÇÕES

y_prob = model.predict_proba(X_test)[:, 1]

# threshold
y_pred = (y_prob >= 0.3).astype(int)

# 5. MATRIZ DE CONFUSÃO

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\nDetalhamento:")
print(f"TN (não-DAC corretamente): {tn}")
print(f"FP (falso DAC): {fp}")
print(f"FN (perdeu DAC): {fn}")
print(f"TP (DAC correto): {tp}")

# 6. MÉTRICAS

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nAUC:")
print(roc_auc_score(y_test, y_prob))

print("\nRelatório:")
print(classification_report(y_test, y_pred))

# 7. MATRIZ VISUAL

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
plt.title("Matriz de Confusão (XGBoost)")

plt.show()

# 8. IMPORTÂNCIA DAS FEATURES

importances = model.feature_importances_

print("\nTop 15 features mais importantes:")

# caso tenha nomes das features
try:

    feature_names = X_train.columns

    importance_dict = dict(
        zip(feature_names, importances)
    )

    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for name, score in sorted_features[:15]:
        print(f"{name}: {score:.4f}")

except:
    print("Não foi possível recuperar nomes das features")

# 9. DISTRIBUIÇÃO

print("\nDistribuição treino:")
print(y_train.value_counts())