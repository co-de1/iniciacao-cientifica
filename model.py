import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from feature_extraction import *

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.3).astype(int)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\nDetalhamento:")
print(f"TN (não-DAC corretamente): {tn}")
print(f"FP (falso DAC): {fp}")
print(f"FN (perdeu DAC): {fn}")
print(f"TP (DAC correto): {tp}")

#cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

labels = ['não-DAC', 'DAC']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=labels,
            yticklabels=labels)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

print(np.unique(y_pred, return_counts=True))

print("Accuracy:", model.score(X_test, y_test))
print(y_train.value_counts())