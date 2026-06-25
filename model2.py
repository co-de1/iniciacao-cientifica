from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# importa dados já processados
from feature_extraction import X_train, X_test, y_train, y_test

# 1. MODELO SVM (CORRETO)
model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)

# 3. PREDIÇÃO COM THRESHOLD
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\nDetalhamento:")
print(f"TN (não-DAC corretamente): {tn}")
print(f"FP (falso DAC): {fp}")
print(f"FN (perdeu DAC): {fn}")
print(f"TP (DAC correto): {tp}")

labels = ['não-DAC', 'DAC']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão (SVM)")
plt.show()

print("\nAccuracy:", model.score(X_test, y_test))
print("\nDistribuição treino:")
print(y_train.value_counts())