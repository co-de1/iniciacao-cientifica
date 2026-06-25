import wfdb
import numpy as np
import pandas as pd
import pycatch22
import neurokit2 as nk

from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

from main import df_small

# 1. GARANTIR DADOS VÁLIDOS

df_small = df_small.dropna(subset=['label'])

print("Quantidade de ECGs:", len(df_small))

# 2. FILTRO PASSA-BANDA

def bandpass_filter(signal, low=0.5, high=40, fs=100, order=4):

    nyquist = 0.5 * fs

    low = low / nyquist
    high = high / nyquist

    b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, signal)

# 3. EXTRAÇÃO DE FEATURES

def extract_features(signal, fs=100):

    features = {}

    # NORMALIZAÇÃO

    signal = (signal - np.mean(signal)) / np.std(signal)

    # FEATURES ESTATÍSTICAS

    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)
    features['median'] = np.median(signal)
    features['var'] = np.var(signal)
    features['ptp'] = np.ptp(signal)
    features['energy'] = np.sum(signal ** 2)

    # ZERO CROSSINGS

    features['zero_crossings'] = np.sum(
        np.diff(np.sign(signal)) != 0
    )
    # FFT

    fft = np.fft.rfft(signal)

    fft_abs = np.abs(fft)

    freqs = np.fft.rfftfreq(len(signal), d=1/fs)

    features['fft_mean'] = np.mean(fft_abs)
    features['fft_std'] = np.std(fft_abs)
    features['fft_max'] = np.max(fft_abs)

    dominant_freq = freqs[np.argmax(fft_abs)]

    features['dominant_freq'] = dominant_freq

    # DETECÇÃO DE R-PEAKS

    try:

        _, rpeaks = nk.ecg_peaks(
            signal,
            sampling_rate=fs
        )

        peaks = rpeaks["ECG_R_Peaks"]

    except:

        peaks = []

    features['num_peaks'] = len(peaks)

    # HRV FEATURES

    if len(peaks) > 2:

        rr = np.diff(peaks)

        features['rr_mean'] = np.mean(rr)
        features['rr_std'] = np.std(rr)

        # RMSSD
        diff_rr = np.diff(rr)

        features['rmssd'] = np.sqrt(
            np.mean(diff_rr ** 2)
        )

        # SDNN
        features['sdnn'] = np.std(rr)

        # pNN50
        nn50 = np.sum(np.abs(diff_rr) > 50)

        features['pnn50'] = nn50 / len(diff_rr)

    else:

        features['rr_mean'] = 0
        features['rr_std'] = 0
        features['rmssd'] = 0
        features['sdnn'] = 0
        features['pnn50'] = 0

    # CATCH22

    try:

        catch22 = pycatch22.catch22_all(signal)

        for name, value in zip(
            catch22["names"],
            catch22["values"]
        ):

            features[name] = value

    except Exception as e:

        print("Erro catch22:", e)

    return features


# 4. PROCESSAR ECGs

features_list = []

for i, row in df_small.iterrows():

    try:

        path = row['filename_hr']

        record = wfdb.rdrecord(path)

        # USAR MÉDIA DAS LEADS

        signal = np.mean(record.p_signal, axis=1)

        # FILTRAR ECG

        signal = bandpass_filter(signal)

        # EXTRAIR FEATURES

        feats = extract_features(signal)

        feats['label'] = row['label']
        feats['strat_fold'] = row['strat_fold']

        features_list.append(feats)

        if i % 50 == 0:
            print(f"ECGs processados: {i}")

    except Exception as e:

        print(f"Erro no ECG {i}: {e}")

        continue

# 5. DATAFRAME FINAL

df_features = pd.DataFrame(features_list)

if df_features.empty:
    raise ValueError(
        "Nenhum ECG foi processado"
    )

print("\nShape final:", df_features.shape)

print("\nDistribuição das classes:")
print(df_features['label'].value_counts())


# 6. PREPARAR DADOS

X = df_features.drop(
    columns=['label', 'strat_fold']
)

y = df_features['label']

# 7. SPLIT TREINO / TESTE

print("\nDistribuição dos folds:")
print(df_features['strat_fold'].value_counts())

test_fold = df_features['strat_fold'].max()

print(f"\nFold de teste: {test_fold}")

train_idx = df_features['strat_fold'] != test_fold
test_idx = df_features['strat_fold'] == test_fold

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]


# 8. NORMALIZAÇÃO SEM DATA LEAKAGE

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 9. SMOTE (BALANCEAMENTO)

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("\nApós SMOTE:")
print(pd.Series(y_train).value_counts())

# 10. FEATURE SELECTION

selector = SelectKBest(
    score_func=f_classif,
    k=25
)

X_train = selector.fit_transform(
    X_train,
    y_train
)

X_test = selector.transform(X_test)

print("\nNúmero final de features:")
print(X_train.shape[1])

# 11. MODELO SVM

model = SVC(
    kernel='rbf',
    class_weight='balanced',
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)

# 12. PREDIÇÃO

y_prob = model.predict_proba(X_test)[:, 1]

# 13. THRESHOLD ÓTIMO

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

optimal_idx = np.argmax(tpr - fpr)

optimal_threshold = thresholds[optimal_idx]

print("\nThreshold ótimo:")
print(optimal_threshold)

y_pred = (
    y_prob >= optimal_threshold
).astype(int)

# 14. AVALIAÇÃO

print("\nAUC:")
print(roc_auc_score(y_test, y_prob))

print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório:")
print(classification_report(y_test, y_pred))

# 15. IMPORTÂNCIA DAS FEATURES

selected_features = X.columns[
    selector.get_support()
]

print("\nFeatures selecionadas:")

for feat in selected_features:
    print(feat)

# 16. SALVAR DATASET

df_features.to_csv(
    "dataset_features_ecg.csv",
    index=False
)

print("\nDataset salvo!")

print(df_features.head())