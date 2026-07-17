import os
import wfdb
import numpy as np
import pandas as pd
import pycatch22
import neurokit2 as nk

from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

# 1. CARREGAR DATASET

DATASET_PATH = "dataset_ecg_basico.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Arquivo não encontrado: {DATASET_PATH}")

df_small = pd.read_csv(DATASET_PATH)

required_columns = {"label", "strat_fold", "filename_hr"}
missing_columns = required_columns - set(df_small.columns)

if missing_columns:
    raise ValueError(f"Colunas ausentes: {missing_columns}")

df_small = df_small.dropna(
    subset=["label", "strat_fold", "filename_hr"]
).copy()

df_small["label"] = df_small["label"].astype(int)
df_small["strat_fold"] = df_small["strat_fold"].astype(int)

print("Quantidade de ECGs:", len(df_small))

# 2. FILTRO PASSA-BANDA

def bandpass_filter(signal, fs, low=0.5, high=40.0, order=4):
    signal = np.asarray(signal, dtype=float)

    nyquist = fs / 2
    low_norm = low / nyquist
    high_norm = min(high / nyquist, 0.99)

    if not 0 < low_norm < high_norm < 1:
        raise ValueError("Frequências inválidas para o filtro.")

    b, a = butter(
        order,
        [low_norm, high_norm],
        btype="band"
    )

    if len(signal) <= 3 * max(len(a), len(b)):
        raise ValueError("Sinal muito curto para o filtro.")

    return filtfilt(b, a, signal)

# 3. EXTRAIR FEATURES

def extract_features(signal, fs):
    signal = np.asarray(signal, dtype=float)

    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "min": np.min(signal),
        "max": np.max(signal),
        "median": np.median(signal),
        "var": np.var(signal),
        "ptp": np.ptp(signal),
        "energy": np.sum(signal ** 2)
    }

    std = np.std(signal)

    if std > 0 and np.isfinite(std):
        normalized_signal = (
            signal - np.mean(signal)
        ) / std
    else:
        normalized_signal = signal - np.mean(signal)

    # Zero crossings
    features["zero_crossings"] = np.sum(
        np.diff(np.sign(normalized_signal)) != 0
    )

    # FFT
    fft_abs = np.abs(np.fft.rfft(normalized_signal))
    frequencies = np.fft.rfftfreq(
        len(normalized_signal),
        d=1 / fs
    )

    features["fft_mean"] = np.mean(fft_abs)
    features["fft_std"] = np.std(fft_abs)
    features["fft_max"] = np.max(fft_abs)

    if len(fft_abs) > 1:
        fft_abs[0] = 0
        features["dominant_freq"] = frequencies[
            np.argmax(fft_abs)
        ]
    else:
        features["dominant_freq"] = 0.0

    # R-peaks
    try:
        clean_signal = nk.ecg_clean(
            signal,
            sampling_rate=fs
        )

        _, info = nk.ecg_peaks(
            clean_signal,
            sampling_rate=fs
        )

        peaks = np.asarray(
            info["ECG_R_Peaks"],
            dtype=int
        )

    except Exception:
        peaks = np.array([], dtype=int)

    features["num_peaks"] = len(peaks)

    # Intervalos RR
    rr_ms = np.array([])

    if len(peaks) > 2:
        rr_ms = np.diff(peaks) / fs * 1000

        rr_ms = rr_ms[
            np.isfinite(rr_ms)
            & (rr_ms >= 300)
            & (rr_ms <= 2000)
        ]

    if len(rr_ms) > 1:
        diff_rr = np.diff(rr_ms)

        features["rr_mean_ms"] = np.mean(rr_ms)
        features["rr_std_ms"] = np.std(rr_ms, ddof=1)
        features["rmssd_ms"] = np.sqrt(
            np.mean(diff_rr ** 2)
        )
        features["sdnn_ms"] = np.std(rr_ms, ddof=1)
        features["pnn50"] = np.mean(
            np.abs(diff_rr) > 50
        )

    else:
        features.update({
            "rr_mean_ms": 0.0,
            "rr_std_ms": 0.0,
            "rmssd_ms": 0.0,
            "sdnn_ms": 0.0,
            "pnn50": 0.0
        })

    # Catch22
    try:
        catch22 = pycatch22.catch22_all(
            normalized_signal
        )

        features.update(
            dict(zip(
                catch22["names"],
                catch22["values"]
            ))
        )

    except Exception:
        pass

    return features

# 4. PROCESSAR ECGs

def process_ecgs(dataframe, lead_index=1):
    features_list = []
    total = len(dataframe)

    for count, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1
    ):
        try:
            record = wfdb.rdrecord(row["filename_hr"])

            if record.p_signal is None:
                raise ValueError("Registro sem sinal.")

            number_of_leads = record.p_signal.shape[1]

            if not 0 <= lead_index < number_of_leads:
                raise ValueError("Índice de derivação inválido.")

            signal = record.p_signal[:, lead_index]

            if not np.all(np.isfinite(signal)):
                signal = (
                    pd.Series(signal)
                    .replace([np.inf, -np.inf], np.nan)
                    .interpolate(limit_direction="both")
                    .to_numpy()
                )

            if not np.all(np.isfinite(signal)):
                raise ValueError("Sinal contém valores inválidos.")

            fs = float(record.fs)

            signal = bandpass_filter(
                signal,
                fs=fs
            )

            features = extract_features(
                signal,
                fs=fs
            )

            features["label"] = int(row["label"])
            features["strat_fold"] = int(
                row["strat_fold"]
            )

            features_list.append(features)

            if count % 50 == 0 or count == total:
                print(f"ECGs processados: {count}/{total}")

        except Exception as error:
            print(f"Erro no ECG {count}: {error}")

    return pd.DataFrame(features_list)

# 5. EXTRAIR E SALVAR FEATURES

df_features = process_ecgs(
    df_small,
    lead_index=1
)

if df_features.empty:
    raise ValueError("Nenhum ECG foi processado.")

df_features.to_csv(
    "dataset_features_ecg.csv",
    index=False
)

print("\nShape:", df_features.shape)
print("\nClasses:")
print(df_features["label"].value_counts().sort_index())

# 6. DIVIDIR POR FOLDS

X = df_features.drop(
    columns=["label", "strat_fold"]
).replace([np.inf, -np.inf], np.nan)

y = df_features["label"]
folds = df_features["strat_fold"]

train_mask = folds.between(1, 8)
val_mask = folds == 9
test_mask = folds == 10

X_train = X.loc[train_mask].copy()
X_val = X.loc[val_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_val = y.loc[val_mask].copy()
y_test = y.loc[test_mask].copy()

if X_train.empty or X_val.empty or X_test.empty:
    raise ValueError("Treino, validação ou teste está vazio.")

# 7. TRATAR NaN

train_medians = X_train.median()

X_train = X_train.fillna(train_medians).fillna(0)
X_val = X_val.fillna(train_medians).fillna(0)
X_test = X_test.fillna(train_medians).fillna(0)

# 8. PADRONIZAR

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 9. SELECIONAR FEATURES

number_of_features = min(25, X.shape[1])

selector = SelectKBest(
    score_func=f_classif,
    k=number_of_features
)

X_train = selector.fit_transform(
    X_train,
    y_train
)

X_val = selector.transform(X_val)
X_test = selector.transform(X_test)

selected_features = X.columns[
    selector.get_support()
].tolist()

print("\nFeatures selecionadas:")
print(selected_features)

# 10. SMOTE SOMENTE NO TREINO

minority_count = pd.Series(y_train).value_counts().min()

if minority_count < 2:
    raise ValueError(
        "A classe minoritária precisa ter pelo menos 2 amostras para usar SMOTE."
    )

smote = SMOTE(
    random_state=42,
    k_neighbors=min(5, minority_count - 1)
)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("\nDistribuição antes do SMOTE:")
print(pd.Series(y.loc[train_mask]).value_counts().sort_index())

print("\nDistribuição após SMOTE:")
print(pd.Series(y_train).value_counts().sort_index())