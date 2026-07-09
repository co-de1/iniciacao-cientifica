import pandas as pd
import ast

# 1. Carregar dataset
df = pd.read_csv("ptbxl_database.csv", on_bad_lines='skip')

df_small = df[['scp_codes', 'filename_hr', 'filename_lr', 'strat_fold']].copy()

# 2. Carregar mapeamento SCP
df_scp = pd.read_csv("scp_statements.csv", index_col=0)

# manter apenas diagnósticos reais
df_scp = df_scp[df_scp['diagnostic'] == 1]

# criar dicionário: código SCP -> superclasse diagnóstica
scp_to_class = df_scp['diagnostic_class'].to_dict()

print("\nExemplo de mapeamento:")
for i, (k, v) in enumerate(scp_to_class.items()):
    print(k, "→", v)
    if i > 5:
        break

# 3. Limpeza da coluna scp_codes
df_small = df_small.dropna(subset=['scp_codes'])
df_small = df_small[df_small['scp_codes'].str.startswith('{')]

# 4. Mapear SCP codes para superclasses
def map_to_superclass(scp_codes_str):
    try:
        scp_dict = ast.literal_eval(scp_codes_str)
        labels = list(scp_dict.keys())

        classes = [
            scp_to_class[label]
            for label in labels
            if label in scp_to_class
        ]

        return list(set(classes))

    except Exception:
        return []

df_small['labels'] = df_small['scp_codes'].apply(map_to_superclass)

# 5. Remover linhas sem label diagnóstica
df_small = df_small[df_small['labels'].map(len) > 0]

# 6. Criar label binária DAC vs não-DAC
def map_to_dac(labels):
    dac_classes = {'MI', 'STTC'}
    non_dac_classes = {'NORM', 'CD', 'HYP'}

    if any(label in dac_classes for label in labels):
        return 1  # DAC

    if any(label in non_dac_classes for label in labels):
        return 0  # não-DAC

    return None

df_small['label'] = df_small['labels'].apply(map_to_dac)

# remover casos indefinidos, se existirem
df_small = df_small.dropna(subset=['label'])
df_small['label'] = df_small['label'].astype(int)

# 7. Debug
print("\nDistribuição DAC vs não-DAC:")
print(df_small['label'].value_counts())

print("\nDistribuição dos folds:")
print(df_small['strat_fold'].value_counts().sort_index())

# 8. Treino / teste
if 10 in df_small['strat_fold'].values:
    test_fold = 10
else:
    test_fold = df_small['strat_fold'].max()

print(f"\nUsando fold {test_fold} como teste")

train = df_small[df_small['strat_fold'] != test_fold]
test = df_small[df_small['strat_fold'] == test_fold]

print("\nShape treino:", train.shape)
print("Shape teste:", test.shape)

print("\nDistribuição treino:")
print(train['label'].value_counts())

print("\nDistribuição teste:")
print(test['label'].value_counts())

# 9. Salvar datasets
df_small.to_csv("dataset_ecg_basico.csv", index=False)
train.to_csv("train_ecg_basico.csv", index=False)
test.to_csv("test_ecg_basico.csv", index=False)

print("\nDataset salvo com sucesso!")
print(df_small[['scp_codes', 'labels', 'label', 'strat_fold']].head(10))