import pandas as pd
import ast

# 1. CARREGAR DATASET
df = pd.read_csv("ptbxl_database.csv", on_bad_lines='skip')

df_small = df[['scp_codes', 'filename_hr', 'filename_lr', 'strat_fold']].copy()

# 2. CARREGAR MAPEAMENTO SCP
df_scp = pd.read_csv("scp_statements.csv")

# manter apenas diagnósticos reais
df_scp = df_scp[df_scp['diagnostic'] == 1]

# GARANTIR COLUNA CORRETA DO CÓDIGO SCP
if 'scp_code' in df_scp.columns:
    key_col = 'scp_code'
else:
    key_col = df_scp.columns[0]  # fallback seguro

scp_to_class = dict(zip(df_scp[key_col], df_scp['diagnostic_class']))

print("\nExemplo de mapeamento:")
for i, (k, v) in enumerate(scp_to_class.items()):
    print(k, "→", v)
    if i > 5:
        break

# 3. LIMPEZA scp_codes
df_small = df_small.dropna(subset=['scp_codes'])
df_small = df_small[df_small['scp_codes'].str.startswith('{')]

# 4. MAPEAR PARA SUPERCLASSES
def map_to_superclass(scp_codes_str):
    try:
        scp_dict = ast.literal_eval(scp_codes_str)
        labels = list(scp_dict.keys())

        classes = [scp_to_class[l] for l in labels if l in scp_to_class]

        return list(set(classes))  # remove duplicados
    except:
        return []

df_small['labels'] = df_small['scp_codes'].apply(map_to_superclass)

# 5. REMOVER LINHAS SEM LABEL
df_small = df_small[df_small['labels'].map(len) > 0]

# 6. SINGLE LABEL
def map_to_dac(labels):
    dac_classes = {'MI', 'STTC'}

    if any(l in dac_classes for l in labels):
        return 1  # DAC
    else:
        return 0  # não-DAC

df_small['label'] = df_small['labels'].apply(map_to_dac)

# 7. DEBUG (IMPORTANTE)
print("Distribuição das classes:")
print(df_small['label'].value_counts())

print("\nDistribuição dos folds:")
print(df_small['strat_fold'].value_counts())

# 8. TREINO / TESTE (ROBUSTO)
if 10 in df_small['strat_fold'].values:
    test_fold = 10
else:
    test_fold = df_small['strat_fold'].max()

print(f"\nUsando fold {test_fold} como teste")

train = df_small[df_small['strat_fold'] != test_fold]
test = df_small[df_small['strat_fold'] == test_fold]

# 9. SALVAR DATASET
df_small.to_csv("dataset_ecg_basico.csv", index=False)

print("\nShape df_small:", df_small.shape)
print(df_small[['scp_codes', 'labels']].head(10))

print("\nDataset salvo com sucesso!")
print(df_small.head())

print("\nDistribuição DAC vs não-DAC:")
print(df_small['label'].value_counts())