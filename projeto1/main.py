import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Garantia da reprodutibilidade do projeto
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Carregamento do dataset
df = pd.read_csv("dataset_projeto1.csv")

# Primeiras linhas do dataset
print("\nPrimeiras linhas do dataset:")
print(df.head())

# Dimensão do dataset
print("\nDimensão do dataset:", df.shape)

print("\nValores ausentes:")
print(df.isnull().sum())

# Estatísticas descritivas
print("\nEstatísticas do dataset:")
print(df.describe())


# Visualização dos dados
plt.figure(figsize=(8, 5))
plt.scatter(df["x"], df["y"])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Dataset de regressão")
plt.show()


# Separação de entrada e alvo
X = df[["x"]].values
y = df[["y"]].values


# Divisão de 10% treino, 10% validação e 80% teste:

# 80% teste e 20% restante:
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.80, random_state=SEED)

# Dos 20% restantes: 10% do total deve ser validação e 10% do total deve ser treino
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

print("\nNúmero de amostras após divisão:")
print("Treino:", len(X_train))
print("Validação:", len(X_val))
print("Teste:", len(X_test))


# Escalonamento:

# Scalers separados para entrada e alvo
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# IMPORTANTE: fit SOMENTE no conjunto de treino
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# Nos conjuntos de validação e teste usa-se apenas transform
X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val)

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# IMPORTANTE: Não deve-se usar fit_transform antes da divisão de dados, pois a média e o desvio padrão seriam calculados
# com os dados de validação e teste, resultando em data leakage (vazamento de dados)

# PARA APRESENTAR RESULTADOS NO RELATÓRIO, VOLTAR Y PARA A ESCALA ORIGINAL:
# y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
# Assim, MSE, RMSE e MAE podem ser interpretados na escala original de y.


# Conversão para Tensores PyTorch:
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

print("\nTensores:")
print("X_train:", X_train_tensor.shape)
print("y_train:", y_train_tensor.shape)