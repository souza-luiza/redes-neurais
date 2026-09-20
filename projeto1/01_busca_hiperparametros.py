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

# ============================================================
# MODELO MLP
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_size, neurons, activation="relu"):
        super().__init__()

        layers = []
        in_features = input_size

        for hidden_size in neurons:
            layers.append(nn.Linear(in_features, hidden_size))

            if activation == "relu":
                layers.append(nn.ReLU())

            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())

            elif activation == "tanh":
                layers.append(nn.Tanh())

            else:
                raise ValueError(f"Ativação desconhecida: {activation}")

            in_features = hidden_size

        # Camada de saída para regressão
        layers.append(nn.Linear(in_features, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================
# DATASETS E DATALOADERS
# ============================================================

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


def create_loaders(batch_size):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


# ============================================================
# TREINAMENTO DE UMA ÉPOCA
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(predictions, y_batch)

        loss.backward()

        optimizer.step()

        batch_size = X_batch.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# AVALIAÇÃO
# ============================================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            predictions = model(X_batch)

            loss = criterion(predictions, y_batch)

            batch_size = X_batch.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# TREINAMENTO COM EARLY STOPPING
# ============================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, max_epochs=2000, patience=100):

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Verifica se houve melhoria na validação
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            break

    # Recupera os pesos correspondentes ao melhor ponto
    model.load_state_dict(best_state)

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses
    }

    return model, history, best_epoch

# ============================================================
# ESPAÇO DE BUSCA
# ============================================================

opcoes_arquiteturas = [
    [32],
    [32, 16],
    [32, 16, 8]
]

opcoes_ativacoes = [
    "relu",
    "leaky_relu",
    "tanh"
]

opcoes_lotes = [
    16,
    32
]

opcoes_lr = [0.01, 0.10, 0.15, 0.20]

print("\nEspaço de busca:")
print("Arquiteturas:", opcoes_arquiteturas)
print("Ativações:", opcoes_ativacoes)
print("Batch sizes:", opcoes_lotes)
print("Learning rates:", opcoes_lr)

total_experimentos = (
    len(opcoes_arquiteturas)
    * len(opcoes_ativacoes)
    * len(opcoes_lotes)
    * len(opcoes_lr)
)

print("\nTotal de combinações:", total_experimentos)

# ============================================================
# BUSCA CONJUNTA DE HIPERPARÂMETROS
# ============================================================

resultados_busca = []

criterio = nn.MSELoss()

experimento = 0

for arquitetura in opcoes_arquiteturas:

    for ativacao in opcoes_ativacoes:

        for lr in opcoes_lr:

            for batch_size in opcoes_lotes:

                experimento += 1

                # --------------------------------------------
                # Reprodutibilidade
                # --------------------------------------------

                random.seed(SEED)
                np.random.seed(SEED)
                torch.manual_seed(SEED)

                # --------------------------------------------
                # DataLoaders
                # --------------------------------------------

                train_loader, val_loader, test_loader = create_loaders(
                    batch_size
                )

                # --------------------------------------------
                # Modelo
                # --------------------------------------------

                model = MLP(
                    input_size=1,
                    neurons=arquitetura,
                    activation=ativacao
                )

                # --------------------------------------------
                # SGD padrão
                # --------------------------------------------

                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=lr,
                    momentum=0.0
                )

                # --------------------------------------------
                # Treinamento
                # --------------------------------------------

                model, history, best_epoch = train_model(
                    model,
                    train_loader,
                    val_loader,
                    criterio,
                    optimizer,
                    max_epochs=2000,
                    patience=100
                )

                # --------------------------------------------
                # Recupera os erros no melhor ponto
                # --------------------------------------------

                best_val_mse = min(history["val_loss"])

                best_index = np.argmin(history["val_loss"])

                best_train_mse = history["train_loss"][best_index]

                epochs_executed = len(history["train_loss"])

                # --------------------------------------------
                # Armazena resultados
                # --------------------------------------------

                resultados_busca.append({
                    "arquitetura": str(arquitetura),
                    "ativacao": ativacao,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "train_mse": best_train_mse,
                    "val_mse": best_val_mse,
                    "best_epoch": best_epoch,
                    "epochs_executed": epochs_executed
                })

                print(
                    f"Experimento {experimento:02d}/90 | "
                    f"Arquitetura={arquitetura} | "
                    f"Ativação={ativacao} | "
                    f"LR={lr} | "
                    f"Batch={batch_size} | "
                    f"Val MSE={best_val_mse:.6f}"
                )


# ============================================================
# TABELA DE RESULTADOS
# ============================================================

resultados_df = pd.DataFrame(resultados_busca)

resultados_df = resultados_df.sort_values(
    by="val_mse",
    ascending=True
).reset_index(drop=True)

print("\nTop 10 configurações:")
print(resultados_df.head(10).to_string(index=False))

print("\nResumo das combinações por batch size:")
print(
    resultados_df.groupby("batch_size")["val_mse"]
    .agg(["min", "mean", "max"])
)

print("\nResumo das combinações por learning rate:")
print(
    resultados_df.groupby("learning_rate")["val_mse"]
    .agg(["min", "mean", "max"])
)

print("\nMelhores resultados por arquitetura:")
print(
    resultados_df
    .groupby("arquitetura")
    .first()
    [["ativacao", "learning_rate", "batch_size",
      "train_mse", "val_mse", "best_epoch"]]
    .sort_values("val_mse")
)