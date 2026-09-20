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


# Implementação de MLP:
# Classe genérica para MLP, capaz de representar várias arquiteturas:
class MLP(nn.Module):
    def __init__(self, input_size, neurons, activation="relu", dropout_p=0.0):
        super().__init__()

        layers = []

        # Número de entradas da primeira camada
        in_features = input_size

        # Camadas ocultas
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

            # Dropout somente quando solicitado
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            
            in_features = hidden_size

        # Camada de saída
        layers.append(nn.Linear(in_features, 1))

        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# Preparação dos dados para treinamento:
# TensorDataset = une as variáveis de entrada (X) com suas respostas corretas (y)
# DataLoader = indica de onde vem dados, 
# batch_size=16: rede neural vai olhar 16 exemplos, calcular erro, atualizar os pesos e pegar próximos 16.
# shuffle = no treinamento é importante embaralhar para a rede aprender padrões reais, porém na validação e teste, como estamos
# apenas avaliando o modelo, o resultado final da avaliação será o mesmo, então não há necessidade de embaralhar.
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Treinamento da rede neural por uma época:
def train_one_epoch(model, loader, criterion, optimizer, l1_lambda=0.0):

    model.train() # modo treino

    total_loss = 0.0 # acumulo de erro total
    total_samples = 0 # numero de dados processados

    for X_batch, y_batch in loader:

        optimizer.zero_grad() # zera gradientes acumulados do passo anterior

        predictions = model(X_batch) # passa dados de entrada pelo modelo para gerar previsões

        loss = criterion(predictions, y_batch) # calcula erro comparando com respostas reais (funcao de perda (critério))

        # Regularização L1
        if l1_lambda > 0:

            l1_penalty = 0.0

            for parameter in model.parameters():
                l1_penalty += torch.sum(torch.abs(parameter))

            loss = loss + l1_lambda * l1_penalty

        loss.backward() # gradiente de erro em relação a todos parametros (pesos) da rede neural 

        optimizer.step() # atualiza os pesos usando o gradiente calculado para ajustar os pesos tentando diminuir erro no proximo passo

        batch_size = X_batch.size(0) # quantos exemplos havia no batch

        total_loss += loss.item() * batch_size # Multiplica o erro médio do grupo pelo número de exemplos e soma ao total.
        total_samples += batch_size

    return total_loss / total_samples # Retorna a perda média ponderada da época inteira.


# Durante validação, pesos não são atualizados:
def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad(): # desativa cálculo dos gradientes

        for X_batch, y_batch in loader:

            predictions = model(X_batch)

            loss = criterion(predictions, y_batch)

            batch_size = X_batch.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


# FUNÇÃO COMPLETA DE TREINAMENTO:
def train_model(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience, l1_lambda=0.0):

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, l1_lambda)

        val_loss = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Verifica se houve melhoria
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

    # Restaura os melhores pesos
    model.load_state_dict(best_state)

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses
    }

    return model, history, best_epoch


# Modelo baseline
model = MLP(input_size=1, neurons=[32, 16, 8], activation="leaky_relu")

# Função de perda
criterion = nn.MSELoss()

# SGD padrão, sem momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.15, momentum=0.0)

# Treinamento
model, history, best_epoch = train_model(model, train_loader, val_loader, criterion, optimizer, max_epochs=2000, patience=100)

print("\nResultado da baseline:")
print("Épocas executadas:", len(history["train_loss"]))
print("Melhor época:", best_epoch)
print("Menor MSE de validação:", min(history["val_loss"]))
print("MSE de treino na melhor época:",
      history["train_loss"][best_epoch - 1])

# ============================================================
# FIGURA 1 - CURVA DE TREINO E VALIDAÇÃO: BASELINE VANILLA
# ============================================================

epochs = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(8, 5))

plt.plot(epochs, history["train_loss"], label="Treino")
plt.plot(epochs, history["val_loss"], label="Validação")

plt.xlabel("Época")
plt.ylabel("MSE")
plt.title("Curva de Treino e Validação: Baseline Vanilla")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("Figura1_Baseline_Vanilla.png", dpi=300)
plt.show()