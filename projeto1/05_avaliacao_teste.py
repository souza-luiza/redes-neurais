import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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


def executar_experimento(nome, momentum=0.0, l1_lambda=0.0, l2_lambda=0.0, dropout_p=0.0):

    # Reinicia a semente para garantir comparabilidade
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # DataLoaders novos para cada experimento
    train_loader_exp = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    val_loader_exp = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )

    # Modelo
    model = MLP(
        input_size=1,
        neurons=[32, 16, 8],
        activation="leaky_relu",
        dropout_p=dropout_p
    )

    # MSE
    criterion = nn.MSELoss()

    # SGD
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.15,
        momentum=momentum,
        weight_decay=l2_lambda
    )

    # Treinamento
    model, history, best_epoch = train_model(
        model,
        train_loader_exp,
        val_loader_exp,
        criterion,
        optimizer,
        max_epochs=2000,
        patience=100,
        l1_lambda=l1_lambda
    )

    # Resultado
    best_val = min(history["val_loss"])
    best_train = history["train_loss"][best_epoch - 1]

    print("\n" + "=" * 60)
    print(nome)
    print("=" * 60)
    print("Arquitetura: [32, 16, 8]")
    print("Ativação: LeakyReLU")
    print("Learning rate: 0.15")
    print("Batch size: 16")
    print("Momentum:", momentum)
    print("L1 lambda:", l1_lambda)
    print("L2 lambda:", l2_lambda)
    print("Dropout:", dropout_p)
    print("Patience: 100")
    print("-" * 60)
    print("Melhor época:", best_epoch)
    print("Épocas executadas:", len(history["val_loss"]))
    print("MSE treino na melhor época:", best_train)
    print("MSE validação:", best_val)
    print("=" * 60)

    return {
        "nome": nome,
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_train": best_train,
        "best_val": best_val
    }


# ============================================================
# AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# ============================================================

def avaliar_teste(model):

    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).numpy()

    # Valores reais e previsões voltam para a escala original
    y_test_original = scaler_y.inverse_transform(
        y_test_tensor.numpy()
    )

    y_pred_original = scaler_y.inverse_transform(
        y_pred_scaled
    )

    # Métricas na escala original
    mse = mean_squared_error(
        y_test_original,
        y_pred_original
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_test_original,
        y_pred_original
    )

    r2 = r2_score(
        y_test_original,
        y_pred_original
    )

    return {
        "y_test": y_test_original.flatten(),
        "y_pred": y_pred_original.flatten(),
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }



# ============================================================
# EXPERIMENTOS DE ABLAÇÃO - MELHORES CONFIGURAÇÕES
# ============================================================

# Baseline
baseline = executar_experimento(
    nome="BASELINE"
)


# Melhor configuração de Momentum
momentum_best = executar_experimento(
    nome="MOMENTUM beta=0.5",
    momentum=0.5
)


# Melhor configuração de L2
l2_best = executar_experimento(
    nome="L2 lambda=1e-5",
    l2_lambda=1e-5
)


# Melhor configuração de L1
l1_best = executar_experimento(
    nome="L1 lambda=1e-3",
    l1_lambda=1e-3
)


# Melhor configuração de Dropout
dropout_best = executar_experimento(
    nome="DROPOUT p=0.1",
    dropout_p=0.1
)


# ============================================================
# AVALIAÇÃO DOS MODELOS NO CONJUNTO DE TESTE
# ============================================================

resultado_baseline = avaliar_teste(
    baseline["model"]
)

resultado_momentum = avaliar_teste(
    momentum_best["model"]
)

resultado_l2 = avaliar_teste(
    l2_best["model"]
)

resultado_l1 = avaliar_teste(
    l1_best["model"]
)

resultado_dropout = avaliar_teste(
    dropout_best["model"]
)


# ============================================================
# TABELA FINAL - CONJUNTO DE TESTE
# ============================================================

resultados_teste = [
    (
        "Baseline Vanilla",
        baseline,
        resultado_baseline
    ),
    (
        "Momentum (beta=0.5)",
        momentum_best,
        resultado_momentum
    ),
    (
        "L2 (lambda=1e-5)",
        l2_best,
        resultado_l2
    ),
    (
        "L1 (lambda=1e-3)",
        l1_best,
        resultado_l1
    ),
    (
        "Dropout (p=0.1)",
        dropout_best,
        resultado_dropout
    )
]

print("\n")
print("=" * 100)
print("RESULTADOS FINAIS - CONJUNTO DE TESTE")
print("=" * 100)

print(
    f"{'Modelo':<25}"
    f"{'MSE':>12}"
    f"{'RMSE':>12}"
    f"{'MAE':>12}"
    f"{'R²':>12}"
    f"{'Melhor época':>16}"
)

print("-" * 100)

for nome, experimento, resultado in resultados_teste:

    print(
        f"{nome:<25}"
        f"{resultado['MSE']:>12.6f}"
        f"{resultado['RMSE']:>12.6f}"
        f"{resultado['MAE']:>12.6f}"
        f"{resultado['R2']:>12.6f}"
        f"{experimento['best_epoch']:>16}"
    )

print("=" * 100)


# ============================================================
# FIGURA 3 - GRÁFICO DE PARIDADE: BASELINE VS. MELHOR MODELO
# ============================================================

plt.figure(figsize=(7, 7))

# Dados do conjunto de teste - escala original
y_real = resultado_baseline["y_test"]

y_pred_baseline = resultado_baseline["y_pred"]
y_pred_l2 = resultado_l2["y_pred"]

# Pontos do Baseline
plt.scatter(
    y_real,
    y_pred_baseline,
    alpha=0.65,
    label="Baseline"
)

# Pontos do melhor modelo: Baseline + L2
plt.scatter(
    y_real,
    y_pred_l2,
    alpha=0.65,
    label=r"Baseline + L2 ($\lambda=10^{-5}$)"
)

# Linha de predição ideal: y_pred = y_real
valor_min = min(
    y_real.min(),
    y_pred_baseline.min(),
    y_pred_l2.min()
)

valor_max = max(
    y_real.max(),
    y_pred_baseline.max(),
    y_pred_l2.max()
)

plt.plot(
    [valor_min, valor_max],
    [valor_min, valor_max],
    linestyle="--",
    label="Predição ideal"
)

plt.xlabel(r"$Y_{\mathrm{real}}$")
plt.ylabel(r"$Y_{\mathrm{pred}}$")
plt.title(r"Gráfico de Paridade ($Y_{\mathrm{real}}$ vs. $Y_{\mathrm{pred}}$) -- Baseline vs. Melhor Modelo")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "Figura3_Grafico_Paridade.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FIGURA 4 - DISTRIBUIÇÃO DOS RESÍDUOS
# ============================================================

residuos = (
    resultado_l2["y_test"]
    - resultado_l2["y_pred"]
)

plt.figure(figsize=(8, 5))

plt.hist(
    residuos,
    bins=20,
    edgecolor="black"
)

plt.xlabel("Resíduo (y real - y predito)")
plt.ylabel("Frequência")
plt.title("Distribuição dos Resíduos - Modelo Baseline + L2")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "Figura4_Distribuicao_Residuos.png",
    dpi=300
)

plt.show()