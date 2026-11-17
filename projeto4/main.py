import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import copy

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

df = pd.read_csv('steamdb_chart.csv')

df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values('DateTime')

valores = df[['In-Game']].values

n_total = len(valores) # 2792
n_train = int(0.7 * n_total) # 70% para treino
n_val = int(0.15 * n_total) # 15% para validação

treino_bruto = valores[:n_train]
val_bruto = valores[n_train : n_val + n_train]
test_bruto = valores[n_val + n_train:]

scaler = MinMaxScaler()
treino_norm = scaler.fit_transform(treino_bruto)
val_norm = scaler.transform(val_bruto)
test_norm = scaler.transform(test_bruto)

def criar_janelas(serie, tamanho_janela, k):
  x, y = [], []
  for i in range(len(serie) - tamanho_janela - k + 1):
    x.append(serie[i : i + tamanho_janela])
    y.append(serie[i + tamanho_janela + k - 1])
  return np.array(x), np.array(y)

TAMANHO_JANELA = 21
K_DIAS_FUTUROS = 5

x_train, y_train = criar_janelas(treino_norm, TAMANHO_JANELA, K_DIAS_FUTUROS)
x_val, y_val = criar_janelas(val_norm, TAMANHO_JANELA, K_DIAS_FUTUROS)
x_test, y_test = criar_janelas(test_norm, TAMANHO_JANELA, K_DIAS_FUTUROS)

# tensores pytorch:
x_train, y_train = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
x_val, y_val = torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
x_test, y_test = torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

BATCH_SIZE = 32

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

class LSTM(nn.Module):
  def __init__(self, input_size=1, hidden_size=32):
    super().__init__()
    self.hidden_size = hidden_size

    self.lstm1 = nn.LSTMCell(input_size, hidden_size)
    self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)

    self.fc1 = nn.Linear(hidden_size, 32)
    self.fc2 = nn.Linear(32, input_size)

    self.leaky_relu = nn.LeakyReLU()

  def forward(self, x):
    batch_size = x.size(0)
    seq_len = x.size(1) # quantos dias cada janela tem

    # limpando memoria para janela nova:
    h1 = torch.zeros(batch_size, self.hidden_size) # memorias de curto prazo
    c1 = torch.zeros(batch_size, self.hidden_size) # memorias de longo prazo

    h2 = torch.zeros(batch_size, self.hidden_size)
    c2 = torch.zeros(batch_size, self.hidden_size)

    for t in range(seq_len):
      x_t = x[:, t, :] # pega valor de jogadores do dia t
      h1, c1 = self.lstm1(x_t, (h1, c1)) # usa memoria de dias anteriores
      h2, c2 = self.lstm2(h1, (h2, c2))
    
    out = self.leaky_relu(self.fc1(h2))
    prediction = self.fc2(out)
    return prediction

model = LSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 70

historico_train_loss = []
historico_val_loss = []

# ------------------------------------------
melhor_estado = None
best_val_loss = float('inf')
melhor_epoca = 0
# ------------------------------------------

for epoch in range(epochs):
  model.train()

  total_train_loss = 0

  for x, y in (train_loader):

    optimizer.zero_grad() # para nao armazenar os gradientes dos lotes anteriores

    pred = model(x)

    loss = criterion(pred, y)

    # retropropagação
    loss.backward() # calcula quanto cada peso da rede contribuiu para erro final
    optimizer.step() # usa valores calculados para atualizar pesos na direção correta

    total_train_loss += loss.item()
  
  avg_train_loss = total_train_loss / len(train_loader)

  model.eval()

  total_val_loss = 0

  with torch.no_grad():
    for x_val, y_val in (val_loader):
      pred_val = model(x_val)
      loss_val = criterion(pred_val, y_val)
      total_val_loss += loss_val.item()

  avg_val_loss = total_val_loss / len(val_loader)

  historico_train_loss.append(avg_train_loss)
  historico_val_loss.append(avg_val_loss)

  # --- LÓGICA PARA SALVAR O MELHOR MODELO ---
  if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    melhor_estado = copy.deepcopy(model.state_dict()) # Faz uma cópia real dos pesos atuais
    melhor_epoca = epoch
    print(f"--> Novo melhor modelo encontrado na Época {epoch + 1}! Erro de Validação: {avg_val_loss:.6f}. Salvo com sucesso.")
  # ------------------------------------------

  print(f"Epoch [{epoch + 1}/{epochs}] | Train_Loss: [{avg_train_loss:.6f}] | Val_Loss: [{avg_val_loss:.6f}]")


plt.figure(figsize=(10, 5))
plt.plot(historico_train_loss, label="Erro de Treino (Train Loss)", linewidth=2)
plt.plot(historico_val_loss, label="Erro de Validação (Val Loss)", linewidth=2)
plt.axvline(x=melhor_epoca, color='red', linestyle='--', linewidth=1.5, label=f"Melhor Época (Época {melhor_epoca})")

plt.title("Evolução do Erro (Loss) durante o Treinamento", fontsize=14)
plt.xlabel("Épocas", fontsize=12)
plt.ylabel("Valor do Erro", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.show()


# =========================================================
# FASE DE TESTE FINAL
# =========================================================
if melhor_estado is not None:
    model.load_state_dict(melhor_estado)

model.eval() # modo de avaliação
total_test_loss = 0

with torch.no_grad(): # sem calcular gradientes
  test_preds_norm = model(x_test).numpy()

# calcula o MSE na escala normalizada para comparar com treino e validação
test_mse_norm = mean_squared_error(y_test.numpy(), test_preds_norm)

y_test_real = scaler.inverse_transform(y_test.numpy())
y_pred_real = scaler.inverse_transform(test_preds_norm)

rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae = mean_absolute_error(y_test_real, y_pred_real)
r2 = r2_score(y_test_real, y_pred_real)

print(f"\n--- Resultados no Conjunto de Teste (Previsão para t+{K_DIAS_FUTUROS} dias) ---")
print(f"MSE (Normalizado): {test_mse_norm}")
print(f"RMSE: {rmse} jogadores")
print(f"MAE:  {mae} jogadores")
print(f"R2: {r2}")


fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(y_test_real, label='Real (In-Game)', color='royalblue', linewidth=1.5)
ax.plot(
    y_pred_real,
    label=f'Previsto pela LSTM (t+{K_DIAS_FUTUROS}d)',
    color='darkorange',
    linestyle='--',
    linewidth=1.5,
)

ax.set_title(
    f'Jogadores In-Game na Steam: Real vs Previsto (k={K_DIAS_FUTUROS})',
    fontsize=12,
    fontweight='bold',
)
ax.set_xlabel('Dias (Conjunto de Teste)', fontsize=10)
ax.set_ylabel('Jogadores In-Game', fontsize=10)

ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.1f}M'.replace('.', ','))
)

ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()