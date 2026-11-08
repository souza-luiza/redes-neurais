import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import copy

SEED = 42
torch.manual_seed(SEED)

BATCH_SIZE = 128

data_transform = transforms.Compose([transforms.ToTensor()]) # converte imagens do dataset para tensores pytorch

full_train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=data_transform, download=True) # 60.000 imagens de treino total

# divide-se o treino em treino e validação:
train_size = int(0.8 * len(full_train_dataset)) # 48.000 imagens de treino
val_size = len(full_train_dataset) - train_size # 12.000 imagens de validação

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

test_dataset = datasets.FashionMNIST(root="./data", train=False, transform=data_transform, download=True) # 10.000 imagens de teste

# DataLoaders para três conjuntos:
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True) # somente embaralhar o treino
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Tamanho de treino: {len(train_dataset)}")
print(f"Tamanho de validação: {len(val_dataset)}")
print(f"Tamanho de teste: {len(test_dataset)}")

latent_dim = 2

class VAE(nn.Module):
  def __init__(self, input_dim=784, hidden_dim1 = 512, hidden_dim2=256, latent_dim=latent_dim):
    super().__init__()

    # Encoder:
    self.fc1 = nn.Linear(input_dim, hidden_dim1)
    self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
    self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
    self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)

    # Decoder:
    self.fc3 = nn.Linear(latent_dim, hidden_dim2)
    self.fc4 = nn.Linear(hidden_dim2, hidden_dim1)
    self.fc5 = nn.Linear(hidden_dim1, input_dim)

    self.leaky_relu = nn.LeakyReLU()
    self.sigmoid = nn.Sigmoid()

  def encode(self, x):
    h1 = self.leaky_relu(self.fc1(x))
    h2 = self.leaky_relu(self.fc2(h1))
    mu = self.fc_mu(h2)
    logvar = self.fc_logvar(h2)
    return mu, logvar

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5*logvar) # desvio padrao = tira raiz quadrada do expoente de logvar
    eps = torch.randn_like(std) # epsilon da msm distribuicao do desvio padrao (randn_like = distribuicao normal padrao, gaussiana com média 0 e desvio padrão 1)
    z = mu + eps*std
    return z

  def decode(self, z):
    h3 = self.leaky_relu(self.fc3(z))
    h4 = self.leaky_relu(self.fc4(h3))
    out = self.sigmoid(self.fc5(h4))
    return out

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    x_reconstructed = self.decode(z)
    return x_reconstructed, mu, logvar

def vae_loss(x_recon, x, mu, logvar):
  recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum")

  kl_loss = 0.5*torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)

  return recon_loss + kl_loss

model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 60

historico_train_loss = []
historico_val_loss = []

# ------------------------------------------
melhor_estado = None
best_val_loss = float('inf')
melhor_epoca = 0
# ------------------------------------------

for epoch in range(epochs):
  model.train() # garante que modelo esta em modo de treino

  total_train_loss = 0

  for x, _ in (train_loader): # _ para rotulos que irao ser recebidos, mas nao usados.

    x = x.view(-1, 784) # para transformar (128, 1, 28, 28) em (128, 784) / manter o tamanho do lote e achatar o resto em 784

    optimizer.zero_grad() # para nao armazenar os gradientes dos lotes anteriores

    x_recon, mu, logvar = model(x)

    loss = vae_loss(x_recon, x, mu, logvar)

    # retropropagação
    loss.backward() # calcula quanto cada peso da rede contribuiu para erro final
    optimizer.step() # usa valores calculados para atualizar pesos na direção correta

    total_train_loss += loss.item() # .item() extrai valor numerico do tensor loss

  avg_train_loss = total_train_loss/len(train_loader.dataset) # train_loss médio por imagem, total_train_loss acumulou a soma dos erros de todas imagens processadas

  model.eval() # coloca o modelo em modo de avaliação

  total_val_loss = 0

  with torch.no_grad(): # desativo o calculo de gradientes
    for x_val, _ in (val_loader):
      x_val = x_val.view(-1, 784)
      x_recon_val, mu_val, logvar_val = model(x_val)
      loss_val = vae_loss(x_recon_val, x_val, mu_val, logvar_val)
      total_val_loss += loss_val.item()

  avg_val_loss = total_val_loss/len(val_loader.dataset)

  historico_train_loss.append(avg_train_loss)
  historico_val_loss.append(avg_val_loss)

  # --- SALVAR O MELHOR MODELO ---
  if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    melhor_estado = copy.deepcopy(model.state_dict()) # faz uma cópia real dos pesos atuais
    melhor_epoca = epoch
    print(f"--> Novo melhor modelo encontrado na Época {epoch + 1}! Erro de Validação: {avg_val_loss:.4f}. Salvo com sucesso.")
  # ------------------------------------------

  print(f"Epoch [{epoch + 1}/{epochs}] | Train_Loss: [{avg_train_loss:.4f}] | Val_Loss: [{avg_val_loss:.4f}]")

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
    for x_test, _ in test_loader:
        x_test = x_test.view(-1, 784)

        # passa os dados de teste pelo modelo final
        x_recon_test, mu_test, logvar_test = model(x_test)

        # calcula o erro final
        loss_test = vae_loss(x_recon_test, x_test, mu_test, logvar_test)
        total_test_loss += loss_test.item()

# calcula a perda média final por imagem do teste
avg_test_loss = total_test_loss / len(test_loader.dataset)

print("\n================ EVALIAÇÃO FINAL ================")
print(f"Loss Final no Conjunto de Teste: {avg_test_loss:.4f}")
print("=================================================")

model.eval()

with torch.no_grad(): # desliga gradientes
  x, _ = next(iter(test_loader)) # extrai lote de 128 imagens do carregador de dados, _ ignora os rotulos.
  x = x.view(-1, 784) # achata imagens 28x28 para o vetor com 784 elementos

  x_recon, _, _ = model(x) # retorna recon, mu e logvar, mas so quero o recon agora

  n = 8
  plt.figure(figsize=(12, 3))
  for i in range(n):
    plt.subplot(2, n, i + 1)
    plt.imshow(x[i].view(28, 28), cmap="gray") # para imagem ser exibida da forma correta
    plt.axis("off")

    plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_recon[i].view(28, 28), cmap="gray")
    plt.axis("off")

  plt.show()

# z aleatorios
with torch.no_grad():
  z = torch.randn(16, latent_dim)
  samples = model.decode(z)

  plt.figure(figsize=(6, 6))
  for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(samples[i].view(28, 28), cmap="gray")
    plt.axis("off")
  plt.show()


model.eval()

todos_mus = []
todos_rotulos = []

with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test = x_test.view(-1, 784)

        mu, _ = model.encode(x_test)

        todos_mus.append(mu)
        todos_rotulos.append(y_test)

todos_mus = torch.cat(todos_mus, dim=0).numpy()   
todos_rotulos = torch.cat(todos_rotulos, dim=0).numpy()

eixo_x = todos_mus[:, 0] # primeira dimensão do espaço latente
eixo_y = todos_mus[:, 1] # segunda dimensão do espaço latente

nomes_classes = ['Camiseta', 'Calça', 'Suéter', 'Vestido', 'Casaco',
                 'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']

plt.figure(figsize=(10, 8))

grafico = plt.scatter(eixo_x, eixo_y, c=todos_rotulos, cmap='tab10', alpha=0.7, edgecolors='none', s=15)

barra_cores = plt.colorbar(grafico, ticks=range(10))
barra_cores.set_ticklabels(nomes_classes)

plt.title("Espaço Latente 2D do VAE (FashionMNIST)", fontsize=14)
plt.xlabel("Dimensão Latente 1 (μ1)", fontsize=12)
plt.ylabel("Dimensão Latente 2 (μ2)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()