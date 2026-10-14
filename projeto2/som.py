import numpy as np
from PIL import Image

imagem = Image.open("imagens/familia.jpg").convert("RGB")

imagem_array = np.array(imagem)

print("Tamanho da imagem: ", imagem.size)
print("Shape do array: ", imagem_array.shape)

pixels = imagem_array / 255

pixels = pixels.reshape(-1, 3)

print("Shape dos pixels: ", pixels.shape)

def treino_som(pixels, grid_linhas, grid_colunas, learning_rate_inicial, sigma_inicial, iteracoes, rng):

    n_neuronios = grid_linhas * grid_colunas

    pesos = rng.random((n_neuronios, 3))

    posicoes = []

    for linha in range(grid_linhas):
        for coluna in range(grid_colunas):
            posicoes.append([linha, coluna])

    posicoes = np.array(posicoes)

    tau_learning = iteracoes
    tau_sigma = iteracoes / np.log(sigma_inicial)

    for iteracao in range(iteracoes):

        indice = rng.integers(len(pixels))
        pixel = pixels[indice]

        distancias = np.linalg.norm(pesos - pixel, axis=1)

        bmu = np.argmin(distancias)

        distancias_grid = np.linalg.norm(posicoes - posicoes[bmu], axis=1)

        learning_rate = (learning_rate_inicial * np.exp(-iteracao / tau_learning))

        sigma = (sigma_inicial * np.exp(-iteracao / tau_sigma))

        influencia = np.exp(-(distancias_grid ** 2) / (2 * sigma ** 2))

        pesos += (learning_rate * influencia[:, np.newaxis] * (pixel - pesos))

    return pesos, posicoes


rng = np.random.default_rng(42)

pesos, posicoes = treino_som(
    pixels=pixels,
    grid_linhas=16,
    grid_colunas=16,
    learning_rate_inicial=0.5,
    sigma_inicial=8.0,
    iteracoes=40000,
    rng=rng
)


def erro_quantizacao(pixels, pesos):
    erros = []

    for pixel in pixels:

        distancias = np.linalg.norm(pesos - pixel, axis=1)

        bmu = np.argmin(distancias)

        erro = distancias[bmu]

        erros.append(erro)
    
    return np.mean(erros)


def sao_vizinhos(pos1, pos2):

    distancia = (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    return distancia == 1


def erro_topografico(pixels, pesos, posicoes):
    erros = []

    for pixel in pixels:

        distancias = np.linalg.norm(pesos - pixel, axis=1)

        melhores = np.argsort(distancias)

        bmu = melhores[0]
        segundo = melhores[1]

        if sao_vizinhos(posicoes[bmu], posicoes[segundo]):
            erro = 0
        else:
            erro = 1
        
        erros.append(erro)

    return np.mean(erros)


eq = erro_quantizacao(pixels, pesos)

print("Erro de quantização:", eq)

et = erro_topografico(pixels, pesos, posicoes)

print("Erro topográfico:", et)

def reconstruir_imagem(pixels, pesos):
    pixels_reconstruidos = []

    for pixel in pixels:
        distancias = np.linalg.norm(pesos - pixel,axis=1)

        bmu = np.argmin(distancias)

        nova_cor = pesos[bmu]

        pixels_reconstruidos.append(nova_cor)

    return np.array(pixels_reconstruidos)


pixels_reconstruidos = reconstruir_imagem(pixels, pesos)

altura, largura, canais = imagem_array.shape

imagem_reconstruida = pixels_reconstruidos.reshape(
        altura,
        largura,
        3
    )

imagem_reconstruida = (
        imagem_reconstruida * 255
    ).clip(0, 255).astype(np.uint8)

img_pil = Image.fromarray(imagem_reconstruida)

img_pil.save("imagem_reconstruida.jpg")

img_pil.show()

cores_unicas = np.unique(
    imagem_reconstruida.reshape(-1, 3),
    axis=0
)

print(
    "Número de cores:",
    len(cores_unicas)
)