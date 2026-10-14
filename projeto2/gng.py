import numpy as np
from PIL import Image

imagem = Image.open("imagens/familia.jpg").convert("RGB")

imagem_array = np.array(imagem)

print("Tamanho da imagem: ", imagem.size)
print("Shape do array: ", imagem_array.shape)

pixels = imagem_array / 255

pixels = pixels.reshape(-1, 3)

print("Shape dos pixels: ", pixels.shape)

def treino_gng(pixels, eta_b, eta_n, a_max, lamb, beta, alpha, iteracoes, rng, n_neuronios_finais):

    n_neuronios = 2

    pesos = rng.random((n_neuronios, 3))

    erros = np.zeros(n_neuronios)

    conexoes = {
        0: {1: 0},
        1: {0: 0}
    }

    for iteracao in range(iteracoes):

        indice = rng.integers(len(pixels))
        pixel = pixels[indice]

        distancias = np.linalg.norm(pesos - pixel, axis=1)
        melhores = np.argsort(distancias)

        s1 = melhores[0]
        s2 = melhores[1]

        conexoes[s1][s2] = 0
        conexoes[s2][s1] = 0

        erro = distancias[s1] ** 2
        erros[s1] += erro


        pesos[s1] += eta_b * (pixel - pesos[s1])
        for vizinho in list(conexoes[s1]):
            if vizinho != s1:
                pesos[vizinho] += eta_n * (pixel - pesos[vizinho])


        for vizinho in list(conexoes[s1]):
            idade = conexoes[s1][vizinho] + 1

            conexoes[s1][vizinho] = idade
            conexoes[vizinho][s1] = idade

            if idade > a_max:
                del conexoes[s1][vizinho]
                del conexoes[vizinho][s1]
                

        # Remover neurônios sem conexão
        isolados = [k for k in conexoes if len(conexoes[k]) == 0]

        if len(pesos) - len(isolados) >= 2:
            manter = [k for k in range(len(pesos)) if k not in isolados]
            pesos = pesos[manter]
            erros = erros[manter]

            novo_indice = {
                antigo: novo
                for novo, antigo in enumerate(manter)
            }

            conexoes = {
                novo_indice[k]: {
                    novo_indice[v]: idade
                    for v, idade in conexoes[k].items()
                    if v in novo_indice
                }
                for k in manter
            }

        # Caso a condição seja satisfeita:
        if (iteracao + 1) % lamb == 0 and len(pesos) < n_neuronios_finais:
            # Determinar o neurônio q com o maior erro acumulado
            q = int(np.argmax(erros))

            # Dentre os vizinhos de q, determinar o neurônio f com maior erro acumulado
            if len(conexoes[q]) > 0:
                f = max(conexoes[q], key=lambda vizinho: erros[vizinho])
            
                # Adicionar a nova unidade r a rede interpolando seu vetor de pesos a partir de q e f
                peso_r = (pesos[q] + pesos[f]) / 2

                r = len(pesos)
                pesos = np.vstack([pesos, peso_r])
                erros = np.append(erros, 0.0)

                # Inserir conexões entre r-q, r-f e remover conexão q-f
                del conexoes[q][f]
                del conexoes[f][q]

                conexoes[r] = {
                    q: 0,
                    f: 0
                }

                conexoes[q][r] = 0
                conexoes[f][r] = 0

                # Diminuir o erro E das unidades q e f
                erros[q] += -alpha * erros[q]
                erros[f] += -alpha * erros[f]

                # Configurar o erro de r a partir dos erros de q
                erros[r] = erros[q]
        
        erros += -beta * erros

    return pesos, conexoes

rng = np.random.default_rng(42)

pesos, conexoes = treino_gng(
    pixels=pixels,
    eta_b=0.01,
    eta_n=0.0001,
    a_max=90,
    lamb=50,
    beta=0.005,
    alpha=0.5,
    iteracoes=40000,
    rng=rng,
    n_neuronios_finais=256
)

def erro_quantizacao(pixels, pesos):
    erros = []

    for pixel in pixels:

        distancias = np.linalg.norm(pesos - pixel, axis=1)

        bmu = np.argmin(distancias)

        erro = distancias[bmu]

        erros.append(erro)
    
    return np.mean(erros)


def erro_topografico(pixels, pesos, posicoes):
    erros = []

    for pixel in pixels:

        distancias = np.linalg.norm(pesos - pixel, axis=1)

        melhores = np.argsort(distancias)

        bmu = melhores[0]
        segundo = melhores[1]

        if segundo in conexoes.get(bmu, {}):
            erro = 0
        else:
            erro = 1
        
        erros.append(erro)

    return np.mean(erros)


eq = erro_quantizacao(pixels, pesos)

print("Erro de quantização:", eq)

et = erro_topografico(pixels, pesos, conexoes)

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