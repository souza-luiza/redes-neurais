# Projeto 1 — Regressão com MLP e Estudo de Ablação

Projeto desenvolvido na disciplina de Redes Neurais com o objetivo de avaliar uma rede **Multi-Layer Perceptron (MLP)** em uma tarefa de regressão e analisar o efeito de diferentes técnicas de otimização e regularização.

## Estrutura

- `dataset_projeto1.csv` — conjunto de dados utilizado no projeto.
- `figuras/` — gráficos finais do projeto utilizados no relatório.
- `relatorio.pdf` — relatório final do projeto.

## Código

Os scripts estão organizados de acordo com as etapas do experimento:

1. `01_busca_hiperparametros.py` — busca das configurações do modelo Vanilla.
2. `02_baseline_vanilla.py` — treinamento e análise do modelo baseline selecionado.
3. `03_busca_ablacao.py` — busca dos valores das técnicas de ablação.
4. `04_ablacao.py` — execução das melhores configurações de cada técnica.
5. `05_avaliacao_teste.py` — avaliação final dos modelos no conjunto de teste.
