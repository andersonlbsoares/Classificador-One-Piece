# Classificador de Personagens de One Piece

Este projeto implementa um modelo de aprendizado profundo para classificar personagens da popular série de anime One Piece usando técnicas de reconhecimento de imagem.

## Visão Geral do Projeto

Desenvolvemos um classificador de personagens utilizando uma rede neural convolucional (CNN) para identificar personagens de One Piece a partir de imagens. O projeto utiliza transfer learning com a arquitetura MobileNetV2, escolhida pelo seu equilíbrio entre precisão e eficiência computacional.

### Principais Características

- Utiliza a arquitetura MobileNetV2 pré-treinada no ImageNet
- Implementa transfer learning para classificação de personagens de One Piece
- Alcança 87% de precisão no conjunto de teste

## Conjunto de Dados

O conjunto de dados consiste em imagens de vários personagens de One Piece. Realizamos as seguintes etapas de preparação de dados:

1. Removemos imagens invertidas (negativas) para melhorar o desempenho do modelo
2. Dividimos os dados em conjuntos de treinamento (80%), validação (10%) e teste (10%)
3. Redimensionamos as imagens para 256x256 pixels para processamento ideal

## Arquitetura do Modelo

Usamos a arquitetura MobileNetV2 com as seguintes modificações:

- Congelamos as camadas pré-treinadas
- Modificamos a camada de classificação para produzir o número correto de classes
- Adicionamos uma camada LogSoftmax para melhor estabilidade numérica

## Treinamento

O modelo foi treinado com os seguintes parâmetros:

- Número de épocas: 100
- Tamanho do lote (batch size): 300
- Otimizador: Adam
- Função de perda: Negative Log Likelihood Loss (NLLLoss)

## Resultados

Nosso modelo final alcançou uma acurácia de 87%

## Uso

Para usar o modelo treinado para previsões:

1. Carregue o modelo:
   ```python
   modelo = torch.load('./modelos/modelo_final.pt')
   modelo.eval()
   ```

2. Use a função `predicao_one_piece` para fazer previsões em novas imagens:
   ```python
   resultado = predicao_one_piece(modelo, 'caminho/para/sua/imagem.jpg')
   ```

## Melhorias Futuras

- Experimentar técnicas de aumento de dados
- Tentar métodos de ensemble com outras arquiteturas
- Coletar dados mais diversos para melhorar a generalização

## Dependências

- PyTorch
- torchvision
- numpy
- Pillow
- matplotlib
- scikit-learn