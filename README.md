# Modelo de Classificação Naive Bayes para Predição de Obesidade

## Introdução

A obesidade é um problema de saúde pública global que afeta milhões de pessoas, aumentando significativamente o risco de doenças crônicas como diabetes tipo 2, doenças cardiovasculares, hipertensão e alguns tipos de câncer. De acordo com a Organização Mundial da Saúde (OMS), a prevalência de obesidade quase triplicou desde 1975, com mais de 650 milhões de adultos obesos em todo o mundo.

Este projeto implementa um modelo de classificação utilizando o algoritmo Naive Bayes para prever a probabilidade de obesidade com base em diversos fatores de risco e características individuais. A detecção precoce e a classificação precisa da obesidade podem ajudar profissionais de saúde a desenvolver intervenções personalizadas e estratégias de prevenção mais eficazes.

## Sobre o Algoritmo Naive Bayes

O Naive Bayes é um algoritmo de classificação probabilístico baseado no Teorema de Bayes, com a "suposição ingênua" (naive) de independência entre os atributos preditores. Apesar desta simplificação, o Naive Bayes é particularmente adequado para problemas de classificação relacionados à obesidade pelas seguintes razões:

- **Eficiência computacional**: O algoritmo apresenta alta performance mesmo com grandes conjuntos de dados.
- **Bom desempenho com poucos dados de treinamento**: Funciona bem mesmo quando o conjunto de treinamento não é muito extenso.
- **Capacidade de lidar com atributos categóricos e numéricos**: Ideal para dados de saúde que frequentemente combinam diferentes tipos de variáveis.
- **Interpretabilidade**: Os resultados são probabilísticos e oferecem insights sobre a relevância de cada fator na predição.
- **Robustez a atributos irrelevantes**: O modelo consegue identificar os preditores mais importantes.

Neste projeto, utilizamos a implementação GaussianNB do scikit-learn, que é especialmente adequada para atributos contínuos como idade, índices antropométricos e frequência de consumo alimentar.

## Instalação e Configuração

### Pré-requisitos

- Python 3.11+
- pipenv (opcional, para gerenciamento de ambiente virtual)

### Passos de instalação

1. Clone o repositório:
```bash
git clone https://github.com/srmatheusmaciel/classificacao_naive_bayes.git
cd classificacao_naive_bayes
```

2. Instale as dependências:
```bash
# Usando pipenv
pipenv install

# Ou usando pip
pip install pandas plotly matplotlib statsmodels scikit-learn ipykernel ipywidgets sweetviz flask optuna
```

3. Ative o ambiente virtual (se estiver usando pipenv):
```bash
pipenv shell
```

## Como Usar

### Executando o notebook

1. Inicie o Jupyter Notebook:
```bash
jupyter notebook
```


## Avaliação de Desempenho

O modelo foi avaliado usando as seguintes métricas:

### Modelo Baseline (Naive Bayes sem seleção de características)
- **Acurácia**: 0.81
- **Precisão**: 0.81
- **Recall**: 0.77
- **F1-Score**: 0.76

### Modelo Otimizado (Naive Bayes com SelectKBest)
- Melhorou a classificação de casos negativos (não obesos)
- Manteve alta sensibilidade para detecção de casos positivos (obesos)
- Reduzindo falsos positivos e falsos negativos

### Matriz de Confusão
O modelo final apresenta uma matriz de confusão equilibrada, com boa capacidade tanto para identificar corretamente indivíduos obesos quanto não obesos. A otimização de hiperparâmetros e seleção de características contribuíram para um modelo mais robusto.
