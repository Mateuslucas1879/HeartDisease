# Heart Disease Prediction

![Heart Disease Prediction](https://img.shields.io/badge/Flask-1.1.2-blue) ![Python](https://img.shields.io/badge/Python-3.12%2B-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

## 📝 Sobre o Projeto

O **Heart Disease Prediction** é uma aplicação web que utiliza técnicas de aprendizado de máquina para prever o risco de doenças cardíacas com base em informações médicas fornecidas pelo usuário. A aplicação é desenvolvida usando **Flask** para o backend e **Scikit-learn** para a modelagem preditiva.

### Funcionalidade

A aplicação oferece um modelo interativo onde os usuários podem:
- Treinar modelos de aprendizado de máquina com diferentes algoritmos.
- Inserir dados pessoais para prever o risco de um ataque cardíaco.
- Obter métricas de avaliação do modelo, como precisão, sensibilidade.
- Visualizar a matriz de confusão para análise de desempenho do modelo.

## 📊 Detalhes do Projeto

### 1. **Base de Dados**

O projeto utiliza um conjunto de dados sobre doenças cardíacas, com as seguintes características:
- **age**: Idade do paciente
- **sex**: Sexo do paciente (1 = feminino, 0 = masculino)
- **cp**: Tipo de dor no peito (0 a 3)
- **trestbps**: Pressão arterial em repouso (em mm Hg)
- **chol**: Colesterol sérico (em mg/dl)
- **fbs**: Nível de açúcar no sangue em jejum (0 = menor que 120 mg/dl, 1 = maior ou igual a 120 mg/dl)
- **restecg**: Resultados do eletrocardiograma em repouso (0 a 2)
- **thalach**: Frequência cardíaca máxima alcançada
- **exang**: Angina induzida por exercício (0 = não, 1 = sim)
- **oldpeak**: Depressão do segmento ST induzida por exercício
- **slope**: Inclinação do segmento ST (0 a 2)
- **ca**: Número de vasos sanguíneos coloridos (0 a 3)
- **thal**: Tipo de talassemia (0 a 3)
- **target**: Resultado da presença de doenças cardíacas (0 = não, 1 = sim)

### 2. **Pré-processamento dos Dados**

Os dados são pré-processados com os seguintes passos:
- **Codificação One-Hot**: Variáveis categóricas, como `cp` e `restecg`, são convertidas em variáveis dummy.
- **Escalonamento**: Utilizamos o `StandardScaler` para normalizar os dados, garantindo que todas as características tenham a mesma escala.

### 3. **Modelos de Machine Learning**

O projeto suporta três tipos de modelos:
- **Random Forest Classifier**: Um modelo de ensemble que combina múltiplas árvores de decisão para melhorar a precisão.
- **K-Nearest Neighbors (KNN)**: Um classificador baseado na proximidade dos dados de entrada.
- **Logistic Regression**: Um modelo de regressão usado para problemas de classificação binária, que fornece probabilidades de pertença às classes.

### 4. **Treinamento do Modelo**

O treinamento é realizado com as seguintes etapas:
- **Divisão dos Dados**: Os dados são divididos em conjuntos de treinamento e teste (80% treino, 20% teste).
- **Treinamento**: O modelo é treinado com os dados de treinamento e avaliado com os dados de teste.
- **Métricas**: São calculadas a precisão do modelo e a matriz de confusão (para classificação).

### 5. **Previsão**

Para fazer previsões:
- **Entrada de Dados**: O usuário insere dados pessoais através de um formulário.
- **Preprocessamento**: Os dados são pré-processados da mesma forma que os dados de treinamento.
- **Predição**: O modelo prevê a classe (para classificadores) ou a probabilidade (para regressão) do risco de doença cardíaca.

### 6. **Interface do Usuário**

A aplicação possui duas páginas principais:
- **Página Inicial**: Permite treinar o modelo e visualizar métricas.
- **Resultados da Previsão**: Mostra a probabilidade de um ataque cardíaco ou a classe prevista, além de outras métricas relevantes.

## 🛠️ Tecnologias Utilizadas

- **Flask**: Framework web para Python.
- **Scikit-learn**: Biblioteca de machine learning para Python.
- **Pandas**: Biblioteca para manipulação e análise de dados.
- **HTML/CSS**: Para a criação da interface web.

## 📦 Instalação e Execução

1. **Clone o Repositório**
   ```bash
   git clone https://github.com/seu-usuario/heart-disease-prediction.git
   cd heart-disease-prediction
