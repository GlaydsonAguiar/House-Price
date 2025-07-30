# 📘 Projeto – Previsão com Regressão Linear

## 🧠 Objetivo
Desenvolver um modelo de **regressão linear** usando o scikit-learn para prever um valor-alvo com base em variáveis numéricas do dataset `train.csv`. O foco foi aplicar conceitos de **machine learning supervisionado** com validação e avaliação de desempenho.

---

## 📁 Dataset
- Arquivo: `train.csv`
- Contém colunas numéricas e categóricas (ex: `MSSubClass`, `LotArea`, etc.).
- Usado para fins de aprendizado e validação de modelo preditivo.

---

## 🔧 Tecnologias e Bibliotecas
- **Python**
- **Pandas** – Manipulação e limpeza dos dados
- **NumPy** – Operações numéricas
- **Matplotlib** – Visualização de dados
- **Scikit-learn** – Treinamento, validação e métrica do modelo de regressão

---

## 📌 Etapas do Projeto
1. 📥 **Leitura do Dataset**  
   - `pd.read_csv("train.csv")`

2. 🧹 **Exploração e Limpeza de Dados**  
   - Verificação de colunas, tipos e valores ausentes.
   - Seleção de variáveis relevantes.

3. 🔀 **Separação em Treino e Teste**  
   - Usando `train_test_split` (80% treino / 20% teste).

4. 📈 **Treinamento do Modelo de Regressão Linear**  
   - `LinearRegression()` do scikit-learn.

5. 🧪 **Avaliação do Modelo**  
   - Métrica usada: **Erro Médio Absoluto (MAE)**.

6. 📉 **Visualização de Resultados**  
   - Gráfico de dispersão real vs. previsto.

---

## 📊 Exemplo de Código
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Separação em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Previsões e avaliação
y_pred = modelo.predict(X_test)
erro = mean_absolute_error(y_test, y_pred)
```

---

## 📈 Resultados
- O modelo obteve um **MAE** de aproximadamente `25` .
- A visualização mostra que a regressão teve desempenho razoável, mas com margem de erro em casos extremos.

---

## 🚀 Possíveis Melhorias
- Normalizar as variáveis de entrada
- Testar outros algoritmos como Ridge, Lasso, RandomForest
- Ajustar hiperparâmetros com GridSearchCV
- Tratar variáveis categóricas

---


