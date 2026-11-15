# Adult Income Classification

## 📌 Sinopse do Programa
Aplicação em Python para classificação preditiva de renda individual (`<=50K` vs `>50K`) a partir do conjunto de dados censitários **UCI Adult Census Income** (48.842 registros). O software implementa um fluxo completo e rigoroso de Machine Learning projetado para evitar vazamento de dados (*data leakage*), aplicando todas as transformações de aprendizado estritamente dentro de pipelines sob validação cruzada estratificada de 10 folds (*10-Fold Stratified Cross-Validation*).

> 📖 **Para uma análise detalhada, metodologia formal, tabelas e gráficos comparativos, consulte o [Relatório Técnico Completo (RELATORIO.md)](RELATORIO.md).**

---

## 💻 Sinopse do Código e Módulos

O código é modularizado em etapas sequenciais e reproduzíveis:

* **`prep.py`**: Módulo base do projeto. Realiza a carga determinística dos dados brutos, limpeza determinística, montagem dos transformadores (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `FunctionTransformer`) e execução da validação cruzada sem vazamento de dados.
* **`01_eda.py`**: Script de análise exploratória. Diagnostica a distribuição das classes (76% `<=50K` vs 24% `>50K`), identifica dados faltantes codificados como `?` em `workclass`, `occupation` e `native-country`, detecta redundâncias (`education` vs `education-num`) e analisa a assimetria severa em ganhos e perdas de capital.
* **`02_ablacao_preproc.py`**: Estudo sistemático de ablação de pré-processamento. Avalia o impacto isolado de estratégias de imputação (categoria `"Unknown"` vs moda vs remoção), eliminação de ruído/redundância (`fnlwgt`, `education`), transformação `log1p` e seleção de atributos (`SelectKBest`).
* **`03_experimentos.py`**: Treinamento e benchmark dos modelos preditivos. Executa validação cruzada de 10 folds comparando 4 algoritmos em 3 configurações distintas cada: **Random Forest**, **Árvore de Decisão**, **K-Nearest Neighbors (KNN)** e **Gaussian Naive Bayes**.
* **`04_desbalanceamento.py`**: Avaliação de técnicas para mitigar o desbalanceamento da classe minoritária (`>50K`), comparando o baseline com ponderação de classes (`class_weight='balanced'`), subamostragem aleatória (*RandomUnderSampler*) e sobreamostragem sintética (*SMOTE*).
* **`05_figuras_tabelas.py`**: Módulo de consolidação estatística e saída. Compila as tabelas de métricas (Acurácia, Precisão, Recall, F1-Score, ROC-AUC), calcula as matrizes de confusão acumuladas, gera os gráficos em `figures/` e realiza os testes pareados de hipótese de Wilcoxon.
* **`visualizar_dados.py`**: Utilitário interativo de linha de comando para inspecionar amostras, tipos de variáveis e distribuições no terminal ou via janelas gráficas.

---

## ⚙️ Como Executar

### 1. Dependências
```bash
pip install -r requirements.txt
```

### 2. Base de Dados
Certifique-se de que os arquivos `adult.data` e `adult.test` estejam posicionados no diretório `data/`.

### 3. Execução dos Scripts
Execute os módulos sequencialmente conforme o fluxo do estudo:

```bash
python 01_eda.py              # Análise exploratória de dados
python 02_ablacao_preproc.py  # Ablações de pré-processamento
python 03_experimentos.py     # Benchmark com 10-Fold CV
python 04_desbalanceamento.py # Avaliação de balanceamento de classes
python 05_figuras_tabelas.py  # Geração de tabelas e testes de Wilcoxon
```
