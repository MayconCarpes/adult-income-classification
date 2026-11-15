# 🎤 Roteiro de Apresentação — Adult Income Classification

Este roteiro foi estruturado para uma apresentação de **10 a 15 minutos** (ideal para bancas acadêmicas, reuniões de equipe ou defesas de projeto).

---

## ⏱️ Estrutura e Tempo Estimado

| Slide / Bloco | Assunto | Tempo Sugerido | Imagem / Recurso de Apoio |
| :---: | :--- | :---: | :--- |
| **1** | Capa e Introdução ao Problema | 1 min | Título do Repositório |
| **2** | Análise Exploratória & Desafios dos Dados | 2 min | `figures/fig1_eda.png` |
| **3** | Metodologia & Rigor (Prevenção de Vazamento) | 2 min | Diagrama do Pipeline |
| **4** | Estudo de Ablação do Pré-processamento | 2 min | Tabela de Ablações |
| **5** | Comparativo de Algoritmos (Benchmark) | 3 min | `figures/fig2_f1_configuracoes.png`, `fig3_melhores_metricas.png` |
| **6** | Validação Estatística (Teste de Wilcoxon) | 2 min | `results/testes_wilcoxon.csv` |
| **7** | Desbalanceamento de Classes (SMOTE vs Ponderação) | 2 min | `figures/fig6_desbalanceamento.png` |
| **8** | Conclusões & Demonstração ao Vivo | 1 a 2 min | `python demo_apresentacao.py` |

---

## 🗣️ O que falar em cada Slide

### Slide 1: Capa & Contexto do Negócio
- **Objetivo**: Prever se uma pessoa ganha mais de US$ 50.000 ao ano com base em dados do censo norte-americano (UCI Adult Dataset).
- **Fala Sugerida**:
  > *"Boa noite a todos. Hoje apresento nosso projeto de classificação e benchmarking de Machine Learning sobre a base UCI Adult Income. O objetivo central é prever se a renda anual de um indivíduo ultrapassa 50 mil dólares com base em características demográficas e profissionais, garantindo uma abordagem metodológica estrita contra data leakage e validação estatística formal dos resultados."*

---

### Slide 2: Análise Exploratória de Dados (EDA)
- **Recurso**: `figures/fig1_eda.png`
- **Pontos-chave**:
  - Total de **48.842 instâncias** (unindo treino e teste originais para validação cruzada 10-fold).
  - **Desbalanceamento natural**: ~76% da classe `<=50K` e apenas ~24% da classe `>50K`.
  - **Valores ausentes**: Identificados pelo caractere `?` em `workclass` (5,7%), `occupation` (5,7%) e `native-country` (1,8%).
  - **Assimetria extrema**: Variáveis de ganho e perda de capital (`capital-gain` e `capital-loss`) têm mais de 91% de zeros e cauda muito longa.
- **Fala Sugerida**:
  > *"Na análise exploratória, identificamos dois grandes desafios: primeiro, o desbalanceamento da base (três quartos dos indivíduos recebem até 50 mil dólares); segundo, dados ausentes disfarçados pelo caractere de interrogação em ocupação e país de origem. Além disso, variáveis financeiras possuíam assimetria altíssima com mais de 90% de valores zerados."*

---

### Slide 3: Metodologia & Rigor Metodológico
- **Pontos-chave**:
  - Validação Cruzada Estratificada de 10 Folds (`StratifiedKFold`, `random_state=42`).
  - **Sem Vazamento de Dados (*No Data Leakage*)**: Imputadores, padronizadores, one-hot encoders e reamostradores ajustados exclusivamente no fold de treino usando `imblearn.pipeline`.
- **Fala Sugerida**:
  > *"Um dos maiores diferenciais técnicos deste trabalho foi o rigor metodológico. Não fizemos nenhuma transformação com aprendizado de parâmetros antes da validação cruzada. Todo o pré-processamento foi encapsulado dentro de Pipelines do Scikit-Learn e Imbalanced-Learn. Isso garante que a média, desvio-padrão ou frequências de categorias nunca vazem do conjunto de treino para o conjunto de teste de cada fold."*

---

### Slide 4: Estudo de Ablação do Pré-processamento
- **Pontos-chave**:
  - `?` tratado como categoria `"Unknown"` superou a remoção de linhas e a imputação pela moda.
  - Remoção de `education`: 100% redundante com `education-num` (reduz dimensionalidade sem perder informação).
  - Remoção de `fnlwgt`: Peso amostral do Censo que gerava ruído e piorava a generalização.
  - Transformação `log1p` estabilizou a variância de `capital-gain`/`loss`.
- **Fala Sugerida**:
  > *"Antes de treinar os algoritmos finais, realizamos um estudo sistemático de ablação de pré-processamento. Constatamos que manter os dados faltantes sob a categoria 'Unknown' preservou mais sinal preditivo do que descartar linhas. Também eliminamos atributos como 'fnlwgt' (peso censitário que causava ruído) e 'education' (que mapeia 1 para 1 com a versão numérica), além de aplicar transformação logarítmica nos ganhos de capital."*

---

### Slide 5: Comparativo dos Algoritmos (Benchmark)
- **Recursos**: `figures/fig2_f1_configuracoes.png`, `fig3_melhores_metricas.png`, `fig4_matrizes_confusao.png`
- **Resultados**:
  - **Random Forest**: Campeão absoluto em F1 (0.6829), Acurácia (86.65%) e ROC-AUC (0.9201).
  - **Árvore de Decisão**: Bom equilíbrio com poda (`leaf>=20`), alcançando F1 de 0.6747.
  - **KNN**: Sensível à alta dimensionalidade, melhorou conforme `k` subiu para 31 (F1 de 0.6658).
  - **Naive Bayes**: Alta revocação (86.95%), porém baixa precisão (48.69%), gerando muitos falsos positivos.
- **Fala Sugerida**:
  > *"Avaliamos 4 famílias de algoritmos em 3 configurações distintas cada, totalizando 12 experimentos sob os mesmos 10 folds. O Random Forest com 200 árvores e profundidade máxima 20 foi o campeão geral, com 86,65% de acurácia, F1-score de 0,683 e ROC-AUC de 0,92. O Naive Bayes teve comportamento oposto: identificou muitos positivos (alta revocação), mas com muitos falsos positivos."*

---

### Slide 6: Testes Estatísticos de Hipótese (Wilcoxon)
- **Recurso**: `results/testes_wilcoxon.csv`
- **Pontos-chave**:
  - Teste pareado não-paramétrico de Wilcoxon com nível de significância de 5% (α = 0.05).
  - RF superou KNN e Naive Bayes com **p-valor = 0.00195** (em 10 de 10 folds).
  - Em acurácia, RF superou Árvore de Decisão com **p-valor = 0.00195**.
- **Fala Sugerida**:
  > *"Para não ficarmos apenas na média das métricas, rodamos o teste não-paramétrico pareado de postos sinalizados de Wilcoxon nos 10 folds. O Random Forest venceu KNN e Naive Bayes em 10 de 10 folds com p-valor de 0,00195, confirmando que a superioridade não foi por acaso, mas estatisticamente significativa."*

---

### Slide 7: Desbalanceamento de Classes
- **Recurso**: `figures/fig6_desbalanceamento.png`
- **Pontos-chave**:
  - A classe minoritária (`>50K`) tem apenas 60% de revocação no baseline.
  - Com **SMOTE**, a revocação sobe para **82,86%** e o F1 da classe atinge **0.6981**.
  - Com **`class_weight='balanced'`**, a revocação atinge **86,33%** com custo computacional mínimo.
- **Fala Sugerida**:
  > *"Em cenários de negócio onde identificar quem ganha mais de 50 mil é crítico (como campanhas de produtos premium), o baseline deixa passar 40% desse público. Ao aplicarmos SMOTE ou ponderação de classes balanced, elevamos a revocação para mais de 86%, tornando o modelo muito mais sensível à classe de interesse."*

---

### Slide 8: Conclusão & Demonstração ao Vivo
- **Ação**: Executar no terminal `python demo_apresentacao.py`.
- **Fala Sugerida**:
  > *"Em conclusão, validamos um pipeline completo, reproduzível e livre de vazamento de dados. Vou agora executar uma demonstração ao vivo de 3 segundos que carrega o modelo campeão e classifica 3 perfis fictícios da vida real com suas respectivas probabilidades."*

---

## ❓ Possíveis Perguntas da Banca e Como Responder

### 1. "Por que você usou o teste de Wilcoxon em vez do teste t de Student pareado?"
> **Resposta**: *"O teste t de Student pressupõe que as diferenças entre os folds sigam uma distribuição normal, o que raramente se sustenta em métricas de aprendizado de máquina com apenas 10 folds. O teste de Wilcoxon é não-paramétrico, baseado em postos (ranks), sendo a recomendação metodológica padrão da literatura (como proposto por Janez Demšar, 2006) para comparação de classificadores."*

### 2. "Por que você descartou o atributo `fnlwgt`?"
> **Resposta**: *"O `fnlwgt` (final weight) é um peso amostral atribuído pelo Census Bureau para projetar a população representada pelo indivíduo. Ele reflete a distribuição geográfica/amostral do censo de 1994, e não uma característica intrínseca socioeconômica da pessoa. Em nosso estudo de ablação, a remoção do `fnlwgt` simplificou o modelo e melhorou a métrica de teste."*

### 3. "Por que você priorizou F1-Score e ROC-AUC em vez de olhar só a Acurácia?"
> **Resposta**: *"Como a base é desbalanceada (76% contra 24%), um classificador ingênuo que sempre previsse `<=50K` teria 76% de acurácia, mas zero utilidade prática. O F1-Score captura o equilíbrio entre precisão e revocação na classe minoritária, e o ROC-AUC avalia a capacidade discriminativa geral independente de threshold."*

### 4. "Como você garantiu que não houve Data Leakage no pré-processamento?"
> **Resposta**: *"Utilizamos o `Pipeline` do `imbalanced-learn`. Toda transformação que aprende do conjunto de dados — como o `StandardScaler` e o `OneHotEncoder` — só é ajustada (`fit`) no conjunto de treino de cada fold da validação cruzada. No conjunto de validação/teste, aplicamos apenas o `transform`."*
