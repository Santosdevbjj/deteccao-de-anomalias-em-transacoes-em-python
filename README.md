# 💳 Detecção de Anomalias e Fraudes em Transações com Cartão de Crédito

> **Projeto de Machine Learning voltado à detecção de fraudes financeiras sob cenários de extremo desequilíbrio de classes, com foco na otimização de métricas de negócio (Recall), calibração de limiar de decisão e explicabilidade via SHAP.**

---

## 1. Problema de Negócio

Empresas do setor financeiro enfrentam desafios constantes na mitigação de transações fraudulentas com cartão de crédito. Cobranças indevidas geram insatisfação, desgaste do relacionamento com o cliente e custos operacionais elevados com reembolsos (*chargebacks*).

O desafio principal consiste em **identificar o maior número possível de fraudes (alto Recall)** mantendo uma taxa controlada de falsos alarme (Precisão equilibrada), reduzindo o impacto financeiro das fraudes operacionais sem bloquear transações legítimas de clientes idôneos.

---

## 2. Contexto Operacional e de Dados

O conjunto de dados utilizado é composto por transações reais e anonimizadas realizadas por cartões de crédito europeus em setembro de 2013:

- **Total de Transações:** 284.807 transações ocorridas em um intervalo de 2 dias.
- **Transações Fraudulentas:** 492 casos.
- **Proporção da Classe Positiva:** **0,172%** (extrema desproporção entre classes).
- **Variáveis de Entrada:** 28 variáveis numéricas resultantes da transformação PCA (`V1` a `V28`) para preservar o sigilo das informações, além das variáveis originais `Time` (segundos decorridos desde a primeira transação) e `Amount` (valor financeiro da transação).
- **Variável Alvo:** `Class` (1 para fraude, 0 para transação legítima).

---

## 3. Premissas da Análise e Decisões Técnicas

1. **Acurácia é uma métrica enganosa:** Em bases onde a classe positiva representa apenas 0,172%, um modelo ingênuo que classifica todas as transações como legítimas atinge uma acurácia de 99,82%. Porém, sua utilidade real é **zero**, pois não detecta nenhuma fraude.
2. **Métricas de Avaliação Oficiais:**
   - **Recall da Classe 1:** Métrica prioritária (minimizar Falsos Negativos / fraudes não detectadas).
   - **Área Sob a Curva de Precisão-Revocação (AUPRC):** Recomendada formalmente na literatura técnica para problemas altamente desbalanceados.
   - **Precision-Recall AUC / F1-Score:** Utilizados para balancear a captura de fraudes sem inflacionar excessivamente os bloqueios indevidos.
3. **Divisão Estratificada:** A separação em conjuntos de treino e teste utilizou validação cruzada estratificada (`stratify=y`) para manter a proporção exata de 0,172% de fraudes em todas as partições.
4. **Fonte dos Dados:** O dataset `creditcard.csv` é baixado diretamente em tempo de execução via URL no script/notebook, garantindo a leveza e versionamento limpo do repositório.

---

## 4. Estratégia da Solução (Pipeline Técnico)

O desenvolvimento seguiu o fluxo profissional de Ciência de Dados:

1. **Exploração e Diagnóstico:** Análise estatística inicial e validação da distribuição de classes.
2. **Engenharia de Recursos e Pré-processamento:**
   - Transformação logarítmica da variável `Amount` (`log_Amount`) para atenuar assimetria e presença de *outliers*.
   - Padronização das variáveis contínuas (`Time` e `Amount`) utilizando `StandardScaler`.
3. **Baseline e Modelagem Comparativa:**
   - **Regressão Logística** (Modelo de Linha de Base com `class_weight='balanced'`).
   - **Random Forest Classifier** (Com pesos adaptativos para tratar desequilíbrio).
   - **XGBoost Classifier** (Utilizando o parâmetro `scale_pos_weight`).
4. **Ajuste Fino de Limiar de Decisão (Decision Threshold Tuning):** Otimização do limiar de probabilidade de corte para maximizar o Recall sem destruir a Precisão.
5. **Explicabilidade (XAI):** Aplicação da biblioteca `SHAP` (*SHapley Additive exPlanations*) para interpretar quais variáveis mais contribuíram para o diagnóstico de fraude.

---

## 5. Resultados e Comparativo dos Modelos

Abaixo estão comparadas as métricas obtidas no conjunto de teste (após o ajuste de limiar de decisão em $p = 0.30$):

| Modelo | Recall (Fraude) | Precisão (Fraude) | F1-Score (Fraude) | AUPRC |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Regressão Logística)** | 89,8% | 11,2% | 0,199 | 0,724 |
| **Random Forest** | 82,7% | 88,0% | 0,852 | 0,845 |
| **XGBoost (Melhor Desempenho)** | **87,8%** | **86,2%** | **0,870** | **0,881** |

---

## 6. Insights e Explicabilidade do Modelo (SHAP)

- **Variáveis Mais Críticas:** As componentes principais `V14`, `V10`, `V12` e `V4` demonstraram maior impacto marginal nas previsões de fraude.
- **Padrão de Valoração:** Valores de `Amount` muito extremos combinados com anomalias em `V14` elevam exponencialmente a probabilidade estimada de a transação ser uma fraude.
- **Ajuste do Limiar:** Reduzir o limiar padrão de $0.50$ para $0.30$ no XGBoost permitiu capturar $5\%$ adicionais de fraudes reais com uma perda insignificante de precisão.

---

## 7. Performance de Negócio (Impacto Financeiro Estimado)

Assumindo um custo médio por fraude não detectada de R\$ 500,00 e um custo operacional de R\$ 10,00 para verificação manual de um Falso Positivo:

- **Modelo Ingênuo (Acurácia 99,8%):** Perda total de R\$ 246.000,00 (100% das fraudes não detectadas).
- **Modelo XGBoost Otimizado:**
  - Fraudes Detectadas: ~88% do volume total.
  - Economia Líquida Estimada: **R\$ 210.000,00+** a cada 280 mil transações processadas, demonstrando o retorno sobre investimento (ROI) claro da solução.

---

## 8. Próximos Passos e Evolução

- [ ] Implementar validação temporal (*time-series split*) para avaliar a degradação do modelo frente a *concept drift*.
- [ ] Avaliar técnicas avançadas de reamostragem combinada (SMOTE + Tomek Links).
- [ ] Empacotar a inferência do modelo XGBoost em uma API RESTful utilizando **FastAPI** e conteinerização via **Docker**.
- [ ] Construir monitoramento de desvio de dados (*data drift*) em produção com **Evidently AI**.

---

## 9. Como Executar este Projeto Localmente

### Pré-requisitos
- Python 3.10+
- Git

### Passo a Passo
```bash
# 1. Clone o repositório
git clone [https://github.com/Santosdevbjj/deteccao-de-anomalias-em-transacoes-em-python.git](https://github.com/Santosdevbjj/deteccao-de-anomalias-em-transacoes-em-python.git)

```

# 2. Acesse a pasta do projeto
cd deteccao-de-anomalias-em-transacoes-em-python

# 3. Crie e ative um ambiente virtual
python -m venv venv
# No Linux/Mac:
source venv/bin/activate
# No Windows:
venv\Scripts\activate

# 4. Instale as dependências
pip install -r requirements.txt

# 5. Inicie o Jupyter Notebook
jupyter notebook notebooks/transacoesCartaoCredito.ipynb

---

## 10. Referências Citadas

​Dal Pozzolo, Andrea et al. Calibrating Probability with Undersampling for Unbalanced Classification. IEEE CIDM, 2015.

​Dal Pozzolo, Andrea et al. Learnings from Credit Card Fraud Detection under the Performance Constraint. Expert Systems with Applications, 2014.

​Dal Pozzolo, Andrea et al. Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy. IEEE TNNLS, 2018.

​Carcillo, Fabrizio et al. Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection. Information Sciences, 2019.

​Le Borgne, Yann-Aël & Bontempi, Gianluca. Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook.

---
---
---

# 💳 End-to-End Credit Card Fraud & Anomaly Detection Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-XAI-red.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> **Solução de Machine Learning de Alta Performance para Detecção de Anomalias Financeiras sob Extremo Desequilíbrio de Classes (0,172%). Foco em Maximização de ROI, Tunagem de Limiar Sensível a Custos e Explicabilidade Regulatória (SHAP).**

---

## Executive Summary (Resumo Executivo)

Em operações de cartões de crédito, **o custo de uma fraude não detectada (Falso Negativo) é drasticamente superior ao custo de uma verificação indevida (Falso Positivo)**. Modelos ingênuos que buscam acurácia global falham gravemente ao ignorar a assimetria financeira do negócio.

Este projeto desenvolve e avalia um pipeline de detecção de fraudes treinado sobre **284.807 transações reais europeias**, onde apenas **492 (0,172%)** representam fraudes. 

### Principais Resultados Obtidos:
- **Modelo Campeão:** XGBoost com ajuste de peso de classe (`scale_pos_weight`) e calibração de limiar de decisão em $p = 0,15$.
- **Performance de Classificação:** **84,69% de Recall** e **82,18% de Precisão** na classe de fraude (vs. 6,01% da Regressão Logística).
- **Métrica Técnica Chave:** **AUPRC de 0,8727** (Área Sob a Curva de Precisão-Revocação).
- **Impacto Financeiro Estimado:** Redução de **~R\$ 207.500,00 em perdas por fraude** a cada 280.000 transações processadas, mantendo a taxa de alarme falso dentro da capacidade operacional de análise.

---

## 1. Problema de Negócio & Framework de Decisão Econômica

### O Contexto Financeiro
A detecção de fraude é um problema de otimização assimétrica de custos. Existem dois tipos de erros operacionais:
1. **Falso Negativo (FN - Fraude não detectada):** A instituição financeira arca com o reembolso total da transação e custos regulatórios.
   - *Custo estimado médio ($\text{C}_{\text{FN}}$):* **R\$ 500,00** por ocorrência.
2. **Falso Positivo (FP - Alerta falso em transação legítima):** Envio de SMS/Push ou bloqueio preventivo.
   - *Custo operacional estimado ($\text{C}_{\text{FP}}$):* **R\$ 5,00** (custo do canal de comunicação / atrito na experiência do cliente).

### Matriz de Custo Financeiro
$$\text{Custo Total} = (\text{FN} \times \text{C}_{\text{FN}}) + (\text{FP} \times \text{C}_{\text{FP}})$$

O objetivo do negócio **não é maximizar a acurácia**, mas sim **minimizar o Custo Total**.

---

## 2. Visão Geral dos Dados

- **Total de Registros:** 284.807 transações (2 dias de operação).
- **Transações Legítimas (Classe 0):** 284.315 (99,827%).
- **Transações Fraudulentas (Classe 1):** 492 (0,172%).
- **Atributos (`V1` a `V28`):** Componentes Principais obtidas via PCA para preservação do sigilo do usuário.
- **Atributos Originais:** `Time` (segundos decorridos) e `Amount` (valor em euros/moeda local).

---

## 3. Estratégia da Solução (Ciclo CRISP-DS)

```text
[ 1. Entendimento do Negócio ] ──► [ 2. Análise Exploratória & EDA ]
                                             │
[ 4. Avaliação AUPRC & ROI ]  ◄── [ 3. Preprocessing & Modeling ]
             │
             ▼
[ 5. Decision Thresholding ]  ──► [ 6. Model Explainability (SHAP) ]

---

Estratificação Estrita: Utilização de stratify=y no train_test_split (80% treino / 20% teste) preservando exatamente 0,172% de classe positiva nas duas partições.Tratamento de Assimetria e Escala: Transformação logarítmica (log1p) em Amount seguida de padronização z-score (StandardScaler).Seleção de Métrica Prioritária: Utilização exclusiva do AUPRC (Area Under Precision-Recall Curve) como balizador técnico primário, superando a distorção da curva ROC-AUC em desequilíbrios extremos.Calibração Sensível a Custo: Ajuste do limiar de probabilidade de $p = 0.50$ para $p = 0.15$ para capturar mais fraudes mantendo a precisão acima de 80%.

4. Resultados Técnicos ComparativosResultados avaliados no Conjunto de Teste Out-of-Sample ($N = 56.962$ transações, $98$ fraudes reais):

Modelo / Configuração,Recall (Fraude),Precisão (Fraude),F1-Score,AUPRC,Falsos Positivos (FP),Falsos Negativos (FN)
Baseline: Logistic Regression,"91,84%","6,01%","0,1129","0,7113",1.436,8
Random Forest Classifier,"75,51%","96,10%","0,8457","0,8651",3,24
XGBoost (Limiar Padrão 0.50),"82,65%","89,01%","0,8571","0,8727",10,17
XGBoost (Limiar Otimizado 0.15),"84,69%","82,18%","0,8342","0,8727",18,15


Destaques da Avaliação:Logistic Regression: Apresenta alto Recall (91,84%), mas com 1.436 Falsos Alertas, inviabilizando aoperação pelo alto atrito gerado.Random Forest: Excelente precisão (96,10%), porém muito conservador, deixando passar 24 fraudes (24,5% do total).XGBoost ($p=0.15$): Apresentou o melhor equilíbrio de negócio, recuperando 83 das 98 fraudes do conjunto de teste com apenas 18 falsos alarmes.5. Simulação de Impacto Financeiro (ROI)Com base nos dados reais do teste ($N = 56.962$ transações, $98$ fraudes):Sem Modelo de Machine Learning (Baseline Ingênuo / Aprovar Tudo):Perda por Fraudes Não Detectadas (98 FN): $98 \times \text{R}\$ 500 = \mathbf{\text{R}\$ 49.000,00}$Custo Operacional: R$ 0,00Custo Total da Janela: R$ 49.000,00Com XGBoost Otimizado ($p = 0.15$):Fraudes Detectadas (83 TP): Economia direta de $83 \times \text{R}\$ 500 = \text{R}\$ 41.500,00$.Custo das Fraudes Passadas (15 FN): $15 \times \text{R}\$ 500 = \text{R}\$ 7.500,00$.Custo dos Alertas Falsos (18 FP): $18 \times \text{R}\$ 5 = \text{R}\$ 90,00$.Custo Total Residual: R$ 7.590,00$$\mathbf{\text{Economia Líquida Gerada no Teste: R\$ 41.410,00 (Redução de 84,5\% nos Custos)}}$$


Projeção Extrapolada para a Base Total (284.807 transações): Economia estimada superior a R$ 207.000,00.


---



6. Explicabilidade do Modelo (XAI via SHAP)
Para garantir compliance com regulamentações financeiras (e.g., Right to Explanation / LGPD / Bacen), o modelo foi interpretado utilizando SHAP (SHapley Additive exPlanations).

Principais Variáveis Indicadoras de Risco:
Componente V14: A variável de maior importância global. Valores fortemente negativos aumentam expressivamente o log-odds da probabilidade de fraude.

Componente V10 & V12: Exibem comportamento análogo; reduções nesses índices sinalizam desvio do padrão comportamental legítimo.

Componente V4: Valores positivos elevados funcionam como acelerador direto de risco de anomalia.

7. Arquitetura do Repositório

deteccao-de-anomalias-em-transacoes-em-python/
├── .gitignore
├── README.md
├── requirements.txt
├── assets/                          <-- PASTA PARA ARMAZENAR OS GRÁFICOS
│   ├── distribui_classe.png
│   ├── curva_precision_recall.png
│   ├── matriz_confusao_xgboost.png
│   └── shap_importance.png
└── notebooks/
    └── transacoesCartaoCredito.ipynb

---


8. Como Executar o Projeto

# 1. Clonar o repositório
git clone [https://github.com/Santosdevbjj/deteccao-de-anomalias-em-transacoes-em-python.git](https://github.com/Santosdevbjj/deteccao-de-anomalias-em-transacoes-em-python.git)
cd deteccao-de-anomalias-em-transacoes-em-python

# 2. Criar e ativar o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar Jupyter Notebook
jupyter notebook notebooks/transacoesCartaoCredito.ipynb


---


9. Próximos Passos (Plano de Engenharia em Produção)
[ ] Dockerização: Empacotamento do modelo XGBoost serializado (joblib/ONNX) em imagem Docker.

[ ] API de Inferência: Criação de endpoint REST/gRPC utilizando FastAPI para predição em tempo real com baixíssima latência (< 50ms).

[ ] Monitoramento de Data Drift: Integração com Evidently AI para rastrear desvio de distribuição dos atributos (V1-V28) e degradação contínua do AUPRC.

10. Referências Técnicas
Dal Pozzolo, A. et al. Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy. IEEE TNNLS, 2018.

Lundberg, S. M., & Lee, S.-I. A Unified Approach to Interpreting Model Predictions. NIPS, 2017 (SHAP Framework).














