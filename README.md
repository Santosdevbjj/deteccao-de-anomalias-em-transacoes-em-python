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



