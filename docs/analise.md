## 📊 1. Análise Técnica e Diagnóstico das Imagens Geradas

## ​Analisando os gráficos gerados

**​Curva Precision-Recall (Imagem 1):**

​Mostra graficamente por que o XGBoost (linha verde) e o Random Forest (linha laranja) dominam a Regressão Logística (linha azul).  

​Evidencia como o XGBoost sustenta alta precisão mesmo quando o Recall é elevado para perto de 85\%.  

​**Matriz de Confusão do XGBoost com Limiar p=0.15 (Imagens 2 e 3):**

**​Verdadeiros Negativos (TN):** 56.846 transações legítimas identificadas corretamente.  


​**Falsos Positivos (FP):** Apenas 18 transações legítimas marcadas como fraude (baixo custo operacional de verificação).  

**​Falsos Negativos (FN):** Apenas 15 fraudes não detectadas.  

**​Verdadeiros Positivos (TP):** 83 fraudes reais capturadas (84,69\% de Recall).  


## ​Gráfico de Importância de Features via SHAP (Imagem 3):

​As variáveis V14, V4, V12, V10 e V11 são disparadamente as mais determinantes para indicar o risco de fraude.  

​Atributos como Time_scaled e Amount_scaled possuem papel secundário se comparados às variáveis latentes extraídas pelo PCA.  



​
