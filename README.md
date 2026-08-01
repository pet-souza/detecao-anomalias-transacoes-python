# Projeto de Detecção de Fraude em Transações com Cartão de Crédito
# Importação de datasets e bibliotecas necessárias
import pandas as pd

url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'

df = pd.read_csv(url)

df.head(10)

# Verificando a proporção de classes no dataset
df["Class"].value_counts(normalize=True)


# Criando uma nova variável para o logaritmo do valor da transação amount_log
import numpy as np

df['Amount_log'] = np.log1p(df['Amount'])

# Normalizando a variável Amount para melhorar o desempenho do modelo. Padronização dos dados usando StandardScaler do sklearn
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])

# Preparação dos dados para o modelo de machine learning
# Separação dos dados em conjuntos de treino e teste, com 70% para treino e 30% para teste, mantendo a proporção das classes (fraude e não fraude) usando o parâmetro stratify
from sklearn.model_selection import train_test_split

x = df.drop('Class', axis=1)
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

----->------>-------->--------->------->

# Treinamento do modelo de regressão logística para detecção de fraude. Previsão das classes (fraude e não fraude) usando o modelo treinado e avaliação do desempenho do modelo usando métricas como acurácia, precisão, recall e F1-score.
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Avaliação do desempenho do modelo usando métricas como acurácia, precisão, recall e F1-score
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# Curva ROC e AUC para avaliar o desempenho do modelo. Visualização da curva ROC usando matplotlib e cálculo da AUC usando roc_auc_score do sklearn 
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_pred_probs = model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)

plt.plot(fpr, tpr)
plt.title('Curva ROC')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.show()

print('AUC: ', roc_auc_score(y_test, y_pred_probs))

# Verificação da curva Precision-Recall para avaliar o desempenho do modelo em relação à precisão e recall, especialmente em datasets desbalanceados. Visualização da curva Precision-Recall usando matplotlib e cálculo da precisão e recall usando precision_recall_curve do sklearn
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)

plt.plot(recall, precision)
plt.title('Curva Precision-Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# Formas de Balanceamento de Dados para Melhorar o Desempenho do Modelo.
# Under-sampling: Reduzir o número de exemplos da classe majoritária (não fraude) para equilibrar as classes. Isso pode ser feito removendo aleatoriamente exemplos da classe majoritária.
fraudes = df[df['Class'] == 1]
normais = df[df['Class'] == 0].sample(len(fraudes), random_state=42)

df_under = pd.concat([fraudes, normais])

# Over-sampling: Aumentar o número de exemplos da classe minoritária (fraude) para equilibrar as classes. Isso pode ser feito replicando exemplos da classe minoritária ou usando técnicas como SMOTE (Synthetic Minority Over-sampling Technique).
from imblearn.over_sampling import SMOTE

smote = SMOTE()

x_res, y_res = smote.fit_resample(x, y)


# verificação do desempenho do modelo de Random Forest para detecção de fraude. Treinamento do modelo usando o conjunto de treino e avaliação do desempenho do modelo usando métricas como acurácia, precisão, recall e F1-score.
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=42)

rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

print(classification_report(y_test, y_pred_rf))

# Pipeline de pré-processamento e treinamento do modelo de Random Forest para detecção de fraude. O pipeline inclui a padronização dos dados usando StandardScaler e o treinamento do modelo de Random Forest usando o conjunto de treino. Avaliação do desempenho do modelo usando métricas como acurácia, precisão, recall e F1-score.
from sklearn.pipeline import Pipeline

pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression (max_iter=1000))])

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

# Thresholding: Ajustar o limiar de decisão do modelo para melhorar a precisão ou recall, dependendo do objetivo do negócio. Isso pode ser feito ajustando o valor de corte usado para classificar uma transação como fraude ou não fraude.
threshold = 0.5

y_pred_custom = (y_pred_probs >= threshold).astype(int)

print(classification_report(y_test, y_pred_custom))

# XGBoost: Treinamento do modelo de XGBoost para detecção de fraude. Avaliação do desempenho do modelo usando métricas como acurácia, precisão, recall e F1-score.
from xgboost import XGBClassifier

xgb = XGBClassifier(scale_pos_weight=10, use_label_encoder=False, eval_metric='logloss')

xgb.fit(x_train, y_train)

y_pred_xgb = xgb.predict(x_test)

# Avaliação do desempenho do modelo de XGBoost usando métricas como acurácia, precisão, recall e F1-score
print(classification_report(y_test, y_pred_xgb))

# Importancia das variáveis para o modelo de XGBoost. Visualização da importância das variáveis usando matplotlib e cálculo da importância das variáveis usando feature_importances_ do modelo de XGBoost.
import matplotlib.pyplot as plt

importancias = xgb.feature_importances_

plt.bar(range(len(importancias)), importancias)
plt.title('Importância das Variáveis - XGBoost')
plt.show()  

# Ajuste de hiperparâmetros do modelo de XGBoost usando GridSearchCV do sklearn. Avaliação do desempenho do modelo usando métricas como acurácia, precisão, recall e F1-score.
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [3, 5], "n_estimators": [50, 100]}

grid = GridSearchCV(XGBClassifier(eval_metric='logloss'), param_grid, scoring='recall', cv=3)

grid.fit(x_train, y_train)

print("Melhores parâmetros: ", grid.best_params_)

# Explicabilidade do modelo de XGBoost usando SHAP (SHapley Additive exPlanations). Visualização da importância das variáveis usando matplotlib e cálculo da importância das variáveis usando shap_values do modelo de XGBoost.
import shap
explainer = shap.Explainer(xgb)
shap_values = explainer(x_test[:100])

shap.plots.bar(shap_values)
