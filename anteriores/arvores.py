# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 15:16:53 2025

@author: ulf
"""

# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Preparacao final dos dados para geracao dos modelo.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: Ulf Bergmann

"""

import pandas as pd
import numpy as np

import time

import sys

sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

import funcoes as f
from funcoes import Log

import funcoes_ulf as uf
import bib_graficos as ug

import statsmodels.api as sm # estimação de modelos
import statsmodels.formula.api as smf # estimação do modelo logístico binário
from statstests.process import stepwise # procedimento Stepwise
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
                                                        
import category_encoders as ce 
                                                        
from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score , precision_score , f1_score,  roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
                                                        
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV


#%% CARGA DADOS


log = Log()
log.carregar_log("log_BaseModelagemFinal")
df = f.leitura_arquivo_parquet("BaseModelagemFinal")

print(log.logar_acao_realizada("Carga Dados", "Carregamento dos dados para modelagem", df.shape[0]))

var_dep = "_Gerada_RESFINAL"

# df_encoded_train = df_encoded.sample(50000)


df_working = df.groupby(var_dep, group_keys=False).apply(lambda x: x.sample(250000 , random_state=1))
# df_working = df
df_restante = df[~df.apply(tuple, axis=1).isin(df_working.apply(tuple, axis=1))]


file_name_rf = 'rf3'
file_name_xgb = 'xgb1'

print(log.logar_acao_realizada("Carga Dados", "Trabalhando com uma amostra dos registros", df_working.shape[0]))

X = df_working.drop(columns=[var_dep])
y = df_working[var_dep]

# Vamos escolher 70% das observações para treino e 30% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=100)


#%%######################### Random Forest ####################################

#%% Criacao do Modelo
## Alguns hiperparâmetros:

# n_estimators: qtde de árvores estimadas
# max_depth: profundidade máxima das árvores
# max_features: qtde de variáveis preditoras consideradas nos splits
# min_samples_split: qtde mínima de observações exigidas para dividir o nó
# min_samples_leaf: qtde mínima de observações exigidas para ser nó folha


# Vamos especificar a lista de hiperparâmetros desejados e seus valores
# param_grid_rf = {
#     'n_estimators': [5 , 7 , 10 , 20 , 50 ],
#     'max_features': [2, 3 , 10 ],
#     'min_samples_split': [20, 50]
# }



# param_grid_rf = {
#     'n_estimators': [750 , 1000 ], # Número de árvores no ensemble. Mais árvores podem melhorar a acurácia, mas podem aumentar o custo computacional e levar ao overfitting.
#                                           # Experimente valores entre 100 e 1000
#     'max_features': [10 , 20],  # Profundidade máxima das árvores. Valores maiores permitem que o modelo capture padrões complexos, mas aumentam o risco de overfitting.
#                                          # Teste valores entre 3 e 10.
#     'min_samples_split': [30 , 40], # Número mínimo de amostras necessário para dividir um nó. Valores maiores simplificam as árvores e ajudam a evitar overfitting.
#                                        # Sugestão: Tente valores entre 2 e 10.
# }

ccp_alpha_range = np.linspace(0.05, 0.05, 3) 

param_grid_rf = {
    'ccp_alpha': ccp_alpha_range , # controlar a complexidade das árvores no conjunto florestal e prevenir overfitting. O parâmetro ccp_alpha (minimal cost-complexity pruning) controla a poda das árvores de decisão no modelo.
}

# Identificar o algoritmo em uso

rf_grid = RandomForestClassifier(random_state=100 , n_estimators= 1000, min_samples_split = 30, max_features = 10)  

# rf_grid = RandomForestClassifier(random_state=100)  # usar um modelo nao treinado

    
# Treinar os modelos para o grid search
# rf_grid_model = GridSearchCV(estimator = rf_grid, 
#                              param_grid = param_grid_rf,
#                              scoring='accuracy',
#                              cv=5, verbose=2)


rf_grid_model = RandomizedSearchCV(estimator = rf_grid, param_distributions = param_grid_rf , verbose=2 )
# HalvingRandomSearchCV

start_time = time.time()

rf_grid_model.fit(X_train, y_train)

end_time = time.time()
print( log.logar_acao_realizada( "Modelagem", "criacao do modelo Randon Forest",  end_time - start_time  ))

# Verificando os melhores parâmetros obtidos
print("Melhores parametros")
print(rf_grid_model.best_params_)

# Gerando o modelo com os melhores hiperparâmetros
rf_best = rf_grid_model.best_estimator_

#%% Predicoes

# Predict na base de treino
rf_pred_train_class = rf_best.predict(X_train)
rf_pred_train_prob = rf_best.predict_proba(X_train)

# Predict na base de testes
rf_pred_test_class = rf_best.predict(X_test)
rf_pred_test_prob = rf_best.predict_proba(X_test)

# Predict na base que nao foi usada na montagem do modelo e nos testes
X_comp = df_restante.drop(columns=[var_dep])
y_comp = df_restante[var_dep]
df_restante['rf_pred_class'] = rf_best.predict(X_comp)

#%% Resultados

# ug.plot_matriz_confusao(rf_pred_train_class , y_train , cutoff=None)
# ug.plot_matriz_confusao(rf_pred_test_class , y_test , cutoff=None)
# ug.plot_matriz_confusao(df_restante['rf_pred_class'] , y_comp , cutoff=None)


resumo = pd.DataFrame()
a_nome_coluna = f'Randon Forest {file_name_rf} - Sample  {X.shape[0]}'
resumo.at['Parametros Testados Grid' , a_nome_coluna] = str(param_grid_rf) 
resumo.at['Parametros Escolhidos Grid' , a_nome_coluna] = str(rf_grid_model.best_params_)
resumo.at['Parametros Modelo' , a_nome_coluna] = str(rf_best.get_params())
resumo.at['Tempo Treinamento' , a_nome_coluna] = str(end_time - start_time)


indicadores = ug.calcula_indicadores_predicao_classificacao(f'{file_name_rf} Sample {X.shape[0]} - Train ' , rf_best , y_train , rf_pred_train_class , cutoff = None)
aux_ind = ug.calcula_indicadores_predicao_classificacao(f'{file_name_rf} Sample {X.shape[0]} - Test' , rf_best , y_test , rf_pred_test_class , cutoff = None)
indicadores = pd.concat([indicadores, aux_ind], axis=0)
aux_ind = ug.calcula_indicadores_predicao_classificacao(f'{file_name_rf} Sample {X.shape[0]} - Completa' , rf_best , y_comp , df_restante['rf_pred_class'] , cutoff = None)
indicadores = pd.concat([indicadores, aux_ind], axis=0)

an_obj_dict = {
    'modelo' : rf_best,
    'resumo' : resumo,
    'indicadores' : indicadores,
    }

file_name = f'{file_name_rf}_{X.shape[0]}'
f.save_objects(an_obj_dict , file_name)

# Importância das variáveis preditoras
rf_features = pd.DataFrame({'features':X.columns.tolist(),
                            'importance':rf_best.feature_importances_})

print(rf_features)

# ug.plot_curvas_roc_1(rf_best, X_train, y_train, X_test, y_test)
# ug.plot_curvas_roc(df_restante , var_dep , ['rf_pred_class'])



#%%######################### XGBoost ##########################################
###############################################################################
#%% Criacao do Modelo

# n_estimators: qtde de árvores no modelo
# max_depth: profundidade máxima das árvores
# colsample_bytree: percentual de variáveis X subamostradas para cada árvore
# learning_rate: taxa de aprendizagem


# Especificar a lista de hiperparâmetros
param_grid_xgb = {
    'n_estimators': [2000 , 2500], # Número de árvores no ensemble. Mais árvores podem melhorar a acurácia, mas podem aumentar o custo computacional e levar ao overfitting.
                                          # Experimente valores entre 100 e 1000
    'max_depth': [7 , 8], # Profundidade máxima das árvores. Valores maiores permitem que o modelo capture padrões complexos, mas aumentam o risco de overfitting.
                      # Teste valores entre 3 e 10.
    'learning_rate': [0.05 , 0.075 , 0.1], # Taxa de aprendizado que controla o impacto de cada árvore. Valores menores (ex.: 0.01) exigem mais árvores, mas podem melhorar a generalização.
                                           #  Comece com valores como 0.1 ou 0.01.
}

    # 'colsample_bytree': [0.3], # Proporção de features usadas para cada árvore. Valores menores ajudam na regularização.
    #                            # Teste valores entre 0.5 e 1.
    # 'min_child_weight' : [3 , 5], # Número mínimo de amostras em um nó folha. Valores altos fazem o modelo focar em divisões significativas, reduzindo o overfitting.
    #                          # Comece com valores entre 1 e 10
    # 'reg_alpha' : [0 , 0.1], # Regularização L1 (adiciona sparsidade ao modelo). Útil para dados com muitas variáveis irrelevantes.
    #                   # Sugerido: Teste valores como 0, 0.1 ou 1.                             
    # 'gamma' : [] , # Define o ganho mínimo necessário para dividir um nó. Valores altos tornam o modelo mais conservador.
    #                # Teste entre 0 e 5.
    # 'reg_lambda': [] , # Regularização L2 (controla o tamanho dos coeficientes). Reduz o risco de overfitting.
    #                    # Sugerido: Teste entre 1 e 10.

# Identificar o algoritmo em uso
xgb = XGBClassifier(random_state=100)               # usar um modelo nao treinado
# xgb = f.obter_modelo_salvo('xgb_500000')        # usar modelo ja treinado com outro grupo de parametros  

# Treinar os modelos para o grid search
# xgb_grid_model = GridSearchCV(estimator = xgb, 
#                               param_grid = param_grid_xgb,
#                               scoring='accuracy', 
#                               cv=5, verbose=2) 

xgb_grid_model = RandomizedSearchCV(estimator = xgb, param_distributions = param_grid_xgb , verbose=2 )

start_time = time.time()

xgb_grid_model.fit(X_train, y_train)

end_time = time.time()
print( log.logar_acao_realizada( "Modelagem", "criacao do modelo XGBoost",  end_time - start_time  ))


print("Melhores parametros")
print(xgb_grid_model.best_params_)

# Gerando o modelo com os melhores hiperparâmetros
xgb_best = xgb_grid_model.best_estimator_



#%% Predicoes

# Predict na base de treinamento
xgb_pred_train_class = xgb_best.predict(X_train)
xgb_pred_train_prob = xgb_best.predict_proba(X_train)

# Predict na base de testes
xgb_pred_test_class = xgb_best.predict(X_test)
xgb_pred_test_prob = xgb_best.predict_proba(X_test)

# Predict na base completa

X_comp = df_restante.drop(columns=[var_dep , 'rf_pred_class']) # tirar pois o modelo nao foi treinado com a variavel
y_comp = df_restante[var_dep]

df_restante['xgb_pred_class'] = xgb_best.predict(X_comp)
# xgb_pred_comp_prob = xgb_best.predict_proba(X_comp)

#%% Resultados

# ug.plot_matriz_confusao(xgb_pred_train_class , y_train , cutoff=None)
# ug.plot_matriz_confusao(xgb_pred_test_class , y_test , cutoff=None)
# ug.plot_matriz_confusao(df_restante['xgb_pred_class'] , y_comp , cutoff=None)

resumo = pd.DataFrame()
a_nome_coluna = f'XGBoost {file_name_xgb} - Sample  {X.shape[0]}'
resumo.at['Parametros Testados Grid' , a_nome_coluna] = str(param_grid_xgb) 
resumo.at['Parametros Escolhidos Grid' , a_nome_coluna] = str(xgb_grid_model.best_params_)
resumo.at['Parametros Modelo' , a_nome_coluna] = str(xgb_best.get_params())
resumo.at['Tempo Treinamento' , a_nome_coluna] = str(end_time - start_time)


indicadores = ug.calcula_indicadores_predicao_classificacao(f'XGB {file_name_xgb} Sample {X.shape[0]} - Train ' , xgb_best , y_train , xgb_pred_train_class , cutoff = None)
aux_ind = ug.calcula_indicadores_predicao_classificacao(f'XGB {file_name_xgb} Sample {X.shape[0]} - Test' , xgb_best , y_test , xgb_pred_test_class , cutoff = None)
indicadores = pd.concat([indicadores, aux_ind], axis=0)
aux_ind = ug.calcula_indicadores_predicao_classificacao(f'XGB {file_name_xgb} Sample {X.shape[0]} - Completa' , xgb_best , y_comp , df_restante['xgb_pred_class'] , cutoff = None)
indicadores = pd.concat([indicadores, aux_ind], axis=0)

an_obj_dict = {
    'modelo' : xgb_best,
    'resumo' : resumo,
    'indicadores' : indicadores,
    }


file_name = f'{file_name_xgb}_{X.shape[0]}'
f.save_objects(an_obj_dict , file_name)

# Importância das variáveis preditoras
xgb_features = pd.DataFrame({'features':X.columns.tolist(),
                            'importance':xgb_best.feature_importances_})

# ug.plot_curvas_roc_1(xgb_best, X_train, y_train, X_test, y_test)
# ug.plot_curvas_roc(df_restante , var_dep , ['xgb_pred_class'])

#%% Analise de overfit

# escolher o melhor modelo e analisar o overfitting:
    # RF1: 500 mil
    # {'n_estimators': 750, 'min_samples_split': 40, 'max_features': 10}
    
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer, roc_auc_score

# df_working = df.groupby(var_dep, group_keys=False).apply(lambda x: x.sample(10000 , random_state=1))
df_working = df
X = df_working.drop(columns=[var_dep])
y = df_working[var_dep]

score = 'accuracy'


# # Criando um scorer personalizado baseado em roc_auc
# roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
model = RandomForestClassifier(random_state=100 , n_estimators= 750, min_samples_split = 40, max_features = 10 )  

# model = XGBClassifier(n_estimators = 500, max_depth = 10, learning_rate = 0.1 , colsample_bytree = 0.3)

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring=score, n_jobs=-1
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Treinamento')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Teste')
plt.xlabel('Tamanho do Conjunto de Treinamento')
plt.ylabel(score)
plt.legend()
plt.title('Curva de Aprendizado ')
plt.show()

