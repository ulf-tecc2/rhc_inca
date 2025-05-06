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


def testa_amostra(label , rf_grid , X_train, X_test, y_train, y_test):
    ccp_alpha_range = np.linspace(0.2 , 0.3) 
    
    param_grid_rf = {'n_estimators': [50, 200 , 300]}
    
    # Identificar o algoritmo em uso
    #n_estimators= 1000,
     
    
    rf_grid_model = RandomizedSearchCV(estimator = rf_grid, param_distributions = param_grid_rf , return_train_score=True , scoring=['accuracy','recall' , 'precision' ,'roc_auc' ] , refit='accuracy', verbose=2 )
    
    start_time = time.time()
    
    rf_grid_model.fit(X_train, y_train)
    
    end_time = time.time()
    
    rf_best = rf_grid_model.best_estimator_

    # Predict na base de treino
    rf_pred_train_class = rf_best.predict(X_train)

    # Predict na base de testes
    rf_pred_test_class = rf_best.predict(X_test)

    indicadores = ug.calcula_indicadores_predicao_classificacao(label + ' - Train' , rf_best , y_train , rf_pred_train_class , cutoff = None)
    aux_ind = ug.calcula_indicadores_predicao_classificacao(label + ' - Test'  , rf_best , y_test , rf_pred_test_class , cutoff = None)
    indicadores = pd.concat([indicadores, aux_ind], axis=0)
    
    print('--------------------------------------------------------')
    print(f'Interacao realizada: {label} - {end_time - start_time}s ')
    print(f'Parametros Escolhidos Grid: {rf_grid_model.best_params_}')
    print(indicadores)
    
    return indicadores , rf_grid_model
    
    
#%% Carga Dados
log = Log()
log.carregar_log("log_BaseModelagemFinal")
df = f.leitura_arquivo_parquet("BaseModelagemFinal")

print(log.logar_acao_realizada("Carga Dados", "Carregamento dos dados para modelagem", df.shape[0]))

var_dep = "_Gerada_RESFINAL"

# df_encoded_train = df_encoded.sample(50000)
indicadores = pd.DataFrame()
resumo = pd.DataFrame()
a_nome_coluna = 'RF Alpha 2 '
a_nome_arquivo = a_nome_coluna

#%% Execucao Testes
tamanhos = [200000 ]

for tamanho_amostra in tamanhos:
    df_working = df.groupby(var_dep, group_keys=False).apply(lambda x: x.sample(int(tamanho_amostra/2) , random_state=1))
    
    X = df_working.drop(columns=[var_dep])
    y = df_working[var_dep]
    
    # Vamos escolher 70% das observações para treino e 30% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3, 
                                                        random_state=100)
    
    rf_grid = RandomForestClassifier(random_state=100 , ccp_alpha= 0.01) 
    
    aux_ind , rf_grid_model = testa_amostra(f'Alpha_{tamanho_amostra}' , rf_grid , X_train, X_test, y_train, y_test)

    
    indicadores = pd.concat([indicadores, aux_ind], axis=0)
    aux_nome_coluna = a_nome_coluna + f'{tamanho_amostra}'
    resumo.at['Parametros Escolhidos Grid' , aux_nome_coluna] = str(rf_grid_model.best_params_) 

an_obj_dict = {
    'resumo' : resumo,
    'indicadores' : indicadores,
    }    
f.save_objects(an_obj_dict , "testeGS_1")  
    
#  
    
#%%

# # Verificando os melhores parâmetros obtidos
# print("Melhores parametros")
# print(rf_grid_model.best_params_)

# # Gerando o modelo com os melhores hiperparâmetros
# rf_best = rf_grid_model.best_estimator_

# # Predict na base de treino
# rf_pred_train_class = rf_best.predict(X_train)
# rf_pred_train_prob = rf_best.predict_proba(X_train)

# # Predict na base de testes
# rf_pred_test_class = rf_best.predict(X_test)
# rf_pred_test_prob = rf_best.predict_proba(X_test)


# # Resultados

# # ug.plot_matriz_confusao(rf_pred_train_class , y_train , cutoff=None)
# # ug.plot_matriz_confusao(rf_pred_test_class , y_test , cutoff=None)
# # ug.plot_matriz_confusao(df_restante['rf_pred_class'] , y_comp , cutoff=None)


# resumo = pd.DataFrame()
# a_nome_coluna = f'Randon Forest {file_name_rf} - Sample  {X.shape[0]}'
# resumo.at['Parametros Testados Grid' , a_nome_coluna] = str(param_grid_rf) 
# resumo.at['Parametros Escolhidos Grid' , a_nome_coluna] = str(rf_grid_model.best_params_)
# resumo.at['Parametros Modelo' , a_nome_coluna] = str(rf_best.get_params())
# resumo.at['Tempo Treinamento' , a_nome_coluna] = str(end_time - start_time)


# indicadores = ug.calcula_indicadores_predicao_classificacao(f'{file_name_rf} Sample {X.shape[0]} - Train ' , rf_best , y_train , rf_pred_train_class , cutoff = None)
# aux_ind = ug.calcula_indicadores_predicao_classificacao(f'{file_name_rf} Sample {X.shape[0]} - Test' , rf_best , y_test , rf_pred_test_class , cutoff = None)
# indicadores = pd.concat([indicadores, aux_ind], axis=0)


# an_obj_dict = {
#     'modelo' : rf_best,
#     'resumo' : resumo,
#     'indicadores' : indicadores,
#     }

# # file_name = f'{file_name_rf}_{X.shape[0]}'
# # f.save_objects(an_obj_dict , file_name)

# # Importância das variáveis preditoras
# rf_features = pd.DataFrame({'features':X.columns.tolist(),
#                             'importance':rf_best.feature_importances_})

# print(rf_features)

# # ug.plot_curvas_roc_1(rf_best, X_train, y_train, X_test, y_test)
# # ug.plot_curvas_roc(df_restante , var_dep , ['rf_pred_class'])

