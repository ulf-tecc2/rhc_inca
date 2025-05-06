# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 15:16:53 2025

@author: ulf
"""

# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Treinamento com reducao do overfittin.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: Ulf Bergmann

"""

import pandas as pd
import numpy as np

import time

from tabulate import tabulate

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

from itertools import product


#%% Funcoes

def calc_ind(index ,label , observado, predicao , parametros):
    sensitividade = recall_score(observado, predicao, pos_label=1)
    especificidade = recall_score(observado, predicao, pos_label=0)
    acuracia = accuracy_score(observado, predicao)
    precisao = precision_score(observado, predicao)
    f1_scor = f1_score(observado, predicao)
    auc = roc_auc_score(observado , predicao)

    column_list = [
                    'Index',
                    'Label' ,
                    'tamanho' ,                             
                    'Sensitividade',
                    'Especificidade',
                    'Acuracia',
                    'Precisao',
                    'F1_Score',
                    'AUC',
                    'bootstrap',
                    'ccp_alpha',
                    'class_weight',
                    'criterion',
                    'max_depth',
                    'max_features',
                    'max_leaf_nodes',
                    'max_samples',
                    'min_impurity_decrease',
                    'min_samples_leaf',
                    'min_samples_split',
                    'n_estimators',
                    'random_state',
                    'warm_start',
                    ]
    
    ind_train = pd.DataFrame(columns = column_list)
    ind_train.at[0,'Index'] = index
    ind_train.at[0,'Label'] = label
    ind_train.at[0,'tamanho'] =  observado.shape[0]
    ind_train.at[0,'Sensitividade'] = sensitividade
    ind_train.at[0,'Especificidade'] = especificidade
    ind_train.at[0,'Acuracia'] = acuracia
    ind_train.at[0,'Precisao'] = precisao
    ind_train.at[0,'F1_Score'] = f1_scor
    ind_train.at[0,'AUC'] = auc

    for key, value in parametros.items():
        ind_train.at[0 , key] = value
        
    return ind_train

def ulf_grid_search(fixed_param ,  X_train, X_test, y_train, y_test , param_distributions , score = ['accuracy']):
    combinacoes = [
        dict(zip(param_distributions.keys(), valores))
        for valores in product(*param_distributions.values())
    ]
    modelos = []
    indicadores = pd.DataFrame()
    print(f'Iniciando as interacoes. Total de {len(combinacoes)}')
    int_count = 1
    for combinacao in combinacoes:
        start_time = time.time()
        combinacao.update(fixed_param)
        modelo = RandomForestClassifier(**combinacao)  
    
        modelo.fit(X_train , y_train)
        modelos.append(modelo)
        y_train_pred = modelo.predict(X_train)
        aux_ind_train = calc_ind(int_count , 'Treino' , y_train , y_train_pred , combinacao )
        indicadores = pd.concat([indicadores, aux_ind_train], axis=0)

        y_test_pred = modelo.predict(X_test)
        aux_ind_test = calc_ind(int_count , 'Teste' , y_test , y_test_pred , combinacao)
        indicadores = pd.concat([indicadores, aux_ind_test], axis=0)
        
        end_time = time.time() 
        print(f'------ Interacao Nr {int_count} Realizada - Tempo: {end_time - start_time}')
        a_maxcolwidths=[30, 30, 30, 30, 30, 30, 30, 30, 30]
        print(tabulate(aux_ind_train, headers='keys', tablefmt='grid', numalign='center' , maxcolwidths = a_maxcolwidths) )
        print(tabulate(aux_ind_test, headers='keys', tablefmt='grid', numalign='center' , maxcolwidths = a_maxcolwidths) )
        print('\n\n')
        
        if (int_count % 5) == 0:
            f.salvar_excel_conclusao(indicadores, f'inter{int_count}')  
            print(f'Arquivo intermediario salvo: inter{int_count}')
            print(f'Ultimos parametros usados: {combinacao}')
            
        int_count = int_count + 1
        
        
    return indicadores , modelos
    


#%% Carga dos dados


log = Log()
log.carregar_log("log_BaseModelagemFinal")
df = f.leitura_arquivo_parquet("BaseModelagemFinalAposPCA")

print(log.logar_acao_realizada("Carga Dados", "Carregamento dos dados para modelagem", df.shape[0]))

var_dep = "_Gerada_RESFINAL"

tamanho_amostra = df.shape[0]
df_working = df

# tamanho_amostra = 200000
# df_working = df.groupby(var_dep, group_keys=False).apply(lambda x: x.sample(int(tamanho_amostra/2) , random_state=1))


print(log.logar_acao_realizada("Carga Dados", "Trabalhando com uma amostra dos registros", df_working.shape[0]))

X = df_working.drop(columns=[var_dep])
y = df_working[var_dep]

# Vamos escolher 70% das observações para treino e 30% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=100)



#%% Modelagem

#T1
# param_distributions = {
#     'n_estimators': [50, 100 , 300],
#     'max_depth' : [None , 10 , 20],
#     'min_samples_split': [2 , 5 , 10],
#     'min_samples_leaf' : [1 , 2 , 4],
#     'max_features' : [3 , 5]
# }
# a_fixed_param = {'random_state': 100}


#T2
# param_distributions = {
#     'n_estimators': [50, 100 , 300],
#     'min_samples_split': [2 , 5 , 10],
# }

# a_fixed_param = {'random_state': 100 , 'max_depth' : 20 , 'max_features' : 3 ,  'min_samples_leaf' : 4}

# #T3
# param_distributions = {
#     'n_estimators': [400 , 700 , 800],
#     'min_samples_split': [6 , 7],
# }

# a_fixed_param = {'random_state': 100 , 'max_depth' : 20 , 'max_features' : 3 ,  'min_samples_leaf' : 4}

# #T4
# param_distributions = {
#     'ccp_alpha' : [0 , 0.01 , 0.02 , 0.03 , 0.04 , 0.05 , 0.1]
# }

# a_fixed_param = {'n_estimators': 400 , 'random_state': 100 , 'max_depth' : 25 ,   'min_samples_leaf' : 4 , 'min_samples_split': 10 , 'max_features' : 10}




# =============================================================================
# param_distributions = {
#     'n_estimators': [340 , 350],
#     'max_features' : [7,8],
# }
# 
# a_fixed_param = {'max_depth' : 25 , 'min_samples_leaf' : 1 , 'min_samples_split': 2 , 'random_state': 100   }

# RF_Adicional1
# param_distributions = {'min_samples_leaf': [1, 3], 'max_leaf_nodes': [10, 100], 'max_depth': [10, 20]}
# a_fixed_param = {'random_state': 100}


# RF_Adicional2
param_distributions = {'n_estimators': [200,750, 1000], 'max_features': [5,10, 20], 'min_samples_split': [10,20,30, 40 , 50]}
a_fixed_param = {'random_state': 100}


indicadores , modelos = ulf_grid_search(fixed_param = a_fixed_param , X_train = X_train , X_test = X_test, y_train = y_train , y_test = y_test, param_distributions = param_distributions )

f.salvar_excel_conclusao(indicadores, 'RF_Adicional2')  
# =============================================================================

#%% Analises do Melhor Modelo
    
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer, roc_auc_score

best_model_param = {'max_depth' : 25 , 'min_samples_leaf' : 1 , 'min_samples_split': 2 , 'n_estimators': 340 , 'max_features' : 7 , 'random_state': 100   }

#%%% learning_curve
score = 'accuracy'

model = RandomForestClassifier(**best_model_param)  

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

print(log.logar_acao_realizada("Analises", "Learning Curve" , ""))



#%%% CROSS VALIDATION
# # from sklearn.model_selection import cross_val_score

# # start_time = time.time()

# # scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy', n_jobs=-1)

# # # Resultados
# # print("Acurácias em cada fold:", scores)
# # print(f"Acurácia média: {scores.mean():.4f}")
# # print(f"Desvio padrão: {scores.std():.4f}")

# # end_time = time.time()
# # print( log.logar_acao_realizada( "Modelagem", "criacao do modelo Randon Forest - Cross Validation",  end_time - start_time  ))


from sklearn.model_selection import cross_validate

# Avaliando múltiplas métricas
model = RandomForestClassifier(**best_model_param)  

resultados = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'roc_auc' ], n_jobs=-1, return_train_score=True , verbose=2)
#recall -> sensitividade
#precision
# Exibindo resultados
print("Acurácia média na validação:", resultados['test_accuracy'].mean())
print("Acurácia std na validação:", resultados['test_accuracy'].std())
print("roc_auc médio na validação:", resultados['roc_auc'].mean())
print("roc_auc std na validação:", resultados['roc_auc'].std())

result_df = pd.DataFrame(resultados)

f.salvar_excel_conclusao(result_df, 'IndicadoresRFCrossValidation')  

print(log.logar_acao_realizada("Analises", "Cross Validation" , ""))
