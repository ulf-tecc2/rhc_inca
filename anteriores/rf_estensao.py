# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:40:47 2025

@author: ulf
"""

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


# df_working = df.groupby(var_dep, group_keys=False).apply(lambda x: x.sample(250000 , random_state=1))
df_working = df

file_name_rf = 'rf1 Cross Validation'
# file_name_xgb = 'xgb1'

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

# rf = RandomForestClassifier(random_state=100)
# rf.fit(X_train, y_train)
# path = rf.cost_complexity_pruning_path(X_train, y_train)  # CCP Path na base de treino
# ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # 'n_estimators': [200 , 750 ], # Número de árvores no ensemble. Mais árvores podem melhorar a acurácia, mas podem aumentar o custo computacional e levar ao overfitting.
    #                                       # Experimente valores entre 100 e 1000
    # 'max_features': [5 , 10],  # Profundidade máxima das árvores. Valores maiores permitem que o modelo capture padrões complexos, mas aumentam o risco de overfitting.
    #                                      # Teste valores entre 3 e 10.
    # 'min_samples_split': [40, 50], # Número mínimo de amostras necessário para dividir um nó. Valores maiores simplificam as árvores e ajudam a evitar overfitting.
    #                                    # Sugestão: Tente valores entre 2 e 10.



    # 'min_samples_leaf':[],# Número mínimo de amostras em cada nó folha. Valores maiores criam nós mais robustos e reduzem o risco de overfitting.
    #                   Tente valores entre 1 e 5.
    # 'max_leaf_nodes': [], # Número máximo de nós folha.Limitar os nós folha simplifica a árvore e melhora a generalização.
                            # Sugestão: Experimente valores como 10, 50 ou 100
    # 'max_depth': [], # Profundidade máxima de cada árvore. Profundidades maiores capturam padrões complexos, mas podem causar overfitting.
                        # Teste valores como 10, 20 ou deixe como None (valor padrão) para não limitar. 
    # 'class_weight': [] ,  # Ajusta os pesos das classes, especialmente útil para dados desbalanceados.
                            # Use balanced para compensar automaticamente o desbalanceamento.               
    # 'max_samples': [],   # Permite limitar o número de amostras usadas para treinar cada árvore. Útil para conjuntos de dados muito grandes.                        


rf_model = RandomForestClassifier(random_state=100 , n_estimators= 1000, min_samples_split = 30, max_features = 10)  
# n_estimators': 1000, 'max_features': 10, 'min_samples_split': 30		

#%% TREINAR COM K-Folds

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configuração do K-Fold (5 folds neste exemplo)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

indicadores = pd.DataFrame()
# Loop pelos folds

i = 0
for treino_index, teste_index in skf.split(X):
    X_treino, X_teste = X[treino_index], X[teste_index]
    y_treino, y_teste = y[treino_index], y[teste_index]
    
    # Treinando o modelo nos dados de treinamento
    rf_model.fit(X_treino, y_treino)
    
    # Fazendo previsões nos dados de teste
    predicoes_treino = rf_model.predict(X_treino)
    predicoes_teste = rf_model.predict(X_teste)

    aux_ind = ug.calcula_indicadores_predicao_classificacao(f'{file_name_rf} K_Fold {i} - Treino ' , rf_model , y_treino , predicoes_treino , cutoff = None)
    indicadores = pd.concat([indicadores, aux_ind], axis=0)
    aux_ind =  ug.calcula_indicadores_predicao_classificacao(f'{file_name_rf} K_Fold {i} - Teste ' , rf_model , y_teste , predicoes_teste , cutoff = None)
    indicadores = pd.concat([indicadores, aux_ind], axis=0)
    
    i = i + 1

# armazenando os resultados

an_obj_dict = {
    'indicadores' : indicadores,
    }

file_name = 'k_fold_indicadores'
f.save_objects(an_obj_dict , file_name)


#%% CROSS VALIDATION
# from sklearn.model_selection import cross_val_score

# start_time = time.time()

# scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy', n_jobs=-1)

# # Resultados
# print("Acurácias em cada fold:", scores)
# print(f"Acurácia média: {scores.mean():.4f}")
# print(f"Desvio padrão: {scores.std():.4f}")

# end_time = time.time()
# print( log.logar_acao_realizada( "Modelagem", "criacao do modelo Randon Forest - Cross Validation",  end_time - start_time  ))


from sklearn.model_selection import cross_validate

# Avaliando múltiplas métricas
resultados = cross_validate(rf_model, X, y, cv=5, scoring=['accuracy', 'roc_auc' ], n_jobs=-1, return_train_score=True , verbose=2)
#recall -> sensitividade
#precision
# Exibindo resultados
# print("Acurácia média na validação:", resultados['test_accuracy'].mean())
# print("Acurácia std na validação:", resultados['test_accuracy'].std())
# print("roc_auc médio na validação:", resultados['roc_auc'].mean())
# print("roc_auc std na validação:", resultados['roc_auc'].std())

resumo = pd.DataFrame()
a_nome_coluna = 'Randon Forest - Cross Validation'
resumo.at['Parametros Modelo' , a_nome_coluna] = 'n_estimators= 1000, min_samples_split = 30, max_features = 10'

result_df = pd.DataFrame(resultados)
an_obj_dict = {
    'modelo' : rf_model,
    'resumo' : resumo,
    'resultados' : result_df,
    }

file_name = 'RF_cross_valid'
f.save_objects(an_obj_dict , file_name)

f.salvar_excel_conclusao(result_df, 'IndicadoresRFCrossValidation')  


#%%
start_time = time.time()

rf_model.fit(X_train, y_train)

end_time = time.time()
print( log.logar_acao_realizada( "Modelagem", "criacao do modelo Randon Forest",  end_time - start_time  ))

#%% Predicoes

# Predict na base de treino
rf_pred_train_class = rf_model.predict(X_train)
rf_pred_train_prob = rf_model.predict_proba(X_train)

# Predict na base de testes
rf_pred_test_class = rf_model.predict(X_test)
rf_pred_test_prob = rf_model.predict_proba(X_test)

#%% Resultados

ug.plot_matriz_confusao(rf_pred_train_class , y_train , cutoff=None)
ug.plot_matriz_confusao(rf_pred_test_class , y_test , cutoff=None)

resumo = pd.DataFrame()
a_nome_coluna = f'Randon Forest {file_name_rf} - Sample  {X.shape[0]}'
resumo.at['Parametros' , a_nome_coluna] = 'random_state=100 , n_estimators= 750, min_samples_split = 40, max_features = 10' 
resumo.at['Tempo Treinamento' , a_nome_coluna] = str(end_time - start_time)


indicadores = ug.calcula_indicadores_predicao_classificacao(f'{file_name_rf} Sample {X.shape[0]} - Train ' , rf_model , y_train , rf_pred_train_class , cutoff = None)
aux_ind = ug.calcula_indicadores_predicao_classificacao(f'{file_name_rf} Sample {X.shape[0]} - Test' , rf_model , y_test , rf_pred_test_class , cutoff = None)
indicadores = pd.concat([indicadores, aux_ind], axis=0)

an_obj_dict = {
    'modelo' : rf_model,
    'resumo' : resumo,
    'indicadores' : indicadores,
    }

file_name = f'{file_name_rf}_{X.shape[0]}'
f.save_objects(an_obj_dict , file_name)

# Importância das variáveis preditoras
rf_features = pd.DataFrame({'features':X.columns.tolist(),
                            'importance':rf_model.feature_importances_})

print(rf_features)

# ug.plot_curvas_roc_1(rf_best, X_train, y_train, X_test, y_test)
# ug.plot_curvas_roc(df_restante , var_dep , ['rf_pred_class'])


 