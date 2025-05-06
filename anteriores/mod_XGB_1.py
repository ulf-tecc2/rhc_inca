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
from sklearn.metrics import make_scorer, accuracy_score

from itertools import product


#%% Funcoes

## Define uma função personalizada para calcular a métrica desejada
def metrica_melhor_modelo(model, X_train, y_train, X_test, y_test , min_ratio):
    print('Chamada da funcao')
    # Treinar o modelo
    model.fit(X_train, y_train)
    
    # Calcular a acurácia no conjunto de treinamento
    acc_train = accuracy_score(y_train, model.predict(X_train))
    
    # Calcular a acurácia no conjunto de teste
    acc_test = accuracy_score(y_test, model.predict(X_test))
    
    # Calcular a razão entre a acurácia do teste e do treinamento
    ratio = acc_test / acc_train if acc_train != 0 else 0
    
    # Decisao do melhor modelo: 
        # ratio > minimo => usar a acuracia para buscar a maior
        # ratio < minimo => retornar 0 para nao utilizar esta alternativa
    
    if ratio >= min_ratio:
        result =  acc_train
    else:
        result =  ratio
        
    print(result)
    return result

class CustomScorer:
    def __init__(self, model, X_train, y_train, X_test, y_test , min_ratio = 0.8):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.min_ratio = min_ratio

    def __call__(self, params):
        return metrica_melhor_modelo(self.model.set_params(**params), self.X_train, self.y_train, self.X_test, self.y_test , self.min_ratio)


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
                    'learning_rate',
                    'gamma',	
                    'lambda',	
                    'alpha',
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
        modelo = XGBClassifier(**combinacao)  
    
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
        # a_maxcolwidths=[30, 30, 30, 30, 30, 30, 30, 30, 30 , 30, 30, 30, 30, 30, 30, 30, 30,30, 30, 30, 30, 30]
        # print(tabulate(aux_ind_train, headers='keys', tablefmt='grid', numalign='center' , maxcolwidths = a_maxcolwidths) )
        # print(tabulate(aux_ind_test, headers='keys', tablefmt='grid', numalign='center' , maxcolwidths = a_maxcolwidths) )
        # print('\n\n')
        
        if (int_count % 10) == 0:
            f.salvar_excel_conclusao(indicadores, f'inter{int_count}')  
            print(f'Arquivo intermediario salvo: inter{int_count}')
            print(f'Ultimos parametros usados: {combinacao}')
        int_count = int_count + 1
        
        
    return indicadores , modelos
    


#%% Carga dos dados


log = Log()
log.carregar_log("log_BaseModelagemFinal")
df = f.leitura_arquivo_parquet("BaseModelagemFinal")

print(log.logar_acao_realizada("Carga Dados", "Carregamento dos dados para modelagem", df.shape[0]))

var_dep = "_Gerada_RESFINAL"

# tamanho_amostra = df.shape[0]
# df_working = df

tamanho_amostra = 200000
df_working = df.groupby(var_dep, group_keys=False).apply(lambda x: x.sample(int(tamanho_amostra/2) , random_state=1))


print(log.logar_acao_realizada("Carga Dados", "Trabalhando com uma amostra dos registros", df_working.shape[0]))

X = df_working.drop(columns=[var_dep])
y = df_working[var_dep]

# Vamos escolher 70% das observações para treino e 30% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=100)



#%% Modelagem




# param_grid = {
#     'n_estimators': [100, 200, 500],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0],
#     'gamma': [0, 0.1],
#     'lambda': [1, 5],
#     'alpha': [0, 1]
# }

# #XGB1
# param_distributions = {
#     'n_estimators': [200, 300,500],
#     'learning_rate': [0.05, 0.075, 0.1],
#     'max_depth': [ 5, 7 , 8],
# }
# a_fixed_param = {'random_state': 100   }

#XGB2
# param_distributions = {
#     'n_estimators': [500 , 700 , 900],
#     'learning_rate': [0.1, 0.15 , 0.2],
#     'max_depth': [ 8 , 9 , 10],
# }
# a_fixed_param = {'random_state': 100   }

# file_name = 'XGB3'
# param_distributions = {
#     'n_estimators': [700 , 900 , 1100],
#     'learning_rate': [0.01 , 0.02 , 0.03 , 0.05],
# }
# a_fixed_param = {'random_state': 100 , 'max_depth': 8  }

# file_name = 'XGB4'
# param_distributions = {
#     'n_estimators': [1000 , 1100 , 1200],
#     'learning_rate': [0.04 , 0.05 , 0.06],
#     'gamma': [0, 0.1],
#     'lambda': [1, 5],
#     'max_depth': [ 8 , 10 , 12]
# }

# a_fixed_param = {'random_state': 100   }

# file_name = 'XGB5'
# param_distributions = {
#     'n_estimators': [1000 , 1100 , 1200],
#     'lambda': [0.5 , 1.0 , 1.5],
#     'alpha': [0 , 1 , 5],
# }

# a_fixed_param = {'random_state': 100 , 'max_depth': 12 , 'gamma':  0.01 , 'learning_rate': 0.05 ,   }

#pesquisa com os melhores resultados anteriores com a quantidade total de dados
# file_name = 'XGB7'
# param_distributions = {
#     'max_depth': [ 8 , 9 , 10 , 12],
#     'n_estimators': [1000 , 1100 , 1200],
#     'learning_rate': [0.04 , 0.05 , 0.06 , 0.1],
#     'gamma': [0, 0.1], 
#     'lambda': [1 , 1],

# }

# file_name = 'XGB8'
# param_distributions = {
#     'max_depth': [7 , 8],
#     'n_estimators': [1200 , 1300],
#     'learning_rate': [0.01 , 0.05],
#     'subsample': [0.8 , 0.9],
#     'colsample_bytree': [0.7 , 0.8],
#     'gamma': [0.1 , 0.2]
# }

# file_name = 'XGB9'
# param_distributions = {
#     'subsample': [0.8 , 0.9],
#     'colsample_bytree': [0.7 , 0.8],
# }

# a_fixed_param = {'max_depth' : 9 , 'n_estimators': 1100 , 'learning_rate': 0.05 , 'gamma': 0 , 'lambda': 1 , 'random_state': 100  }

# indicadores , modelos = ulf_grid_search(fixed_param = a_fixed_param , X_train = X_train , X_test = X_test, y_train = y_train , y_test = y_test, param_distributions = param_distributions )

# f.salvar_excel_conclusao(indicadores, file_name)  

#XGB1
param_distributions = {
    'n_estimators': [200, 300],
    # 'learning_rate': [0.05, 0.1],
    # 'max_depth': [ 5, 8],
}

#%%
modelo = XGBClassifier() 
scorer = make_scorer(CustomScorer(modelo, X_train, y_train, X_test, y_test , min_ratio = 0.1))
# search = RandomizedSearchCV(modelo, param_distributions=param_distributions, n_iter=10, scoring=scorer, random_state=100 , verbose=3 , return_train_score = True)
search = GridSearchCV(modelo, param_grid=param_distributions, scoring=scorer , verbose=3 , return_train_score = True)

## Executar a busca
search.fit(X_train, y_train)

#%%
## Exibir os melhores parâmetros e a melhor pontuação
print("Melhores parâmetros:", search.best_params_)
print("Melhor métrica (razão de acurácia):", search.best_score_)


#%% Analises do Melhor Modelo
    
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer, roc_auc_score

#XGB2
# param_distributions = {
#     'n_estimators': [500 , 1000],
#     'learning_rate': [0.05 , 0.1],
#     'max_depth': [ 9 , 10],
# }
# a_fixed_param = {'random_state': 100   }



indicadores = pd.DataFrame()
best_model_param1 = {'max_depth' : 9 , 'n_estimators': 1100 , 'learning_rate': 0.05 , 'gamma': 0 , 'lambda': 1 , 'random_state': 100 , 'subsample': 0.9,  'colsample_bytree': 0.7,}

param = best_model_param1


#%%% learning_curve
score = 'accuracy'

model = XGBClassifier(**param)  

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring=score, n_jobs=-1
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Treinamento')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Teste')
plt.xlabel('Tamanho do Conjunto de Treinamento')
plt.ylabel(score)
plt.legend()
plt.title('Curva de Aprendizado - Modelo 3')

plt.show()  

print(log.logar_acao_realizada("Analises", "Learning Curve" , ""))



#%%% CROSS VALIDATION

from sklearn.model_selection import cross_validate

# Avaliando múltiplas métricas
model = XGBClassifier(**param)

resultados = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'roc_auc' ], n_jobs=-1, return_train_score=True , verbose=2)
#recall -> sensitividade
#precision
# Exibindo resultados
print("Acurácia média na validação:", resultados['test_accuracy'].mean())
print("Acurácia std na validação:", resultados['test_accuracy'].std())
print("roc_auc médio na validação:", resultados['test_roc_auc'].mean())
print("roc_auc std na validação:", resultados['test_roc_auc'].std())

result_df = pd.DataFrame(resultados)

f.salvar_excel_conclusao(result_df, 'IndicadoresXGBCrossValidationMod3')  

print(log.logar_acao_realizada("Analises", "Cross Validation Mod1" , ""))

#%%% VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor


vif_data = pd.DataFrame({
    "Variável": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print(vif_data)

#%%%% Teste de eliminacao de variaveis

# var_eliminar = ['_Gerada_TIPOHIST_BIOLOGICO_3' , '_Gerada_BASMAIMP_TUMPRIM' , '_Gerada_TIPOHIST_CELULAR_ENC']
var_eliminar = ['_Gerada_TIPOHIST_BIOLOGICO_3' , '_Gerada_BASMAIMP_TUMPRIM' , ] # manter '_Gerada_TIPOHIST_CELULAR_ENC' pois não teve influencia

y = df_working[var_dep]
X = df_working.drop(columns=[var_dep])

X = X.drop(columns=var_eliminar)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=100)

file_name = 'XGB9_1'
param_distributions = {
    'subsample': [0.8 , 0.9],
    'colsample_bytree': [0.7 , 0.8],
}

a_fixed_param = {'max_depth' : 9 , 'n_estimators': 1100 , 'learning_rate': 0.05 , 'gamma': 0 , 'lambda': 1 , 'random_state': 100  }

indicadores , modelos = ulf_grid_search(fixed_param = a_fixed_param , X_train = X_train , X_test = X_test, y_train = y_train , y_test = y_test, param_distributions = param_distributions )

f.salvar_excel_conclusao(indicadores, file_name) 

vif_data1 = pd.DataFrame({
    "Variável": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print(vif_data1)
 