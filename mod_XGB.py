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

tamanho_amostra = df.shape[0]
df_working = df

# tamanho_amostra = 50000
# df_working = df.groupby(var_dep, group_keys=False).apply(lambda x: x.sample(int(tamanho_amostra/2) , random_state=1))


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


#%% Escolha do melhor testando K-folds
param_grid = {
    'max_depth' : [ 10 , 12],	
    'n_estimators' : [1000 , 1100 , 1200],	
    'learning_rate' :	[0.05 , 0.06 , 0.1],
    'gamma': [0 , 0.1],			
    'lambda': [1]	
}

xgb = XGBClassifier(random_state=100)  
scoring = {"AUC": "roc_auc", "F1": "f1"}             
# Treinar os modelos para o grid search
xgb_grid_model = GridSearchCV(estimator = xgb, 
                              param_grid = param_grid,
                              scoring=scoring,
                              refit="F1",
                              cv=4, verbose=2,
                              return_train_score=True) 

start_time = time.time()

xgb_grid_model.fit(X_train, y_train)

end_time = time.time()
print( log.logar_acao_realizada( "Modelagem", "criacao do modelo XGBoost",  end_time - start_time  ))
a = xgb_grid_model.cv_results_
a = pd.DataFrame(a)
print("Melhores parametros")
print(xgb_grid_model.best_params_)

# Gerando o modelo com os melhores hiperparâmetros
xgb_best = xgb_grid_model.best_estimator_

f.salvar_excel_conclusao(a, 'IndicadoresXGBKfold_Final') 
f.save_model('ResultadosXGBKfold', xgb_grid_model.cv_results_)

#%%% Analise grafica dos parametros. Ajuste fino do parametro n_estimators

param_grid = {
    'max_depth' : [10],	
    'n_estimators' : range(900, 1300, 10),	
    'learning_rate' :	[0.05],
    'gamma': [0.1],			
    'lambda': [1]	
}

xgb = XGBClassifier(random_state=100) 
 
scoring = {"AUC": "roc_auc", "F1": "f1"}             
# Treinar os modelos para o grid search
xgb_grid_model = GridSearchCV(estimator = xgb, 
                              param_grid = param_grid,
                              scoring=scoring,
                              refit="F1",
                              cv=2, verbose=2,
                              return_train_score=True) 

xgb_grid_model.fit(X_train, y_train)


print( log.logar_acao_realizada( "Modelagem", "Termino do fit para ajuste fino",  ""  ))

results = xgb_grid_model.cv_results_
a = pd.DataFrame(results)
f.salvar_excel_conclusao(a, 'IndicadoresXGBKfold_AjusteFino') 
#%%
a_param_name = 'n_estimators'
plt.figure(figsize=(13, 13))
# plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)

plt.xlabel(a_param_name)
plt.ylabel("Score")

ax = plt.gca()
# ax.set_xlim(900, 1300)
ax.set_ylim(0.65, 0.90)

X_axis = np.array(results[f"param_{a_param_name}"].data, dtype=float)

for scorer, color in zip(sorted(scoring), ["g", "k"]):
    for sample, style in (("train", "--"), ("test", "-")):
        sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
        sample_score_std = results["std_%s_%s" % (sample, scorer)]
        ax.fill_between(
            X_axis,
            sample_score_mean - sample_score_std,
            sample_score_mean + sample_score_std,
            alpha=0.1 if sample == "test" else 0,
            color=color,
        )
        ax.plot(
            X_axis,
            sample_score_mean,
            style,
            color=color,
            alpha=1 if sample == "test" else 0.7,
            label="%s (%s)" % (scorer, sample),
        )

    best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
    best_score = results["mean_test_%s" % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot(
        [
            X_axis[best_index],
        ]
        * 2,
        [0, best_score],
        linestyle="-.",
        color=color,
        marker="x",
        markeredgewidth=3,
        ms=8,
    )

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid(False)
plt.show()

#%% Analises do Melhor Modelo
    

from sklearn.metrics import make_scorer, roc_auc_score

#XGB2
# param_distributions = {
#     'n_estimators': [500 , 1000],
#     'learning_rate': [0.05 , 0.1],
#     'max_depth': [ 9 , 10],
# }
# a_fixed_param = {'random_state': 100   }



indicadores = pd.DataFrame()

# best_model_param1 = {'max_depth' : 12 , 'n_estimators': 1000 , 'learning_rate': 0.05 , 'gamma': 0 , 'lambda': 1 , 'random_state': 100  }
# best_model_param2 = {'max_depth' : 9 , 'n_estimators': 1200 , 'learning_rate': 0.06 , 'gamma': 0 , 'lambda' : 1 , 'random_state': 100 }
# best_model_param3 = {'max_depth' : 9 , 'n_estimators': 1100 , 'learning_rate': 0.05 , 'gamma': 0 , 'lambda': 1 , 'random_state': 100 , 'subsample': 0.9,  'colsample_bytree': 0.7,}
# best_model_param4 = {'max_depth' : 10 , 'n_estimators': 1100 , 'learning_rate': 0.05 , 'gamma': 0 , 'lambda': 1 , 'random_state': 100  }
# lista_parametros_modelos = [best_model_param1 , best_model_param2 , best_model_param3 , best_model_param4]
# param = best_model_param4

best_model_param1 = {'gamma': 0.1, 'lambda': 1, 'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 1000}
best_model_param2 = {'gamma': 0.1, 'lambda': 1, 'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 1100}
best_model_param3 = {'gamma': 0.1, 'lambda': 1, 'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 1200}
param = best_model_param1


#%%% learning_curve
from sklearn.model_selection import learning_curve

model = XGBClassifier(**param)  

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='f1', n_jobs=-1 , verbose=2
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Treinamento')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Teste')
plt.legend()
ax = plt.gca()

## Definir título e personalizar a fonte
ax.set_xlabel('Tamanho base treinamento', fontsize=11, fontfamily='arial')  ## Tamanho e tipo de fonte
ax.set_ylabel('F1 Score', fontsize=11, fontfamily='arial')  ## Tamanho e tipo de fonte

ax.spines['bottom'].set_linewidth(1.5)  ## Eixo X
ax.spines['left'].set_linewidth(1.5)    ## Eixo Y
ax.spines['top'].set_linewidth(0)  ## Eixo X
ax.spines['right'].set_linewidth(0)    ## Eixo Y
# plt.yticks([])  # Remove os valores do eixo Y

plt.show()  

#%%
model = XGBClassifier(**param)  

train_sizes_auc, train_scores_auc, test_scores_auc = learning_curve(
    model, X, y, cv=5, scoring='roc_auc', n_jobs=-1 , verbose=2
)

plt.plot(train_sizes_auc, train_scores_auc.mean(axis=1), label='Treinamento')
plt.plot(train_sizes_auc, test_scores_auc.mean(axis=1), label='Teste')
plt.legend()
ax = plt.gca()

## Definir título e personalizar a fonte
ax.set_xlabel('Tamanho base treinamento', fontsize=11, fontfamily='arial')  ## Tamanho e tipo de fonte
ax.set_ylabel('AUC', fontsize=11, fontfamily='arial')  ## Tamanho e tipo de fonte

ax.spines['bottom'].set_linewidth(1.5)  ## Eixo X
ax.spines['left'].set_linewidth(1.5)    ## Eixo Y
ax.spines['top'].set_linewidth(0)  ## Eixo X
ax.spines['right'].set_linewidth(0)    ## Eixo Y
# plt.yticks([])  # Remove os valores do eixo Y

plt.show()  

#%%%


print(log.logar_acao_realizada("Analises", "Learning Curve" , ""))



#%%% CROSS VALIDATION

from sklearn.model_selection import cross_validate

a_count = 0
for param  in lista_parametros_modelos:
    # Avaliando múltiplas métricas
    model = XGBClassifier(**param)
    
    resultados = cross_validate(model, X, y, cv=5, scoring=['f1', 'roc_auc' ], n_jobs=-1, return_train_score=True , verbose=2)
    result_df = pd.DataFrame(resultados)
    
    f.salvar_excel_conclusao(result_df, f'IndicadoresXGBCrossValidationMod_{a_count}')  
    
    print(log.logar_acao_realizada("Analises", "Cross Validation" , a_count))
    a_count = a_count + 1


 