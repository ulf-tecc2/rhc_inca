# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Analise dos modelos construidos.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: Ulf Bergmann

"""


import pandas as pd
import numpy as np

import time
import os
import sys

import seaborn as sns

import matplotlib.pyplot as plt

sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

import funcoes as f
from funcoes import Log

import funcoes_ulf as uf
import bib_graficos as ug

import statsmodels.api as sm # estimação de modelos
import statsmodels.formula.api as smf # estimação do modelo logístico binário
from statstests.process import stepwise # procedimento Stepwise
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos

import statsmodels.formula.api as smf # estimação do modelo logístico binário

                                                        
import category_encoders as ce 
                                                        
from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score , precision_score , f1_score,  roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

#%% Funcoes
def consolidar_resultados_modelos():
    dict_resultados = {}
    
    dir_padrao = "dados/workdir/"
    for a_name in os.listdir(dir_padrao):
        a_file_name = os.path.join(dir_padrao , a_name)
    
        if ~(os.path.isfile(a_file_name)):
            print('dir '+ a_name)
            dict_resultados = dict_resultados |  { a_name : f.load_objects(a_name) } 
            
    
    indicadores = pd.DataFrame()
    resumos = pd.DataFrame()
    for i in dict_resultados.keys():
        if 'indicadores' in dict_resultados[i].keys():
            indicadores = pd.concat([indicadores , dict_resultados[i]['indicadores']] , axis = 0)
            resumos = pd.concat([resumos , dict_resultados[i]['resumo']] , axis = 1)
        else:
            print(dict_resultados[i].keys())
    
    f.salvar_excel_conclusao(indicadores, 'IndicadoresModelos')  
    f.salvar_excel_conclusao(resumos, 'ResumosModelos')  

#gera e salva os modelos escolhidos
def gera_modelos(file_sufix = ''):
    # mod_logistico = f.obter_modelo_salvo('rf1_500000')
    
    # param_mod_log = '_Gerada_RESFINAL ~ IDADE + _Gerada_tempo_para_inicio_tratamento + _Gerada_distancia_tratamento + _Gerada_ALCOOLIS + _Gerada_BASMAIMP_CLIN + _Gerada_BASMAIMP_PESQ + _Gerada_BASMAIMP_IMG + _Gerada_BASMAIMP_MET + _Gerada_DIAGANT_DIAG + _Gerada_EXDIAG_IMG + _Gerada_EXDIAG_END_CIR + _Gerada_EXDIAG_PAT + _Gerada_EXDIAG_MARC + _Gerada_HISTFAMC + _Gerada_LATERALI_ESQ + _Gerada_LATERALI_DIR + _Gerada_MAISUMTU + _Gerada_ORIENC_SUS + _Gerada_TABAGISM + _Gerada_TIPOHIST_BIOLOGICO_1 + _Gerada_TIPOHIST_BIOLOGICO_2 + _Gerada_TNM_M_1 + _Gerada_TNM_M_X + _Gerada_TNM_N_1 + _Gerada_TNM_N_2 + _Gerada_TNM_N_3 + _Gerada_TNM_T_2 + _Gerada_TNM_T_3 + _Gerada_TNM_T_4 + _Gerada_TNM_T_I + _Gerada_TNM_T_X + SEXO_2 + _Gerada_TIPOHIST_CELULAR_ENC + LOCTUDET_ENC + ESTADIAM_ENC'
    # mod_log = smf.glm(formula=param_mod_log, data=df_working, family=sm.families.Binomial()).fit()
    # f.save_model('Logistic_Model' + file_sufix, mod_log)
    # print('Gerado o modelo Logistico com os parametros:')
    # print(param_mod_log)
    
    # param_mod_rf = {'max_depth' : 25 , 'min_samples_leaf' : 1 , 'min_samples_split': 2 , 'n_estimators': 340 , 'max_features' : 7 , 'random_state': 100   }							
    # mod_rf = RandomForestClassifier(**param_mod_rf)  
    # mod_rf.fit(X_train , y_train)
    # f.save_model('RandonForest_Model' + file_sufix, mod_rf)
    # print('Gerado o modelo RandonForest com os parametros:')
    # print(param_mod_rf)
    
    
    param_mod_xgb =  {'gamma': 0.1, 'lambda': 1, 'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 1000}
    mod_xgb = XGBClassifier(**param_mod_xgb)  
    mod_xgb.fit(X_train , y_train)
    f.save_model('XGBoost_Model' + file_sufix, mod_xgb) 
    print('Gerado o modelo XGBClassifier com os parametros:')
    print(param_mod_xgb)

#%%% Analises

def analisa_arvore(mod):
    print('----- Parametros -----')
    print(mod.get_params())

    n_leaf_nodes = [arvore.tree_.n_leaves for arvore in mod.estimators_]
    
    # Exibindo o resultado
    # print(f"Nós folha por árvore: {n_leaf_nodes}")
    print(f"Média de nós folha: {sum(n_leaf_nodes) / len(n_leaf_nodes):.2f}")
    print(f"Total de nós folha: {sum(n_leaf_nodes)}")
    
    n_leaf_nodes = [arvore.tree_.n_leaves for arvore in mod.estimators_]
    # print(f"Nós folha por árvore: {n_leaf_nodes}")
    print(f"Média de nós folha: {sum(n_leaf_nodes) / len(n_leaf_nodes):.2f}")
    print(f"Total de nós folha: {sum(n_leaf_nodes)}")

    alturas = [arvore.tree_.max_depth for arvore in mod.estimators_]
    # print(f"Alturas das árvores: {alturas}")
    print(f"Altura média: {sum(alturas) / len(alturas):.2f}")
    print(f"Altura máxima: {max(alturas)}")
    print(f"Altura mínima: {min(alturas)}")
    
    # # Quantidade de samples nas folhas de cada árvore
    # samples_por_folha = [arvore.tree_.n_node_samples[arvore.tree_.children_left == -1] for arvore in mod.estimators_]
    
    # # Exibindo o resultado para cada árvore
    # for i, samples in enumerate(samples_por_folha):
    #     print(f"Árvore {i + 1}: {max(samples)}")  # Exibe a quantidade de samples por folha em cada árvore
    
def descritiva(df_, var, vresp, max_classes=5):
    """
    Gera um gráfico descritivo da taxa de sobreviventes por categoria da variável especificada.
    
    Parâmetros:
    df : DataFrame - Base de dados a ser analisada.
    var : str - Nome da variável categórica a ser analisada.
    """
    
    df = df_.copy()
    
    if df[var].nunique()>max_classes:
        df[var] = pd.qcut(df[var], max_classes, duplicates='drop')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.pointplot(data=df, y=vresp, x=var, ax=ax1)
    
    # Criar o segundo eixo y para a taxa de sobreviventes
    ax2 = ax1.twinx()
    sns.countplot(data=df, x=var, palette='viridis', alpha=0.5, ax=ax2)
    ax2.set_ylabel('Frequência', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)  # Tornar o fundo do eixo 1 transparente
    
    # Exibir o gráfico
    plt.show()
    
def diagnóstico(df_, var, vresp , pred , max_classes=5):
    """
    Gera um gráfico descritivo da taxa de sobreviventes por categoria da variável especificada.
    
    Parâmetros:
    df : DataFrame - Base de dados a ser analisada.
    var : str - Nome da variável categórica a ser analisada.
    """
    
    df = df_.copy()
    
    # if df[var].nunique()>max_classes:
    #     df[var] = pd.qcut(df[var], max_classes, duplicates='drop')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.pointplot(data=df, y=vresp, x=var, ax=ax1)
    sns.pointplot(data=df, y=pred, x=var, ax=ax1, color='red', linestyles='--', ci=None)
    
    # Criar o segundo eixo y para a taxa de sobreviventes
    ax2 = ax1.twinx()
    sns.countplot(data=df, x=var, palette='viridis', alpha=0.5, ax=ax2)
    ax2.set_ylabel('Frequência', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)  # Tornar o fundo do eixo 1 transparente
    
    # Exibir o gráfico
    plt.show()
    
def calcula_indicadores( label , modelo ,  X_train, X_test, y_train, y_test , cutoff = None):
   
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)
    
    observado = y_train
    predicao = y_train_pred
    if cutoff != None:
        predicao_binaria = []
        for item in predicao:
            if item < cutoff:
                predicao_binaria.append(0)
            else:
                predicao_binaria.append(1) 
        predicao = predicao_binaria
    
    sensitividade = recall_score(observado, predicao, pos_label=1)
    especificidade = recall_score(observado, predicao, pos_label=0)
    acuracia_train = accuracy_score(observado, predicao)
    precisao = precision_score(observado, predicao)
    f1_scor = f1_score(observado, predicao)
    auc = roc_auc_score(observado , predicao)
   
    ind_train = pd.DataFrame()
    ind_train.at[0,'Label'] = label
    ind_train.at[0,'Tipo'] = 'Treino'
    ind_train.at[0,'Sensitividade'] = sensitividade
    ind_train.at[0,'Especificidade'] = especificidade
    ind_train.at[0,'Acuracia'] = acuracia_train
    ind_train.at[0,'Precisao'] = precisao
    ind_train.at[0,'F1_Score'] = f1_scor
    ind_train.at[0,'AUC'] = auc
    
    observado = y_test
    predicao = y_test_pred
    if cutoff != None:
        predicao_binaria = []
        for item in predicao:
            if item < cutoff:
                predicao_binaria.append(0)
            else:
                predicao_binaria.append(1) 
        predicao = predicao_binaria
    sensitividade = recall_score(observado, predicao, pos_label=1)
    especificidade = recall_score(observado, predicao, pos_label=0)
    acuracia_test = accuracy_score(observado, predicao)
    precisao = precision_score(observado, predicao)
    f1_scor = f1_score(observado, predicao)
    auc = roc_auc_score(observado , predicao)
   
    ind_train.at[1,'Label'] = label
    ind_train.at[1,'Tipo'] = 'Teste'
    ind_train.at[1,'Sensitividade'] = sensitividade
    ind_train.at[1,'Especificidade'] = especificidade
    ind_train.at[1,'Acuracia'] = acuracia_test
    ind_train.at[1,'Precisao'] = precisao
    ind_train.at[1,'F1_Score'] = f1_scor
    ind_train.at[1,'AUC'] = auc
    
    ind_train.at[0,'Acuracia Test/Train'] = acuracia_test / acuracia_train
    ind_train.at[1,'Acuracia Test/Train'] = acuracia_test / acuracia_train
    
    return ind_train
    
def obtem_importancia_features(model):
    importances = model.feature_importances_
    feature_columns = model.feature_names_in_
    feature_importances = pd.Series(importances, index=feature_columns).sort_values(ascending=False)
    
    return feature_importances

#%% Carga dos dados

log = Log()
log.carregar_log("log_BaseModelagemFinal")
# df = f.leitura_arquivo_parquet("BaseModelagemFinal")
df = f.leitura_arquivo_parquet("BaseModelagemFinal")


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




#%% Principal

# gera_modelos('_com_pca')
gera_modelos('_sem_pca')

mod_log , mod_rf , mod_xgb = f.carrega_modelos('_sem_pca')

# mod_log = f.load_model('LogisticoBinario_Escolhido' + '_sem_pca')

df_ind = pd.DataFrame()
ind = calcula_indicadores('XGB sem PCA' , mod_xgb , X_train, X_test, y_train, y_test)
df_ind = pd.concat([df_ind, ind], axis=0)

ind = calcula_indicadores('RF sem PCA' , mod_rf , X_train, X_test, y_train, y_test)
df_ind = pd.concat([df_ind, ind], axis=0)

ind = calcula_indicadores('LOG sem PCA' , mod_log , X_train, X_test, y_train, y_test , cutoff=0.5)
df_ind = pd.concat([df_ind, ind], axis=0)

f.salvar_excel(df_ind , 'ind_sem_pca')

# analisa_arvore(mod_rf)

# for col in df_working.columns:
#     descritiva(df, col, var_dep)

# for col in df_working.columns:
#     diagnóstico(df, col, var_dep , y, 'pred_rf')

# print(obtem_importancia_features(mod_rf))

# print(obtem_importancia_features(mod_xgb))
