# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Construcao do Modelo Logistico.

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
                    
from sklearn.linear_model import LogisticRegression                                    
                                                     
import warnings
warnings.filterwarnings('ignore')



#%% CARGA DADOS


log = Log()
log.carregar_log("log_BaseModelagemFinal")
df = f.leitura_arquivo_parquet("BaseModelagemFinal")

print(log.logar_acao_realizada("Carga Dados", "Carregamento dos dados para modelagem", df.shape[0]))

var_dep = "_Gerada_RESFINAL"

tamanho_amostra = 500000
df_encoded_train = df.groupby(var_dep, group_keys=False).apply(lambda x: x.sample(int(tamanho_amostra / 2) , random_state=1))

var_eliminar = ['_Gerada_TIPOHIST_BIOLOGICO_3' , '_Gerada_BASMAIMP_TUMPRIM' , ] # manter '_Gerada_TIPOHIST_CELULAR_ENC' pois não teve influencia
df_encoded_train = df_encoded_train.drop(columns=var_eliminar)
df_restante = df[~df.apply(tuple, axis=1).isin(df_encoded_train.apply(tuple, axis=1))]

print(log.logar_acao_realizada("Carga Dados", "Trabalhando com uma amostra dos registros", df_encoded_train.shape[0]))

#%% Aplicacao do stepwise para definicao das variaveis

start_time = time.time()

#Analise das variaveis pelo Stepwise ol
X = df_encoded_train.drop(columns=[var_dep])
y = df_encoded_train[var_dep]

result = uf.stepwise_ols(X, y , threshold_in=0.05, threshold_out = 0.05 )

end_time = time.time()

print('----- Variaveis selecionadas -------')
print(result)

#%% Criacao do Modelo

X = X[result]

start_time = time.time()

modelo = LogisticRegression(random_state=0).fit(X, y)
a = modelo.coef_
b = modelo.intercept_
end_time = time.time()
tempo_mod_inicial = end_time - start_time
print( log.logar_acao_realizada( "Modelagem", "criacao do modelo Logístico Binário", tempo_mod_inicial  ))



#%% Salvar dados do modelo

cutoff=0.5

# Calcular o AUC para a massa de testes de treino e teste
X['phat'] = modelo.predict(X)

resumo = pd.DataFrame()
a_nome_coluna = f'Logistico Binario - Modelo LogisticRegression  - Sample {tamanho_amostra}' # Descricao
resumo.at['Variaveis utilizadas' , a_nome_coluna] = str(result) 
resumo.at['Tamanho Base Modelagem' , a_nome_coluna] = df_encoded_train.shape[0] 
resumo.at['Tamanho Base' , a_nome_coluna] = df_restante.shape[0] 
resumo.at['Tempo de construcao do modelo inicial' , a_nome_coluna] = tempo_mod_inicial


indicadores = ug.calcula_indicadores_predicao_classificacao(f'Logistico Binario - Apos PCA  e eliminacao de alto VIF - Sample {tamanho_amostra} - Base modelagem'  , modelo , y , X['phat_train'] , cutoff)

modelo_step_summary = modelo.summary()

an_obj_dict = {
    'modelo' : modelo,
    'resumo' : resumo,
    'summary' : modelo_step_summary,
    'cutoff' : cutoff,
    'indicadores' : indicadores,
    }

f.save_model('Logistic_Model_sem_pca', modelo_step)
f.save_objects(an_obj_dict , f'log_PCA_VIF{tamanho_amostra}')

#%% Analise dos resultados finais

cutoff = 0.5

X['phat'] = modelo.predict(X)


# # # Plotagem de um gráfico que mostra a variação da especificidade e da sensitividade em função do cutoff
# # #Somente para entendimento e não para definir o cutoff
tabela = ug.plota_analise_indicadores_predicao_classificacao('LogisticRegression' , modelo , y , X['phat'])


ug.plot_matriz_confusao(y,X['phat'] , cutoff)


# # Construção da curva ROC
# ug.plot_curvas_roc(df_encoded_train , var_dep , ['phat_inicial' , 'phat_step'])

# from sklearn.metrics import classification_report
# a = np.where(df_encoded_train['phat_train'] >= cutoff , 1 , 0)
# print(classification_report(df_encoded_train[var_dep] ,a))

