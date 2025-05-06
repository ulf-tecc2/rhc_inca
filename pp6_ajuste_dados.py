# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:38:57 2025

@author: ulf
"""

# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Modelo Logistico.

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



#%% CARGA DADOS


log = Log()

log.carregar_log("log_BaseModelagem")
df = f.leitura_arquivo_parquet("BaseModelagem")

print(log.logar_acao_realizada("Carga Dados", "Carregamento dos dados para modelagem", df.shape[0]))

df.isnull().sum()
df.shape[0]

df.info()



#%% Análise da Base - Distribuicao dos valores
ug.plot_frequencias_valores_atributos(df , ['UFUH'] , bins = 25 , title = 'Registros por UF')

ug.plot_frequencias_valores_atributos(df , ['_Gerada_RESFINAL'] , bins = 2 , title = 'Registros por resultado do Tratamento')

ug.plot_frequencias_valores_atributos(df , ['DTPRICON'] , bins = 22 , title = 'Registros por ano da consulta')


# com = df.loc[df['_Gerada_RESFINAL'] == 'com resposta'].shape[0] / df.shape[0]
# sem = df.loc[df['_Gerada_RESFINAL'] == 'sem resposta'].shape[0] / df.shape[0]

# print(f'Distribuição dos registros pelo resultado (_Gerada_RESFINAL) com resposta:{com} sem resposta:{sem}')


#%%AJUSTE NAS VARIAVEIS
var_dep = "_Gerada_RESFINAL"

#Variavei usadas apenas na analise final dos dados... Remover agora
colunas_remover = ['UFUH','DTPRICON' ]

# existe alta correlação entre PRIMTRATH e se fez_cirurgia ....
colunas_remover = colunas_remover + [
    '_Gerada_Fez_Cirurgia', '_Gerada_Fez_Radioterapia',
    '_Gerada_Fez_Quimioterapia', '_Gerada_Fez_Hormonioterapia',
    '_Gerada_Fez_Transplante', '_Gerada_Fez_Imunoterapia',
    '_Gerada_Fez_OutroTrat',]

df = df.drop(columns=colunas_remover)
print(log.logar_acao_realizada("Ajustes nos Dados", "Remocao de colunas com alta correlacao", ''))

#transformar a resposta em inteiros
df[var_dep] = np.where(df[var_dep] == 'com resposta', 1 , 0)
print(log.logar_acao_realizada("Ajustes nos Dados", "Transformacao da variavel resposta em inteiros", ""))

colunas_int = [ '_Gerada_tempo_para_inicio_tratamento', '_Gerada_distancia_tratamento', '_Gerada_PRITRATH_NrTratamentos' ] #, '_Gerada_tempo_para_diagnostico']


#%% Análise da Base - Graficos para analise das variaveis em relacao ao resultado do tratamento
from matplotlib import pyplot as plt
import seaborn as sns

colunas_higt_card = ['_Gerada_TIPOHIST_CELULAR' , 'LOCTUDET' , 'ESTADIAM']

colunas = [item for item in df.columns if item not in colunas_int]
# colunas = [item for item in colunas if item not in colunas_higt_card]
colunas = [item for item in colunas if item not in ['IDADE' , var_dep ]]

# colunas = colunas[0:6]

aux_df = df.copy()

for i in colunas_higt_card:
    top_categorias = df[i].value_counts().index[:5]  #
    aux_df[i] = df[i].apply(lambda x: x if x in top_categorias else 'Outros')


plt.figure(figsize=(20*0.75,35*0.67))
for j,i in enumerate(colunas):
    if j < 15:
        plt.subplot(5,3,j+1)
        sns.histplot(
            data=aux_df,
            x=var_dep, hue=i,
            multiple="fill", stat="count",
            discrete=True, shrink=.8
        )
        ax = plt.gca()

        ## Definir título e personalizar a fonte
        ax.set_xlabel('', fontsize=11, fontfamily='arial')  ## Tamanho e tipo de fonte
        ax.set_ylabel('', fontsize=11, fontfamily='arial')  ## Tamanho e tipo de fonte

        ax.spines['bottom'].set_linewidth(1.5)  ## Eixo X
        ax.spines['left'].set_linewidth(1.5)    ## Eixo Y
        ax.spines['top'].set_linewidth(0)  ## Eixo X
        ax.spines['right'].set_linewidth(0)    ## Eixo Y
        plt.yticks([])  # Remove os valores do eixo Y
        
plt.show()
        
plt.figure(figsize=(20*0.75,35*0.67))
for j,i in enumerate(colunas):
    if j >= 15:
        plt.subplot(5,3,j+1-15)
        ax = sns.histplot(
            data=aux_df,
            x=var_dep, hue=i,
            multiple="fill", stat="count",
            discrete=True, shrink=.8
        )
        ax = plt.gca()

        ## Definir título e personalizar a fonte
        ax.set_xlabel('', fontsize=11, fontfamily='arial')  ## Tamanho e tipo de fonte
        ax.set_ylabel('', fontsize=11, fontfamily='arial')  ## Tamanho e tipo de fonte

        ax.spines['bottom'].set_linewidth(1.5)  ## Eixo X
        ax.spines['left'].set_linewidth(1.5)    ## Eixo Y
        ax.spines['top'].set_linewidth(0)  ## Eixo X
        ax.spines['right'].set_linewidth(0)    ## Eixo Y
        plt.yticks([])  # Remove os valores do eixo Y
plt.show()
        

colunas = colunas_int 
# plt.figure(figsize=(20*0.7,35*0.7))
plt.figure(figsize=(20*0.75,35*0.67))
for j,i in enumerate(colunas):
    if j < 28:
        plt.subplot(9,3,j+1)
        # sns.histplot(data=df, x=var_dep, hue=i, bins=10, kde=False, alpha=0.7 , multiple='dodge')
        sns.histplot(data=df, x=i, hue=var_dep, bins=10, kde=False, alpha=0.7, multiple='dodge')
        # plt.xlabel("")  # Remove nome do eixo X
        
        ax = plt.gca()

        ## Definir título e personalizar a fonte
        ax.set_xlabel(i, fontsize=11, fontfamily='arial')  ## Tamanho e tipo de fonte
        ax.set_ylabel('', fontsize=11, fontfamily='arial')  ## Tamanho e tipo de fonte

        ax.spines['bottom'].set_linewidth(1.5)  ## Eixo X
        ax.spines['left'].set_linewidth(1.5)    ## Eixo Y
        ax.spines['top'].set_linewidth(0)  ## Eixo X
        ax.spines['right'].set_linewidth(0)    ## Eixo Y
        plt.yticks([])  # Remove os valores do eixo Y
        
    else:
        print(f'Faltou {i}')
plt.show()

#%% PADRONIZAÇCAO E CODIFICACAO DAS VARIAVEIS

#%%% padronização dos dados das variaveis quantitativas

scaler = StandardScaler()
df[colunas_int] = scaler.fit_transform(df[colunas_int])
a = df[colunas_int].describe()

print(log.logar_acao_realizada("Ajustes nos Dados", "padronização dos dados das variaveis quantitativas", colunas_int))

#%%% OneHot Encoder (dummies)
colunas_onehot = [
    '_Gerada_PRITRATH_1',
    '_Gerada_TIPOHIST_BIOLOGICO',
    '_Gerada_TNM_M',
    '_Gerada_TNM_N',
    '_Gerada_TNM_T',
    'SEXO',
    ]

df['SEXO'] = df['SEXO'].cat.remove_unused_categories()

df_encoded = pd.get_dummies(df, columns=colunas_onehot, dtype=int, drop_first=True)
print(log.logar_acao_realizada("Codificacao dos Dados", "Codificação por OneHotEncoding", colunas_onehot))

df_encoded.info()
#%%% TargetEncoder - high cardinality categorical

# tratar variaveis com quantidade muito grande de categorias: high cardinality categorical features logistic regression
# LOCTUDET
# ESTADIAM
# _Gerada_TIPOHIST_CELULAR
    
# freq = df[a_var].value_counts()

colunas_higt_card = ['_Gerada_TIPOHIST_CELULAR' , 'LOCTUDET' , 'ESTADIAM']

encoder = ce.TargetEncoder(cols = colunas_higt_card)
df_aux = encoder.fit_transform(df[colunas_higt_card], df[var_dep])

colunas_higt_card_encoded = colunas_higt_card.copy()
for coluna in colunas_higt_card:
    new_var_name = f'{coluna}_ENC'
    colunas_higt_card_encoded.append(new_var_name) 
    df_encoded[new_var_name] = df_aux[coluna]
    
#salvar os valores originais e os mapeados para serem analisados no shap
mapping_df = df_encoded[colunas_higt_card_encoded]
# mapping_df = mapping_df.drop_duplicates() Tirei para poder contar a quantidade de elementos no shap
f.salvar_parquet(mapping_df, "BaseMapeamentoTargetEncoder")


df_encoded = df_encoded.drop(colunas_higt_card , axis = 1)

print(log.logar_acao_realizada("Codificacao dos Dados", "Codificação por TargetEncoder (muitas categorias)", colunas_higt_card))

df_encoded.info()

#%% Analise da Base - Verificacao final 

#%%% Nulos
a = df_encoded.isnull().sum()

#%%% dominios de valores das variaveis
a_dict = {}
lista_colunas = df_encoded.columns
for c in lista_colunas:
    a_dict[c] = list(df_encoded[c].unique())
    
#%%% Correlacoes    
corr = df_encoded.corr()


## Transforma a matriz de correlação em uma série
corr_abs = corr.abs()
corr_stack = corr_abs.stack()

## Remove duplicatas (correlação A-A)
corr_stack = corr_stack[corr_stack.index.get_level_values(0) != corr_stack.index.get_level_values(1)]

corr_stack = corr_stack.sort_index().drop_duplicates()

## Obtém os maiores valores
maiores_correlações = corr_stack.nlargest(30) 
maiores_correlações = maiores_correlações.reset_index()
f.salvar_excel_conclusao(corr, 'correlacao_completa_Final')  
f.salvar_excel_conclusao(maiores_correlações, 'correlacao_maiores_Final')  

#%%%
colunas_sem_significancia_depois_do_inicio_tratamento = [
    '_Gerada_PRITRATH_NrTratamentos',
    '_Gerada_PRITRATH_1_2',
    '_Gerada_PRITRATH_1_3',
    '_Gerada_PRITRATH_1_4',
    '_Gerada_PRITRATH_1_5',
    '_Gerada_PRITRATH_1_6',
    '_Gerada_PRITRATH_1_7',
    '_Gerada_PRITRATH_1_8',

    ]

df_encoded = df_encoded.drop(columns= colunas_sem_significancia_depois_do_inicio_tratamento)

a = df_encoded.columns


#%%% Analise da multicolineralidade - Modelo Logístico

#%%%% Calculo do VIF para cada variável
# cálculo do Variance Inflation Factor (VIF) pode ser importante para modelos de classificação usando regressão logística binária. Embora a regressão logística seja um modelo de classificação, ela ainda se baseia na estimativa de coeficientes para as variáveis independentes, semelhante à regressão linear.
# O VIF é uma medida projetada para detectar multicolinearidade entre variáveis independentes em modelos de regressão, onde a interpretação dos coeficientes é importante. Em modelos de classificação, como árvores de decisão ou Random Forest, a estrutura do modelo é diferente, e a interpretação direta dos coeficientes não se aplica.
# A multicolinearidade — quando duas ou mais variáveis independentes estão altamente correlacionadas

from statsmodels.stats.outliers_influence import variance_inflation_factor

a = df_encoded
# a = df_encoded.drop(columns = ['_Gerada_TIPOHIST_BIOLOGICO_3' , '_Gerada_BASMAIMP_TUMPRIM'])
vif_data = pd.DataFrame({
    "Variável": a.columns,
    "VIF": [variance_inflation_factor(a.values, i) for i in range(a.shape[1])]
})
print(vif_data.sort_values(by='VIF'))
a2= vif_data

f.salvar_excel_conclusao(vif_data, 'VIF_Final')  

#%%% analise da distribuicao das variaveis com alto VIF (> 5)
# IDADE
import matplotlib.pyplot as plt  # visualização gráfica
df_encoded['IDADE'].plot(kind='kde')
plt.title('Estimativa de Densidade (KDE)')
plt.show()


#%%% Teste de normalidade. 
# A regressão logística não assume que as variáveis independentes sejam normalmente distribuídas, ao contrário de muitos métodos de ajuste de modelos lineares. Assim, a não normalidade das variáveis independentes geralmente não é um problema para a aplicação da regressão logística.
from scipy.stats import shapiro

stat, p = shapiro(df_encoded['IDADE'])
print('IDADE tem distribuicao normal (p > 0,05):', stat, 'Valor-p:', p)


colunas_vif_alto = [
    '_Gerada_TIPOHIST_BIOLOGICO_3',
    '_Gerada_BASMAIMP_TUMPRIM',
    '_Gerada_TIPOHIST_BIOLOGICO_2',
    'ESTADIAM_ENC',
    '_Gerada_TNM_T_2',
    '_Gerada_TIPOHIST_CELULAR_ENC',
    '_Gerada_TNM_T_3',
    '_Gerada_TNM_T_1',
    'LOCTUDET_ENC',
    # 'IDADE',
    '_Gerada_TNM_T_4',
    '_Gerada_BASMAIMP_CIT',
    '_Gerada_BASMAIMP_IMG',
    '_Gerada_BASMAIMP_CLIN',
    '_Gerada_BASMAIMP_MET',
    '_Gerada_BASMAIMP_PESQ',
    ]

for a_var in colunas_vif_alto:
    b = df_encoded[a_var].value_counts(normalize=True)
    print('*** ' + a_var)
    print(b)


# df_encoded = df_encoded.drop(columns= colunas_vif_alto)
# =============================================================================
# APLICAR PCA: tem multicolineridade
# =============================================================================

# ug.plot_correlation_heatmap(df_encoded , df_encoded.columns)

#%% Salvar os dados

log.salvar_log("log_BaseModelagemFinal")
f.salvar_parquet(df_encoded, "BaseModelagemFinal")
