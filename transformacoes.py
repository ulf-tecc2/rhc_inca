# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 18:55:27 2024

@author: ulf
"""

import dbfread as db
import pandas as pd
import numpy as np
import re 
import os
import glob
from datetime import date
import locale

import matplotlib.pyplot as plt
import seaborn as sns


import sys
sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

import funcoes_ulf as ulfpp
import bib_graficos as ug

from funcoes import Log
import funcoes as f

from tabulate import tabulate

#%% ETAPA 4 :  CARREGAR ANTERIOR PARA CONTINUIDADE
# =============================================================================
log = Log()
log.carregar_log('log_etapa4')
df_unico = f.leitura_arquivo_parquet('etapa4')
a=log.asString()


#%% TRANSFORMACOES NOS DADOS E GERACAO DE NOVOS CAMPOS DERIVADOS
# =============================================================================

#Dividir PRITRATH em primeiro tratamento e os demais

def PRITRATH_dividir_tratamentos(df):

    df['PRITRATH_Primeiro'] = df['PRITRATH'].apply(lambda x: x[0]) 
    
    df['PRITRATH_Seguintes'] = df['PRITRATH'].apply(lambda x: x[1:])
    df['PRITRATH_Seguintes'] = df['PRITRATH'].apply(lambda x: np.nan if (x == '') else x)
    
    df['PRITRATH_NrTratamentos'] = df['PRITRATH'].apply(lambda x: len(x)) 
    
    return df

def busca_data_sp_iniciou_mais_que_1_trat(df):
    aux_sp = df[df['UFUH'] == 'SP']
    a = aux_sp.loc[aux_sp['PRITRATH_NrTratamentos'] > 1]
    a.sort_values(by='DATAINITRT' , ascending=[True]).reset_index(drop=True)
    
    return  a['DATAINITRT'].iloc[0]


df_unico = PRITRATH_dividir_tratamentos(df_unico)

data_inicio = busca_data_sp_iniciou_mais_que_1_trat(df_unico)
print(log.logar_acao_realizada('Informacao', 'Data do inicio de envio de mais de um tratamento por SP', data_inicio))

#%%
df = df_unico.sample(n=50000, random_state=1)
df = df.dropna(subset=['DATAINITRT', 'PRITRATH_NrTratamentos'])


c = df['DATAINITRT']
c.unique()

aux_ts = pd.Series(df['PRITRATH_NrTratamentos'].values , index=df['DATAINITRT'])


aux_ts.tail(20)
aux_ts.unique()

df_unico['DATAINITRT'].describe()
a = df_unico.groupby(['DATAINITRT'] , observed=True).size()

pd.to_datetime(df_unico['DATAINITRT']).describe()

df[a_col] = pd.to_datetime(df[a_col] , format="%d/%m/%Y" , errors= 'coerce')

# In[16]: Fazendo o grafico (Selecionar todos os comandos)
plt.figure(figsize=(10, 6))
plt.plot(aux_ts)
plt.title("Total de Passageiros no Transporte Aereo BR")
plt.xlabel("Jan/2011 a Mai/2024")
plt.ylabel("Total de Passageiros Mensal")
plt.show()



c.head(100)

b = df_unico.groupby(['PRITRATH_NrTratamentos'] , observed=True).size()
 b = a['DATAINITRT']
 b.iloc[0]
 
aux_sp = df_unico[df_unico['UFUH'] == 'SP']

df_unico.reset_index()
df_unico.index
aux_sp.reset_index
a = df_unico.loc[aux_sp['PRITRATH'].str.len() > 1]
a1 = df_unico.groupby(['PRITRATH_NrTratamentos'] , observed=True).size()
 
 
 
