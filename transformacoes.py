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

def gera_intervalos(df):
    """Geracao de intervalos de tempo e selecao dos registros.
    
     Gerar intervalos:
        _tempo_diagnostico_tratamento = dias desde DTDIAGNO e DATAINITRT
        _tempo_consulta_tratamento = dias desde DATAPRICON e DATAINITRT
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """

    
    df['_tempo_diagnostico_tratamento'] = (df['DATAINITRT'] - df['DTDIAGNO']).dt.days
    df['_tempo_diagnostico_tratamento'].astype(int)
    
    df['_tempo_consulta_tratamento'] = (df['DATAINITRT'] - df['DATAPRICON']).dt.days
    df['_tempo_consulta_tratamento'].astype(int)
    
    print(log.logar_acao_realizada('Gerar Dados' , 'Geracao das variaveis _tempo_diagnostico_tratamento e _tempo_consulta_tratamento ' ,''))
    
    return df
    
gera_intervalos(df)




def  define_valor_esperado(df):
    """Analisa ESTDFIMT e cria a variavel booleana _RESFINAL com sucesso / insucesso. .
    
    Resultado esperado.  ESTDFIMT ==>> Variável dependente _RESFINAL
    1.Sem evidência da doença (remissão completa); 2.Remissão parcial; ===>>> SUCESSO
    3.Doença estável; 4.Doença em progressão; 5.Suporte terapêutico oncológico; 6. Óbito; 8. Não se aplica; 9. Sem informação ===>>> INSUCESSO
  
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    # gerar variavel dependente binaria => sucesso ou nao
    df['_RESFINAL'] =  np.where((df['ESTDFIMT'] == '1') | (df['ESTDFIMT'] == '2') , True , False)
    
    return df






df = define_valor_esperado(df)

def transforma_nulos_naoinformados(df):
    """Transforma os valores nulos para nao informados. .
    
    Nulo => 9
        ALCOOLIS	Histórico de consumo de bebida alcoólica	1.Nunca; 2.Ex-consumidor; 3.Sim; 4.Não avaliado; 8.Não se aplica; 9.Sem informação
        TABAGISM	Histórico de consumo de tabaco	1.Nunca; 2.Ex-consumidor; 3.Sim; 4.Não avaliado; 8.Não se aplica;  9.Sem informação 
        HISTFAMC	Histórico familiar de câncer	1.Sim; 2.Não; 9.Sem informação
        ORIENC	Origem do encaminhamento	1.SUS; 2.Não SUS; 3.Veio por conta própria;8.Não se aplica; 9. Sem informação
        ESTCONJ	Estado conjugal atual	1.Solteiro; 2.Casado; 3.Viúvo;4.Separado judicialmente; 5.União consensual; 9.Sem informação
        DIAGANT	Diagnóstico e tratamento anteriores	1.Sem diag./Sem trat.; 2.Com diag./Sem trat.; 3.Com diag./Com trat.; 4.Outros; 9. Sem informação
        BASMAIMP	Base mais importante para o diagnóstico do tumor	1.Clínica; 2.Pesquisa clínica; 3.Exame por imagem; 4.Marcadores tumorais; 5.Citologia; 6.Histologia da metástase; 7.Histologia do tumor primário; 9. Sem informação
        EXDIAG	Exames relevantes para o diagnóstico e planejamento da terapêutica do tumor	1.Exame clínico e patologia clínica; 2.Exames por imagem; 3.Endoscopia e cirurgia exploradora; 4.Anatomia patológica; 5.Marcadores tumorais; 8.Não se aplica; 9. Sem informação
        LATERALI	Lateralidade do tumor	1.Direita; 2. Esquerda; 3.Bilateral; 8.Não se aplica; 9.Sem informação
        
    Nulo => 1
        MAISUMTU	Ocorrência de mais um tumor primário	1.Não; 2.Sim; 3.Duvidoso

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado
     

    Returns:
        (DataFrame): df modificado
       
    """    
    values = {'ALCOOLIS': '9' , 'TABAGISM': '9' , 'HISTFAMC': '9' , 'ORIENC': '9' ,
              'ESTCONJ': '9' , 'MAISUMTU' : '1' , 'DIAGANT' : '9' , 'BASMAIMP' : '9' ,
              'EXDIAG' : '9' , 'LATERALI' : '9'}
    df = df.fillna(value=values , inplace = False)
    
    print(log.logar_acao_realizada('Dados Nulos' , 'Setar valores nulos para sem informacao' ,f'{values}'))
    
    return df

df = transforma_nulos_naoinformados(df)

def remover_colunas_naosignificativas(df):
    """Elimina as variaveis (colunas) que nao possuem significancia .

    Colunas:
        'ESTDFIMT', 'RZNTR','DTDIAGNO' , 'DTTRIAGE' , 'DATAPRICON' , 'DATAINITRT' , 'DATAOBITO', 
        'TPCASO', 'LOCALNAS' , 'BASDIAGSP' , 'VALOR_TOT' , 
        'AnaliseLOCTUDET', 'AnaliseLOCTUDET_tipo', 'AnaliseLOCTUPRI', 'AnaliseLOCTUPRO','AnaliseTNM', 'AnaliseESTADIAM' ,
        'CLIATEN' , 'CLITRAT' , 'CNES' , 'DTINITRT' , 'LOCTUPRO' , 'ESTADRES', 'OUTROESTA' , 'OCUPACAO' , 'PROCEDEN' , 'ESTADIAG'
        
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    #remocao de variaveis nao significativas
    colunas_a_remover = ['ESTDFIMT', 'RZNTR','DTDIAGNO' , 'DTTRIAGE' , 'DATAPRICON' , 'DATAINITRT' , 'DATAOBITO', 'RZNTR'
                         'TPCASO', 'LOCALNAS' , 'BASDIAGSP' , 'VALOR_TOT' , 
                         'AnaliseLOCTUDET', 'AnaliseLOCTUDET_tipo', 'AnaliseLOCTUPRI', 'AnaliseLOCTUPRO','AnaliseTNM', 'AnaliseESTADIAM' ,
                         'CLIATEN' , 'CLITRAT' , 'CNES' , 'DTINITRT' , 'LOCTUPRO' , 'ESTADRES', 'OUTROESTA' , 'OCUPACAO' , 'PROCEDEN' , 'ESTADIAG' ]
    
    df_aux = df.drop(columns=colunas_a_remover , axis=1)
    
    print(log.logar_acao_realizada('Remocao Registros' , 'Eliminacao de colunas com dados sem significancia' ,f'{colunas_a_remover}'))
    
    return df_aux

df = remover_colunas_naosignificativas(df)

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
 
 
 
