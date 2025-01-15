# -*- coding: utf-8 -*-

"""Registros Hospitalares de Câncer (RHC) - Extracao e selecao dos dados para construcao dos modelos
MBA em Data Science e Analytics - USP/Esalq - 2025

@author: ulf Bergmann

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

import funcoes as f
from funcoes import Log

from tabulate import tabulate


log = Log()
log.carregar_log('log_analise_valores')
df_base = f.leitura_arquivo_parquet('analise_valores')

a=log.asString()

base_inicial = df_base.isnull().sum()

#%% SELECAO DA BASE QUE ATENDE AOS CRITERIOS
#=============================================================================

#remocao de variaveis nao significativas
colunas_a_remover = []

# =============================================================================
# Resultado esperado. Variável dependente ==> ESTDFIMT
# 1.Sem evidência da doença (remissão completa); 2.Remissão parcial; 3.Doença estável; 4.Doença em progressão; 
# 5.Suporte terapêutico oncológico; 6. Óbito; 8. Não se aplica; 9. Sem informação
# =============================================================================

q_inicial = df_base.shape[0]
#retirar valores 8 e 9
df_base.drop(df_base[(df_base['ESTDFIMT'] == '8') | (df_base['ESTDFIMT'] == '9')].index, inplace = True)
#retirar valores null
df_base.dropna(subset = ['ESTDFIMT'], inplace=True)
print(log.logar_acao_realizada('Selecao Dados' , 'Eliminacao de registros com ESTDFIMT invalido (8 , 9 e nulos)' ,f'{q_inicial - df_base.shape[0]}'))

# gerar variavel dependente binaria => sucesso ou nao
df_base['_RESFINAL'] =  np.where(df_base['ESTDFIMT'] == '1' , True , False)

colunas_a_remover = colunas_a_remover + ['ESTDFIMT']
# =============================================================================
# ESTADIAM
#Codificação do grupamento do estádio clínico segundo classificação TNM
# =============================================================================

q_inicial = df_base.shape[0]
#retirar valores que nao sao exatos
df_base.drop(df_base[(df_base['AnaliseESTADIAM'] == 'nulo') | (df_base['AnaliseESTADIAM'] == 'demais')].index, inplace = True)
print(log.logar_acao_realizada('Selecao Dados' , 'Eliminacao de registros com ESTADIAM que nao sao exatos' ,f'{q_inicial - df_base.shape[0]}'))


# =============================================================================
# Geracao de intervalos de tempo e eliminacao de datas
# ----- REMOVER -----
# DTDIAGNO	Data do primeiro diagnóstico
# DTTRIAGE	Data da triagem
# DATAPRICON	Data da 1ª consulta
# DATAINITRT	Data do início do primeiro tratamento específico para o tumor, no hospital
# DATAOBITO	Data do óbito

# ----- GERAR -----
# _tempo_diagnostico_tratamento = dias desde DTDIAGNO e DATAINITRT
# _tempo_consulta_tratamento = dias desde DATAPRICON e DATAINITRT

# =============================================================================

# remover sem data de inicio do tratamento
q_inicial = df_base.shape[0]
#retirar valores que nao sao exatos
df_base.dropna(subset = ['DATAINITRT'], inplace=True)
print(log.logar_acao_realizada('Dados Nulos' , 'Eliminacao de registros com DATAINITRT nulo' ,f'{q_inicial - df_base.shape[0]}'))

# remover sem data de diagnostico
q_inicial = df_base.shape[0]
#retirar valores que nao sao exatos
df_base.dropna(subset = ['DTDIAGNO'], inplace=True)
print(log.logar_acao_realizada('Dados Nulos' , 'Eliminacao de registros com DTDIAGNO nulo' ,f'{q_inicial - df_base.shape[0]}'))

df_base['_tempo_diagnostico_tratamento'] = (df_base['DATAINITRT'] - df_base['DTDIAGNO']).dt.days
df_base['_tempo_diagnostico_tratamento'].astype(int)

df_base['_tempo_consulta_tratamento'] = (df_base['DATAINITRT'] - df_base['DATAPRICON']).dt.days
df_base['_tempo_consulta_tratamento'].astype(int)

print(log.logar_acao_realizada('Gerar Dados' , 'Geracao das variaveis _tempo_diagnostico_tratamento e _tempo_consulta_tratamento ' ,''))

colunas_a_remover = colunas_a_remover + ['DTDIAGNO' , 'DTTRIAGE' , 'DATAPRICON' , 'DATAINITRT' , 'DATAOBITO']

# =============================================================================
# Ajustes em variaveis
# # ALCOOLIS	Histórico de consumo de bebida alcoólica	1.Nunca; 2.Ex-consumidor; 3.Sim; 4.Não avaliado; 8.Não se aplica; 9.Sem informação
# # TABAGISM	Histórico de consumo de tabaco	1.Nunca; 2.Ex-consumidor; 3.Sim; 4.Não avaliado; 8.Não se aplica;  9.Sem informação 
# # HISTFAMC	Histórico familiar de câncer	1.Sim; 2.Não; 9.Sem informação
# # ORIENC	Origem do encaminhamento	1.SUS; 2.Não SUS; 3.Veio por conta própria;8.Não se aplica; 9. Sem informação
# # ESTCONJ	Estado conjugal atual	1.Solteiro; 2.Casado; 3.Viúvo;4.Separado judicialmente; 5.União consensual; 9.Sem informação
# # DIAGANT	Diagnóstico e tratamento anteriores	1.Sem diag./Sem trat.; 2.Com diag./Sem trat.; 3.Com diag./Com trat.; 4.Outros; 9. Sem informação
# # BASMAIMP	Base mais importante para o diagnóstico do tumor	1.Clínica; 2.Pesquisa clínica; 3.Exame por imagem; 4.Marcadores tumorais; 5.Citologia; 6.Histologia da metástase; 7.Histologia do tumor primário; 9. Sem informação
# # EXDIAG	Exames relevantes para o diagnóstico e planejamento da terapêutica do tumor	1.Exame clínico e patologia clínica; 2.Exames por imagem; 3.Endoscopia e cirurgia exploradora; 4.Anatomia patológica; 5.Marcadores tumorais; 8.Não se aplica; 9. Sem informação
# # LATERALI	Lateralidade do tumor	1.Direita; 2. Esquerda; 3.Bilateral; 8.Não se aplica; 9.Sem informação

# # Nulo => 9 
#
# # MAISUMTU	Ocorrência de mais um tumor primário	1.Não; 2.Sim; 3.Duvidoso
# # Nulo => 1
# =============================================================================

values = {'ALCOOLIS': '9' , 'TABAGISM': '9' , 'HISTFAMC': '9' , 'ORIENC': '9' ,
          'ESTCONJ': '9' , 'MAISUMTU' : '1' , 'DIAGANT' : '9' , 'BASMAIMP' : '9' ,
          'EXDIAG' : '9' , 'LATERALI' : '9'}
df_base = df_base.fillna(value=values)

print(log.logar_acao_realizada('Dados Nulos' , 'Setar valores nulos para sem informacao' ,f'{values}'))

# =============================================================================
# Remover registros com nulos
# # SEXO
# TIPOHIST
# =============================================================================
q_inicial = df_base.shape[0]
df_base.dropna(subset = ['SEXO' , 'TIPOHIST'], inplace=True)
print(log.logar_acao_realizada('Dados Nulos' , 'Eliminacao de registros com SEXO ou TIPOHIST nulo' ,f'{q_inicial - df_base.shape[0]}'))

# =============================================================================
# Remover quem nao fez tratamento
# # RZNTR diferente de nulo
# =============================================================================


df_base = df_base[df_base['RZNTR'].notna()]
colunas_a_remover = colunas_a_remover + ['RZNTR']




#remocao de variaveis nao significativas
colunas_a_remover = colunas_a_remover + ['TPCASO', 'LOCALNAS' , 'BASDIAGSP' , 'VALOR_TOT' , 
                                         'AnaliseLOCTUDET', 'AnaliseLOCTUDET_tipo', 'AnaliseLOCTUPRI', 'AnaliseLOCTUPRO','AnaliseTNM', 'AnaliseESTADIAM' ,
                                         'CLIATEN' , 'CLITRAT' , 'CNES' , 'DTINITRT' ,
                                         'LOCTUPRO' , 'ESTADRES', 'OUTROESTA' , 'OCUPACAO' , 'PROCEDEN' , 'ESTADIAG' ]

df_base = df_base.drop(columns=colunas_a_remover , axis=1)

print(log.logar_acao_realizada('Remocao Registros' , 'Eliminacao de colunas com dados sem significancia' ,f'{colunas_a_remover}'))


print(log.logar_acao_realizada('Informacao' , 'Quantidade de registros com valores validos' ,f'{df_base.shape[0]}'))
a = df_base.isnull().sum()
print(log.logar_acao_realizada('Informacao' , 'Quantidade de registros com valores nulos' , a))


a = log.asString()


# df.isnull().sum()


b = df_base['ESTADIAG'].value_counts(dropna=False, normalize=False)

# # retirar as que são detalhamento de outras, por exemplo 





#%% SALVAR MUDANCAS

log.salvar_log('log_extracao_dados') 
f.salvar_parquet(df_base , 'extracao_dados')


