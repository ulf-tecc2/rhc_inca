# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Analise dos dados e escolha dos métodos e técnicas.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: Ulf Bergmann

"""

import pandas as pd
import numpy as np

import seaborn as sns  # visualização gráfica
import matplotlib.pyplot as plt  # visualização gráfica

import sys

sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

import funcoes as f
from funcoes import Log

import funcoes_ulf as uf
import bib_graficos as ug

log = Log()

df_unico = f.leitura_arquivo_parquet("BaseModelagem")
print(
    log.logar_acao_realizada(
        "Carga Dados", "Carregamento dos dados para calculo dos indicadores", " "
    )
)

a = df_unico.columns
c = df_unico.isnull().sum()
# Variavel dependente
var_dep = "_RESFINAL"

a = uf.tabela_frequencias(df_unico , var_dep)
ug.plot_frequencias_valores_atributos(df_unico , [var_dep] , bins = 20 , title = 'Distribuição dos Registros pelo Ano')
ug.plot_frequencias_valores_atributos(df_unico , ['DTPRICON'] , bins = 20 , title = 'Distribuição dos Registros pelo Ano')


ug.descritiva(df_unico, "_tempo_para_tratamento", vresp="_RESFINAL", max_classes=10)
ug.descritiva_metrica(df_unico, "_tempo_para_tratamento", vresp="_RESFINAL")
# ug.plot_boxplot_for_variables(df_unico , ['_tempo_para_tratamento' ])

ug.descritiva(df_unico, "PRITRATH_1", vresp="_RESFINAL", max_classes=10)
ug.descritiva(df_unico, "PRITRATH_NrTratamentos", vresp="_RESFINAL", max_classes=10)


ug.descritiva(df_unico, "ESTADIAM", vresp="_RESFINAL", max_classes=23)
ug.descritiva(df_unico, "IDADE", vresp="_RESFINAL", max_classes=10)

b = uf.search_for_categorical_variables(df_unico)











#Contagem dos nao informados (9)
colunas = ['DTPRICON','ESTADIAM','HISTFAMC','INSTRUC','LATERALI','ORIENC','PRITRATH_1','RACACOR','TABAGISM']
a_tab = pd.DataFrame()
for var in colunas:
    df = df_unico.drop(
            df_unico[df_unico[var] == "9"].index,
            inplace=False,
        )
    i = df_unico.shape[0] -  df.shape[0] 
    a_tab.at[var , "quant"] = i
    a_tab = a_tab.reset_index()
