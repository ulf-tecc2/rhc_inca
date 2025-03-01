# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Analise das inconsistencias.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: Ulf Bergmann


"""

# =============================================================================
# Definicao de inconsistencia (variavel VAR)
#     - valores de VAR fora do dominio especifico 
#     - valores de VAR inconsistente com outras variaveis
# =============================================================================




import pandas as pd

import sys
sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")


import funcoes as f
from funcoes import Log


import funcoes_ulf as uf
import bib_graficos as ug

df_inicial = f.leitura_arquivo_csv("BaseConsolidada")
# df_apos_tipos = f.leitura_arquivo_parquet("BaseInicial")
df_apos_analise = f.leitura_arquivo_parquet("BaseSanitizada")
# df_apos_transf = f.leitura_arquivo_parquet("BaseTransfor")
# df_apos_extracao = f.leitura_arquivo_parquet("BaseModelagem")

df = df_apos_analise
df_validos = f.filtrar_registros_validos(df)

def verifica_TNM_PTNM(df , a_var = 'TNM'):
    a_result_list = []
    
    #identificar valores fora do domínio => invalido e demais
    a_analise = '_Analise' + a_var
    a_inconsistencia = f'_{a_var}_Inconsistente'
    
    df[a_inconsistencia] = False
    
    df.loc[(df[a_analise] == 'demais') | (df[a_analise] == 'invalido') , a_inconsistencia] = True 

    q_inconsistente = df.loc[(df[a_analise] == 'demais') | (df[a_analise] == 'invalido')].shape[0]
   
    if q_inconsistente > 0:
        a_dict = {
            'var' : a_var ,
            'criterio' : 'Fora do dominio' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente
            }    
        a_result_list.append(a_dict)
    
    #verificar casos que tem TNM => exato incompleto e e´ pediatrico
    df.loc[((df[a_analise] == 'exato') | (df[a_analise] == 'incompleto')) & (df['IDADE'] < 20) , a_inconsistencia] = True
    q_inconsistente = df.loc[((df[a_analise] == 'exato') | (df[a_analise] == 'incompleto')) & (df['IDADE'] < 20)].shape[0]
    
    if q_inconsistente > 0:
        a_dict = {
            'var' : a_var ,
            'criterio' : 'Inconsistente com Tumor Pediatrico' ,
            'referencia' : f'Tumores pediatricos não tem {a_var}' , 
            'quantidade' : q_inconsistente
            }
        a_result_list.append(a_dict)
    
    # Hemato 'nao se aplica - Hemato'
    df.loc[((df[a_analise] == 'exato') | (df[a_analise] == 'incompleto')) & (df['_AnaliseLOCTUDET_tipo'] == 'Hemato') , a_inconsistencia] = True
    q_inconsistente = df.loc[((df[a_analise] == 'exato') | (df[a_analise] == 'incompleto')) & (df['_AnaliseLOCTUDET_tipo'] == 'Hemato') ].shape[0]

    if q_inconsistente > 0:
        a_dict = {
            'var' : a_var ,
            'criterio' : 'Inconsistente com Tumor Hematologico' ,
            'referencia' : f'Tumores hematologicos não tem {a_var}' , 
            'quantidade' : q_inconsistente
            }
        a_result_list.append(a_dict)
        
    # df[a_inconsistencia].fillna(False , inplace = True)

    return df , a_result_list

def verifica_ESTADIAM(df):
    a_result_list = []
    
    a_inconsistencia = '_ESTADIAM_Inconsistente'
    
    df[a_inconsistencia] = False
    
    df.loc[(df['_AnaliseESTADIAM'] == 'demais') , a_inconsistencia] = True 
    q_inconsistente = df.loc[(df['_AnaliseESTADIAM'] == 'demais') ].shape[0]
    
    a_dict = {
        'var' : 'ESTADIAM' ,
        'criterio' : 'Fora do dominio' ,
        'referencia' : 'Manual do RHC' , 
        'quantidade' : q_inconsistente
        }    
    a_result_list.append(a_dict)
    

    #verificar casos que tem ESTADIAM => exato  e e´ pediatrico
    df.loc[((df['_AnaliseESTADIAM'] == 'exato')) & (df['IDADE'] < 20) , a_inconsistencia] = True
    q_inconsistente = df.loc[(df['_AnaliseESTADIAM'] == 'exato') & (df['IDADE'] < 20)].shape[0]
    
    if q_inconsistente > 0:
        a_dict = {
            'var' : 'ESTADIAM' ,
            'criterio' : 'Inconsistente com Tumor Pediatrico' ,
            'referencia' : f'Tumores pediatricos não tem ESTADIAM' , 
            'quantidade' : q_inconsistente
            }
        a_result_list.append(a_dict)


    #verificar casos que tem ESTADIAM => exato  e e´ pediatrico
    df.loc[((df['_AnaliseESTADIAM'] == 'exato')) & (df['_AnaliseLOCTUDET_tipo'] == 'Hemato') , a_inconsistencia] = True
    q_inconsistente = df.loc[(df['_AnaliseESTADIAM'] == 'exato') & (df['_AnaliseLOCTUDET_tipo'] == 'Hemato')].shape[0]
    
    if q_inconsistente > 0:
        a_dict = {
            'var' : 'ESTADIAM' ,
            'criterio' : 'Inconsistente com Tumor Hematologico' ,
            'referencia' : f'Tumores hematologicos não tem ESTADIAM' , 
            'quantidade' : q_inconsistente
            }
        a_result_list.append(a_dict)

    return df , a_result_list



a_result_list = []

df = df_unico

df , a_list = verifica_TNM_PTNM(df , a_var = 'TNM')  
a_result_list = a_result_list + a_list

df , a_list = verifica_TNM_PTNM(df , a_var = 'PTNM')  
a_result_list = a_result_list + a_list

df , a_list = verifica_ESTADIAM(df)  
a_result_list = a_result_list + a_list

a_response = pd.DataFrame(a_result_list)

a = df[['ESTADIAM' , '_ESTADIAM_Inconsistente' ]].sample(100)


    