# -*- coding: utf-8 -*-

"""Registros Hospitalares de Câncer (RHC) - Geração dos indicadores de incompletude e inconsistência.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: ulf Bergmann

"""



import pandas as pd
import sys
sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")


import funcoes as f
from funcoes import Log


import funcoes_ulf as uf
import bib_graficos as ug

#%% INCONSISTENCIAS

def identifica_inconsistencia_TNM_PTNM(df , a_var = 'TNM'):
    a_result_list = []
    
    #identificar valores fora do domínio => invalido e demais
    a_analise = '_Analise' + a_var
    
    df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
    
    cod_tipo = 'dominio - regra de formato'
    df.loc[(df[a_analise] == 'demais') | (df[a_analise] == 'invalido') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 
    q_inconsistente = df.loc[(df[a_analise] == 'demais') | (df[a_analise] == 'invalido')].shape[0]
   
    if q_inconsistente > 0:
        a_dict = {
            'var' : a_var ,
            'criterio' : 'Inconsistencia' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "demais" ou "invalido"'
            }    
        a_result_list.append(a_dict)
     
# =============================================================================
#     NULOS e SEM INFO
# =============================================================================
    cod_tipo = 'valores invalidos (nulos)'
    q_inconsistente = df.loc[(df[a_analise] == 'nulo')].shape[0]
    if q_inconsistente > 0:
        df.loc[(df[a_analise] == 'nulo') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

        a_dict = {
            'var' : a_var ,
            'criterio' : 'Incompletude' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "nulo"'
            }    
        a_result_list.append(a_dict)

    cod_tipo = 'valores sem informacao'
    q_inconsistente = df.loc[(df[a_analise] == 'sem_info')].shape[0]
    if q_inconsistente > 0:
        df.loc[(df[a_analise] == 'sem_info') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

        a_dict = {
            'var' : a_var ,
            'criterio' : 'Incompletude' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "sem info"'
            }    
        a_result_list.append(a_dict)
    
     
    #verificar casos que tem TNM => exato incompleto e e´ pediatrico
    cod_tipo = 'Inconsistente com pediatrico'
    df.loc[((df[a_analise] == 'exato') | (df[a_analise] == 'incompleto')) & (df['IDADE'] < 20) , a_inconsistencia] =  df[a_inconsistencia] + ';' + cod_tipo
    q_inconsistente = df.loc[((df[a_analise] == 'exato') | (df[a_analise] == 'incompleto')) & (df['IDADE'] < 20)].shape[0]
    
    if q_inconsistente > 0:
        a_dict = {
            'var' : a_var ,
            'criterio' : 'Inconsistencia' ,
            'referencia' : f'Tumores pediatricos não tem {a_var}' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "exato" ou "incompleto" sendo IDADE < 20'

            }
        a_result_list.append(a_dict)
 
    # Hemato 'nao se aplica - Hemato'
    cod_tipo = 'Inconsistente com hematologico'
    df.loc[((df[a_analise] == 'exato') | (df[a_analise] == 'incompleto')) & (df['_AnaliseLOCTUDET_tipo'] == 'Hemato') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
    q_inconsistente = df.loc[((df[a_analise] == 'exato') | (df[a_analise] == 'incompleto')) & (df['_AnaliseLOCTUDET_tipo'] == 'Hemato') ].shape[0]

    if q_inconsistente > 0:
        a_dict = {
            'var' : a_var ,
            'criterio' : 'Inconsistencia' ,
            'referencia' : f'Tumores hematologicos não tem {a_var}' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "exato" ou "incompleto" sendo LOCTUDET hematologico'

            }
        a_result_list.append(a_dict)
        
    # df[a_inconsistencia].fillna(False , inplace = True)

    return df , a_result_list



def cria_coluna_inconsistencia(df , var):
    a_inconsistencia = f.get_nome_coluna_indicador_variavel(var)
    
    # ja existe a coluna
    if a_inconsistencia not in df.columns:
        df[a_inconsistencia] = ''
        
    return df , a_inconsistencia


def identifica_inconsistencia_ESTADIAM(df):
    a_result_list = []
    
    df , a_inconsistencia = cria_coluna_inconsistencia( df , 'ESTADIAM')
    
    cod_tipo = 'dominio - regra de formato'
    df.loc[(df['_AnaliseESTADIAM'] == 'demais') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
    q_inconsistente = df.loc[(df['_AnaliseESTADIAM'] == 'demais') ].shape[0]

    
    a_dict = {
        'var' : 'ESTADIAM' ,
        'criterio' : 'Inconsistencia' ,
        'referencia' : 'Manual do RHC' , 
        'quantidade' : q_inconsistente,
        'codigo tipo' : cod_tipo,
        'detalhamento' : 'Análise "demais"'

        }    
    a_result_list.append(a_dict)
    
# =============================================================================
#     NULOS e SEM INFO
# =============================================================================
    cod_tipo = 'valores invalidos (nulos)'
    q_inconsistente = df.loc[(df['_AnaliseESTADIAM'] == 'nulo')].shape[0]
    if q_inconsistente > 0:
        df.loc[(df['_AnaliseESTADIAM'] == 'nulo') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

        a_dict = {
            'var' : 'ESTADIAM' ,
            'criterio' : 'Incompletude' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "nulo"'
            }    
        a_result_list.append(a_dict)

    cod_tipo = 'valores sem informacao'
    q_inconsistente = df.loc[(df['_AnaliseESTADIAM'] == 'sem_info')].shape[0]
    if q_inconsistente > 0:
        df.loc[(df['_AnaliseESTADIAM'] == 'sem_info') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

        a_dict = {
            'var' : 'ESTADIAM' ,
            'criterio' : 'Incompletude' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "sem info"'
            }    
        a_result_list.append(a_dict)

    #verificar casos que tem ESTADIAM => exato  e e´ pediatrico
    cod_tipo = 'Inconsistente com pediatrico'
    df.loc[((df['_AnaliseESTADIAM'] == 'exato')) & (df['IDADE'] < 20) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
    q_inconsistente = df.loc[(df['_AnaliseESTADIAM'] == 'exato') & (df['IDADE'] < 20)].shape[0]
    
    if q_inconsistente > 0:
        a_dict = {
            'var' : 'ESTADIAM' ,
            'criterio' : 'Inconsistencia' ,
            'referencia' : 'Tumores pediatricos não tem ESTADIAM' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "exato" e IDADE < 20'

            }
        a_result_list.append(a_dict)


    #verificar casos que tem ESTADIAM => exato  e e´ pediatrico
    cod_tipo = 'Inconsistente com hematologico'
    df.loc[((df['_AnaliseESTADIAM'] == 'exato')) & (df['_AnaliseLOCTUDET_tipo'] == 'Hemato') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
    q_inconsistente = df.loc[(df['_AnaliseESTADIAM'] == 'exato') & (df['_AnaliseLOCTUDET_tipo'] == 'Hemato')].shape[0]
    
    if q_inconsistente > 0:
        a_dict = {
            'var' : 'ESTADIAM' ,
            'criterio' : 'Inconsistencia' ,
            'referencia' : 'Tumores hematologicos não tem ESTADIAM' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "exato" sendo LOCTUDET hematologico'

            }
        a_result_list.append(a_dict)

    return df , a_result_list

# def identifica_inconsistencia_valores_invalidos(df):
    
#     nan_values = {
#         "ALCOOLIS": ["0"],
#         "BASDIAGSP": [""],
#         "BASMAIMP": ["0"],
#         "CLIATEN": ["0" ,'51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98'],
#         "CLITRAT": ["0" ,'51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98'],
#         "DIAGANT": ["0"],
#         "DTDIAGNO": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
#         "DTTRIAGE": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
#         "DATAPRICON": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
#         "DATAOBITO": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
#         "DATAINITRT": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
#         "ESTADIAM": ["nan"],
#         "ESTADRES": [ "77", "nan"],
#         "ESTADIAG": ["nan"],
#         "ESTCONJ": ["0"],
#         "ESTDFIMT": ["0"],
#         "EXDIAG": ["0", "<NA>"],
#         "HISTFAMC": ["0"],
#         "INSTRUC": [""],
#         "LATERALI": ["0"],
#         "LOCALNAS": [ "nan"],
#         "LOCTUDET": ["nan", "D46", "E05", "N62", "C"],
#         "LOCTUPRI": ["nan", "D46", "E05", "N62", "C .", "C .0"],
#         "LOCTUPRO": ["", ",", ".", "9.", "nan"],
#         "MAISUMTU": ["0"],
#         "ORIENC": ["0"],
#         "OUTROESTA": [
#             "",
#             "nan",
#             ",",
#             ".",
#             "[",
#             "]",
#             "+",
#             "\\",
#             "'",
#             "/",
#             "//",
#             "=",
#         ],
#         "PTNM": [""],
#         "RACACOR": ["99"],
#         "RZNTR": [""],
#         "SEXO": ["0", "3"],
#         "TABAGISM": ["0"],
#         "TIPOHIST": ["nan", "/", "/3)", "1  /", "1   /", "99999", "C090/3"],
#     	"PROCEDEN" : ['nan' , 'SC' , 'RS' , '9999999' , '999' , '99' , '93228' , '8888888' , '7777777'],

#     }
    
#     a_result_list = []
#     cod_tipo = 'dominio - valores iniciais invalidos'
    
#     for a_var in nan_values:
#         df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
        
#         df.loc[ df[a_var].isin(nan_values[a_var]) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
#         q_inconsistente =  df.loc[ df[a_var].isin(nan_values[a_var])].shape[0]
        
#         nome_analise = "_Analise" + a_var
#         b = pd.Series({"Regra Nulo": nan_values[a_var]}, name=nome_analise)
#         a_df = pd.DataFrame(b)
#         a_df.index.names = ["Atributo"]
#         a_df.columns = [a_var]
    
#         if (q_inconsistente) > 0: 
#             a_dict = {
#                 'var' : a_var ,
#                 'criterio' : 'Fora do dominio' ,
#                 'referencia' : 'Manual RHC' , 
#                 'quantidade' : q_inconsistente,
#                 'codigo tipo' : cod_tipo,
#                 'detalhamento' : nan_values[a_var]
#                 }
#             a_result_list.append(a_dict)
            
    
#     return df , a_result_list



def identifica_inconsistencia_anos(df):
    """Identifica valores invalidos de anos.

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    lista_colunas = ["ANOPRIDI", "ANTRI", "DTPRICON", "DTINITRT"]

    a_result_list = []
    for a_var in lista_colunas:
        df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
        
        cod_tipo = 'dominio'
        df.loc[(~df[a_var].isnull()) & ((df[a_var] < 1900) | (df[a_var] > 2023)) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
        q_inconsistente = df.loc[(~df[a_var].isnull()) & ((df[a_var] < 1900) | (df[a_var] > 2023))].shape[0]
        
        if (q_inconsistente) > 0: 
            a_dict = {
                'var' : a_var ,
                'criterio' : 'Inconsistencia' ,
                'referencia' : 'Manual do RHC' , 
                'quantidade' : q_inconsistente,
                'codigo tipo' : cod_tipo,
                'detalhamento' : 'Ano < 1900 OU Ano > 2023'
                }    
            a_result_list.append(a_dict)
        
    return df , a_result_list

def identifica_inconsistencia_datas(df):
    """Identifica inconsistencia de datas.

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    lista_colunas = ["DTDIAGNO", "DTTRIAGE", "DATAPRICON", "DATAOBITO" , 'DATAINITRT']

    a_result_list = []
    for a_var in lista_colunas:
        df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
        
        cod_tipo = 'dominio'
        df.loc[(~df[a_var].isnull()) & ((df[a_var].dt.year < 1900) | (df[a_var].dt.year > 2023)) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
        q_inconsistente = df.loc[(~df[a_var].isnull()) & ((df[a_var].dt.year < 1900) | (df[a_var].dt.year > 2023))].shape[0]
        
        if (q_inconsistente) > 0: 
            a_dict = {
                'var' : a_var ,
                'criterio' : 'Inconsistencia' ,
                'referencia' : 'Manual do RHC' , 
                'quantidade' : q_inconsistente,
                'codigo tipo' : cod_tipo,
                'detalhamento' : 'Data < 1900 OU Data > 2023'
                }    
            a_result_list.append(a_dict)
        
    return df , a_result_list
        
def identifica_inconsistencia_IDADE(df):
    """Identifica inconsistencia de idade. Valida idade entre 0 e 110 anos.

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    a_result_list = []
    a_var = 'IDADE'
    df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
    
    cod_tipo = 'dominio'
    df.loc[(~df[a_var].isnull()) & ((df[a_var] < 0) | (df[a_var] > 110)) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
    q_inconsistente = df.loc[(~df[a_var].isnull()) & ((df[a_var] < 0) | (df[a_var] > 110)) ].shape[0]
    
    if (q_inconsistente) > 0: 
        a_dict = {
            'var' : a_var ,
            'criterio' : 'Inconsistencia' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'IDADE < 0 OU IDADE > 110'
            }    
        a_result_list.append(a_dict)
        
    return df , a_result_list

def identifica_inconsistencia_tratamento(df):
    """Identifica inconsistencia no tratamento.

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    a_result_list = []
    
    #Tratamento realizado: _AnalisePRIMTRATH exato  E   Tem razao para não tratamento - RZNTR
    #Optei pela inconsistencia ser apenas em uma variavel
    a_var = 'RZNTR'
    df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
    cod_tipo = 'entre variaveis'
    df.loc[(df['_AnalisePRITRATH'] == 'exato') & (df['_AnaliseRZNTR'] == 'nao_tratou') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
    q_inconsistente = df.loc[(df['_AnalisePRITRATH'] == 'exato') & (df['_AnaliseRZNTR'] == 'nao_tratou') ].shape[0]
    
    if (q_inconsistente) > 0: 
        a_dict = {
            'var' : a_var ,
            'criterio' : 'Inconsistencia' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Possui PRITRATH mas possui RZNTR'
            }    
        a_result_list.append(a_dict)
        
    #Nao teve tratamento realizado: # _AnalisePRIMTRATH nulo  e Tem data de inicio de tratamento - DATAINITRT - DTINITRT
    a_var = 'DATAINITRT'
    df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
    cod_tipo = 'entre variaveis'
    df.loc[(df['_AnalisePRITRATH'] == 'nulo') & (~df['DATAINITRT'].isnull())  , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
    q_inconsistente =df.loc[(df['_AnalisePRITRATH'] == 'nulo') & (~df['DATAINITRT'].isnull()) ].shape[0]
    
    if (q_inconsistente) > 0: 
        a_dict = {
            'var' : a_var ,
            'criterio' : 'Inconsistencia' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Não possui PRITRATH mas possui DATAINITRT'
            }    
        a_result_list.append(a_dict)        
        
    #Nao teve tratamento realizado: # _AnalisePRIMTRATH nulo  e Tem data de inicio de tratamento - DATAINITRT - DTINITRT
    a_var = 'DTINITRT'
    df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
    cod_tipo = 'entre variaveis'
    df.loc[(df['_AnalisePRITRATH'] == 'nulo') & (~df['DTINITRT'].isnull())  , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
    q_inconsistente =df.loc[(df['_AnalisePRITRATH'] == 'nulo') & (~df['DTINITRT'].isnull()) ].shape[0]
    
    if (q_inconsistente) > 0: 
        a_dict = {
            'var' : a_var ,
            'criterio' : 'Inconsistencia' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Possui PRITRATH mas possui DTINITRT'
            }    
        a_result_list.append(a_dict)  
        
        
    #Nao teve tratamento realizado: # _AnalisePRIMTRATH nulo  e Teve resultado - ESTDFIMT de 1 a 6
    a_var = 'ESTDFIMT'
    df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
    cod_tipo = 'entre variaveis'
    df.loc[(df['_AnalisePRITRATH'] == 'nulo') & df['ESTDFIMT'].isin(range(1,6)) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
    q_inconsistente =df.loc[(df['_AnalisePRITRATH'] == 'nulo') & df['ESTDFIMT'].isin(range(1,6))].shape[0]
    
    if (q_inconsistente) > 0: 
        a_dict = {
            'var' : a_var ,
            'criterio' : 'Inconsistencia' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Possui PRITRATH nulo mas possui ESTDFIMT'
            }    
        a_result_list.append(a_dict)  
    
    return df , a_result_list

def identifica_inconsistencia_ordem_datas(df , var_1 , var_2):
    """Identifica inconsistencia entre as datas (se var_2 foi antes de var_1).

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    a_result_list = []
    
    a_var = var_1
    df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
    cod_tipo = 'entre variaveis'
    
    df.loc[
        (~ df[var_2].isnull()) &
        (~ df[var_1].isnull()) &
        (df[var_2] < df[var_1]) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
    
    q_inconsistente = df.loc[
            (~ df[var_2].isnull()) &
            (~ df[var_1].isnull()) &
            (df[var_2] < df[var_1])].shape[0]
    
    if (q_inconsistente) > 0: 
        a_dict = {
            'var' : a_var ,
            'criterio' : 'Inconsistencia' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : f'Datas fora de ordem ({var_2} antes de {var_1}'
            }    
        a_result_list.append(a_dict)
    
    return df , a_result_list

def identifica_inconsistencia_LOCTU(df):
    """Identifica valores invalidos em LOCTUPRI , LOCTUDET , LOCTUPRO.

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    a_result_list = []
    
    lista_colunas = ['LOCTUPRI' , 'LOCTUDET' , 'LOCTUPRO']
    cod_tipo = 'dominio - regra de formato'
        
    for a_var in lista_colunas:
        df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
        nome_analise = "_Analise" + a_var
        
        df.loc[(df[nome_analise] == 'incompleto') | (df[nome_analise] == 'demais'), a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
        q_inconsistente = df.loc[(df[nome_analise] == 'incompleto') | (df[nome_analise] == 'demais') ] .shape[0]
        
        if (q_inconsistente) > 0: 
            a_dict = {
                'var' : a_var ,
                'criterio' : 'Inconsistencia' ,
                'referencia' : 'Manual do RHC' , 
                'quantidade' : q_inconsistente,
                'codigo tipo' : cod_tipo,
                'detalhamento' : 'Analise = incompleto OU demais'
                }    
            a_result_list.append(a_dict)
            
        # =============================================================================
        #     NULOS e SEM INFO
        # =============================================================================
        cod_tipo = 'valores invalidos (nulos)'
        q_inconsistente = df.loc[(df[nome_analise] == 'nulo')].shape[0]
        if q_inconsistente > 0:
            df.loc[(df[nome_analise] == 'nulo') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

            a_dict = {
                'var' : a_var ,
                'criterio' : 'Incompletude' ,
                'referencia' : 'Manual do RHC' , 
                'quantidade' : q_inconsistente,
                'codigo tipo' : cod_tipo,
                'detalhamento' : 'Análise "nulo"'
                }    
            a_result_list.append(a_dict)

        cod_tipo = 'valores sem informacao'
        q_inconsistente = df.loc[(df[nome_analise] == 'sem_info')].shape[0]
        if q_inconsistente > 0:
            df.loc[(df[nome_analise] == 'sem_info') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

            a_dict = {
                'var' : a_var ,
                'criterio' : 'Incompletude' ,
                'referencia' : 'Manual do RHC' , 
                'quantidade' : q_inconsistente,
                'codigo tipo' : cod_tipo,
                'detalhamento' : 'Análise "sem info"'
                }    
            a_result_list.append(a_dict)
        
    return df , a_result_list


def identifica_inconsistencia_TIPOHIST(df):
    a_result_list = []
    
    df , a_inconsistencia = cria_coluna_inconsistencia( df , 'TIPOHIST')
    
    cod_tipo = 'dominio - regra de formato'
    df.loc[(df['_AnaliseTIPOHIST'] == 'demais') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
    q_inconsistente = df.loc[(df['_AnaliseTIPOHIST'] == 'demais') ].shape[0]

    
    a_dict = {
        'var' : 'TIPOHIST' ,
        'criterio' : 'Inconsistencia' ,
        'referencia' : 'Manual do RHC' , 
        'quantidade' : q_inconsistente,
        'codigo tipo' : cod_tipo,
        'detalhamento' : 'Análise "demais"'

        }    
    a_result_list.append(a_dict)
    
    
# =============================================================================
#     NULOS e SEM INFO
# =============================================================================
    cod_tipo = 'valores invalidos (nulos)'
    q_inconsistente = df.loc[(df['_AnaliseTIPOHIST'] == 'nulo')].shape[0]
    if q_inconsistente > 0:
        df.loc[(df['_AnaliseTIPOHIST'] == 'nulo') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

        a_dict = {
            'var' : 'TIPOHIST' ,
            'criterio' : 'Incompletude' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "nulo"'
            }    
        a_result_list.append(a_dict)

    cod_tipo = 'valores sem informacao'
    q_inconsistente = df.loc[(df['_AnaliseTIPOHIST'] == 'sem_info')].shape[0]
    if q_inconsistente > 0:
        df.loc[(df['_AnaliseTIPOHIST'] == 'sem_info') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

        a_dict = {
            'var' : 'TIPOHIST' ,
            'criterio' : 'Incompletude' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "sem info"'
            }    
        a_result_list.append(a_dict)
        
    return df , a_result_list


def identifica_inconsistencia_municipios(df):
    a_result_list = []
    
    a_var = 'PROCEDEN'
    a_analise = '_Analise' + a_var
    df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
    
    cod_tipo = 'dominio - regra de formato'
    df.loc[(df[a_analise] == 'demais') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
    q_inconsistente = df.loc[(df[a_analise] == 'demais') ].shape[0]
    a_dict = {
        'var' : a_var ,
        'criterio' : 'Inconsistencia' ,
        'referencia' : 'Manual do RHC' , 
        'quantidade' : q_inconsistente,
        'codigo tipo' : cod_tipo,
        'detalhamento' : 'Análise "demais"'

        }    
    a_result_list.append(a_dict)
    
    cod_tipo = 'valores invalidos (nulos)'
    q_inconsistente = df.loc[(df[a_analise] == 'nulo')].shape[0]
    if q_inconsistente > 0:
        df.loc[(df[a_analise] == 'nulo') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 
        a_dict = {
            'var' : a_var ,
            'criterio' : 'Incompletude' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "nulo"'
            }    
        a_result_list.append(a_dict)

    cod_tipo = 'valores sem informacao'
    q_inconsistente = df.loc[(df[a_analise] == 'sem_info')].shape[0]
    if q_inconsistente > 0:
        df.loc[(df[a_analise] == 'sem_info') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

        a_dict = {
            'var' : a_var ,
            'criterio' : 'Incompletude' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "sem info"'
            }    
        a_result_list.append(a_dict)
        
    a_var = 'MUUH'
    a_analise = '_Analise' + a_var
    df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
    
    cod_tipo = 'dominio - regra de formato'
    df.loc[(df[a_analise] == 'demais') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
    q_inconsistente = df.loc[(df[a_analise] == 'demais') ].shape[0]
    a_dict = {
        'var' : a_var ,
        'criterio' : 'Inconsistencia' ,
        'referencia' : 'Manual do RHC' , 
        'quantidade' : q_inconsistente,
        'codigo tipo' : cod_tipo,
        'detalhamento' : 'Análise "demais"'

        }    
    a_result_list.append(a_dict)
    
    cod_tipo = 'valores invalidos (nulos)'
    q_inconsistente = df.loc[(df[a_analise] == 'nulo')].shape[0]
    if q_inconsistente > 0:
        df.loc[(df[a_analise] == 'nulo') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 
        a_dict = {
            'var' : a_var ,
            'criterio' : 'Incompletude' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "nulo"'
            }    
        a_result_list.append(a_dict)

    cod_tipo = 'valores sem informacao'
    q_inconsistente = df.loc[(df[a_analise] == 'sem_info')].shape[0]
    if q_inconsistente > 0:
        df.loc[(df[a_analise] == 'sem_info') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

        a_dict = {
            'var' : a_var ,
            'criterio' : 'Incompletude' ,
            'referencia' : 'Manual do RHC' , 
            'quantidade' : q_inconsistente,
            'codigo tipo' : cod_tipo,
            'detalhamento' : 'Análise "sem info"'
            }    
        a_result_list.append(a_dict)
        
    return df , a_result_list

#%% INCOMPLETUDE

def identifica_incompletude_variaveis(df , lista_var):
    
    nan_values = {
        "ALCOOLIS": ["0"],
        "BASDIAGSP": [""],
        "BASMAIMP": ["0"],
        "CLIATEN": ["0" ,'51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98'],
        "CLITRAT": ["0" ,'51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98'],
        "DIAGANT": ["0"],
        "DTDIAGNO": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
        "DTTRIAGE": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
        "DATAPRICON": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
        "DATAOBITO": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
        "DATAINITRT": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
        # "ESTADIAM": ["nan"],
        "ESTADRES": [ "77", "nan"],
        "ESTADIAG": ["nan"],
        "ESTCONJ": ["0"],
        "ESTDFIMT": ["0"],
        "EXDIAG": ["0", "<NA>"],
        "HISTFAMC": ["0"],
        "INSTRUC": [""],
        "LATERALI": ["0"],
        "LOCALNAS": [ "nan"],
        # "LOCTUDET": ["nan", "D46", "E05", "N62", "C"],
        # "LOCTUPRI": ["nan", "D46", "E05", "N62", "C .", "C .0"],
        # "LOCTUPRO": ["", ",", ".", "9.", "nan"],
        "MAISUMTU": ["0"],
        "ORIENC": ["0"],
        "OUTROESTA": ["","nan" , "," , ".", "[", "]", "+", "\\", "'", "/", "//", "=",],
        # "PTNM": [""],
        "RACACOR": ["99"],
        "RZNTR": [""],
        "SEXO": ["0", "3"],
        "TABAGISM": ["0"],
        "TIPOHIST": ["nan", "/", "/3)", "1  /", "1   /", "99999", "C090/3"],
    }
    
    informacao_ignorada = {
        "ALCOOLIS": ["9"],
        "BASDIAGSP": ["9"],
        "CLIATEN": ["99", "88"],
        "CLITRAT": ["99", "88"],
        "BASMAIMP": ["9"],
        "DIAGANT": ["9"],
        "ESTADIAG": ["9"],
        # "ESTADIAM": ["99", "88"],
        "ESTADRES": ["99"],
        "ESTCONJ": ["9"],
        "ESTDFIMT": ["9"],
        "EXDIAG": ["9"],
        "HISTFAMC": ["9"],
        "INSTRUC": ["9"],
        "LATERALI": ["9"],
        "LOCALNAS": ["99"],
        # "LOCTUPRO": [ "9"],
        "OCUPACAO": ["999", "9999"],
        "ORIENC": ["9"],
        "OUTROESTA": ["99", "88"],
        "PRITRATH": ["9"],
        # "PTNM": ["9"],
        "RACACOR": ["9"],
        "RZNTR": ["9"],
        "TABAGISM": ["9"],
    }
    
    a_result_list = []
    
    
    for a_var in lista_var:
        if a_var in nan_values.keys():
            
            q_inconsistente =  df.loc[ df[a_var].isin(nan_values[a_var])].shape[0]
            if q_inconsistente > 0:
                cod_tipo = 'valores invalidos (nulos)'
                df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
                df.loc[ df[a_var].isin(nan_values[a_var]) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
            
                nome_analise = "_Analise" + a_var
                b = pd.Series({"Regra Nulo": nan_values[a_var]}, name=nome_analise)
                a_df = pd.DataFrame(b)
                a_df.index.names = ["Atributo"]
                a_df.columns = [a_var]
        
                a_dict = {
                    'var' : a_var ,
                    'criterio' : 'Incompletude' ,
                    'referencia' : 'Manual RHC' , 
                    'quantidade' : q_inconsistente,
                    'codigo tipo' : cod_tipo,
                    'detalhamento' : nan_values[a_var]
                    }
                a_result_list.append(a_dict)

        if a_var in informacao_ignorada.keys():
        
            q_inconsistente =  df.loc[ df[a_var].isin(informacao_ignorada[a_var])].shape[0]
            if q_inconsistente > 0:
                cod_tipo = 'valores sem informacao'
                df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
                df.loc[ df[a_var].isin(informacao_ignorada[a_var]) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
            
                nome_analise = "_Analise" + a_var
                b = pd.Series({"Regra Sem Informacao": informacao_ignorada[a_var]}, name=nome_analise)
                a_df = pd.DataFrame(b)
                a_df.index.names = ["Atributo"]
                a_df.columns = [a_var]
        
                a_dict = {
                    'var' : a_var ,
                    'criterio' : 'Incompletude' ,
                    'referencia' : 'Manual RHC' , 
                    'quantidade' : q_inconsistente,
                    'codigo tipo' : cod_tipo,
                    'detalhamento' : informacao_ignorada[a_var]
                    }
                a_result_list.append(a_dict)

    return df , a_result_list

def identificar_inconsistencia_incompletude(df):
    
    a_result_list = []
    
    df , a_list = identifica_inconsistencia_TNM_PTNM(df , a_var = 'TNM')  
    a_result_list = a_result_list + a_list
    df , a_list = identifica_inconsistencia_TNM_PTNM(df , a_var = 'PTNM')  
    a_result_list = a_result_list + a_list
    print(log.logar_acao_realizada("Identificacao de Inconsistencias", "Analisar TNM e PTNM", ""))
    
    df , a_list = identifica_inconsistencia_ESTADIAM(df)  
    a_result_list = a_result_list + a_list
    print(log.logar_acao_realizada("Identificacao de Inconsistencias", "Analisar ESTADIAM", ""))

    df , a_list = identifica_inconsistencia_anos(df)  
    a_result_list = a_result_list + a_list
    print(log.logar_acao_realizada("Identificacao de Inconsistencias", "Analisar variaveis de anos", ""))

    df , a_list = identifica_inconsistencia_datas(df)  
    a_result_list = a_result_list + a_list
    print(log.logar_acao_realizada("Identificacao de Inconsistencias", "Analisar variaveis de datas", ""))
    
    df , a_list = identifica_inconsistencia_IDADE(df)
    a_result_list = a_result_list + a_list
    print(log.logar_acao_realizada("Identificacao de Inconsistencias", "Analisar IDADE", ""))

    df , a_list = identifica_inconsistencia_ordem_datas(df , var_1 = 'DATAPRICON' , var_2 = 'DATAINITRT' )
    a_result_list = a_result_list + a_list
    print(log.logar_acao_realizada("Identificacao de Inconsistencias", "Analisar ordem de datas: DATAPRICON e DATAINITRT", ""))

    df , a_list = identifica_inconsistencia_ordem_datas(df , var_1 = 'DATAINITRT' , var_2 = 'DATAOBITO' )
    a_result_list = a_result_list + a_list
    print(log.logar_acao_realizada("Identificacao de Inconsistencias", "Analisar ordem de datas: DATAINITRT e DATAOBITO", ""))
    
    df , a_list = identifica_inconsistencia_ordem_datas(df , var_1 = 'DTDIAGNO' , var_2 = 'DATAOBITO' )
    a_result_list = a_result_list + a_list
    print(log.logar_acao_realizada("Identificacao de Inconsistencias", "Analisar ordem de datas: DTDIAGNO e DATAOBITO", ""))
    
    df , a_list = identifica_inconsistencia_ordem_datas(df , var_1 = 'DATAPRICON' , var_2 = 'DATAOBITO' )
    a_result_list = a_result_list + a_list
    print(log.logar_acao_realizada("Identificacao de Inconsistencias", "Analisar ordem de datas: DATAPRICON e DATAOBITO", ""))
    
    df , a_list = identifica_inconsistencia_LOCTU(df)
    a_result_list = a_result_list + a_list
    print(log.logar_acao_realizada("Identificacao de Inconsistencias", "Analisar as variaveis LOCTU", ""))
   
    df , a_list = identifica_inconsistencia_TIPOHIST(df)
    a_result_list = a_result_list + a_list
    print(log.logar_acao_realizada("Identificacao de Inconsistencias", "Analisar as variaveis TIPOHIST", ""))
    
    df , a_list = identifica_inconsistencia_municipios(df)
    a_result_list = a_result_list + a_list
    print(log.logar_acao_realizada("Identificacao de Inconsistencias", "Analisar as variaveis Municipios", ""))
    
    lista_var = [
     'ALCOOLIS',
     'BASDIAGSP',
     'BASMAIMP',
     'CLIATEN',
     'CLITRAT',
     'DATAINITRT',
     'DATAOBITO',
     'DATAPRICON',
     'DIAGANT',
     'DTDIAGNO',
     'DTTRIAGE',
     'ESTADIAG',
     'ESTADIAM',
     'ESTADRES',
     'ESTCONJ',
     'ESTDFIMT',
     'EXDIAG',
     'HISTFAMC',
     'INSTRUC',
     'LATERALI',
     'LOCALNAS',
     'LOCTUDET',
     'LOCTUPRI',
     'LOCTUPRO',
     'MAISUMTU',
     'OCUPACAO',
     'ORIENC',
     'OUTROESTA',
     'PRITRATH',
     'PROCEDEN',
     'PTNM',
     'RACACOR',
     'RZNTR',
     'SEXO',
     'TABAGISM',
     'TIPOHIST']

    df , a_list = identifica_incompletude_variaveis(df, lista_var)  
    a_result_list = a_result_list + a_list
        
    a_response = pd.DataFrame(a_result_list)
    
    return df , a_response

#%% INICIO - MAIN - IDENTIFICACAO INCONSITENCIAS
# Carrega a base ja analisada e gera os indicadores da Base Completa e da Base de Casos Validos
if __name__ == "__main__":
    log = Log()
    log.carregar_log("log_BaseAnalisada")
    df = f.leitura_arquivo_parquet("BaseAnalisada")

    print(
        log.logar_acao_realizada(
            "Carga Dados",
            "Carregamento da base dos dados a serem analisados - Casos completos",
            df.shape[0],
        )
    )

    df , a_response = identificar_inconsistencia_incompletude(df)
    a=df.columns
    
    a_nome_arquivo = "AnaliseVariaveisBaseCompleta"
    f.salvar_excel_conclusao(a_response, a_nome_arquivo)
    
    print(log.logar_acao_realizada("Identificacao de Incompletudes", "Analisar valores incompletos e inconsistentes para Base Completa", df.shape[0]))
    
    log.salvar_log("log_BaseIndicadores")
    f.salvar_parquet(df, "BaseIndicadores")
    
    print(log.logar_acao_realizada("Salvar Base", "BaseIndicadores salva", df.shape[0]))


    df = f.filtrar_registros_validos(df)
    print(
        log.logar_acao_realizada(
            "Carga Dados",
            "Filtragem da base dos dados a serem analisados - Casos Validos",
            df.shape[0],
        )
    )

    df , a_response = identificar_inconsistencia_incompletude(df)
    
    
    a_nome_arquivo = "AnaliseVariaveisBaseValida"
    f.salvar_excel_conclusao(a_response, a_nome_arquivo)
    
    print(log.logar_acao_realizada("Identificacao de Incompletudes", "Analisar valores incompletos e inconsistentes para Base Valida", df.shape[0]))
