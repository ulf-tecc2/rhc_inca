# -*- coding: utf-8 -*-

"""Registros Hospitalares de Câncer (RHC) - Carregamento e consolidacao inicial dos dados.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: ulf Bergmann

#Roteiro de execucao:
    1. Identifica valores nulos por variavel na Base de Dados inicial 
    2. Identifica valores nulos por variavel apos definicao dos tipos
    3. Identifica valores nulos por variavel apos analise dos dados
    4. Identifica valores invalidos por variavel apos analise dos dados
    5. Registra registros removidos na seleção da base
    6. Salvar o arquivo de indicadores

"""

import pandas as pd
from datetime import datetime

import sys
sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

import funcoes as f
from funcoes import Log

def insere_indicador_nulos(df , tipo , response):
    """Gera o indicador de registros nulos para cada variavel do DataFrame.
    
    Parameters:
        df (DataFrame): DataFrame a ser analisado
        tipo (String): tipo do df / indicador
        response (DataFrame): DataFrame com indicadores

    Returns:
        (DataFrame): DataFrame com indicadores atualizados
    """
    a = df.isnull().sum().astype('int64')   
    a_df = pd.DataFrame(a).T  
    a_df.insert(0, "Indicador", [tipo])

    return pd.concat([response , a_df] , axis = 0)

def insere_indicador_invalidos(df , response):
    """Gera o indicador de registros invalidos para cada variavel do DataFrame.
    
    Parameters:
        df (DataFrame): DataFrame a ser analisado
        response (DataFrame): DataFrame com indicadores

    Returns:
        (DataFrame):  DataFrame com indicadores atualizados
    """
    a_dict = {}
    
    a_dict['TNM'] = df_apos_analise[(df_apos_analise['AnaliseTNM'] == 'incompleto') |
                                    (df_apos_analise['AnaliseTNM'] == 'demais') ].shape[0]
    
    a_dict['LOCTUDET'] = df_apos_analise[(df_apos_analise['AnaliseLOCTUDET'] == 'incompleto') |
                                    (df_apos_analise['AnaliseLOCTUDET'] == 'demais') ].shape[0]
    
    a_dict['LOCTUPRI'] = df_apos_analise[(df_apos_analise['AnaliseLOCTUPRI'] == 'incompleto') |
                                    (df_apos_analise['AnaliseLOCTUPRI'] == 'demais') ].shape[0]
    
    a_dict['LOCTUPRO'] = df_apos_analise[(df_apos_analise['AnaliseLOCTUPRO'] == 'incompleto') |
                                    (df_apos_analise['AnaliseLOCTUPRO'] == 'demais') ].shape[0]
    
    a_dict['ESTADIAM'] = df_apos_analise[(df_apos_analise['AnaliseESTADIAM'] == 'incompleto') |
                                    (df_apos_analise['AnaliseESTADIAM'] == 'demais') ].shape[0]
    
    a_dict['ESTADIAM'] = df_apos_analise[(df_apos_analise['AnaliseESTADIAM'] == 'incompleto') |
                                    (df_apos_analise['AnaliseESTADIAM'] == 'demais') ].shape[0]
    
    a_dict['IDADE'] = df_apos_analise[df_apos_analise['IDADE'] == -1].shape[0]
    
        
    a_df = pd.DataFrame([a_dict])
    a_df.insert(0, "Indicador", ['Invalidos apos analise'])
    
    
    return pd.concat([response , a_df] , axis = 0)


def busca_data_sp_iniciou_mais_que_1_trat(df):
    """Identifica a data de quando SP iniciou o registro de mais de um tratamento.
    
    Parameters:
        df (DataFrame): DataFrame a ser pesquisado

    Returns:
        (Date):  data do inicio
    """
    
    aux_sp = df[df['UFUH'] == 'SP']
    a = aux_sp.loc[aux_sp['PRITRATH_NrTratamentos'] > 1]
    a.sort_values(by='DATAINITRT' , ascending=[True]).reset_index(drop=True)
    
    return  a['DATAINITRT'].iloc[0]



def main_indicadores_variaveis(response_df):
    """Funcao principal de geracao dos indicadores das variaveis.
    
    Parameters:
        response_df (DataFrame): DataFrame com indicadores
    Returns:
        (DataFrame):  DataFrame com indicadores atualizados       
    """ 
    response_df = insere_indicador_nulos(df = df_inicial, tipo = 'Nulos - Inicial' , response = response_df)
    
    response_df = insere_indicador_nulos(df = df_apos_tipos, tipo = 'Nulos - Após Tipos Definidos', response = response_df)
        
    response_df = insere_indicador_nulos(df = df_apos_analise, tipo = 'Nulos - Após Analise dos Dados', response = response_df)
    
    response_df = insere_indicador_invalidos(df = df_apos_analise, response = response_df)
    
    return response_df


def main_indicadores_globais():
    """Funcao principal de geracao dos indicadores globais.
    
    Returns:
        (DataFrame):  DataFrame com indicadores globais       
    """     
    a_lista = []
    
    a_dict = {'Indicador' : 'Quantidade de Registros' ,
             'Etapa' : 'Carga Inicial - Dados Brutos' ,
             'Valor' : df_inicial.shape[0]
             }
    a_lista.append(a_dict)
    
    a_dict = {'Indicador' : 'Quantidade de Registros' ,
             'Etapa' : 'Final da Analise - Dados Completos' ,
             'Valor' : df_apos_analise.shape[0]
             }
    a_lista.append(a_dict)
   
    a_dict = {'Indicador' : 'Quantidade de Registros' ,
             'Etapa' : 'Extracao - Dados de Casos Analiticos' ,
             'Valor' : df_apos_extracao.shape[0]
             }
    a_lista.append(a_dict)
    
    data_sp = busca_data_sp_iniciou_mais_que_1_trat(df_apos_transf)

    a_dict = {'Indicador' : 'Data de Inicio de mais de um trat em SP' ,
             'Etapa' : 'Final da Analise - Dados Completos' ,
             'Valor' : data_sp.strftime('%m/%Y')
             }
    a_lista.append(a_dict)
    
    a_df = pd.DataFrame(a_lista)
    
    return a_df

if __name__ == "__main__":
    log = Log()

    df_inicial = f.leitura_arquivo_csv('Consolidado_Integrador_Inca')
    df_apos_tipos = f.leitura_arquivo_parquet('BaseCompleta')
    df_apos_analise =  f.leitura_arquivo_parquet('analise_valores')
    df_apos_transf = f.leitura_arquivo_parquet('transformacoes')
    df_apos_extracao =  f.leitura_arquivo_parquet('extracao_dados')
    
 
    result_df = pd.DataFrame()
    result_df = main_indicadores_variaveis(result_df)
    
    a_df = f.leitura_arquivo_excel_conclusao('parcial_extracao')
    result_df = pd.concat([result_df , a_df] , axis = 0)

    result_df = result_df.fillna(0) 
    result_df.reset_index(drop = True , inplace=True)
    
    a_file_name = 'indicadores_variaveis'
    f.salvar_excel_conclusao(result_df , a_file_name)
    
    result_df = pd.DataFrame()
    result_df = main_indicadores_globais()
    
    result_df = result_df.fillna(0) 
    result_df.reset_index(drop = True , inplace=True)
    
    a_file_name = 'indicadores_globais'
    f.salvar_excel_conclusao(result_df , a_file_name)



# df = df_apos_analise.loc[:, df_apos_analise.columns.str.startswith('Analise')]


# a_dict['Indicador'] = 'Valores invalidos'
# for uma_analise in df.columns:
#     variavel = uma_analise.replace('Analise' , '')
#     variavel = uma_analise.replace('_tipo' , '')
#     a_dict[variavel] = 100
