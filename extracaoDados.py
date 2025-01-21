# -*- coding: utf-8 -*-

"""Registros Hospitalares de Câncer (RHC) - Extracao e selecao dos dados para construcao dos modelos.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: Ulf Bergmann


#Roteiro de execucao:
    1. Elimina registros sem tratamento 
    2. Elimina registros com ESTDFIMT nao validos
    3. Elimina registros com ESTADIAM  nao validos
    4. Elimina registros com TNM nao validos
    5. Elimina registros com DATAS  nao validos
    6. Elimina registros com SEXO ou TIPOHIST nulos
    10. Salvar o arquivo como parquet

"""

import pandas as pd

import sys
sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")


import funcoes as f
from funcoes import Log

def seleciona_ESTDFIMT(df):
    """Elimina os valores invalidos ou nulos de ESTDFIMT. .
    
    Regras:
        ESTDFIMT ==>> Variável ESTDFIMT = 8. Não se aplica; 9. Sem informação  ===>>> ELIMINAR
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
        (Dictionary):  dict com indicadores
    """
    q_inicial = df.shape[0]
    #retirar valores 8 e 9
    df = df.drop(df[(df['ESTDFIMT'] == '8') | (df['ESTDFIMT'] == '9')].index, inplace = False)
    #retirar valores null
    df = df.dropna(subset = ['ESTDFIMT'], inplace=False)
    
    a_dict = {}
    a_dict['ESTDFIMT'] = q_inicial - df.shape[0]
    
    print(log.logar_acao_realizada('Selecao Dados' , 'Eliminacao de registros com ESTDFIMT invalido (8 , 9 e nulos)' ,f'{q_inicial - df.shape[0]}'))
    return df , a_dict

def seleciona_ESTADIAM(df):
    """Elimina os valores invalidos ou nulos de ESTADIAM. .
    
    ESTADIAM = codificação do grupamento do estádio clínico segundo classificação TNM
    
    Regras:
        AnaliseESTADIAM ==>> nulo | demais  ===>>> ELIMINAR
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado
       
    Returns:
        (DataFrame): df modificado
        (Dictionary):  dict com indicadores
    """
    q_inicial = df.shape[0]
    #retirar valores que nao sao exatos
    df = df.drop(df[(df['AnaliseESTADIAM'] == 'nulo') | (df['AnaliseESTADIAM'] == 'demais')].index, inplace = False)
    
    a_dict = {}
    a_dict['ESTADIAM'] = q_inicial - df.shape[0]
      
    print(log.logar_acao_realizada('Selecao Dados' , 'Eliminacao de registros com ESTADIAM que nao sao exatos' ,f'{q_inicial - df.shape[0]}'))
    return df , a_dict


def seleciona_TNM(df):
    """Elimina os valores invalidos ou nulos de TNM. .
    
    TNM   <AnaliseTNM>
 
    Regras:
        AnaliseTNM ==>> invalido | demais  ===>>> ELIMINAR
        Analisar o que deve ser feito com < nao se aplica - Hemato | nao se aplica - Geral>
        
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado
        

    Returns:
        (DataFrame): df modificado
        (Dictionary):  dict com indicadores
    """
    q_inicial = df.shape[0]
    #retirar valores que nao sao exatos
    df = df.drop(df[(df['AnaliseTNM'] == 'invalido') | (df['AnaliseTNM'] == 'demais')].index, inplace = False)
    
    a_dict = {}
    a_dict['TNM'] = q_inicial - df.shape[0]
     
    print(log.logar_acao_realizada('Selecao Dados' , 'Eliminacao de registros com TNM que nao sao exatos ou incompletos. \n  Analisar o que deve ser feito com < nao se aplica - Hemato | nao se aplica - Geral>' ,f'{q_inicial - df.shape[0]}'))

    return df , a_dict

def seleciona_DATAS(df):
    """Selecao de datas () nao nulas.
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado
       

    Returns:
        (DataFrame): df modificado
        (Dictionary):  dict com indicadores
    """
    a_dict = {}
    
    q_inicial = df.shape[0]
    #retirar valores que nao sao exatos
    df = df.dropna(subset = ['DATAINITRT'], inplace=False)
    a_dict['DATAINITRT'] = q_inicial - df.shape[0]
    print(log.logar_acao_realizada('Dados Nulos' , 'Eliminacao de registros com DATAINITRT nulo' ,f'{q_inicial - df.shape[0]}'))
    
    # remover sem data de diagnostico
    q_inicial = df.shape[0]
    #retirar valores que nao sao exatos
    df = df.dropna(subset = ['DTDIAGNO'], inplace=False)
    a_dict['DTDIAGNO'] = q_inicial - df.shape[0]
    print(log.logar_acao_realizada('Dados Nulos' , 'Eliminacao de registros com DTDIAGNO nulo' ,f'{q_inicial - df.shape[0]}'))
    
    return df , a_dict
    


def seleciona_naonulos(df , lista_variaveis):
    """Elimina os valores nulos das variaveis passadas como parametros.

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado
        lista_variaveis (list): variaveis cujos valores nulos serão removidos
       

    Returns:
        (DataFrame): df modificado
        (Dictionary):  dict com indicadores
    """
    q_inicial = df.shape[0]
    
    a = df_base[lista_variaveis].isnull().sum().astype('int64')
    a_dict = a.to_dict()

    df = df.dropna(subset = lista_variaveis, inplace=False)
    print(log.logar_acao_realizada('Dados Nulos' , f'Eliminacao dos registros {lista_variaveis} com valores nulos' ,f'{q_inicial - df.shape[0]}'))
    
    return df , a_dict

def elimina_sem_tratamento(df):
    """Elimina os registros de quem nao fez tratamento (RZNTR de 1 ate 7).

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado
       

    Returns:
        (DataFrame): df modificado
        (Dictionary):  dict com indicadores
    """
    
    q_inicial = df.shape[0]
    
    values = ['1', '2' , '3' , '4' , '5' , '6' , '7']
    b = df_base[df_base['RZNTR'].isin(values)]

    df = df.drop(b.index, inplace = False)
    
    a_dict = {}
    a_dict['RZNTR'] = q_inicial - df.shape[0]
   
    print(log.logar_acao_realizada('Dados Nulos' , f'Eliminacao dos registros de quem nao fez tratamento (RZNTR nao nulo)' ,f'{q_inicial - df.shape[0]}'))
    return df ,  a_dict


def coleta_sumario(df):
    """Insere no log as informacoes dos dados.
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """    
    print(log.logar_acao_realizada('Informacao' , 'Quantidade de registros com valores validos' ,f'{df.shape[0]}'))
    a = df.isnull().sum()
    print(log.logar_acao_realizada('Informacao' , 'Quantidade de registros com valores nulos' , a))


def main(df):
    """Funcao principal.
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado
    Returns:
        (DataFrame): df modificado     
        (DataFrame): df indicadores    
    """ 
    a_dict = {}
    

    df , aux_dict = elimina_sem_tratamento(df)
    a_dict = a_dict | aux_dict
    
    df , aux_dict = seleciona_ESTDFIMT(df)
    a_dict = a_dict | aux_dict
    
    df , aux_dict = seleciona_ESTADIAM(df)
    a_dict = a_dict | aux_dict
    
    df , aux_dict = seleciona_TNM(df)
    a_dict = a_dict | aux_dict
    
    df , aux_dict = seleciona_DATAS(df)
    a_dict = a_dict | aux_dict
    
    df , aux_dict = seleciona_naonulos(df , lista_variaveis = ['SEXO' , 'TIPOHIST'])    
    a_dict = a_dict | aux_dict
 
    
    coleta_sumario(df)
    
    an_ind_df = pd.DataFrame([a_dict])
    an_ind_df.astype('int64')
    an_ind_df.insert(0, "Indicador", ['Registros eliminados na seleção dos dados'])
    
    
    return df , an_ind_df

if __name__ == "__main__":
    log = Log()
    log.carregar_log('log_analise_valores')
    df_base = f.leitura_arquivo_parquet('analise_valores')
 
    print( log.logar_acao_realizada('Carga Dados' , 'Carregamento da base dos dados para seleção' , df_base.shape[0]) )

    result_df = pd.DataFrame()
    df_base , result_df = main(df_base) 
    
    
    log.salvar_log('log_extracao_dados') 
    f.salvar_parquet(df_base , 'extracao_dados')
    f.salvar_excel_conclusao(result_df , 'parcial_extracao')
    
    a = log.asString()
