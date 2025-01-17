# -*- coding: utf-8 -*-

"""Registros Hospitalares de Câncer (RHC) - Carregamento e consolidacao inicial dos dados.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: ulf Bergmann

"""

import dbfread as db
import pandas as pd
import numpy as np
import re 
import os

import funcoes as f
from funcoes import Log

# =============================================================================
#  FUNCOES 
# =============================================================================

 

def leitura_inicial_dados(dir_bases = 'dados\\'):
    """Leitura dos arquivos baixados do Integrador RHC - INCA e criacao de um DataFrame unico.
    
    Le todos os arquivos do diretorio egerando um df unico

    Parameters:
        dir_bases (string): diretorio raiz dos arquivos a serem lidos

    Returns:
        (DataFrame): df unico com os dados dos varios arquivos 
    """ 
    count = 0
    
    df_aux = pd.DataFrame()
    for a_dir , _, lista_arquivos in os.walk(dir_bases): #percorre toda a estrutura olhando os subdiretorios
        for nome_arquivo in lista_arquivos:  #para cada subdiretorio
            arquivo = os.path.join( a_dir , nome_arquivo)
            print("Processando arquivo:" + arquivo)
            aux = db.DBF(arquivo , encoding='utf-8' , load=True)
            a_df = pd.DataFrame(iter(aux))
            df_aux = pd.concat([df_aux, a_df])
            count = count +1
            
            if count > 100:
                break
        break
    return df_aux

        
# ACertar os tipos das variaveis
def definir_tipos_variaveis(df):
    """Definicao dos tipos das variaveis.
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """ 
    colunas_anos = ['ANOPRIDI' , 'ANTRI' , 'DTPRICON' , 'DTINITRT' ]
    colunas_numericos = ['IDADE']
    
    a_str = 'ALCOOLIS BASMAIMP BASDIAGSP CLIATEN CLITRAT CNES DIAGANT ESTADIAM ESTADIAG ESTADRES ESTCONJ ESTDFIMT EXDIAG HISTFAMC INSTRUC LATERALI LOCALNAS LOCTUDET LOCTUPRI LOCTUPRO MAISUMTU MUUH OCUPACAO ORIENC PTNM OUTROESTA PRITRATH PROCEDEN RACACOR RZNTR SEXO TABAGISM TIPOHIST TNM TPCASO UFUH'
    colunas_categoricos = a_str.split(' ')
    
    for a_col in colunas_numericos:
        df[a_col] = pd.to_numeric(df[a_col], errors='coerce').fillna(0).astype(np.int64)
    for a_col in colunas_anos:
        df[a_col] = pd.to_numeric(df[a_col], errors='coerce').fillna(0).astype(np.int64)
    for a_col in colunas_categoricos:
        col_float = ['EXDIAG' , 'BASMAIMP']
        if a_col in col_float:  # variaveis que estavam como float e sao int
            df[a_col] = pd.to_numeric(df[a_col], errors='coerce').fillna(0).astype(np.int64)
        df[a_col] = df[a_col].astype(str)
        df[a_col] = df[a_col].astype('category')
        
    return df
    
def strip_spaces(a_str):
    return str(a_str).replace(' ', '')

def valida_datas_acerta_tipo(df):
    """Define o tipo das variaveis Data e substitui valores invalidos por None.
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """ 
    lista_colunas = ['DTDIAGNO', 'DTTRIAGE', 'DATAPRICON', 'DATAOBITO' , 'DATAINITRT']
    nan_values = ['88/88/8888' , '99/99/9999' , '00/00/0000' , '//']
    
    ind_antes=df[lista_colunas].isnull().sum()
    
    for a_var in lista_colunas:
        # df[a_var] = df[a_var].apply(lambda x: strip_spaces(x) )
        df[a_var] = df[a_var].apply(lambda x: np.nan if (strip_spaces(x) in nan_values) else strip_spaces(x))
        df[a_var] = pd.to_datetime(df[a_var] , format="%d/%m/%Y" , errors= 'coerce')
        df[a_var] = df[a_var].apply(lambda x: np.nan if (x.year < 1984 or x.year > 2023) else x)
        
    ind_depois=df[lista_colunas].isnull().sum()

    df_aux = pd.DataFrame()
    df_aux['Nulos antes'] = ind_antes
    df_aux['Nulos depois'] = ind_depois
    df_aux['diferenca'] = ind_depois- ind_antes

    a_file_name = 'valores_datas_tornados_null'
    f.salvar_excel_conclusao(df_aux , a_file_name)

    print(log.logar_acao_realizada('Valores Invalidos' , 'Corrigir valores de datas invalidas para null. Ver arquivo valores_datas_tornados_null.xslx' , ""))

    return df


def tratar_codigo_municipio(a_str):
    """Funcao auxiiar. Verifica se o codigo tem tamanho 7.
    
    Parameters:
        a_str (String): codigo do municipio

    Returns:
        (String): s_str ou None 
    """ 
    a_str = str(a_str)
    a_str = re.sub("\D", "", a_str)
    
    if len(a_str) == 7:
        return a_str
    
    if(len(a_str) > 7):
        return a_str[0:7]
        
    return np.nan

def tratar_variavel_municipio(df):
    """Valida os codigos do municipios transformando em None se invalidos.
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado 
    """ 
    aux_nulos = df['PROCEDEN'].isnull().sum()
    df['PROCEDEN'] = df['PROCEDEN'].apply(tratar_codigo_municipio)
    aux_nulos1 = df['PROCEDEN'].isnull().sum()
    print(log.logar_acao_realizada('Valores Invalidos' , 'Variavel PROCEDEN - Código Municipio invalido ' , aux_nulos1 - aux_nulos ))
    
    aux_nulos = df['MUUH'].isnull().sum()
    df['MUUH'] = df['MUUH'].apply(tratar_codigo_municipio)
    aux_nulos1 = df['MUUH'].isnull().sum()
    print(log.logar_acao_realizada('Valores Invalidos' , 'Variavel MUUH - Código Municipio invalido ' , aux_nulos1 - aux_nulos ))

    return df

def main():
    """Funcao principal.
    
    Parameters:

    Returns:
        (DataFrame): df carregado       
    """ 

    
    df_unico = f.leitura_arquivo_csv('Consolidado_Integrador_Inca')
    print( log.logar_acao_realizada('Carga Dados' , 'Carregamento da base bruta' , df_unico.shape[0]) )
       
    #ACERTOS INICIAIS - TRATAR VALIDACOES A AJUSTES APLICAVEIS A TODA A BASE, SEM SER ESPECIFICO DOS CASOS QUE SERAO ABORDADOS (ANALITICOS)
    df_unico = valida_datas_acerta_tipo(df_unico)
    df_unico = definir_tipos_variaveis(df_unico)
    df_unico = tratar_variavel_municipio(df_unico)
    
    # ETAPA INICIAL - SALVAR TRATAMENTO INICIAL 
    f.salvar_parquet(df_unico , 'BaseCompleta')
    log.salvar_log('log_BaseCompleta')

    df_analiticos = df_unico.loc[df_unico['TPCASO'] == '1']
    f.salvar_parquet(df_analiticos , 'BaseAnaliticos')
    return df_unico


if __name__ == "__main__":
    log = Log()
    df_unico = main()



#%% ANALISE TNM

# def computa_resultados_TNM(df):
#     tamanho_base = df.shape[0]
#     tamanho_base_sp = df[df['UFUH'] == 'SP'].shape[0]
    
#     df_resultado = pd.DataFrame(columns=['Formato', 'Descricao', 'Quantidade Total' , 'Quantidade Total %' , "Quantidade SP" , 'Quantidade SP %' ])
#     df_resultado = df_resultado.astype(dtype={'Formato':'object','Descricao':'object','Quantidade Total':'int' ,  'Quantidade Total %':'float', 'Quantidade SP':'int' , 'Quantidade SP %' : 'float'}  )    

# # r_hemato
# # r_invalido
# # r_nao_se_aplica_geral  'nao se aplica - Geral'
# # r_exato
# # r_incompleto
# # r_regrasp

#     a_tipo = 'nao se aplica - Hemato'
#     q_total =  df[(df['AnaliseTNM'] == a_tipo)].shape[0]
#     q_sp = df[(df['UFUH'] == 'SP') & (df['AnaliseTNM'] == a_tipo)].shape[0]
#     aNewDF = pd.DataFrame([{'Formato' : 'CID-O C81 a C85 / C91 a C97', 'Descricao' : 'Não se aplica ao tumor Hemato', 'Quantidade Total' : q_total , 'Quantidade Total %' : f'{q_total/tamanho_base * 100 :.2f}%'  , 'Quantidade SP' : q_sp , 'Quantidade SP %' : f'{q_sp/tamanho_base_sp * 100 :.2f}%'}])
#     df_resultado = pd.concat([df_resultado , aNewDF ], ignore_index=True)

#     a_tipo = 'invalido'
#     q_total =  df[(df['AnaliseTNM'] == a_tipo)].shape[0]
#     q_sp = df[(df['UFUH'] == 'SP') & (df['AnaliseTNM'] == a_tipo)].shape[0]
#     aNewDF = pd.DataFrame([{'Formato' : r_invalido, 'Descricao' : 'Marcado como inexistente', 'Quantidade Total' : q_total , 'Quantidade Total %' : f'{q_total/tamanho_base * 100 :.2f}%' , 'Quantidade SP' : q_sp , 'Quantidade SP %' : f'{q_sp/tamanho_base_sp * 100 :.2f}%'}])
#     df_resultado = pd.concat([df_resultado , aNewDF ], ignore_index=True)

#     a_tipo = 'nao se aplica - Geral'
#     q_total =  df[(df['AnaliseTNM'] == a_tipo)].shape[0]
#     q_sp = df[(df['UFUH'] == 'SP') & (df['AnaliseTNM'] == a_tipo)].shape[0]
#     aNewDF = pd.DataFrame([{'Formato' : r_nao_se_aplica_geral, 'Descricao' : 'Marcado como não se aplica', 'Quantidade Total' : q_total , 'Quantidade Total %' : f'{q_total/tamanho_base * 100 :.2f}%' , 'Quantidade SP' : q_sp , 'Quantidade SP %' : f'{q_sp/tamanho_base_sp * 100 :.2f}%'}])
#     df_resultado = pd.concat([df_resultado , aNewDF ], ignore_index=True)

#     a_tipo = 'exato'
#     q_total =  df[(df['AnaliseTNM'] == a_tipo)].shape[0]
#     q_sp = df[(df['UFUH'] == 'SP') & (df['AnaliseTNM'] == a_tipo)].shape[0]
#     aNewDF = pd.DataFrame([{'Formato' : r_exato, 'Descricao' : 'Possui exatamente 3 digitos conforme especificado', 'Quantidade Total' : q_total , 'Quantidade Total %' : f'{q_total/tamanho_base * 100 :.2f}%' , 'Quantidade SP' : q_sp , 'Quantidade SP %' : f'{q_sp/tamanho_base_sp * 100 :.2f}%'}])
#     df_resultado = pd.concat([df_resultado , aNewDF ], ignore_index=True)
    
#     a_tipo = 'incompleto'
#     q_total =  df[(df['AnaliseTNM'] == a_tipo)].shape[0]
#     q_sp = df[(df['UFUH'] == 'SP') & (df['AnaliseTNM'] == a_tipo)].shape[0]
#     aNewDF = pd.DataFrame([{'Formato' : r_incompleto, 'Descricao' : 'Possui X em algum dos digitos TNM', 'Quantidade Total' : q_total , 'Quantidade Total %' : f'{q_total/tamanho_base * 100 :.2f}%' , 'Quantidade SP' : q_sp , 'Quantidade SP %' : f'{q_sp/tamanho_base_sp * 100 :.2f}%'}])
#     df_resultado = pd.concat([df_resultado , aNewDF ], ignore_index=True)
    
#     a_tipo = 'nao se aplica - regra SP'
#     q_total =  df[(df['AnaliseTNM'] == a_tipo)].shape[0]
#     q_sp = df[(df['UFUH'] == 'SP') & (df['AnaliseTNM'] == a_tipo)].shape[0]
#     aNewDF = pd.DataFrame([{'Formato' : r_regrasp, 'Descricao' : 'Preenchido com YYY ou XXX. Não se aplica ao tumor (Regra SP)', 'Quantidade Total' : q_total , 'Quantidade Total %' : f'{q_total/tamanho_base * 100 :.2f}%'  , 'Quantidade SP' : q_sp , 'Quantidade SP %' : f'{q_sp/tamanho_base_sp * 100 :.2f}%'}])
#     df_resultado = pd.concat([df_resultado , aNewDF ], ignore_index=True)
        
#     a_tipo = 'demais'
#     q_total =  df[(df['AnaliseTNM'] == a_tipo)].shape[0]
#     q_sp = df[(df['UFUH'] == 'SP') & (df['AnaliseTNM'] == a_tipo)].shape[0]
#     aNewDF = pd.DataFrame([{'Formato' : "" ,  'Descricao' : 'Valores indefinidos. Requer aprofundamento', 'Quantidade Total' : q_total , 'Quantidade Total %' : f'{q_total/tamanho_base * 100 :.2f}%'  , 'Quantidade SP' : q_sp , 'Quantidade SP %' : f'{q_sp/tamanho_base_sp * 100 :.2f}%'}])
#     df_resultado = pd.concat([df_resultado , aNewDF ], ignore_index=True)
    
#     a = df.loc[df['AnaliseTNM'] == 'demais']
#     b = a.groupby('TNM' , observed = True).size()
#     df_sem_conclusao = b.reset_index()
#     df_sem_conclusao = df_sem_conclusao.sort_values(by=[0], ascending=[False]).reset_index(drop=True)
    
#     return df_resultado , df_sem_conclusao




# df_resultado_analise_tnm, df_sem_conclusao = computa_resultados_TNM(df_unico)
# tab_df_resultado_analise_tnm = tabulate(df_resultado_analise_tnm, headers='keys', tablefmt='simple_grid', numalign='right' , floatfmt=".0f" )
# a_nome_arquivo = 'analiseTNM_completo'
# f.salvar_excel_conclusao(df_resultado_analise_tnm , a_nome_arquivo + '_sumario')
# f.salvar_excel_conclusao(df_sem_conclusao , a_nome_arquivo + '_fora_padrao')

# print(log.logar_acao_realizada('Analise Variavel' , 'Resultados da analise do TNM - Base completa' , f'ver arquivos {a_nome_arquivo}'))


# df_analitico = df_unico[df_unico['TPCASO'] == '1']

# df_analitico.shape[0]

# df_resultado_analise_tnm_analitico, df_sem_conclusao_analitico = computa_resultados_TNM(df_analitico)
# tab_df_resultado_analise_tnm_analitico = tabulate(df_resultado_analise_tnm_analitico, headers='keys', tablefmt='simple_grid', numalign='right' , floatfmt=".0f" )
# a_nome_arquivo = 'analiseTNM_analiticos'
# f.salvar_excel_conclusao(df_resultado_analise_tnm_analitico , a_nome_arquivo + '_sumario')
# f.salvar_excel_conclusao(df_sem_conclusao_analitico , a_nome_arquivo + '_fora_padrao')

# print(log.logar_acao_realizada('Analise Variavel' , 'Resultados da analise do TNM - Base de casos analiticos' , f'ver arquivos {a_nome_arquivo}'))


# # a = pd.DataFrame()
# # a['teste'] = np.where(df_unico['TNM'].str.contains('XXX', regex= False, na=False) , 'achou YYY', 
# #              np.where(df_unico['TNM'].str.contains(r"^[0-4XIA][0-4X][0-1X]$", regex= True, na=False) ,'Exato',
# #             'nao achou'))
# # b = a.groupby('teste' , observed = True).size()










