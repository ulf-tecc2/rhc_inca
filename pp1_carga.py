# -*- coding: utf-8 -*-

"""Registros Hospitalares de Câncer (RHC) - Carregamento e consolidacao inicial dos dados.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: ulf Bergmann

#Roteiro de execucao:
    1. Leitura dos arquivos isolados
    2. Validacao das Datas e definição do tipo
    3. Definir tipos das variáveis / colunas do DataFrame
    4. Salvar o arquivo como parquet

"""

import dbfread as db
import pandas as pd
import numpy as np

import os

import funcoes as f
from funcoes import Log

# =============================================================================
#  FUNCOES
# =============================================================================


def leitura_inicial_dados(dir_bases="dados\\"):
    """Leitura dos arquivos baixados do Integrador RHC - INCA e criacao de um DataFrame unico.

    Le todos os arquivos do diretorio egerando um df unico

    Parameters:
        dir_bases (string): diretorio raiz dos arquivos a serem lidos

    Returns:
        (DataFrame): df unico com os dados dos varios arquivos
    """
    count = 0

    df_aux = pd.DataFrame()
    for a_dir, _, lista_arquivos in os.walk(
        dir_bases
    ):  # percorre toda a estrutura olhando os subdiretorios
        for nome_arquivo in lista_arquivos:  # para cada subdiretorio
            arquivo = os.path.join(a_dir, nome_arquivo)
            print("Processando arquivo:" + arquivo)
            aux = db.DBF(arquivo, encoding = "Windows-1252" , load=True) # inicial => encoding="utf-8" update => "Windows-1252"
            a_df = pd.DataFrame(iter(aux))
            df_aux = pd.concat([df_aux, a_df])
            count = count + 1

            if count > 100:
                break
        break
    return df_aux


def strip_spaces(a_str):
    return str(a_str).replace(" ", "")


# ACertar os tipos das variaveis
def definir_tipos_variaveis(df):
    """Definicao dos tipos das variaveis.

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    colunas_anos = ["ANOPRIDI", "ANTRI", "DTPRICON", "DTINITRT"]
    colunas_numericos = ["IDADE"]
    colunas_datas = ["DTDIAGNO", "DTTRIAGE", "DATAPRICON", "DATAOBITO", "DATAINITRT"]

    a_str = "ALCOOLIS BASMAIMP BASDIAGSP CLIATEN CLITRAT CNES DIAGANT ESTADIAM ESTADIAG ESTADRES ESTCONJ ESTDFIMT EXDIAG HISTFAMC INSTRUC LATERALI LOCALNAS LOCTUDET LOCTUPRI LOCTUPRO MAISUMTU MUUH OCUPACAO ORIENC PTNM OUTROESTA PRITRATH PROCEDEN RACACOR RZNTR SEXO TABAGISM TIPOHIST TNM TPCASO UFUH"
    colunas_categoricos = a_str.split(" ")

    for a_col in colunas_numericos:
        df[a_col] = pd.to_numeric(df[a_col], errors="coerce").fillna(0).astype(np.int64)
    for a_col in colunas_anos:
        df[a_col] = pd.to_numeric(df[a_col], errors="coerce").fillna(0).astype(np.int64)
    for a_col in colunas_categoricos:
        col_float = ["EXDIAG", "BASMAIMP"]
        if a_col in col_float:  # variaveis que estavam como float e sao int
            df[a_col] = (
                pd.to_numeric(df[a_col], errors="coerce").fillna(0).astype(np.int64)
            )
        df[a_col] = df[a_col].astype(str)
        df[a_col] = df[a_col].astype("category")

    for a_col in colunas_datas:
        df[a_col] = df[a_col].apply(lambda x: strip_spaces(x))
        df[a_col] = pd.to_datetime(df[a_col], format="%d/%m/%Y", errors="coerce")

    print(
        log.logar_acao_realizada(
            "Definicao de Tipos", "Colunas do Tipo datas -> datetime", colunas_datas
        )
    )
    print(
        log.logar_acao_realizada(
            "Definicao de Tipos", "Colunas do Tipo numerico -> int64", colunas_numericos
        )
    )
    print(
        log.logar_acao_realizada(
            "Definicao de Tipos",
            "Colunas do Tipo categorica -> category",
            colunas_categoricos,
        )
    )

    return df


# def valida_datas_acerta_tipo(df):
#     """Define o tipo das variaveis Data e substitui valores invalidos por None.

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado
#     """
#     lista_colunas = ['DTDIAGNO', 'DTTRIAGE', 'DATAPRICON', 'DATAOBITO' , 'DATAINITRT']
#     nan_values = ['88/88/8888' , '99/99/9999' , '00/00/0000' , '//']

#     ind_antes=df[lista_colunas].isnull().sum()

#     for a_var in lista_colunas:
#         # df[a_var] = df[a_var].apply(lambda x: strip_spaces(x) )
#         df[a_var] = df[a_var].apply(lambda x: np.nan if (strip_spaces(x) in nan_values) else strip_spaces(x))
#         df[a_var] = pd.to_datetime(df[a_var] , format="%d/%m/%Y" , errors= 'coerce')
#         df[a_var] = df[a_var].apply(lambda x: np.nan if (x.year < 1984 or x.year > 2023) else x)

#     ind_depois=df[lista_colunas].isnull().sum()

#     df_aux = pd.DataFrame()
#     df_aux['Nulos antes'] = ind_antes
#     df_aux['Nulos depois'] = ind_depois
#     df_aux['diferenca'] = ind_depois- ind_antes

#     a_file_name = 'valores_datas_tornados_null'
#     f.salvar_excel_conclusao(df_aux , a_file_name)

#     print(log.logar_acao_realizada('Valores Invalidos' , 'Corrigir valores de datas invalidas para null. Ver arquivo valores_datas_tornados_null.xslx' , ""))

#     return df


def main():
    """Funcao principal.

    Returns:
        (DataFrame): df carregado
    """
    df_unico = f.leitura_arquivo_parquet('BaseInicialAtualizada') 
    # df_unico = f.leitura_arquivo_csv("BaseConsolidada")
    print(
        log.logar_acao_realizada(
            "Carga Dados", "Carregamento da base bruta", df_unico.shape[0]
        )
    )

    # ACERTOS INICIAIS - TRATAR VALIDACOES A AJUSTES APLICAVEIS A TODA A BASE, SEM SER ESPECIFICO DOS CASOS QUE SERAO ABORDADOS (ANALITICOS)
    df_unico = definir_tipos_variaveis(df_unico)

    # ETAPA INICIAL - SALVAR TRATAMENTO INICIAL
    f.salvar_parquet(df_unico, "BaseInicialAtualizada_com_tipos")
    log.salvar_log("log_BaseInicial")

    # df_analiticos = df_unico.loc[df_unico['TPCASO'] == '1']
    # f.salvar_parquet(df_analiticos , 'BaseAnaliticos')
    return df_unico


def main_leitura_inicial():
    """Funcao para unir todos os arquivos que estao separados por anos."""
    df_unico = leitura_inicial_dados(dir_bases="dados/originais/update")
    print(
        log.logar_acao_realizada(
            "Carga Dados", "Consolidacao dos arquivos separados", df_unico.shape[0]
        )
    )

    f.salvar_parquet(df_unico, "BaseInicialAtualizada")


if __name__ == "__main__":
    log = Log()
    df_unico = main()
