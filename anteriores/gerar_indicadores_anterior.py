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
import numpy as np

import sys

sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

import funcoes as f
from funcoes import Log


# def insere_indicador_nulos(df, tipo, response):
#     """Gera o indicador de registros nulos para cada variavel do DataFrame.

#     Parameters:
#         df (DataFrame): DataFrame a ser analisado
#         tipo (String): tipo do df / indicador
#         response (DataFrame): DataFrame com indicadores

#     Returns:
#         (DataFrame): DataFrame com indicadores atualizados
#     """
#     a = df.isnull().sum().astype("int64")
#     a_df = pd.DataFrame(a).T
#     a_df.insert(0, "Indicador", [tipo])

#     return pd.concat([response, a_df], axis=0)


def insere_indicador_invalidos(df, response):
    """Gera o indicador de registros invalidos para cada variavel do DataFrame.

    Parameters:
        df (DataFrame): DataFrame a ser analisado
        response (DataFrame): DataFrame com indicadores

    Returns:
        (DataFrame):  DataFrame com indicadores atualizados
    """

    q_TNM = df[
        (df["_AnaliseTNM"] == "incompleto")
        | (df["_AnaliseTNM"] == "demais")
    ].shape[0]

    q_LOCTUDET = df[
        (df["_AnaliseLOCTUDET"] == "incompleto")
        | (df["_AnaliseLOCTUDET"] == "demais")
    ].shape[0]

    q_LOCTUPRI = df[
        (df["_AnaliseLOCTUPRI"] == "incompleto")
        | (df["_AnaliseLOCTUPRI"] == "demais")
    ].shape[0]

    q_LOCTUPRO = df[
        (df["_AnaliseLOCTUPRO"] == "incompleto")
        | (df["_AnaliseLOCTUPRO"] == "demais")
    ].shape[0]

    q_ESTADIAM = df[
        (df["_AnaliseESTADIAM"] == "incompleto")
        | (df["_AnaliseESTADIAM"] == "demais")
    ].shape[0]

    q_IDADE = df[df["IDADE"] == -1].shape[0]

    # a_df = pd.DataFrame([a_dict])
    # a_df.insert(0, "Indicador", ['Invalidos apos analise'])

    response.at["Invalidos apos a Analise", "TNM"] = q_TNM
    response.at["Invalidos apos a Analise", "LOCTUDET"] = q_LOCTUDET
    response.at["Invalidos apos a Analise", "LOCTUPRI"] = q_LOCTUPRI
    response.at["Invalidos apos a Analise", "LOCTUPRO"] = q_LOCTUPRO
    response.at["Invalidos apos a Analise", "ESTADIAM"] = q_ESTADIAM
    response.at["Invalidos apos a Analise", "IDADE"] = q_IDADE

    return response


# def gera_indicador_incompletude(df, etapa, lista_colunas, response):
#     """Gera o indicador de INCOMPLETUDE para cada variavel do DataFrame.

#     Incompletitude: proporção de informação ignorada, ou seja, os campos em branco e os códigos atribuídos à informação ignorada

#     Parameters:
#         df (DataFrame): DataFrame a ser analisado
#         etapa (String): momento do calculo do indicador
#         lista_colunas (String):  colunas a serem analisadas
#         response (DataFrame): DataFrame com indicadores

#     Returns:
#         (DataFrame):  DataFrame com indicador de Incompletude
#     """
#     informacao_ignorada = {
#         "ALCOOLIS": ["9"],
#         "BASDIAGSP": ["4"],
#         "BASMAIMP": ["9"],
#         "DIAGANT": ["9"],
#         "ESTCONJ": ["9"],
#         "ESTDFIMT": ["9"],
#         "EXDIAG": ["9"],
#         "HISTFAMC": ["9"],
#         "INSTRUC": ["9"],
#         "LATERALI": ["9"],
#         "OCUPACAO": ["999", "9999"],
#         "ORIENC": ["9"],
#         "PRITRATH": ["9"],
#         "PTNM": ["9"],
#         "RACACOR": ["9"],
#         "RZNTR": ["9"],
#         "TABAGISM": ["9"],
#     }

#     for var in lista_colunas:
#         a = df[var].value_counts(dropna=False, normalize=False).reset_index()
#         count_nulos = 0
#         count_sem_info = 0
#         for i, row in a.iterrows():
#             # print((var in informacao_ignorada_por_atributo.keys()) and (row[var] in informacao_ignorada_por_atributo[var] ))

#             if (
#                 (row.iloc[0] is np.nan)
#                 | (row.iloc[0] == "nan")
#                 | (pd.isna(row.iloc[0]))
#             ):
#                 # print(f' nan -> {var} count -> {row.iloc[1]}')
#                 count_nulos = count_nulos + row.iloc[1]
#             else:
#                 if (var in informacao_ignorada.keys()) and (
#                     row[var] in informacao_ignorada[var]
#                 ):
#                     count_sem_info = count_sem_info + row.iloc[1]

#         response.at[etapa + " - Nr incompletos (nulos)", var] = count_nulos
#         response.at[etapa + etapa + " - % incompletos (nulos)", var] = (
#             count_nulos * 100 / len(df)
#         )

#         response.at[etapa + " - Nr incompletos (sem info)", var] = count_sem_info
#         response.at[etapa + etapa + " - % incompletos (sem info)", var] = (
#             count_sem_info * 100 / len(df)
#         )

#     return response


def busca_data_sp_iniciou_mais_que_1_trat(df):
    """Identifica a data de quando SP iniciou o registro de mais de um tratamento.

    Parameters:
        df (DataFrame): DataFrame a ser pesquisado

    Returns:
        (Date):  data do inicio
    """
    aux_sp = df[df["UFUH"] == "SP"]
    a = aux_sp.loc[aux_sp["PRITRATH_NrTratamentos"] > 1]
    a.sort_values(by="DATAINITRT", ascending=[True]).reset_index(drop=True)

    return a["DATAINITRT"].iloc[0]


def main_indicadores_variaveis(response_df):
    """Funcao principal de geracao dos indicadores das variaveis.

    Parameters:
        response_df (DataFrame): DataFrame com indicadores
    Returns:
        (DataFrame):  DataFrame com indicadores atualizados
    """
    # response_df = insere_indicador_nulos(df = df_inicial, tipo = 'Nulos - Inicial' , response = response_df)

    # response_df = insere_indicador_nulos(df = df_apos_tipos, tipo = 'Nulos - Após Tipos Definidos', response = response_df)

    # response_df = insere_indicador_nulos(df = df_apos_analise, tipo = 'Nulos - Após Analise dos Dados', response = response_df)

    # print(log.logar_acao_realizada('Indicadores' , 'Calculo dos indicadores de NULOS' , ' '))

    response_df = gera_indicador_incompletude(
        df_inicial, "Inicial", df_inicial.columns, response_df
    )
    response_df = gera_indicador_incompletude(
        df_apos_tipos, "Apos Tipos definidos", df_apos_tipos.columns, response_df
    )
    response_df = gera_indicador_incompletude(
        df_apos_analise, "Após Analise dos Dados", df_apos_analise.columns, response_df
    )
    response_df = gera_indicador_incompletude(
        df_apos_transf, "Apos Transformacoes", df_apos_transf.columns, response_df
    )
    response_df = gera_indicador_incompletude(
        df_apos_extracao, "Apos Extracao", df_apos_extracao.columns, response_df
    )

    print(
        log.logar_acao_realizada(
            "Indicadores",
            "Calculo dos indicadores de COMPLETUDE (porcentagem de nulos ou sem informação)",
            " ",
        )
    )

    response_df = insere_indicador_invalidos(df=df_apos_analise, response=response_df)

    print(
        log.logar_acao_realizada(
            "Indicadores", "Calculo dos indicadores de INVALIDOS", " "
        )
    )

    return response_df


def main_indicadores_globais():
    """Funcao principal de geracao dos indicadores globais.

    Returns:
        (DataFrame):  DataFrame com indicadores globais
    """
    a_lista = []

    a_dict = {
        "Indicador": "Quantidade de Registros",
        "Etapa": "Carga Inicial - Dados Brutos",
        "Valor": df_inicial.shape[0],
    }
    a_lista.append(a_dict)

    a_dict = {
        "Indicador": "Quantidade de Registros",
        "Etapa": "Final da Analise - Dados Completos",
        "Valor": df_apos_analise.shape[0],
    }
    a_lista.append(a_dict)

    a_dict = {
        "Indicador": "Quantidade de Registros",
        "Etapa": "Extracao - Dados de Casos Analiticos",
        "Valor": df_apos_extracao.shape[0],
    }
    a_lista.append(a_dict)

    data_sp = busca_data_sp_iniciou_mais_que_1_trat(df_apos_transf)

    a_dict = {
        "Indicador": "Data de Inicio de mais de um trat em SP",
        "Etapa": "Final da Analise - Dados Completos",
        "Valor": data_sp.strftime("%m/%Y"),
    }
    a_lista.append(a_dict)

    a_df = pd.DataFrame(a_lista)

    return a_df


if __name__ == "__main__":
    log = Log()
    log.carregar_log("log_BaseModelagem")

    df_inicial = f.leitura_arquivo_csv("BaseConsolidada")
    df_apos_tipos = f.leitura_arquivo_parquet("BaseInicial")
    df_apos_analise = f.leitura_arquivo_parquet("BaseSanitizada")
    df_apos_transf = f.leitura_arquivo_parquet("BaseTransfor")
    df_apos_extracao = f.leitura_arquivo_parquet("BaseModelagem")
    print(
        log.logar_acao_realizada(
            "Carga Dados", "Carregamento dos dados para calculo dos indicadores", " "
        )
    )

    result_df = pd.DataFrame()
    result_df = main_indicadores_variaveis(result_df)

    result_df.reset_index(drop=False, inplace=True)
    result_df.rename(columns={"index": "Indicador"}, inplace=True)

    a_df = f.leitura_arquivo_excel_conclusao("parcial_extracao")
    result_df = pd.concat([result_df, a_df], axis=0)
    print(
        log.logar_acao_realizada(
            "Indicadores", "Registros eliminados na seleção de dados", " "
        )
    )

    result_df = result_df.fillna("nan")
    result_df.reset_index(drop=True, inplace=True)

    a_file_name = "indicadores_variaveis"
    f.salvar_excel_conclusao(result_df.T, a_file_name)

    result_df = pd.DataFrame()
    result_df = main_indicadores_globais()

    result_df = result_df.fillna("nan")
    result_df.reset_index(drop=True, inplace=True)

    a_file_name = "indicadores_globais"
    f.salvar_excel_conclusao(result_df, a_file_name)

    log.salvar_log("log_Indicadores")


# df = df_apos_analise.loc[:, df_apos_analise.columns.str.startswith('Analise')]


# a_dict['Indicador'] = 'Valores invalidos'
# for uma_analise in df.columns:
#     variavel = uma_analise.replace('Analise' , '')
#     variavel = uma_analise.replace('_tipo' , '')
#     a_dict[variavel] = 100


# df_apos_analise =  f.leitura_arquivo_parquet('analise_valores')
# df_apos_tipos = f.leitura_arquivo_parquet('BaseCompleta')

# result_df = pd.DataFrame()
# result_df = main_indicadores_variaveis_qualidade(result_df)

# result_df = result_df.fillna(0)
