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
    7. Elimina registros de casos nao analiticos
    8. Remove colunas nao significantes
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
    # retirar valores 8 e 9
    df = df.drop(
        df[(df["ESTDFIMT"] == "8") | (df["ESTDFIMT"] == "9")].index, inplace=False
    )
    # retirar valores null
    df = df.dropna(subset=["ESTDFIMT"], inplace=False)

    a_dict = {}
    a_dict["ESTDFIMT"] = q_inicial - df.shape[0]

    print(
        log.logar_acao_realizada(
            "Selecao Dados",
            "Eliminacao de registros com ESTDFIMT invalido (8 , 9 e nulos)",
            f"{q_inicial - df.shape[0]}",
        )
    )
    log.logar_indicador(
        "Extracao",
        "Eliminar Dados",
        "Eliminacao de registros com ESTDFIMT invalido (8 , 9 e nulos)",
        "ESTDFIMT",
        q_inicial - df.shape[0],
    )

    return df, a_dict


def seleciona_ESTADIAM(df):
    """Elimina os valores invalidos ou nulos de ESTADIAM.

    ESTADIAM = codificação do grupamento do estádio clínico segundo classificação TNM

    Regras:
        _AnaliseESTADIAM ==>> nulo | demais  ===>>> ELIMINAR

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
        (Dictionary):  dict com indicadores
    """
    q_inicial = df.shape[0]
    # retirar valores que nao sao exatos
    df = df.drop(
        df[
            (df["_AnaliseESTADIAM"] == "nulo") | (df["_AnaliseESTADIAM"] == "demais")
        ].index,
        inplace=False,
    )

    a_dict = {}
    a_dict["ESTADIAM"] = q_inicial - df.shape[0]

    print(
        log.logar_acao_realizada(
            "Selecao Dados",
            "Eliminacao de registros com ESTADIAM que nao sao exatos",
            f"{q_inicial - df.shape[0]}",
        )
    )
    log.logar_indicador(
        "Extracao",
        "Eliminar Dados",
        "Eliminacao de registros com ESTADIAM fora do padrao ou nulos",
        "ESTADIAM",
        q_inicial - df.shape[0],
    )

    return df, a_dict


def seleciona_TNM(df):
    """Elimina os valores invalidos ou nulos de TNM. .

    TNM   <_AnaliseTNM>

    Regras:
        _AnaliseTNM ==>> invalido | demais  ===>>> ELIMINAR
        Analisar o que deve ser feito com < nao se aplica - Hemato | nao se aplica - Geral>

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado


    Returns:
        (DataFrame): df modificado
        (Dictionary):  dict com indicadores
    """
    q_inicial = df.shape[0]
    # retirar valores que nao sao exatos
    df = df.drop(
        df[(df["_AnaliseTNM"] == "invalido") | (df["_AnaliseTNM"] == "demais")].index,
        inplace=False,
    )
    df = df.dropna(subset=["TNM"], inplace=False)

    a_dict = {}
    a_dict["TNM"] = q_inicial - df.shape[0]

    print(
        log.logar_acao_realizada(
            "Selecao Dados",
            "Eliminacao de registros com TNM  nulos ou que nao sao exatos ou incompletos. \n  Analisar o que deve ser feito com < nao se aplica - Hemato | nao se aplica - Geral>",
            f"{q_inicial - df.shape[0]}",
        )
    )
    log.logar_indicador(
        "Extracao",
        "Eliminar Dados",
        "Eliminacao de registros com TNM  nulos ou que nao sao exatos ou incompletos",
        "TNM",
        q_inicial - df.shape[0],
    )

    return df, a_dict





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
    
    
    df = df.dropna(subset=["_tempo_para_tratamento"], inplace=False)
    a_dict["_tempo_para_tratamento"] = q_inicial - df.shape[0]
    print(
        log.logar_acao_realizada(
            "Selecao Dados",
            "Eliminacao de registros com _tempo_para_tratamento nulo",
            f"{q_inicial - df.shape[0]}",
        )
    )
    
    
    
    
    # # retirar valores que nao sao exatos
    # df = df.dropna(subset=["DATAINITRT"], inplace=False)
    # a_dict["DATAINITRT"] = q_inicial - df.shape[0]
    # print(
    #     log.logar_acao_realizada(
    #         "Selecao Dados",
    #         "Eliminacao de registros com DATAINITRT nulo",
    #         f"{q_inicial - df.shape[0]}",
    #     )
    # )
    # log.logar_indicador(
    #     "Extracao",
    #     "Eliminar Dados",
    #     "Eliminacao de registros com DATAINITRT nulo",
    #     "DATAINITRT",
    #     q_inicial - df.shape[0],
    # )

    # # remover sem data de diagnostico
    # q_inicial = df.shape[0]
    # # retirar valores que nao sao exatos
    # df = df.dropna(subset=["DTDIAGNO"], inplace=False)
    # a_dict["DTDIAGNO"] = q_inicial - df.shape[0]
    # print(
    #     log.logar_acao_realizada(
    #         "Selecao Dados",
    #         "Eliminacao de registros com DTDIAGNO nulo",
    #         f"{q_inicial - df.shape[0]}",
    #     )
    # )
    # log.logar_indicador(
    #     "Extracao",
    #     "Eliminar Dados",
    #     "Eliminacao de registros com DTDIAGNO nulo",
    #     "DTDIAGNO",
    #     q_inicial - df.shape[0],
    # )

    return df, a_dict


# def seleciona_naonulos(df, lista_variaveis):
#     """Elimina os valores nulos das variaveis passadas como parametros.

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado
#         lista_variaveis (list): variaveis cujos valores nulos serão removidos


#     Returns:
#         (DataFrame): df modificado
#         (Dictionary):  dict com indicadores
#     """

#     q_base = df.shape[0]

#     a = df_base[lista_variaveis].isnull().sum().astype("int64")

#     a_dict = a.to_dict()

#     for a_var in lista_variaveis:
#         q_inicial = df.shape[0]
#         df = df.dropna(subset=[a_var], inplace=False)
#         log.logar_indicador(
#             "Extracao",
#             "Eliminar Dados",
#             f"Eliminacao de registros com {a_var} nulo",
#             a_var,
#             q_inicial - df.shape[0],
#         )

#     print(
#         log.logar_acao_realizada(
#             "Selecao Dados",
#             f"Eliminacao dos registros {lista_variaveis} com valores nulos",
#             f"{q_base - df.shape[0]}",
#         )
#     )

#     return df, a_dict


# def elimina_sem_tratamento(df):
#     """Elimina os registros de quem nao fez tratamento (RZNTR de 1 ate 7).

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado


#     Returns:
#         (DataFrame): df modificado
#         (Dictionary):  dict com indicadores
#     """

#     q_inicial = df.shape[0]

#     values = ["1", "2", "3", "4", "5", "6", "7"]
#     b = df_base[df_base["RZNTR"].isin(values)]

#     df = df.drop(b.index, inplace=False)

#     a_dict = {}
#     a_dict["RZNTR"] = q_inicial - df.shape[0]

#     print(
#         log.logar_acao_realizada(
#             "Selecao Dados",
#             f"Eliminacao dos registros de quem nao fez tratamento (RZNTR nao nulo)",
#             f"{q_inicial - df.shape[0]}",
#         )
#     )
#     log.logar_indicador(
#         "Extracao",
#         "Eliminar Dados",
#         "Eliminacao dos registros de quem nao fez tratamento (RZNTR nao nulo)",
#         "RZNTR",
#         q_inicial - df.shape[0],
#     )

#     return df, a_dict


# def remover_colunas_naosignificativas(df):
#     """Elimina as variaveis (colunas) que nao possuem significancia .

#     Colunas:
#         'ESTDFIMT', 'RZNTR','DTDIAGNO' , 'DTTRIAGE' , 'DATAPRICON' , 'DATAINITRT' , 'DATAOBITO',
#         'TPCASO', 'LOCALNAS' , 'BASDIAGSP' , 'VALOR_TOT' ,
#         '_AnaliseLOCTUDET', '_AnaliseLOCTUDET_tipo', '_AnaliseLOCTUPRI', '_AnaliseLOCTUPRO','_AnaliseTNM', '_AnaliseESTADIAM' ,
#         'CLIATEN' , 'CLITRAT' , 'CNES' , 'DTINITRT' , 'LOCTUPRO' , 'ESTADRES', 'OUTROESTA' , 'OCUPACAO' , 'PROCEDEN' , 'ESTADIAG'

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado
#     """
#     # remocao de variaveis nao significativas
#     colunas_a_remover = [
#         "ANTRI",
#         "ANOPRIDI",
#         "PRITRATH",
#         "ESTDFIMT",
#         "RZNTR",
#         "DTDIAGNO",
#         "DTTRIAGE",
#         "DATAPRICON",
#         "DATAINITRT",
#         "DATAOBITO",
#         "RZNTR",
#         "TPCASO",
#         "LOCALNAS",
#         "BASDIAGSP",
#         "VALOR_TOT",
#         # "_AnaliseLOCTUDET",
#         # "_AnaliseLOCTUDET_tipo",
#         # "_AnaliseLOCTUPRI",
#         # "_AnaliseLOCTUPRO",
#         # "_AnaliseTNM",
#         # "_AnaliseESTADIAM",
#         "CLIATEN",
#         "CLITRAT",
#         "CNES",
#         "DTINITRT",
#         "LOCTUPRO",
#         "ESTADRES",
#         "OUTROESTA",
#         "OCUPACAO",
#         "PROCEDEN",
#         "ESTADIAG",
#     ]

#     df_aux = df.drop(columns=colunas_a_remover, axis=1)

#     print(
#         log.logar_acao_realizada(
#             "Selecao Dados",
#             "Eliminacao de colunas com dados sem significancia",
#             f"{colunas_a_remover}",
#         )
#     )

#     return df_aux


def main(df):
    """Funcao principal.

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado
    Returns:
        (DataFrame): df modificado
        (DataFrame): df indicadores
    """
    a_dict = {}

    # df, aux_dict = elimina_sem_tratamento(df)
    # a_dict = a_dict | aux_dict

    df, aux_dict = seleciona_ESTDFIMT(df)
    a_dict = a_dict | aux_dict

    df, aux_dict = seleciona_ESTADIAM(df)
    a_dict = a_dict | aux_dict

    df, aux_dict = seleciona_TNM(df)
    a_dict = a_dict | aux_dict

    df, aux_dict = seleciona_DATAS(df)
    a_dict = a_dict | aux_dict

    # df, aux_dict = seleciona_naonulos(df, lista_variaveis=["IDADE","SEXO", "TIPOHIST", "ANTRI"])
    # a_dict = a_dict | aux_dict

    q_inicial = df.shape[0]
    df = df[df["TPCASO"] == "1"]
    print(
        log.logar_acao_realizada(
            "Selecao Dados",
            f"Eliminacao dos registros nao analiticos",
            f"{q_inicial - df.shape[0]}",
        )
    )

    # df = remover_colunas_naosignificativas(df)

    an_ind_df = pd.DataFrame([a_dict])
    an_ind_df.astype("int64")
    an_ind_df.insert(0, "Indicador", ["Registros eliminados na seleção dos dados"])

    return df, an_ind_df


if __name__ == "__main__":
    log = Log()
    log.carregar_log("log_BaseTransfor")
    df_base = f.leitura_arquivo_parquet("BaseTransfor")

    print(
        log.logar_acao_realizada(
            "Carga Dados",
            "Carregamento da base dos dados para seleção",
            df_base.shape[0],
        )
    )

    result_df = pd.DataFrame()
    df_base, result_df = main(df_base)

    log.salvar_log("log_BaseModelagem")
    f.salvar_parquet(df_base, "BaseModelagem")
    f.salvar_excel_conclusao(result_df, "parcial_extracao")

    a = log.asString()
