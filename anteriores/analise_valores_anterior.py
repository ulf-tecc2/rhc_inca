# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Analise dos dados.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: ulf Bergmann

#Roteiro de execucao:
    1. Tornar nulos valores invalidos 
    2. Validar a variavel IDADE
    3. Validar os codigos de municipios
    4. Inferir valores para a variável BASMAIMP
    5. Inferir valores para a variável ESTDFIMT
    6. Analisar a variavel LOCTUDET
    7. Analisar a variavel LOCTUPRI
    8. Analisar a variavel LOCTUPRO
    9. Analisar a variavel TNM
    10. Analisar a variavel ESTADIAM
    11. Salvar o arquivo como parquet
    
"""

import pandas as pd
import numpy as np
import re

import sys

sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

import funcoes as f
from funcoes import Log
import bib_graficos as ug


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

    if len(a_str) > 7:
        return a_str[0:7]

    return np.nan


def tratar_variavel_municipio(df):
    """Valida os codigos do municipios transformando em None se invalidos.

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    aux_nulos = df["PROCEDEN"].isnull().sum()
    df["PROCEDEN"] = df["PROCEDEN"].apply(tratar_codigo_municipio)
    aux_nulos1 = df["PROCEDEN"].isnull().sum()
    log.logar_indicador(
        "Consistencia",
        "Registros errados",
        "registros com o Código do Municipio invalido",
        "PROCEDEN",
        aux_nulos1 - aux_nulos,
    )

    aux_nulos = df["MUUH"].isnull().sum()
    df["MUUH"] = df["MUUH"].apply(tratar_codigo_municipio)
    aux_nulos1 = df["MUUH"].isnull().sum()
    log.logar_indicador(
        "Consistencia",
        "Registros errados",
        "registros com o Código do Municipio invalido",
        "MUUH",
        aux_nulos1 - aux_nulos,
    )

    return df


# def trocar_valores_invalidos_para_nulos(df):
#     """Trocar os valores que representam brancos / nulos (99, 999, ...) por None.

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado
#     """
#     nan_values = {
#         "ALCOOLIS": ["0"],
#         "BASDIAGSP": [""],
#         "BASMAIMP": ["0"],
#         "CLIATEN": ["99", "0"],
#         "CLITRAT": ["99", "0"],
#         "DIAGANT": ["0"],
#         "DTDIAGNO": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
#         "DTTRIAGE": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
#         "DATAPRICON": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
#         "DATAOBITO": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
#         "DATAINITRT": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
#         "ESTADIAM": ["99", "88", "nan"],
#         "ESTADRES": ["99", "77", "nan"],
#         "ESTADIAG": ["nan"],
#         "ESTCONJ": ["0"],
#         "ESTDFIMT": ["0"],
#         "EXDIAG": ["0", "<NA>"],
#         "HISTFAMC": ["0"],
#         "INSTRUC": [""],
#         "LATERALI": ["0"],
#         "LOCALNAS": ["99", "nan"],
#         "LOCTUDET": ["nan", "D46", "E05", "N62", "C"],
#         "LOCTUPRI": ["nan", "D46", "E05", "N62", "C .", "C .0"],
#         "LOCTUPRO": ["", ",", ".", "9", "9.", "nan"],
#         "MAISUMTU": ["0"],
#         "ORIENC": ["0"],
#         "OUTROESTA": [
#             "",
#             "99",
#             "88",
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
#     }

#     response_df = pd.DataFrame()

#     for a_var in nan_values:
#         q_antes = df[a_var].isnull().sum()
#         df[a_var] = df[a_var].apply(lambda x: np.nan if (x in nan_values[a_var]) else x)
#         q_depois = df[a_var].isnull().sum()

#         nome_analise = "_Analise" + a_var
#         b = pd.Series({"Regra Nulo": nan_values[a_var]}, name=nome_analise)
#         a_df = pd.DataFrame(b)
#         a_df.index.names = ["Atributo"]
#         a_df.columns = [a_var]

#         if (q_depois - q_antes) > 0:
#             log.logar_indicador(
#                 "Consistencia",
#                 "Registros errados",
#                 f"registros com codigos inexistentes: {nan_values[a_var]}",
#                 a_var,
#                 q_depois - q_antes,
#             )

#         response_df = pd.concat([response_df, a_df], axis=1)

#     return df, response_df



# def valida_anos(df):
#     """Substitui valores invalidos por None.

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado
#     """
#     lista_colunas = ["ANOPRIDI", "ANTRI", "DTPRICON", "DTINITRT"]

#     for a_var in lista_colunas:
#         q_antes = df[a_var].isnull().sum()
#         df[a_var] = df[a_var].apply(lambda x: np.nan if (x < 1900 or x > 2023) else x)
#         q_depois = df[a_var].isnull().sum()
#         if (q_depois - q_antes) > 0:
#             log.logar_indicador(
#                 "Consistencia",
#                 "Registros errados",
#                 "Valores de anos invalidos (< 1900 ou > 2023)",
#                 a_var,
#                 q_depois - q_antes,
#             )

#     return df


# def valida_intervalo_datas(df):
#     """Substitui valores invalidos por None.

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado
#     """
#     lista_colunas = ["DTDIAGNO", "DTTRIAGE", "DATAPRICON", "DATAOBITO" , 'DATAINITRT']

#     for a_var in lista_colunas:
#         q_antes = df[a_var].isnull().sum()
#         # df[a_var] = df[a_var].apply(lambda x: strip_spaces(x) )
#         # df[a_var] = df[a_var].apply(lambda x: np.nan if (strip_spaces(x) in nan_values) else strip_spaces(x))
#         df[a_var] = df[a_var].apply(
#             lambda x: np.nan if (x.year < 1960 or x.year > 2023) else x
#         )
#         q_depois = df[a_var].isnull().sum()
#         if (q_depois - q_antes) > 0:
#             log.logar_indicador(
#                 "Consistencia",
#                 "Registros errados",
#                 "Valores de datas invalidas (< 1960 ou > 2023)",
#                 a_var,
#                 q_depois - q_antes,
#             )

#     return df


# def valida_idade(df):
#     """Valida idade entre 0 e 110 anos. Torna o valor -1 se for diferente.

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado
#     """
#     q_antes = df["IDADE"].isnull().sum()
#     df["IDADE"] = df["IDADE"].apply(lambda x: np.nan if x < 0 or x > 110 else int(x))

#     q_depois = df["IDADE"].isnull().sum()
#     if (q_depois - q_antes) > 0:
#         log.logar_indicador(
#             "Consistencia",
#             "Registros errados",
#             "Valores de idade invalidas (< 0 ou > 110)",
#             "IDADE",
#             q_depois - q_antes,
#         )

#     return df
    # a = df_unico[a_var].value_counts(dropna=False, normalize=False)
    # df_unico[a_var].info()

def analisa_datas_anos(df):
    
    colunas_datas = ['ANOPRIDI', 'ANTRI', 'DTPRICON' , 'DTINITRT','DTDIAGNO', 'DTTRIAGE', 'DATAPRICON', 'DATAOBITO' , 'DATAINITRT']
    
    analise_datas = pd.DataFrame()
    analise_datas['Nulos'] = df[colunas_datas].isnull().sum()
    analise_datas['Nulos %'] = (df[colunas_datas].isnull().sum() / df.shape[0] * 100).round(2)
    
    #dicionario ANO : DATA 
    dict_datas = {
        'ANOPRIDI' : 'DTDIAGNO' ,
        'ANTRI' : 'DTTRIAGE', 
        'DTPRICON' : 'DATAPRICON' , 
        'DTINITRT' : 'DATAINITRT',
        }
    
    for key in dict_datas:
        var_ano = key
        var_data = dict_datas[key]
    
        aux_df = df[[var_data , var_ano]]
        a = aux_df.loc[ ~aux_df[var_ano].isnull() & ~aux_df[var_data].isnull() & (aux_df[var_data].dt.year != aux_df[var_ano].astype('float64')) ]
        # print(f'Inconsistencias entre as variaveis {var_ano} e {var_data} => {a.shape[0]}')
        analise_datas.at[var_ano , 'Inconsistencia Ano'] = a.shape[0]
        a = aux_df.loc[aux_df[var_ano].isnull() ]
        b = a.loc[ ~ a[var_data].isnull() ]
        # print(f'Possibilidade de gerar {var_ano} usando {var_data} => {b.shape[0]}')
        analise_datas.at[var_ano , 'Possibilidade Gerar Ano'] = b.shape[0]
        aux_df = df[[var_data , var_ano]]
        a = aux_df.loc[aux_df[var_data].isnull() ]
        b = a.loc[ ~ a[var_ano].isnull() ]
        # print(f'Possibilidade de gerar {var_data} usando apenas o ano de {var_ano} => {b.shape[0]}')
        analise_datas.at[var_data , 'Possibilidade Gerar Data'] = b.shape[0]
    
    return analise_datas


#analisar  a ordem e ver se aconteceu antes. Inconsistencia 
# def analisa_ordem_datas(df , lista_campos):
#     resultado_analise = pd.DataFrame()
#     resultado_analise.at[f"Registros analisados" , 'Quantidade'] = df.shape[0]
#     for i in range(0 , len(lista_campos) - 1):
#         var_1 = lista_campos[i]
#         var_2 = lista_campos[i+1]

#         aux_df = df.loc[~ df[var_2].isnull() ]
#         aux_df = aux_df.loc[~ aux_df[var_1].isnull() ]
#         aux_df = aux_df.loc[ aux_df[var_1] > aux_df[var_2] ]
#         aux_df['intervalo'] = (aux_df[var_2] - aux_df[var_1]).dt.days
#         print(f"{var_2} anterior a {var_1} {aux_df.shape[0]}")
#         resultado_analise.at[f"{var_2} anterior a {var_1}" , 'Quantidade'] = aux_df.shape[0]
        
#     return resultado_analise

# def analisaDATAOBITO(df):
#     """Analisa os valores da variavel DATAOBITO gerando a variavel <_intervalo obito> .
    
#     Analisar DATAOBITO DATAINITRT  : 
#         1) calcular intervalo do tratamento com o obito
#         - possui DATAOBITO - Data de fim do intervalo
#         - data a considerar para inicio:   
#             a) DATAINITRT se não nulo
#             b) DATAPRICON demais casos (nao tem nulos))
    
#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado com as colunas inseridas
#     """     
#     df['_intervalo obito'] = np.where( df['DATAOBITO'].isnull()   , np.nan ,
#                                          (df['DATAOBITO'] - np.where(~df['DATAINITRT'].isnull(), df['DATAINITRT'], df['DATAPRICON'])).dt.days)

#     a_valor = df['_intervalo obito'].quantile(0.90)
#     a_media = df['_intervalo obito'].mean()
#     a = df[(df['_intervalo obito'] > 0) & (df['_intervalo obito'] < a_media + a_valor)]
    
#     ug.plot_frequencias_valores_atributos(a , ['_intervalo obito'] , bins = 20 , title = 'Intervalo ate o obito (dias)')
#     ug.plota_histograma_distribuicao(a , '_intervalo obito' )
    
#     print(
#         log.logar_acao_realizada(
#             "Analise de valores",
#             "Geracao da variavel <_intervalo obito>",
#             "",
#         )
#     )
    
#     return df


# def analisa_LOCTUDET(df):
#     """Analisa os valores da variavel LOCTUDET verificando se seguem os formatos padroes.

#     LOCTUDET - Localização primária do tumor pela CID-O, 3 dígitos
#     Insere uma nova coluna <_AnaliseLOCTUDET> com os resultados: <exato | incompleto | nulo | demais>
#     Insere uma nova coluna <_AnaliseLOCTUDET_tipo> com os resultados: <Hemato | demais>

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado com as colunas inseridas
#     """
#     nome_variavel = "LOCTUDET"
#     nome_analise = "_Analise" + nome_variavel

#     r_exato = r"C\d{2}"
#     r_incompleto = r"C\.?"

#     df[nome_analise] = np.where(
#         df[nome_variavel].str.contains(r_exato, regex=True, na=False),
#         "exato",
#         np.where(
#             df[nome_variavel].str.contains(r_incompleto, regex=True, na=False),
#             "incompleto",
#             np.where(df[nome_variavel].isnull(), "nulo", "demais"),
#         ),
#     )

#     a = df[nome_analise].value_counts(dropna=False, normalize=False)
#     a_df = pd.DataFrame(a)
#     a_df.index.names = ["Atributo"]
#     a_df.columns = [nome_variavel]

#     b = pd.Series(
#         {"Regra exata": r_exato, "Regra incompleta": r_incompleto}, name=nome_analise
#     )
#     b_df = pd.DataFrame(b)
#     b_df.index.names = ["Atributo"]
#     b_df.columns = [nome_variavel]

#     a = a.reset_index()
#     fora_padrao = ["incompleto", "demais"]
#     count_invalidos = 0
#     for i, row in a.iterrows():
#         # print((var in informacao_ignorada_por_atributo.keys()) and (row[var] in informacao_ignorada_por_atributo[var] ))
#         if row["Atributo"] in fora_padrao:
#             count_invalidos = count_invalidos + row["count"]
#     log.logar_indicador(
#         "Consistencia",
#         "Fora padrao",
#         f"registros nao atendem aos padroes definidos : {fora_padrao}",
#         nome_variavel,
#         count_invalidos,
#     )

#     hemato_values = {
#         "C81",
#         "C82",
#         "C83",
#         "C84",
#         "C85",
#         "C91",
#         "C92",
#         "C93",
#         "C94",
#         "C95",
#         "C96",
#         "C97",
#     }

#     df[nome_analise + "_tipo"] = df[nome_variavel].apply(
#         lambda x: "Hemato" if (x in hemato_values) else "demais"
#     )

#     ab_df = pd.concat([a_df, b_df], axis=0)

#     return df, ab_df


# def analisa_LOCTUPRI(df):
#     """Analisa os valores da variavel LOCTUPRI verificando se seguem os formatos padroes.

#     LOCTUPRI - Localização primária do tumor pela CID-O, 4 dígitos
#     Insere uma nova coluna <_AnaliseLOCTUPRI> com os resultados: <exato | incompleto | nulo | demais>

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado com a coluna inserida
#     """
#     nome_variavel = "LOCTUPRI"
#     nome_analise = "_Analise" + nome_variavel

#     r_exato = r"C\d{2}\.\d"
#     r_incompleto = r"C\d{2}\.?"

#     df[nome_analise] = np.where(
#         df[nome_variavel].str.contains(r_exato, regex=True, na=False),
#         "exato",
#         np.where(
#             df[nome_variavel].str.contains(r_incompleto, regex=True, na=False),
#             "incompleto",
#             np.where(df[nome_variavel].isnull(), "nulo", "demais"),
#         ),
#     )

#     a = df[nome_analise].value_counts(dropna=False, normalize=False)
#     a_df = pd.DataFrame(a)
#     a_df.index.names = ["Atributo"]
#     a_df.columns = [nome_variavel]

#     b = pd.Series(
#         {"Regra exata": r_exato, "Regra incompleta": r_incompleto}, name=nome_analise
#     )
#     b_df = pd.DataFrame(b)
#     b_df.index.names = ["Atributo"]
#     b_df.columns = [nome_variavel]

#     ab_df = pd.concat([a_df, b_df], axis=0)

#     a = a.reset_index()
#     fora_padrao = ["incompleto", "demais"]
#     count_invalidos = 0
#     for i, row in a.iterrows():
#         # print((var in informacao_ignorada_por_atributo.keys()) and (row[var] in informacao_ignorada_por_atributo[var] ))
#         if row["Atributo"] in fora_padrao:
#             count_invalidos = count_invalidos + row["count"]
#     log.logar_indicador(
#         "Consistencia",
#         "Fora padrao",
#         f"registros nao atendem aos padroes definidos : {fora_padrao}",
#         nome_variavel,
#         count_invalidos,
#     )

#     return df, ab_df


# def analisa_LOCTUPRO(df):
#     """Analisa os valores da variavel LOCTUPRO verificando se seguem os formatos padroes.

#     Localização primária provável do tumor pela CID-O, 4 dígitos
#     Insere uma nova coluna <_AnaliseLOCTUPRO> com os resultados: <exato | incompleto | nulo | demais>

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado com a coluna inserida
#     """
#     nome_variavel = "LOCTUPRO"
#     nome_analise = "_Analise" + nome_variavel

#     r_exato = r"C\d{2}\.\d"
#     r_incompleto = r"C\d{2}\.?"

#     df[nome_analise] = np.where(
#         df[nome_variavel].str.contains(r_exato, regex=True, na=False),
#         "exato",
#         np.where(
#             df[nome_variavel].str.contains(r_incompleto, regex=True, na=False),
#             "incompleto",
#             np.where(df[nome_variavel].isnull(), "nulo", "demais"),
#         ),
#     )

#     a = df[nome_analise].value_counts(dropna=False, normalize=False)
#     a_df = pd.DataFrame(a)
#     a_df.index.names = ["Atributo"]
#     a_df.columns = [nome_variavel]

#     b = pd.Series(
#         {"Regra exata": r_exato, "Regra incompleta": r_incompleto}, name=nome_analise
#     )
#     b_df = pd.DataFrame(b)
#     b_df.index.names = ["Atributo"]
#     b_df.columns = [nome_variavel]

#     ab_df = pd.concat([a_df, b_df], axis=0)

#     a = a.reset_index()
#     fora_padrao = ["incompleto", "demais"]
#     count_invalidos = 0
#     for i, row in a.iterrows():
#         # print((var in informacao_ignorada_por_atributo.keys()) and (row[var] in informacao_ignorada_por_atributo[var] ))
#         if row["Atributo"] in fora_padrao:
#             count_invalidos = count_invalidos + row["count"]
#     log.logar_indicador(
#         "Consistencia",
#         "Fora padrao",
#         f"registros nao atendem aos padroes definidos : {fora_padrao}",
#         nome_variavel,
#         count_invalidos,
#     )

#     return df, ab_df


# def analisa_TNM_PTNM(df , nome_variavel = "TNM" ):
#     """Analisa os valores da variavel TNM verificando se seguem os formatos padroes.

#     TNM: Codificação do estádio clínico segundo classificação TNM
#     T -a extensão do tumor primário
#     N -a ausência ou presença e a extensão de metástase em linfonodos regionais ,
#     M -a ausência ou presença de metástase à distância

#     A adição de números a estes três componentes indica a extensão da doença maligna. Assim temos:
#     T0, TI, T2, T3, T4 - N0, Nl, N2, N3 - M0, Ml

#     Insere uma nova coluna <_AnaliseTNM> com os resultados: <exato | incompleto | nao se aplica - Hemato | nao se aplica - Geral | nulo | demais>

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado com a coluna inserida
#     """
    
#     nome_analise = "_Analise" + nome_variavel

#     r_invalido = r"^999|99$"
#     r_nao_se_aplica_geral = r"^888|88|988|898|889$"
#     r_exato = r"^[0-4IA][0-4][0-1]$"
#     r_incompleto = r"^[0-4Xx][0-4Xx][0-1Xx]$"
#     r_regrasp = r"^YYY|XXX$"
#     r_nao_se_aplica_hemato = "Hemato"

#     df[nome_analise] = np.where(
#         df["_AnaliseLOCTUDET_tipo"].str.contains(
#             r_nao_se_aplica_hemato, regex=False, na=False
#         ),
#         "nao se aplica - Hemato",
#         np.where(
#             df[nome_variavel].str.contains(r_invalido, regex=True, na=False),
#             "invalido",
#             np.where(
#                 df[nome_variavel].str.contains(
#                     r_nao_se_aplica_geral, regex=True, na=False
#                 ),
#                 "nao se aplica - Geral",
#                 np.where(
#                     df[nome_variavel].str.contains(r_exato, regex=True, na=False),
#                     "exato",
#                     np.where(
#                         df[nome_variavel].str.contains(
#                             r_incompleto, regex=True, na=False
#                         ),
#                         "incompleto",
#                         np.where(
#                             df[nome_variavel].str.contains(
#                                 r_regrasp, regex=True, na=False
#                             ),
#                             "nao se aplica - regra SP",
#                             np.where(df[nome_variavel].isnull(), "nulo",
#                                      np.where(df[nome_variavel].isna(), "nulo", "demais")
#                                      )
#                         ),
#                     ),
#                 ),
#             ),
#         ),
#     )

#     a = df[nome_analise].value_counts(dropna=False, normalize=False)
#     a_df = pd.DataFrame(a)
#     a_df.index.names = ["Atributo"]
#     a_df.columns = [nome_variavel]

#     b = pd.Series(
#         {
#             "Regra exata": r_exato,
#             "Regra incompleta": r_incompleto,
#             "Regra não se aplica - Hemato ": "CID-O de Hemato em LOCTUDET",
#             "Regra não se aplica": r_nao_se_aplica_geral,
#             "Regra de SP ": r_regrasp,
#             "Regra invalido": r_invalido,
#         },
#         name=nome_analise,
#     )
#     b_df = pd.DataFrame(b)
#     b_df.index.names = ["Atributo"]
#     b_df.columns = [nome_variavel]

#     ab_df = pd.concat([a_df, b_df], axis=0)

#     a = a.reset_index()
#     fora_padrao = ["invalido", "incompleto", "demais"]
#     count_invalidos = 0
#     for i, row in a.iterrows():
#         # print((var in informacao_ignorada_por_atributo.keys()) and (row[var] in informacao_ignorada_por_atributo[var] ))
#         if row["Atributo"] in fora_padrao:
#             count_invalidos = count_invalidos + row["count"]
#     log.logar_indicador(
#         "Consistencia",
#         "Fora padrao",
#         f"registros nao atendem aos padroes definidos : {fora_padrao}",
#         nome_variavel,
#         count_invalidos,
#     )

#     return df, ab_df


# def analisa_PRITRATH(df):
#     pattern = r'^[1-8]+$'
#     falta implementar
#     1.Nenhum; 2. Cirurgia; 3.Radioterapia; 4.Quimioterapia; 5.Hormonioterapia; 6.Transplante de medula óssea;
# 7.Imunoterapia; 8.Outras; 9.Sem informação

# checar se nenhum tratamento o que fazer. Checar com RZNTR. •	


# def analisa_ESTADIAM(df):
#     """Analisa os valores da variavel ESTADIAM verificando se seguem os formatos padroes.

#     Localização primária provável do tumor pela CID-O, 4 dígitos
#     Insere uma nova coluna <_AnaliseESTADIAM> com os resultados: <exato | nulo | demais>

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado com a coluna inserida
#     """
#     nome_variavel = "ESTADIAM"
#     nome_analise = "_Analise" + nome_variavel

#     r_exato = r"(^I|II|III|IV$)|(^[0-4]$)|(^[0-4][A|B|C]$)|(^0[0-4]$)"

#     r_exato1 = r"^I|II|III|IV$"
#     r_exato2 = r"^[0-4]$"
#     r_exato3 = r"^[0-4][A|B|C]$"
#     r_exato4 = r"^0[0-4]$"

#     df[nome_analise] = np.where(
#         df[nome_variavel].str.contains(r_exato1, regex=True, na=False),
#         "exato",
#         np.where(
#             df[nome_variavel].str.contains(r_exato2, regex=True, na=False),
#             "exato",
#             np.where(
#                 df[nome_variavel].str.contains(r_exato3, regex=True, na=False),
#                 "exato",
#                 np.where(
#                     df[nome_variavel].str.contains(r_exato4, regex=True, na=False),
#                     "exato",
#                     np.where(df[nome_variavel].isnull(), "nulo", "demais"),
#                 ),
#             ),
#         ),
#     )
#     a = df[nome_analise].value_counts(dropna=False, normalize=False)
#     a_df = pd.DataFrame(a)
#     a_df.index.names = ["Atributo"]
#     a_df.columns = [nome_variavel]

#     b = pd.Series({"Regra exata": r_exato}, name=nome_analise)
#     b_df = pd.DataFrame(b)
#     b_df.index.names = ["Atributo"]
#     b_df.columns = [nome_variavel]

#     ab_df = pd.concat([a_df, b_df], axis=0)

#     a = a.reset_index()
#     fora_padrao = ["incompleto", "demais"]
#     count_invalidos = 0
#     for i, row in a.iterrows():
#         # print((var in informacao_ignorada_por_atributo.keys()) and (row[var] in informacao_ignorada_por_atributo[var] ))
#         if row["Atributo"] in fora_padrao:
#             count_invalidos = count_invalidos + row["count"]
#     log.logar_indicador(
#         "Consistencia",
#         "Fora padrao",
#         f"registros nao atendem aos padroes definidos : {fora_padrao}",
#         nome_variavel,
#         count_invalidos,
#     )

#     return df, ab_df




def main(df):
    """Funcao principal.

    Parameters:
        df (DataFrame) : df a ser modificado
    Returns:
        (DataFrame): df modificado
    """
    global df_unico

    # TORNAR VALORES INVALIDOS COMO NULOS - NONE
    # ind_antes=df.isnull().sum()

    # df_unico, df_regras = trocar_valores_invalidos_para_nulos(df_unico)

    # df_unico = valida_idade(df_unico)
    
    # df_unico = valida_intervalo_datas(df_unico)
    # df_unico = valida_anos(df_unico)
    df_unico = tratar_variavel_municipio(df_unico)

    print(
        log.logar_acao_realizada(
            "Valores Invalidos",
            "Corrigir valores invalidos transformando em null. ",
            "",
        )
    )
    
    # lista_campos = ['DTDIAGNO', 'DTTRIAGE', 'DATAPRICON' ,'DATAINITRT', 'DATAOBITO'] # ordem correta
    # a = analisa_ordem_datas(df , lista_campos)
    # f.salvar_excel_conclusao(a , 'ResOrdemDatas')
    
    b = analisa_datas_anos(df)
    f.salvar_excel_conclusao(b , 'ResDatasAnos')

    print(
        log.logar_acao_realizada(
            "Valores Invalidos",
            "Analise da ordem das datas e da consistencia entre variaveis ANO e DATA. Ver arquivos ResOrdemDatas.xlsx e ResDatasAnos.xlsx ",
            "",
        )
    )
    
    # ind_depois=df_unico.isnull().sum()

    # df_aux = pd.DataFrame()
    # df_aux['Nulos antes'] = ind_antes
    # df_aux['Nulos depois'] = ind_depois
    # df_aux['diferenca'] = ind_depois- ind_antes

    # a_file_name = 'valores_tornados_null'
    # f.salvar_excel_conclusao(df_aux , a_file_name)
    
    df_unico = analisaDATAOBITO(df_unico)

    # df_unico, df_regras_aux = analisa_LOCTUDET(df_unico)
    # df_regras = pd.concat([df_regras, df_regras_aux], axis=1)
    # print(
    #     log.logar_acao_realizada(
    #         "Analise de valores", f"Analisar o valor de LOCTUDET", ""
    #     )
    # )

    # df_unico, df_regras_aux = analisa_LOCTUPRI(df_unico)
    # df_regras = pd.concat([df_regras, df_regras_aux], axis=1)
    # print(
    #     log.logar_acao_realizada(
    #         "Analise de valores", f"Analisar o valor de LOCTUPRI", ""
    #     )
    # )

    # df_unico, df_regras_aux = analisa_LOCTUPRO(df_unico)
    # df_regras = pd.concat([df_regras, df_regras_aux], axis=1)
    # print(
    #     log.logar_acao_realizada(
    #         "Analise de valores", f"Analisar o valor de LOCTUPRO", ""
    #     )
    # )

    # df_unico, df_regras_aux = analisa_TNM_PTNM(df_unico , nome_variavel = "TNM")
    # df_regras = pd.concat([df_regras, df_regras_aux], axis=1)
    # print(
    #     log.logar_acao_realizada("Analise de valores", f"Analisar o valor de TNM", "")
    # )
    
    # df_unico, df_regras_aux = analisa_TNM_PTNM(df_unico , nome_variavel = "PTNM")
    # df_regras = pd.concat([df_regras, df_regras_aux], axis=1)
    # print(
    #     log.logar_acao_realizada("Analise de valores", f"Analisar o valor de TNM", "")
    # )

    # df_unico, df_regras_aux = analisa_ESTADIAM(df_unico)
    # df_regras = pd.concat([df_regras, df_regras_aux], axis=1)
    # print(
    #     log.logar_acao_realizada(
    #         "Analise de valores", f"Analisar o valor de ESTADIAM", ""
    #     )
    # )

    # df_regras = df_regras.fillna("")

    # a_nome_arquivo = "RegrasAnaliseValores"
    # f.salvar_excel_conclusao(df_regras.T, a_nome_arquivo)
    
    # print(
    #     log.logar_acao_realizada(
    #         "Analise de valores",
    #         "Resultados consolidados da analise dos valores",
    #         f"ver arquivo {a_nome_arquivo}",
    #     )
    # )
    
    

    return df_unico


if __name__ == "__main__":
    log = Log()
    log.carregar_log("log_BaseInicial")

    # df_unico = f.leitura_arquivo_parquet('BaseAnaliticos')
    # print( log.logar_acao_realizada('Carga Dados' , 'Carregamento da base dos dados a serem analisados - Casos analiticos' , df_unico.shape[0]) )

    df_unico = f.leitura_arquivo_parquet("BaseInicial")

    restringir_periodo = False
    if restringir_periodo:  # usar 2000 a 2020 apenas
        df_unico = df_unico[
            (df_unico["DATAINITRT"].dt.year > 1999)
            & (df_unico["DATAINITRT"].dt.year < 2021)
        ]

    print(
        log.logar_acao_realizada(
            "Carga Dados",
            "Carregamento da base dos dados a serem analisados - Casos completos",
            df_unico.shape[0],
        )
    )

    # if True:
    #     df_unico = filtrar_registros_validos(df_unico)
    #     print(
    #         log.logar_acao_realizada(
    #             "Carga Dados",
    #             "Restricao dos dados a serem analisados - Casos Validos (analiticos e sem razao para nao tratamento",
    #             df_unico.shape[0],
    #         )
    #      )   


    df_unico = main(df_unico)

    log.salvar_log("log_BaseSanitizada")
    f.salvar_parquet(df_unico, "BaseSanitizada")
