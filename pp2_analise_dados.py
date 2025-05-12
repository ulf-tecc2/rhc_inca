# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Analise das inconsistencias.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: Ulf Bergmann


"""

import pandas as pd
import numpy as np
import sys
sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")


import funcoes as f
from funcoes import Log


import funcoes_ulf as uf
import bib_graficos as ug

#%% ANALISE VARIAVEIS
def analisa_LOCTUDET(df):
    """Analisa os valores da variavel LOCTUDET verificando se seguem os formatos padroes.

    LOCTUDET - Localização primária do tumor pela CID-O, 3 dígitos
    Insere uma nova coluna <_AnaliseLOCTUDET> com os resultados: <exato | incompleto | nulo | demais>
    Insere uma nova coluna <_AnaliseLOCTUDET_tipo> com os resultados: <Hemato | demais>

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado com as colunas inseridas
    """
    nome_variavel = "LOCTUDET"
    nome_analise = "_Analise" + nome_variavel

    r_exato = r"C\d{2}"
    r_incompleto = r"C\.?"

    sem_info = []
    nan_values = ["D46", "E05", "N62", "C"]
    df[nome_analise] = np.where(
        df[nome_variavel].str.contains(r_exato, regex=True, na=False),
        "exato",
        np.where(
            df[nome_variavel].str.contains(r_incompleto, regex=True, na=False),
            "incompleto",
            np.where((df[nome_variavel].isnull()) | (df[nome_variavel].isin(nan_values)) , "nulo",
                     np.where((df[nome_variavel].isin(sem_info)), "sem_info", "demais"))
        ),
    )

    # a = df[nome_analise].value_counts(dropna=False, normalize=False)
    # a_df = pd.DataFrame(a)
    # a_df.index.names = ["Atributo"]
    # a_df.columns = [nome_variavel]

    # b = pd.Series(
    #     {"Regra exata": r_exato, "Regra incompleta": r_incompleto}, name=nome_analise
    # )
    # b_df = pd.DataFrame(b)
    # b_df.index.names = ["Atributo"]
    # b_df.columns = [nome_variavel]

    # a = a.reset_index()
    # fora_padrao = ["incompleto", "demais"]
    # count_invalidos = 0
    # for i, row in a.iterrows():
    #     # print((var in informacao_ignorada_por_atributo.keys()) and (row[var] in informacao_ignorada_por_atributo[var] ))
    #     if row["Atributo"] in fora_padrao:
    #         count_invalidos = count_invalidos + row["count"]
    # log.logar_indicador(
    #     "Consistencia",
    #     "Fora padrao",
    #     f"registros nao atendem aos padroes definidos : {fora_padrao}",
    #     nome_variavel,
    #     count_invalidos,
    # )

    hemato_values = {
        "C81",
        "C82",
        "C83",
        "C84",
        "C85",
        "C91",
        "C92",
        "C93",
        "C94",
        "C95",
        "C96",
        "C97",
    }

    df[nome_analise + "_tipo"] = df[nome_variavel].apply(
        lambda x: "Hemato" if (x in hemato_values) else "demais"
    )

    # ab_df = pd.concat([a_df, b_df], axis=0)

    return df


def analisa_LOCTUPRI(df):
    """Analisa os valores da variavel LOCTUPRI verificando se seguem os formatos padroes.

    LOCTUPRI - Localização primária do tumor pela CID-O, 4 dígitos
    Insere uma nova coluna <_AnaliseLOCTUPRI> com os resultados: <exato | incompleto | nulo | demais>

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado com a coluna inserida
    """
    nome_variavel = "LOCTUPRI"
    nome_analise = "_Analise" + nome_variavel

    r_exato = r"C\d{2}\.\d"
    r_incompleto = r"C\d{2}\.?"
    
    sem_info = []
    nan_values = ["nan", "D46", "E05", "N62", "C .", "C .0"]
    
    df[nome_analise] = np.where(
        df[nome_variavel].str.contains(r_exato, regex=True, na=False),
        "exato",
        np.where(
            df[nome_variavel].str.contains(r_incompleto, regex=True, na=False),
            "incompleto",
            np.where((df[nome_variavel].isnull()) | (df[nome_variavel].isin(nan_values)) , "nulo",
                     np.where((df[nome_variavel].isin(sem_info)), "sem_info", "demais"))
        ),
    )

    # a = df[nome_analise].value_counts(dropna=False, normalize=False)
    # a_df = pd.DataFrame(a)
    # a_df.index.names = ["Atributo"]
    # a_df.columns = [nome_variavel]

    # b = pd.Series(
    #     {"Regra exata": r_exato, "Regra incompleta": r_incompleto}, name=nome_analise
    # )
    # b_df = pd.DataFrame(b)
    # b_df.index.names = ["Atributo"]
    # b_df.columns = [nome_variavel]

    # ab_df = pd.concat([a_df, b_df], axis=0)

    # a = a.reset_index()
    # fora_padrao = ["incompleto", "demais"]
    # count_invalidos = 0
    # for i, row in a.iterrows():
    #     # print((var in informacao_ignorada_por_atributo.keys()) and (row[var] in informacao_ignorada_por_atributo[var] ))
    #     if row["Atributo"] in fora_padrao:
    #         count_invalidos = count_invalidos + row["count"]
    # log.logar_indicador(
    #     "Consistencia",
    #     "Fora padrao",
    #     f"registros nao atendem aos padroes definidos : {fora_padrao}",
    #     nome_variavel,
    #     count_invalidos,
    # )

    return df


def analisa_LOCTUPRO(df):
    """Analisa os valores da variavel LOCTUPRO verificando se seguem os formatos padroes.

    Localização primária provável do tumor pela CID-O, 4 dígitos
    Insere uma nova coluna <_AnaliseLOCTUPRO> com os resultados: <exato | incompleto | nulo | demais>

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado com a coluna inserida
    """
    nome_variavel = "LOCTUPRO"
    nome_analise = "_Analise" + nome_variavel

    r_exato = r"C\d{2}\.\d"
    r_incompleto = r"C\d{2}\.?"

    sem_info = [ "9"]
    nan_values = ["", ",", ".", "9.", "nan"]
                  
    df[nome_analise] = np.where(
        df[nome_variavel].str.contains(r_exato, regex=True, na=False),
        "exato",
        np.where(
            df[nome_variavel].str.contains(r_incompleto, regex=True, na=False),
            "incompleto",
           np.where((df[nome_variavel].isnull()) | (df[nome_variavel].isin(nan_values)) , "nulo",
                    np.where((df[nome_variavel].isin(sem_info)), "sem_info", "demais")),
        ),
    )

    # a = df[nome_analise].value_counts(dropna=False, normalize=False)
    # a_df = pd.DataFrame(a)
    # a_df.index.names = ["Atributo"]
    # a_df.columns = [nome_variavel]

    # b = pd.Series(
    #     {"Regra exata": r_exato, "Regra incompleta": r_incompleto}, name=nome_analise
    # )
    # b_df = pd.DataFrame(b)
    # b_df.index.names = ["Atributo"]
    # b_df.columns = [nome_variavel]

    # ab_df = pd.concat([a_df, b_df], axis=0)

    # a = a.reset_index()
    # fora_padrao = ["incompleto", "demais"]
    # count_invalidos = 0
    # for i, row in a.iterrows():
    #     # print((var in informacao_ignorada_por_atributo.keys()) and (row[var] in informacao_ignorada_por_atributo[var] ))
    #     if row["Atributo"] in fora_padrao:
    #         count_invalidos = count_invalidos + row["count"]
    # log.logar_indicador(
    #     "Consistencia",
    #     "Fora padrao",
    #     f"registros nao atendem aos padroes definidos : {fora_padrao}",
    #     nome_variavel,
    #     count_invalidos,
    # )

    return df


def analisa_TNM_PTNM(df , nome_variavel = "TNM" ):
    """Analisa os valores da variavel TNM verificando se seguem os formatos padroes.

    TNM: Codificação do estádio clínico segundo classificação TNM
    T -a extensão do tumor primário
    N -a ausência ou presença e a extensão de metástase em linfonodos regionais ,
    M -a ausência ou presença de metástase à distância

    A adição de números a estes três componentes indica a extensão da doença maligna. Assim temos:
    T0, TI, T2, T3, T4 - N0, Nl, N2, N3 - M0, Ml

    Insere uma nova coluna <_AnaliseTNM> com os resultados: <exato | incompleto | nao se aplica - Hemato | nao se aplica - Geral | nulo | demais>

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado com a coluna inserida
    """
    
    nome_analise = "_Analise" + nome_variavel

    r_invalido = r"^999|99$"
    r_nao_se_aplica_geral = r"^888|88|988|898|889$"
    r_exato = r"^[0-4IA][0-4][0-1]$"
    r_incompleto = r"^[0-4Xx][0-4Xx][0-1Xx]$"
    r_regrasp = r"^YYY|XXX$"
    r_nao_se_aplica_hemato = "Hemato"
    
    sem_info = ['9']
    nan_values = ['']

    df[nome_analise] = np.where(
        df["_AnaliseLOCTUDET_tipo"].str.contains(
            r_nao_se_aplica_hemato, regex=False, na=False
        ),
        "nao se aplica - Hemato",
        np.where(
            df[nome_variavel].str.contains(r_invalido, regex=True, na=False),
            "invalido",
            np.where(
                df[nome_variavel].str.contains(
                    r_nao_se_aplica_geral, regex=True, na=False
                ),
                "nao se aplica - Geral",
                np.where(
                    df[nome_variavel].str.contains(r_exato, regex=True, na=False),
                    "exato",
                    np.where(
                        df[nome_variavel].str.contains(
                            r_incompleto, regex=True, na=False
                        ),
                        "incompleto",
                        np.where(
                            df[nome_variavel].str.contains(
                                r_regrasp, regex=True, na=False
                            ),
                            "nao se aplica - regra SP",
                            np.where((df[nome_variavel].isnull()) | (df[nome_variavel].isin(nan_values)) , "nulo",
                                     np.where((df[nome_variavel].isin(sem_info)), "sem_info", "demais")),
      
                        ),
                    ),
                ),
            ),
        ),
    )

    # a = df[nome_analise].value_counts(dropna=False, normalize=False)
    # a_df = pd.DataFrame(a)
    # a_df.index.names = ["Atributo"]
    # a_df.columns = [nome_variavel]

    # b = pd.Series(
    #     {
    #         "Regra exata": r_exato,
    #         "Regra incompleta": r_incompleto,
    #         "Regra não se aplica - Hemato ": "CID-O de Hemato em LOCTUDET",
    #         "Regra não se aplica": r_nao_se_aplica_geral,
    #         "Regra de SP ": r_regrasp,
    #         "Regra invalido": r_invalido,
    #     },
    #     name=nome_analise,
    # )
    # b_df = pd.DataFrame(b)
    # b_df.index.names = ["Atributo"]
    # b_df.columns = [nome_variavel]

    # ab_df = pd.concat([a_df, b_df], axis=0)

    # a = a.reset_index()
    # fora_padrao = ["invalido", "incompleto", "demais"]
    # count_invalidos = 0
    # for i, row in a.iterrows():
    #     # print((var in informacao_ignorada_por_atributo.keys()) and (row[var] in informacao_ignorada_por_atributo[var] ))
    #     if row["Atributo"] in fora_padrao:
    #         count_invalidos = count_invalidos + row["count"]
    # log.logar_indicador(
    #     "Consistencia",
    #     "Fora padrao",
    #     f"registros nao atendem aos padroes definidos : {fora_padrao}",
    #     nome_variavel,
    #     count_invalidos,
    # )

    return df

def analisa_ESTADIAM(df):
    """Analisa os valores da variavel ESTADIAM verificando se seguem os formatos padroes.

    Localização primária provável do tumor pela CID-O, 4 dígitos
    Insere uma nova coluna <_AnaliseESTADIAM> com os resultados: <exato | nulo | demais>

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado com a coluna inserida
    """
    nome_variavel = "ESTADIAM"
    nome_analise = "_Analise" + nome_variavel

    r_exato = r"(^I|II|III|IV$)|(^[0-4]$)|(^[0-4][A|B|C]$)|(^0[0-4]$)"

    r_exato1 = r"^I|II|III|IV$"
    r_exato2 = r"^[0-4]$"
    r_exato3 = r"^[0-4][A|B|C]$"
    r_exato4 = r"^0[0-4]$"
    
    sem_info = ['88' , '99']
    nan_values = ['nan']

    df[nome_analise] = np.where(
        df[nome_variavel].str.contains(r_exato1, regex=True, na=False),
        "exato",
        np.where(
            df[nome_variavel].str.contains(r_exato2, regex=True, na=False),
            "exato",
            np.where(
                df[nome_variavel].str.contains(r_exato3, regex=True, na=False),
                "exato",
                np.where(
                    df[nome_variavel].str.contains(r_exato4, regex=True, na=False),
                    "exato",
                    np.where((df[nome_variavel].isnull()) | (df[nome_variavel].isin(nan_values)) , "nulo",
                             np.where((df[nome_variavel].isin(sem_info)), "sem_info", "demais")),
                ),
            ),
        ),
    )
    # a = df[nome_analise].value_counts(dropna=False, normalize=False)
    # a_df = pd.DataFrame(a)
    # a_df.index.names = ["Atributo"]
    # a_df.columns = [nome_variavel]

    # b = pd.Series({"Regra exata": r_exato}, name=nome_analise)
    # b_df = pd.DataFrame(b)
    # b_df.index.names = ["Atributo"]
    # b_df.columns = [nome_variavel]

    # ab_df = pd.concat([a_df, b_df], axis=0)

    # a = a.reset_index()
    # fora_padrao = ["incompleto", "demais"]
    # count_invalidos = 0
    # for i, row in a.iterrows():
    #     # print((var in informacao_ignorada_por_atributo.keys()) and (row[var] in informacao_ignorada_por_atributo[var] ))
    #     if row["Atributo"] in fora_padrao:
    #         count_invalidos = count_invalidos + row["count"]
    # log.logar_indicador(
    #     "Consistencia",
    #     "Fora padrao",
    #     f"registros nao atendem aos padroes definidos : {fora_padrao}",
    #     nome_variavel,
    #     count_invalidos,
    # )

    return df

# def analisa_PRITRATH(df):
#     pattern = r'^[1-8]+$'
#     falta implementar
#     1.Nenhum; 2. Cirurgia; 3.Radioterapia; 4.Quimioterapia; 5.Hormonioterapia; 6.Transplante de medula óssea;
# 7.Imunoterapia; 8.Outras; 9.Sem informação

# checar se nenhum tratamento o que fazer. Checar com RZNTR. •	

def analisa_PRITRATH(df):
    """Analisa os valores da variavel ESTADIAM verificando se seguem os formatos padroes. 
    Preenche a informacao se teve ou nao tratamento (_TeveTratamento)

    1.Nenhum; 2. Cirurgia; 3.Radioterapia; 4.Quimioterapia; 5.Hormonioterapia; 6.Transplante de medula óssea; 7.Imunoterapia; 8.Outras; 9.Sem informação

    Insere uma nova coluna <_AnalisePRITRATH> com os resultados: <exato | nulo | demais>

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado com a coluna inserida
    """
    nome_variavel = "PRITRATH"
    nome_analise = "_Analise" + nome_variavel

    r_exato = r'^[1-8]+$'
    r_nulo = r'^9+$'

    sem_info = ['9']
    nan_values = []
    
    df[nome_analise] = np.where(
        df[nome_variavel].str.contains(r_exato, regex=True, na=False),
        "exato", 
        np.where((df[nome_variavel].isnull()) | (df[nome_variavel].isin(nan_values)) , "nulo",
                 np.where((df[nome_variavel].isin(sem_info)), "sem_info", "demais")))
    
    df['_TeveTratamento'] = False
    df.loc[ df[nome_analise] == 'exato' , '_TeveTratamento'] = True



    # a = df[nome_analise].value_counts(dropna=False, normalize=False)
    # a_df = pd.DataFrame(a)
    # a_df.index.names = ["Atributo"]
    # a_df.columns = [nome_variavel]

    # b = pd.Series({"Regra exata": r_exato}, name=nome_analise)
    # b_df = pd.DataFrame(b)
    # b_df.index.names = ["Atributo"]
    # b_df.columns = [nome_variavel]

    # ab_df = pd.concat([a_df, b_df], axis=0)

    # a = a.reset_index()
    # fora_padrao = ["nulo", "demais"]
    # count_invalidos = 0
    # for i, row in a.iterrows():
    #     # print((var in informacao_ignorada_por_atributo.keys()) and (row[var] in informacao_ignorada_por_atributo[var] ))
    #     if row["Atributo"] in fora_padrao:
    #         count_invalidos = count_invalidos + row["count"]
    # log.logar_indicador(
    #     "Consistencia",
    #     "Fora padrao",
    #     f"registros nao atendem aos padroes definidos : {fora_padrao}",
    #     nome_variavel,
    #     count_invalidos,
    # )

    return df

def analisa_RZNTR(df):
    """Analisa os valores da variavel RZNTR verificando se nao realizou tratamento.

    1.Nenhum; 2. Cirurgia; 3.Radioterapia; 4.Quimioterapia; 5.Hormonioterapia; 6.Transplante de medula óssea; 7.Imunoterapia; 8.Outras; 9.Sem informação

    Insere uma nova coluna <_AnaliseRZNTR> com os resultados: <nao_tratou | demais>

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado com a coluna inserida
    """
    
    values = ["1", "2", "3", "4", "5", "6", "7"]

    df['_AnaliseRZNTR'] = np.where(df['RZNTR'].isin(values) , "nao_tratou", "demais")

    return df

def analisa_TIPOHIST(df):
    """Analisa os valores da variavel ESTADIAM verificando se seguem os formatos padroes.

    Tipo histológico do tumor primário
    Insere uma nova coluna <_AnaliseTIPOHIST> com os resultados: <exato | demais>

    # 4 dígitos tipo celular (histologia)
    # 1 dígito comportamento biológico
    # 1 dígito o grau de diferenciação ou fenótipo
    
    #4 digitos para histologia, começa com 8 ou 9 e mais 3 digitos
    # 1 digito Código de comportamento biológico das neoplasias: 0 a 3, 6 ou 9
    # 1 digito código para graduação e diferenciação histológicas: 1 a 9

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado com a coluna inserida
    """
    nome_variavel = "TIPOHIST"
    nome_analise = "_Analise" + nome_variavel

    # r_exato = r"^[89]\d{0,3}(/?[012369])?(\.[1-9])?$"
    r_exato =    r'^[89]\d{3}/[012369]$'

    sem_info = ['99999' , '9999/9']
    nan_values = ['nan' , 'C090/3' , ]
    
    df[nome_analise] = np.where(
        df[nome_variavel].str.contains(r_exato, regex=True, na=False),
        "exato", 
        np.where((df[nome_variavel].isnull()) | (df[nome_variavel].isin(nan_values)) , "nulo",
                 np.where((df[nome_variavel].isin(sem_info)), "sem_info", "demais")))


    return df

def analisa_MUNICIPIOS(df):
    """Analisa os valores das variaveis PROCEDEN e MUUH  verificando se seguem os formatos padroes.

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado com a coluna inserida
    """
    r_exato =    r'^(?!([0-9])\1{6})\d{7}$'

    sem_info = ['9999999' , '8888888' , '7777777']
    nan_values = ['nan']
    
    nome_variavel = "PROCEDEN"
    nome_analise = "_Analise" + nome_variavel
    df[nome_analise] = np.where(
        df[nome_variavel].str.contains(r_exato, regex=True, na=False),
        "exato", 
        np.where((df[nome_variavel].isnull()) | (df[nome_variavel].isin(nan_values)) , "nulo",
                 np.where((df[nome_variavel].isin(sem_info)), "sem_info", "demais")))
    
    nome_variavel = "MUUH"
    nome_analise = "_Analise" + nome_variavel
    df[nome_analise] = np.where(
        df[nome_variavel].str.contains(r_exato, regex=True, na=False),
        "exato", 
        np.where((df[nome_variavel].isnull()) | (df[nome_variavel].isin(nan_values)) , "nulo",
                 np.where((df[nome_variavel].isin(sem_info)), "sem_info", "demais")))

    return df


# #%% IDENTIFICACAO DE INCONSISTENCIAS

# def identifica_inconsistencia_TNM_PTNM(df , a_var = 'TNM'):
#     a_result_list = []
    
#     #identificar valores fora do domínio => invalido e demais
#     a_analise = '_Analise' + a_var
    
#     df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
    
#     cod_tipo = 'dominio - regra de formato'
#     df.loc[(df[a_analise] == 'demais') | (df[a_analise] == 'invalido') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 
#     q_inconsistente = df.loc[(df[a_analise] == 'demais') | (df[a_analise] == 'invalido')].shape[0]
   
#     if q_inconsistente > 0:
#         a_dict = {
#             'var' : a_var ,
#             'criterio' : 'Inconsistencia' ,
#             'referencia' : 'Manual do RHC' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : 'Análise "demais" ou "invalido"'
#             }    
#         a_result_list.append(a_dict)
     
# # =============================================================================
# #     NULOS e SEM INFO
# # =============================================================================
#     cod_tipo = 'valores invalidos (nulos)'
#     q_inconsistente = df.loc[(df[a_analise] == 'nulo')].shape[0]
#     if q_inconsistente > 0:
#         df.loc[(df[a_analise] == 'nulo') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

#         a_dict = {
#             'var' : a_var ,
#             'criterio' : 'Incompletude' ,
#             'referencia' : 'Manual do RHC' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : 'Análise "nulo"'
#             }    
#         a_result_list.append(a_dict)

#     cod_tipo = 'valores sem informacao'
#     q_inconsistente = df.loc[(df[a_analise] == 'sem_info')].shape[0]
#     if q_inconsistente > 0:
#         df.loc[(df[a_analise] == 'sem_info') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

#         a_dict = {
#             'var' : a_var ,
#             'criterio' : 'Incompletude' ,
#             'referencia' : 'Manual do RHC' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : 'Análise "sem info"'
#             }    
#         a_result_list.append(a_dict)
    
     
#     #verificar casos que tem TNM => exato incompleto e e´ pediatrico
#     cod_tipo = 'Inconsistente com pediatrico'
#     df.loc[((df[a_analise] == 'exato') | (df[a_analise] == 'incompleto')) & (df['IDADE'] < 20) , a_inconsistencia] =  df[a_inconsistencia] + ';' + cod_tipo
#     q_inconsistente = df.loc[((df[a_analise] == 'exato') | (df[a_analise] == 'incompleto')) & (df['IDADE'] < 20)].shape[0]
    
#     if q_inconsistente > 0:
#         a_dict = {
#             'var' : a_var ,
#             'criterio' : 'Inconsistencia' ,
#             'referencia' : f'Tumores pediatricos não tem {a_var}' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : 'Análise "exato" ou "incompleto" sendo IDADE < 20'

#             }
#         a_result_list.append(a_dict)
 
#     # Hemato 'nao se aplica - Hemato'
#     cod_tipo = 'Inconsistente com hematologico'
#     df.loc[((df[a_analise] == 'exato') | (df[a_analise] == 'incompleto')) & (df['_AnaliseLOCTUDET_tipo'] == 'Hemato') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
#     q_inconsistente = df.loc[((df[a_analise] == 'exato') | (df[a_analise] == 'incompleto')) & (df['_AnaliseLOCTUDET_tipo'] == 'Hemato') ].shape[0]

#     if q_inconsistente > 0:
#         a_dict = {
#             'var' : a_var ,
#             'criterio' : 'Inconsistencia' ,
#             'referencia' : f'Tumores hematologicos não tem {a_var}' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : 'Análise "exato" ou "incompleto" sendo LOCTUDET hematologico'

#             }
#         a_result_list.append(a_dict)
        
#     # df[a_inconsistencia].fillna(False , inplace = True)

#     return df , a_result_list



# def cria_coluna_inconsistencia(df , var):
#     a_inconsistencia = f.get_nome_coluna_indicador_variavel(var)
    
#     # ja existe a coluna
#     if a_inconsistencia not in df.columns:
#         df[a_inconsistencia] = ''
        
#     return df , a_inconsistencia


# def identifica_inconsistencia_ESTADIAM(df):
#     a_result_list = []
    
#     df , a_inconsistencia = cria_coluna_inconsistencia( df , 'ESTADIAM')
    
#     cod_tipo = 'dominio - regra de formato'
#     df.loc[(df['_AnaliseESTADIAM'] == 'demais') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
#     q_inconsistente = df.loc[(df['_AnaliseESTADIAM'] == 'demais') ].shape[0]

    
#     a_dict = {
#         'var' : 'ESTADIAM' ,
#         'criterio' : 'Inconsistencia' ,
#         'referencia' : 'Manual do RHC' , 
#         'quantidade' : q_inconsistente,
#         'codigo tipo' : cod_tipo,
#         'detalhamento' : 'Análise "demais"'

#         }    
#     a_result_list.append(a_dict)
    
# # =============================================================================
# #     NULOS e SEM INFO
# # =============================================================================
#     cod_tipo = 'valores invalidos (nulos)'
#     q_inconsistente = df.loc[(df['_AnaliseESTADIAM'] == 'nulo')].shape[0]
#     if q_inconsistente > 0:
#         df.loc[(df['_AnaliseESTADIAM'] == 'nulo') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

#         a_dict = {
#             'var' : 'ESTADIAM' ,
#             'criterio' : 'Incompletude' ,
#             'referencia' : 'Manual do RHC' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : 'Análise "nulo"'
#             }    
#         a_result_list.append(a_dict)

#     cod_tipo = 'valores sem informacao'
#     q_inconsistente = df.loc[(df['_AnaliseESTADIAM'] == 'sem_info')].shape[0]
#     if q_inconsistente > 0:
#         df.loc[(df['_AnaliseESTADIAM'] == 'sem_info') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

#         a_dict = {
#             'var' : 'ESTADIAM' ,
#             'criterio' : 'Incompletude' ,
#             'referencia' : 'Manual do RHC' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : 'Análise "sem info"'
#             }    
#         a_result_list.append(a_dict)

#     #verificar casos que tem ESTADIAM => exato  e e´ pediatrico
#     cod_tipo = 'Inconsistente com pediatrico'
#     df.loc[((df['_AnaliseESTADIAM'] == 'exato')) & (df['IDADE'] < 20) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
#     q_inconsistente = df.loc[(df['_AnaliseESTADIAM'] == 'exato') & (df['IDADE'] < 20)].shape[0]
    
#     if q_inconsistente > 0:
#         a_dict = {
#             'var' : 'ESTADIAM' ,
#             'criterio' : 'Inconsistencia' ,
#             'referencia' : f'Tumores pediatricos não tem ESTADIAM' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : 'Análise "exato" e IDADE < 20'

#             }
#         a_result_list.append(a_dict)


#     #verificar casos que tem ESTADIAM => exato  e e´ pediatrico
#     cod_tipo = 'Inconsistente com hematologico'
#     df.loc[((df['_AnaliseESTADIAM'] == 'exato')) & (df['_AnaliseLOCTUDET_tipo'] == 'Hemato') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
#     q_inconsistente = df.loc[(df['_AnaliseESTADIAM'] == 'exato') & (df['_AnaliseLOCTUDET_tipo'] == 'Hemato')].shape[0]
    
#     if q_inconsistente > 0:
#         a_dict = {
#             'var' : 'ESTADIAM' ,
#             'criterio' : 'Inconsistencia' ,
#             'referencia' : f'Tumores hematologicos não tem ESTADIAM' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : 'Análise "exato" sendo LOCTUDET hematologico'

#             }
#         a_result_list.append(a_dict)

#     return df , a_result_list

# # def identifica_inconsistencia_valores_invalidos(df):
    
# #     nan_values = {
# #         "ALCOOLIS": ["0"],
# #         "BASDIAGSP": [""],
# #         "BASMAIMP": ["0"],
# #         "CLIATEN": ["0" ,'51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98'],
# #         "CLITRAT": ["0" ,'51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98'],
# #         "DIAGANT": ["0"],
# #         "DTDIAGNO": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
# #         "DTTRIAGE": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
# #         "DATAPRICON": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
# #         "DATAOBITO": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
# #         "DATAINITRT": ["88/88/8888", "99/99/9999", "00/00/0000", "//"],
# #         "ESTADIAM": ["nan"],
# #         "ESTADRES": [ "77", "nan"],
# #         "ESTADIAG": ["nan"],
# #         "ESTCONJ": ["0"],
# #         "ESTDFIMT": ["0"],
# #         "EXDIAG": ["0", "<NA>"],
# #         "HISTFAMC": ["0"],
# #         "INSTRUC": [""],
# #         "LATERALI": ["0"],
# #         "LOCALNAS": [ "nan"],
# #         "LOCTUDET": ["nan", "D46", "E05", "N62", "C"],
# #         "LOCTUPRI": ["nan", "D46", "E05", "N62", "C .", "C .0"],
# #         "LOCTUPRO": ["", ",", ".", "9.", "nan"],
# #         "MAISUMTU": ["0"],
# #         "ORIENC": ["0"],
# #         "OUTROESTA": [
# #             "",
# #             "nan",
# #             ",",
# #             ".",
# #             "[",
# #             "]",
# #             "+",
# #             "\\",
# #             "'",
# #             "/",
# #             "//",
# #             "=",
# #         ],
# #         "PTNM": [""],
# #         "RACACOR": ["99"],
# #         "RZNTR": [""],
# #         "SEXO": ["0", "3"],
# #         "TABAGISM": ["0"],
# #         "TIPOHIST": ["nan", "/", "/3)", "1  /", "1   /", "99999", "C090/3"],
# #     	"PROCEDEN" : ['nan' , 'SC' , 'RS' , '9999999' , '999' , '99' , '93228' , '8888888' , '7777777'],

# #     }
    
# #     a_result_list = []
# #     cod_tipo = 'dominio - valores iniciais invalidos'
    
# #     for a_var in nan_values:
# #         df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
        
# #         df.loc[ df[a_var].isin(nan_values[a_var]) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
# #         q_inconsistente =  df.loc[ df[a_var].isin(nan_values[a_var])].shape[0]
        
# #         nome_analise = "_Analise" + a_var
# #         b = pd.Series({"Regra Nulo": nan_values[a_var]}, name=nome_analise)
# #         a_df = pd.DataFrame(b)
# #         a_df.index.names = ["Atributo"]
# #         a_df.columns = [a_var]
    
# #         if (q_inconsistente) > 0: 
# #             a_dict = {
# #                 'var' : a_var ,
# #                 'criterio' : 'Fora do dominio' ,
# #                 'referencia' : 'Manual RHC' , 
# #                 'quantidade' : q_inconsistente,
# #                 'codigo tipo' : cod_tipo,
# #                 'detalhamento' : nan_values[a_var]
# #                 }
# #             a_result_list.append(a_dict)
            
    
# #     return df , a_result_list



# def identifica_inconsistencia_anos(df):
#     """Identifica valores invalidos de anos.

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado
#     """
#     lista_colunas = ["ANOPRIDI", "ANTRI", "DTPRICON", "DTINITRT"]

#     a_result_list = []
#     for a_var in lista_colunas:
#         df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
        
#         cod_tipo = 'dominio'
#         df.loc[(~df[a_var].isnull()) & ((df[a_var] < 1900) | (df[a_var] > 2023)) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
#         q_inconsistente = df.loc[(~df[a_var].isnull()) & ((df[a_var] < 1900) | (df[a_var] > 2023))].shape[0]
        
#         if (q_inconsistente) > 0: 
#             a_dict = {
#                 'var' : a_var ,
#                 'criterio' : 'Inconsistencia' ,
#                 'referencia' : 'Manual do RHC' , 
#                 'quantidade' : q_inconsistente,
#                 'codigo tipo' : cod_tipo,
#                 'detalhamento' : 'Ano < 1900 OU Ano > 2023'
#                 }    
#             a_result_list.append(a_dict)
        
#     return df , a_result_list

# def identifica_inconsistencia_datas(df):
#     """Identifica inconsistencia de datas.

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado
#     """
#     lista_colunas = ["DTDIAGNO", "DTTRIAGE", "DATAPRICON", "DATAOBITO" , 'DATAINITRT']

#     a_result_list = []
#     for a_var in lista_colunas:
#         df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
        
#         cod_tipo = 'dominio'
#         df.loc[(~df[a_var].isnull()) & ((df[a_var].dt.year < 1900) | (df[a_var].dt.year > 2023)) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
#         q_inconsistente = df.loc[(~df[a_var].isnull()) & ((df[a_var].dt.year < 1900) | (df[a_var].dt.year > 2023))].shape[0]
        
#         if (q_inconsistente) > 0: 
#             a_dict = {
#                 'var' : a_var ,
#                 'criterio' : 'Inconsistencia' ,
#                 'referencia' : 'Manual do RHC' , 
#                 'quantidade' : q_inconsistente,
#                 'codigo tipo' : cod_tipo,
#                 'detalhamento' : 'Data < 1900 OU Data > 2023'
#                 }    
#             a_result_list.append(a_dict)
        
#     return df , a_result_list
        
# def identifica_inconsistencia_IDADE(df):
#     """Identifica inconsistencia de idade. Valida idade entre 0 e 110 anos.

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado
#     """

#     a_result_list = []
#     a_var = 'IDADE'
#     df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
    
#     cod_tipo = 'dominio'
#     df.loc[(~df[a_var].isnull()) & ((df[a_var] < 0) | (df[a_var] > 110)) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
#     q_inconsistente = df.loc[(~df[a_var].isnull()) & ((df[a_var] < 0) | (df[a_var] > 110)) ].shape[0]
    
#     if (q_inconsistente) > 0: 
#         a_dict = {
#             'var' : a_var ,
#             'criterio' : 'Inconsistencia' ,
#             'referencia' : 'Manual do RHC' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : 'IDADE < 0 OU IDADE > 110'
#             }    
#         a_result_list.append(a_dict)
        
#     return df , a_result_list

# def identifica_inconsistencia_tratamento(df):
#     """Identifica inconsistencia no tratamento.

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado
#     """

#     a_result_list = []
    
#     #Tratamento realizado: _AnalisePRIMTRATH exato  E   Tem razao para não tratamento - RZNTR
#     #Optei pela inconsistencia ser apenas em uma variavel
#     a_var = 'RZNTR'
#     df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
#     cod_tipo = 'entre variaveis'
#     df.loc[(df['_AnalisePRITRATH'] == 'exato') & (df['_AnaliseRZNTR'] == 'nao_tratou') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
#     q_inconsistente = df.loc[(df['_AnalisePRITRATH'] == 'exato') & (df['_AnaliseRZNTR'] == 'nao_tratou') ].shape[0]
    
#     if (q_inconsistente) > 0: 
#         a_dict = {
#             'var' : a_var ,
#             'criterio' : 'Inconsistencia' ,
#             'referencia' : 'Manual do RHC' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : 'Possui PRITRATH mas possui RZNTR'
#             }    
#         a_result_list.append(a_dict)
        
#     #Nao teve tratamento realizado: # _AnalisePRIMTRATH nulo  e Tem data de inicio de tratamento - DATAINITRT - DTINITRT
#     a_var = 'DATAINITRT'
#     df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
#     cod_tipo = 'entre variaveis'
#     df.loc[(df['_AnalisePRITRATH'] == 'nulo') & (~df['DATAINITRT'].isnull())  , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
#     q_inconsistente =df.loc[(df['_AnalisePRITRATH'] == 'nulo') & (~df['DATAINITRT'].isnull()) ].shape[0]
    
#     if (q_inconsistente) > 0: 
#         a_dict = {
#             'var' : a_var ,
#             'criterio' : 'Inconsistencia' ,
#             'referencia' : 'Manual do RHC' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : 'Não possui PRITRATH mas possui DATAINITRT'
#             }    
#         a_result_list.append(a_dict)        
        
#     #Nao teve tratamento realizado: # _AnalisePRIMTRATH nulo  e Tem data de inicio de tratamento - DATAINITRT - DTINITRT
#     a_var = 'DTINITRT'
#     df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
#     cod_tipo = 'entre variaveis'
#     df.loc[(df['_AnalisePRITRATH'] == 'nulo') & (~df['DTINITRT'].isnull())  , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
#     q_inconsistente =df.loc[(df['_AnalisePRITRATH'] == 'nulo') & (~df['DTINITRT'].isnull()) ].shape[0]
    
#     if (q_inconsistente) > 0: 
#         a_dict = {
#             'var' : a_var ,
#             'criterio' : 'Inconsistencia' ,
#             'referencia' : 'Manual do RHC' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : 'Possui PRITRATH mas possui DTINITRT'
#             }    
#         a_result_list.append(a_dict)  
        
        
#     #Nao teve tratamento realizado: # _AnalisePRIMTRATH nulo  e Teve resultado - ESTDFIMT de 1 a 6
#     a_var = 'ESTDFIMT'
#     df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
#     cod_tipo = 'entre variaveis'
#     df.loc[(df['_AnalisePRITRATH'] == 'nulo') & df['ESTDFIMT'].isin(range(1,6)) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
#     q_inconsistente =df.loc[(df['_AnalisePRITRATH'] == 'nulo') & df['ESTDFIMT'].isin(range(1,6))].shape[0]
    
#     if (q_inconsistente) > 0: 
#         a_dict = {
#             'var' : a_var ,
#             'criterio' : 'Inconsistencia' ,
#             'referencia' : 'Manual do RHC' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : 'Possui PRITRATH mas possui ESTDFIMT'
#             }    
#         a_result_list.append(a_dict)  
    
#     return df , a_result_list

# def identifica_inconsistencia_ordem_datas(df , var_1 , var_2):
#     """Identifica inconsistencia entre as datas (se var_2 foi antes de var_1).

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado
#     """

#     a_result_list = []
    
#     a_var = var_1
#     df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
#     cod_tipo = 'entre variaveis'
    
#     df.loc[
#         (~ df[var_2].isnull()) &
#         (~ df[var_1].isnull()) &
#         (df[var_2] < df[var_1]) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
    
#     q_inconsistente = df.loc[
#             (~ df[var_2].isnull()) &
#             (~ df[var_1].isnull()) &
#             (df[var_2] < df[var_1])].shape[0]
    
#     if (q_inconsistente) > 0: 
#         a_dict = {
#             'var' : a_var ,
#             'criterio' : 'Inconsistencia' ,
#             'referencia' : 'Manual do RHC' , 
#             'quantidade' : q_inconsistente,
#             'codigo tipo' : cod_tipo,
#             'detalhamento' : f'Datas fora de ordem ({var_2} antes de {var_1}'
#             }    
#         a_result_list.append(a_dict)
    
#     return df , a_result_list


# def identifica_inconsistencia_LOCTU(df):
#     """Identifica valores invalidos em LOCTUPRI , LOCTUDET , LOCTUPRO

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado

#     Returns:
#         (DataFrame): df modificado
#     """

#     a_result_list = []
    
#     lista_colunas = ['LOCTUPRI' , 'LOCTUDET' , 'LOCTUPRO']
#     cod_tipo = 'dominio - regra de formato'
        
#     for a_var in lista_colunas:
#         df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
#         nome_analise = "_Analise" + a_var
        
#         df.loc[df[nome_analise] != 'exato' , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
#         q_inconsistente = df.loc[df[nome_analise] != 'exato' ] .shape[0]
        
#         if (q_inconsistente) > 0: 
#             a_dict = {
#                 'var' : a_var ,
#                 'criterio' : 'Inconsistencia' ,
#                 'referencia' : 'Manual do RHC' , 
#                 'quantidade' : q_inconsistente,
#                 'codigo tipo' : cod_tipo,
#                 'detalhamento' : 'Analise != exato'
#                 }    
#             a_result_list.append(a_dict)
            
#         # =============================================================================
#         #     NULOS e SEM INFO
#         # =============================================================================
#         cod_tipo = 'valores invalidos (nulos)'
#         q_inconsistente = df.loc[(df[nome_analise] == 'nulo')].shape[0]
#         if q_inconsistente > 0:
#             df.loc[(df[nome_analise] == 'nulo') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

#             a_dict = {
#                 'var' : a_var ,
#                 'criterio' : 'Incompletude' ,
#                 'referencia' : 'Manual do RHC' , 
#                 'quantidade' : q_inconsistente,
#                 'codigo tipo' : cod_tipo,
#                 'detalhamento' : 'Análise "nulo"'
#                 }    
#             a_result_list.append(a_dict)

#         cod_tipo = 'valores sem informacao'
#         q_inconsistente = df.loc[(df[nome_analise] == 'sem_info')].shape[0]
#         if q_inconsistente > 0:
#             df.loc[(df[nome_analise] == 'sem_info') , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 

#             a_dict = {
#                 'var' : a_var ,
#                 'criterio' : 'Incompletude' ,
#                 'referencia' : 'Manual do RHC' , 
#                 'quantidade' : q_inconsistente,
#                 'codigo tipo' : cod_tipo,
#                 'detalhamento' : 'Análise "sem info"'
#                 }    
#             a_result_list.append(a_dict)
        
#     return df , a_result_list


# #%% OUTROS INDICADORES

# def identifica_incompletude_variaveis(df , lista_var):
    
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
#         # "ESTADIAM": ["nan"],
#         "ESTADRES": [ "77", "nan"],
#         "ESTADIAG": ["nan"],
#         "ESTCONJ": ["0"],
#         "ESTDFIMT": ["0"],
#         "EXDIAG": ["0", "<NA>"],
#         "HISTFAMC": ["0"],
#         "INSTRUC": [""],
#         "LATERALI": ["0"],
#         "LOCALNAS": [ "nan"],
#         # "LOCTUDET": ["nan", "D46", "E05", "N62", "C"],
#         # "LOCTUPRI": ["nan", "D46", "E05", "N62", "C .", "C .0"],
#         # "LOCTUPRO": ["", ",", ".", "9.", "nan"],
#         "MAISUMTU": ["0"],
#         "ORIENC": ["0"],
#         "OUTROESTA": ["","nan" , "," , ".", "[", "]", "+", "\\", "'", "/", "//", "=",],
#         # "PTNM": [""],
#         "RACACOR": ["99"],
#         "RZNTR": [""],
#         "SEXO": ["0", "3"],
#         "TABAGISM": ["0"],
#         "TIPOHIST": ["nan", "/", "/3)", "1  /", "1   /", "99999", "C090/3"],
#     	"PROCEDEN" : ['nan' , 'SC' , 'RS' , '9999999' , '999' , '99' , '93228' , '8888888' , '7777777'],

#     }
    
#     informacao_ignorada = {
#         "ALCOOLIS": ["9"],
#         "BASDIAGSP": ["9"],
#         "CLIATEN": ["99", "88"],
#         "CLITRAT": ["99", "88"],
#         "BASMAIMP": ["9"],
#         "DIAGANT": ["9"],
#         "ESTADIAG": ["9"],
#         # "ESTADIAM": ["99", "88"],
#         "ESTADRES": ["99"],
#         "ESTCONJ": ["9"],
#         "ESTDFIMT": ["9"],
#         "EXDIAG": ["9"],
#         "HISTFAMC": ["9"],
#         "INSTRUC": ["9"],
#         "LATERALI": ["9"],
#         "LOCALNAS": ["99"],
#         # "LOCTUPRO": [ "9"],
#         "OCUPACAO": ["999", "9999"],
#         "ORIENC": ["9"],
#         "OUTROESTA": ["99", "88"],
#         "PRITRATH": ["9"],
#         # "PTNM": ["9"],
#         "RACACOR": ["9"],
#         "RZNTR": ["9"],
#         "TABAGISM": ["9"],
#     }
    
#     a_result_list = []
    
    
#     for a_var in lista_var:
#         if a_var in nan_values.keys():
            
#             q_inconsistente =  df.loc[ df[a_var].isin(nan_values[a_var])].shape[0]
#             if q_inconsistente > 0:
#                 cod_tipo = 'valores invalidos (nulos)'
#                 df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
#                 df.loc[ df[a_var].isin(nan_values[a_var]) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
            
#                 nome_analise = "_Analise" + a_var
#                 b = pd.Series({"Regra Nulo": nan_values[a_var]}, name=nome_analise)
#                 a_df = pd.DataFrame(b)
#                 a_df.index.names = ["Atributo"]
#                 a_df.columns = [a_var]
        
#                 a_dict = {
#                     'var' : a_var ,
#                     'criterio' : 'Incompletude' ,
#                     'referencia' : 'Manual RHC' , 
#                     'quantidade' : q_inconsistente,
#                     'codigo tipo' : cod_tipo,
#                     'detalhamento' : nan_values[a_var]
#                     }
#                 a_result_list.append(a_dict)

#         if a_var in informacao_ignorada.keys():
        
#             q_inconsistente =  df.loc[ df[a_var].isin(informacao_ignorada[a_var])].shape[0]
#             if q_inconsistente > 0:
#                 cod_tipo = 'valores sem informacao'
#                 df , a_inconsistencia = cria_coluna_inconsistencia( df , a_var)
#                 df.loc[ df[a_var].isin(informacao_ignorada[a_var]) , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo
            
#                 nome_analise = "_Analise" + a_var
#                 b = pd.Series({"Regra Sem Informacao": informacao_ignorada[a_var]}, name=nome_analise)
#                 a_df = pd.DataFrame(b)
#                 a_df.index.names = ["Atributo"]
#                 a_df.columns = [a_var]
        
#                 a_dict = {
#                     'var' : a_var ,
#                     'criterio' : 'Incompletude' ,
#                     'referencia' : 'Manual RHC' , 
#                     'quantidade' : q_inconsistente,
#                     'codigo tipo' : cod_tipo,
#                     'detalhamento' : informacao_ignorada[a_var]
#                     }
#                 a_result_list.append(a_dict)

#     return df , a_result_list



#%% FUNCOES PRINCIPAIS

def analisar_dados(df):
    
    df = analisa_LOCTUDET(df)
    print(log.logar_acao_realizada("Analise de valores", f"Analisar o valor de LOCTUDET", "")
    )
    
    df = analisa_LOCTUPRI(df)
    print( log.logar_acao_realizada( "Analise de valores", f"Analisar o valor de LOCTUPRI", ""       )    )
    
    df = analisa_LOCTUPRO(df)
    print( log.logar_acao_realizada("Analise de valores", f"Analisar o valor de LOCTUPRO", "" ) )
    
    df = analisa_TNM_PTNM(df , nome_variavel = "TNM")
    print(log.logar_acao_realizada("Analise de valores", f"Analisar o valor de TNM", ""))
    
    df = analisa_TNM_PTNM(df , nome_variavel = "PTNM")
    print( log.logar_acao_realizada("Analise de valores", f"Analisar o valor de PTNM", ""))
    
    df = analisa_ESTADIAM(df)
    print(log.logar_acao_realizada("Analise de valores", f"Analisar o valor de ESTADIAM", "" ) )
    
    df = analisa_PRITRATH(df)
    print(log.logar_acao_realizada( "Analise de valores", f"Analisar o valor de PRITRATH", "" ))
    
    df = analisa_RZNTR(df)
    print( log.logar_acao_realizada( "Analise de valores", f"Analisar o valor de RZNTR", "" ) )
  
    df = analisa_TIPOHIST(df)
    print( log.logar_acao_realizada( "Analise de valores", f"Analisar o valor de TIPOHIST", "" ) )
    
    df = analisa_MUNICIPIOS(df)
    print( log.logar_acao_realizada( "Analise de valores", f"Analisar o valor de Municipios", "" ) )

    return df




# def identificar_inconsistencia_incompletude(df):
    
#     a_result_list = []
    
#     df , a_list = identifica_inconsistencia_TNM_PTNM(df , a_var = 'TNM')  
#     a_result_list = a_result_list + a_list
#     df , a_list = identifica_inconsistencia_TNM_PTNM(df , a_var = 'PTNM')  
#     a_result_list = a_result_list + a_list
#     print(log.logar_acao_realizada("Identificacao de Inconsistencias", f"Analisar TNM e PTMN", ""))
    
#     df , a_list = identifica_inconsistencia_ESTADIAM(df)  
#     a_result_list = a_result_list + a_list
#     print(log.logar_acao_realizada("Identificacao de Inconsistencias", f"Analisar ESTADIAM", ""))

#     df , a_list = identifica_inconsistencia_anos(df)  
#     a_result_list = a_result_list + a_list
#     print(log.logar_acao_realizada("Identificacao de Inconsistencias", f"Analisar variaveis de anos", ""))

#     df , a_list = identifica_inconsistencia_datas(df)  
#     a_result_list = a_result_list + a_list
#     print(log.logar_acao_realizada("Identificacao de Inconsistencias", f"Analisar variaveis de datas", ""))
    
#     df , a_list = identifica_inconsistencia_IDADE(df)
#     a_result_list = a_result_list + a_list
#     print(log.logar_acao_realizada("Identificacao de Inconsistencias", f"Analisar IDADE", ""))

#     df , a_list = identifica_inconsistencia_ordem_datas(df , var_1 = 'DATAPRICON' , var_2 = 'DATAINITRT' )
#     a_result_list = a_result_list + a_list
#     print(log.logar_acao_realizada("Identificacao de Inconsistencias", f"Analisar ordem de datas: DATAPRICON e DATAINITRT", ""))

#     df , a_list = identifica_inconsistencia_ordem_datas(df , var_1 = 'DATAINITRT' , var_2 = 'DATAOBITO' )
#     a_result_list = a_result_list + a_list
#     print(log.logar_acao_realizada("Identificacao de Inconsistencias", f"Analisar ordem de datas: DATAINITRT e DATAOBITO", ""))
    
#     df , a_list = identifica_inconsistencia_LOCTU(df)
#     a_result_list = a_result_list + a_list
#     print(log.logar_acao_realizada("Identificacao de Inconsistencias", f"Analisar as variaveis LOCTU", ""))
    
    
#     lista_var = [
#      'ALCOOLIS',
#      'BASDIAGSP',
#      'BASMAIMP',
#      'CLIATEN',
#      'CLITRAT',
#      'DATAINITRT',
#      'DATAOBITO',
#      'DATAPRICON',
#      'DIAGANT',
#      'DTDIAGNO',
#      'DTTRIAGE',
#      'ESTADIAG',
#      'ESTADIAM',
#      'ESTADRES',
#      'ESTCONJ',
#      'ESTDFIMT',
#      'EXDIAG',
#      'HISTFAMC',
#      'INSTRUC',
#      'LATERALI',
#      'LOCALNAS',
#      'LOCTUDET',
#      'LOCTUPRI',
#      'LOCTUPRO',
#      'MAISUMTU',
#      'OCUPACAO',
#      'ORIENC',
#      'OUTROESTA',
#      'PRITRATH',
#      'PROCEDEN',
#      'PTNM',
#      'RACACOR',
#      'RZNTR',
#      'SEXO',
#      'TABAGISM',
#      'TIPOHIST']

#     df , a_list = identifica_incompletude_variaveis(df, lista_var)  
#     a_result_list = a_result_list + a_list
        
#     a_response = pd.DataFrame(a_result_list)
    
#     return df , a_response



#%% INICIO - MAIN - ANALISE - INCONSISTENICA - INCOMPLETUDE
if __name__ == "__main__":
    log = Log()
    log.carregar_log("log_BaseInicial")
    df = f.leitura_arquivo_parquet("BaseInicial")

    restringir_periodo = False
    if restringir_periodo:  # usar 2000 a 2020 apenas
        df = df[
            (df["DATAINITRT"].dt.year > 1999)
            & (df["DATAINITRT"].dt.year < 2021)
        ]

    print(
        log.logar_acao_realizada(
            "Carga Dados",
            "Carregamento da base dos dados a serem analisados - Casos completos",
            df.shape[0],
        )
    )

# =============================================================================
# A atualização de abril de 2025 inseriu 9830 linhas duplicadas. Elas precisam ser removidas pois esta
# dando erro na geracao de indicadores ao usar a seguinte linha:
# df.loc[ CONDICAO , a_inconsistencia] = df[a_inconsistencia] + ';' + cod_tipo 
# =============================================================================

    df = df.drop_duplicates()



    df = analisar_dados(df)
    
    log.salvar_log("log_BaseAnalisada")
    f.salvar_parquet(df, "BaseAnalisada")

# #%% INICIO - MAIN - IDENTIFICACAO INCONSITENCIAS
# # Carrega a base ja analisada e gera os indicadores da Base Completa e da Base de Casos Validos
# if __name__ == "__main__":
#     log = Log()
#     log.carregar_log("log_BaseSanitizada")
#     df = f.leitura_arquivo_parquet("BaseSanitizada")

#     print(
#         log.logar_acao_realizada(
#             "Carga Dados",
#             "Carregamento da base dos dados a serem analisados - Casos completos",
#             df.shape[0],
#         )
#     )

#     df , a_response = identificar_inconsistencia_incompletude(df)
    
    
#     a_nome_arquivo = "AnaliseVariaveisBaseCompleta"
#     f.salvar_excel_conclusao(a_response, a_nome_arquivo)
    
#     print(log.logar_acao_realizada("Identificacao de Incompletudes", f"Analisar valores incompletos e inconsistentes para Base Completa", df.shape[0]))

#     df = f.filtrar_registros_validos(df)
#     print(
#         log.logar_acao_realizada(
#             "Carga Dados",
#             "Filtragem da base dos dados a serem analisados - Casos Validos",
#             df.shape[0],
#         )
#     )

#     df , a_response = identificar_inconsistencia_incompletude(df)
    
    
#     a_nome_arquivo = "AnaliseVariaveisBaseValida"
#     f.salvar_excel_conclusao(a_response, a_nome_arquivo)
    
#     print(log.logar_acao_realizada("Identificacao de Incompletudes", f"Analisar valores incompletos e inconsistentes para Base Valida", df.shape[0]))
