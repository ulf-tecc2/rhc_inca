# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Transformações dos dados.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: Ulf Bergmann


#Roteiro de execucao:
    1. Gera intervalos de tempo a partir das datas dos eventos 
    2. Cria uma nova variavel para descrever o resultado (sucesso/insucesso) do tratamento
    3. Transforma os valores nulos em Não Informados (normalemente 9)
    4. Desmembra os tratamentos realizados em primeiro trat e seguintes
    5. Salvar o arquivo como parquet

"""
import re

import pandas as pd
import numpy as np
from datetime import timedelta

from tabulate import tabulate

import sys
import gc

sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

from funcoes import Log
import funcoes as f
import bib_graficos as ug

# def gera_intervalo_DATAOBITO(df):
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

def calcular_distancia(linha , tabela_distancias):
    cod_origem = linha['MUUH']
    cod_destino = linha['PROCEDEN']
    
    a_var_analise_proceden = f.get_nome_coluna_indicador_variavel('PROCEDEN')
    a_var_analise_muuh = f.get_nome_coluna_indicador_variavel('MUUH')
    if cod_origem == cod_destino:
        distancia = 0
    else:
        if (len(linha[a_var_analise_proceden]) == 0) & (len(linha[a_var_analise_muuh]) == 0):
            try:
                distancia = tabela_distancias.at[cod_origem , cod_destino]
            except (KeyError , ValueError):
                return np.nan
        else:
            distancia = np.nan
        
    return distancia




def gera_distancias(df):
    """Geracao da distancia entre a residencia e o local de tratamento - MUUH - PROCEDEN

      Gerar intervalos:
        _distancia_tratamento = km entre MUUH - PROCEDEN

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """

    tabela_distancias = pd.read_parquet("dados/consolidado/" + 'tab_distancias' + ".parquet")
    
    df['_Gerada_distancia_tratamento'] = df.apply(lambda linha: calcular_distancia(linha , tabela_distancias) , axis=1)
    
    del tabela_distancias
    gc.collect()
    
    print(
        log.logar_acao_realizada(
            "Gerar / Transformar Dados",
            "Geracao de distancias entre PROCEDEN e MUUH ",
            "",
        )
    )
    
    return df


def gera_intervalos(df):
    """Geracao de intervalos de tempo e selecao dos registros.

      Gerar intervalos:
        _Gerada_tempo_para_diagnostico = dias desde DATAPRICON ate DTDIAGNO
        _Gerada_tempo_para_inicio_tratamento = dias desde DATAPRICON e DATAINITRT

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """


    # _tempo_para_diagnostico => Tempo para o diagnóstico: Tempo em dias da data da 1a consulta até a data do diagnóstico = pode ser positiva, quando o paciente fez o diagnóstico já no hospital; negativa, quando o paciente já chegou com o diagnóstico. Para fins de indicador, devem ser usados os tempos ≥ 0.
    a_var_analise_datapricon = f.get_nome_coluna_indicador_variavel('DATAPRICON')
    a_var_analise_dtdiagno = f.get_nome_coluna_indicador_variavel('DTDIAGNO')
    a_var_analise_datainitrt = f.get_nome_coluna_indicador_variavel('DATAINITRT')

    df['_Gerada_tempo_para_diagnostico'] = np.where( 
            df['DTDIAGNO'].isnull() |  df['DATAPRICON'].isnull() |
            (df[a_var_analise_datapricon].str.len() > 0) |
            (df[a_var_analise_dtdiagno].str.len() > 0),
        np.nan ,     (df['DTDIAGNO'] - df['DATAPRICON']).dt.days)

    # _temp_para_inicio_tratamento => Tempo em dias entre a data do diagnóstico (ou data da consulta (DATAPRICON)), na ausência da data do diagnóstico) até o início do tratamento: deve ser ≥ 0     

    df['_Gerada_tempo_para_inicio_tratamento'] = np.where( df['DATAINITRT'].isnull() | (df[a_var_analise_datainitrt].str.len() > 0)   , np.nan ,
                                         (df['DATAINITRT'] - 
                                              np.where((~df['DTDIAGNO'].isnull()) | (df[a_var_analise_dtdiagno].str.len() > 0) , 
                                                       df['DTDIAGNO'], 
                                                       df['DATAPRICON'])).dt.days)

    # Para sabermos o tempo para óbito, seria o tempo em anos da data do diagnóstico (quando não houver usar DATAPRICON) até a data do óbito. Assim, podemos considerar os mesmos critérios acima: tempo para o diagnóstico ≥ 0 e tempo para início do tratamento ≥ 0, somente dos casos sem diagnóstico e sem tratamento anterior. 
    df['_Gerada_tempo_para_obito'] = np.where( df['DATAOBITO'].isnull()   , np.nan ,
                                         (df['DATAOBITO'] - np.where(~df['DTDIAGNO'].isnull(), df['DTDIAGNO'], df['DATAPRICON'])).dt.days)

    print(
        log.logar_acao_realizada(
            "Gerar / Transformar Dados",
            "Geracao das variaveis de tempo <_Gerada_tempo_para_diagnostico> <_Gerada_tempo_para_inicio_tratamento>  <_Gerada_tempo_para_obito>",
            "",
        )
    )

    return df


# def atualiza_baseado_data_obito(df , i1 , i2):
#     #     Verificar os intervalos maximos:
#     #         - i1 => intervalo para considerar que nao chegou a realizar o tratamento
#     #         - i2 => intervalo para considerar que o obito é consequencia do tratamento 
#     #     se intervalo < i1 => não realizou nenhum tratamento, o obito foi anterior
#     #     se intervalo > i1 e < i2 => obito é consequencia do tratamento. se ESTDFIMT for nulo => setar ESTDFIMT para obito 

#     resultado_analise = pd.DataFrame()
    
#     aux_df = df.loc[(~df['_intervalo obito'].isnull()) & (df['_intervalo obito'] <= i1)]
#     resultado_analise.at[f"Registros com intervalo ate o obito menor que {i1}" , 'Quantidade'] = aux_df.shape[0]
    
#     aux_df = df.loc[(~df['_intervalo obito'].isnull()) & (df['_intervalo obito'] > i1) & (df['_intervalo obito'] <= i2)]
#     resultado_analise.at[f"Registros com intervalo ate o obito entre {i1} e {i2}" , 'Quantidade'] = aux_df.shape[0]
    
    
#     aux_df = df.loc[(~df['_intervalo obito'].isnull()) & (df['_intervalo obito'] > i2)]
#     resultado_analise.at[f"Registros com intervalo ate o obito maior que {i2}" , 'Quantidade'] = aux_df.shape[0]
    
    
#     #ver a funcao infere_ESTDFIMT
    
    
#     print(
#         log.logar_acao_realizada(
#             "Gerar / Transformar Dados",
#             f"Geracao de dados a partir de <_intervalo obito> e limites {i1} e {i2}",
#             "",
#         )
#     )
    
#     print(tabulate(
#         resultado_analise,
#         headers="keys",
#         tablefmt="simple_grid",
#         numalign="right",
#         floatfmt=".2f",
#     ))
#     return df


def define_valor_esperado(df):
    """Analisa ESTDFIMT e cria a variavel categorica _RESFINAL com resposta / sem resposta / sem informacao

    Resultado esperado.  ESTDFIMT ==>> Variável dependente _RESFINAL
    1.Sem evidência da doença (remissão completa); 2.Remissão parcial; ===>>> com resposta
    3.Doença estável; 4.Doença em progressão; 5.Suporte terapêutico oncológico; 6. Óbito;  ===>>> sem resposta
    8. Não se aplica; 9. Sem informação ===>>> sem informacao

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    # gerar variavel dependente binaria => sucesso ou nao
    com = ["1" , "2"]
    sem = ["3" , "4" , "5" , "6"]
    
    df["_Gerada_RESFINAL"] = np.where(df["_Gerada_ESTDFIMT"].isin(com), 'com resposta', 
                               np.where(df["_Gerada_ESTDFIMT"].isin(sem) , 'sem resposta' , 'sem informacao' ))

    print(
        log.logar_acao_realizada(
            "Gerar / Transformar Dados",
            "Analise de ESTDFIMT e criacao da variavel dependente _Gerada_RESFINAL ",
            "",
        )
    )

    return df



def TNM_PTNM_dividir(df , a_var = 'TNM'):
    """Analisa TNM e desmembra em T  N  e   M

    Cria as colunas:
        _Gerada_TNM_T   _Gerada_TNM_N   _Gerada_TNM_M

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    a_analise = '_Analise' + a_var
    df.loc[(df[a_analise] == 'exato') | (df[a_analise] == 'incompleto') , '_Gerada_' + a_var + '_T'] = df[a_var].str[0]
    df.loc[(df[a_analise] == 'exato') | (df[a_analise] == 'incompleto') , '_Gerada_' + a_var + '_N'] = df[a_var].str[1]
    df.loc[(df[a_analise] == 'exato') | (df[a_analise] == 'incompleto') , '_Gerada_' + a_var + '_M'] = df[a_var].str[2]
    
    print(
        log.logar_acao_realizada(
            "Gerar / Transformar Dados", "Desmembramento de " + a_var , ""
        )
    )
    
    return df
    


def PRITRATH_dividir_tratamentos(df):
    """Analisa PRITRATH e desmembra a sequencia de tratamentos.

    Cria as colunas:
        PRITRATH  123 => _Gerada_PRITRATH_Primeiro 1  |  _Gerada_PRITRATH_Seguintes 23  | _Gerada_PRITRATH_NrTratamentos 3
        _Gerada_Fez ... => para cada tratamento
        
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
   
    # df["PRITRATH_Primeiro"] = df["PRITRATH"].apply(lambda x: x[0])

    # df["PRITRATH_Seguintes"] = df["PRITRATH"].apply(lambda x: x[1:])
    # # df['PRITRATH_Seguintes'] = df['PRITRATH'].apply(lambda x: np.nan if (x == '') else x)

    # df["PRITRATH_NrTratamentos"] = df["PRITRATH"].apply(lambda x: len(x))
        
    
    #gerar campo adicional 
    # 2. Cirurgia; 3.Radioterapia; 4.Quimioterapia; 5.Hormonioterapia; 6.Transplante de medula óssea;7.Imunoterapia;8 .Outras

    df["_Gerada_Fez_Cirurgia"] = np.where(df['PRITRATH'].str.contains('2') , 1 , 0)
    df["_Gerada_Fez_Radioterapia"] = np.where(df['PRITRATH'].str.contains('3') , 1 , 0)
    df["_Gerada_Fez_Quimioterapia"] = np.where(df['PRITRATH'].str.contains('4') , 1 , 0)
    df["_Gerada_Fez_Hormonioterapia"] = np.where(df['PRITRATH'].str.contains('5') , 1 , 0)
    df["_Gerada_Fez_Transplante"] = np.where(df['PRITRATH'].str.contains('6') , 1 , 0)
    df["_Gerada_Fez_Imunoterapia"] = np.where(df['PRITRATH'].str.contains('7') , 1 , 0)
    df["_Gerada_Fez_OutroTrat"] = np.where(df['PRITRATH'].str.contains('8')  , 1 , 0)   
    
    df["_Gerada_PRITRATH_NrTratamentos"] = df['PRITRATH'].str.len()
    
    max_length = df['PRITRATH'].str.len().max()
    for i in range(max_length):
        df[f'{"_Gerada_PRITRATH"}_{i+1}'] = df['PRITRATH'].str[i]

    print(
        log.logar_acao_realizada(
            "Gerar / Transformar Dados", "Desmembramento de PRITRATH ", ""
        )
    )

    return df

        
def TIPOHIST_dividir(df):
    """Analisa TIPOHIST e desmembra.

    Cria as colunas:
        _TIPOHIST_CELULAR: 4 dígitos do tipo celular (histologia)
        _TIPOHIST_BIOLOGICO: 1 dígito comportamento biológico
        # 1 dígito o grau de diferenciação ou fenótipo

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
 
    
    # Aplicando a função para criar novas colunas
    df_aux = df.loc[df['_AnaliseTIPOHIST'] == 'exato']

    
    df[['_Gerada_TIPOHIST_CELULAR', '_Gerada_TIPOHIST_BIOLOGICO']] = df['TIPOHIST'].str.split(pat ='/', n=1 , expand=True)

    return df

# VALIDACOES E INFERENCIAS ENTRE VARIAVEIS - Ver manual pagina 337


def infere_BASMAIMP(df):
    """Tenta inferir os valores corretos para os nulos de BASMAIMP através de outras variaveis. Atualiza a variavel global df_unico.

    BASMAIMP e BASDIAGSP: informacoes coerentes entre eles. Preencho os valores 0 com os correspondentes preenchidos de BASDIAGSP
    regras quando BASMAIMP for 0
    '1' => '1'   REGRA APLICADA
    '2' => '2' | '3' | '4'
    '3' => '5' | '6' | '7'


    """
    # =============================================================================

    # =============================================================================

    # a = df_unico['BASMAIMP'].value_counts(dropna=False, normalize=False)
    # b = df_unico['BASDIAGSP'].value_counts(dropna=False, normalize=False)
    # c = df_unico.groupby(['BASMAIMP' , 'BASDIAGSP'] , observed=True).agg({'TPCASO' : 'count'})
    aux_df = df.loc[
        (df["BASMAIMP"].isnull() | (df["BASMAIMP"] == "0")) & (df["BASDIAGSP"] == "1")
    ]
    aux_quant = aux_df.shape[0]
    
    df['_Gerada_BASMAIMP'] = df['BASMAIMP']
    df.loc[
        (df["BASMAIMP"].isnull() | (df["BASMAIMP"] == "0")) & (df["BASDIAGSP"] == "1"),
        "_Gerada_BASMAIMP",
    ] = "1"
    print(
        log.logar_acao_realizada(
            "Gerar / Transformar Dados",
            "Gerar a variavel _Gerada_BASMAIMP com o valor inferido de BASMAIMP a partir de BASDIAGSP. Regra 0 <= 1. Falta definir outras regras aplicaveis",
            f"{aux_quant}",
        )
    )

    log.logar_indicador(
        "Correcoes",
        "Inferencia Dados",
        "Inferir o valor de BASMAIMP a partir de BASDIAGSP. Regra 0 <= 1. ",
        "BASMAIMP",
        aux_quant,
    )

    return df


def infere_ESTDFIMT(df, dias_para_obito=365):
    """Tenta inferir os valores corretos para os nulos de ESTDFIMT através de outras variaveis. Atualiza a variavel global df_unico.

    ESTDFIMT nulo e DATAOBITO ate o intervalo de tempo a partir da DATAINITRT

    Parameters:
        df (DataFrame): dataframe com os dados a serem analisados
        dias_para_obito (int): intervalo de tempo a ser considerado

    """
    
   
    aux_df = df.loc[
        ((df["ESTDFIMT"] == "8") | (df["ESTDFIMT"] == "9"))
        & (~(df["DATAOBITO"].isnull()))
        & (df["_Gerada_tempo_para_obito"] <  dias_para_obito)
    ]

    aux_quant = aux_df.shape[0]
    df['_Gerada_ESTDFIMT'] = df['ESTDFIMT']
    df.loc[
        ((df["ESTDFIMT"] == "8") | (df["ESTDFIMT"] == "9"))
        & (~(df["DATAOBITO"].isnull()))
        & (df["_Gerada_tempo_para_obito"] <  dias_para_obito),
        "_Gerada_ESTDFIMT",
    ] = "6"

    print(
        log.logar_acao_realizada(
            "Gerar / Transformar Dados",
            f"Gerar a variavel _Gerada_ESTDFIMT com o valor de ESTDFIMT inferido a partir do intervalo de {dias_para_obito} do obito",
            f"{aux_quant}",
        )
    )

    log.logar_indicador(
        "Correcoes",
        "Inferencia Dados",
        f"Inferir o valor de ESTDFIMT a partir da DATAOBITO - {dias_para_obito} dias ",
        "ESTDFIMT",
        aux_quant,
    )

    return df

#%% Geracao de informacoes relevantes
def gera_informacao_relevante(df):
    df = trata_ALCOOLIS(df)
    df = trata_BASMAIMP(df)
    df = trata_DIAGANT(df)
    df = trata_EXDIAG(df)
    df = trata_HISTFAMC(df)
    df = trata_LATERALI(df)
    df = trata_MAISUMTU(df)
    df = trata_ORIENC(df)
    df = trata_TABAGISM(df)
    
    print(
        log.logar_acao_realizada(
            "Gerar / Transformar Dados",
            "Geracao de novas variaveis a partir dos dados relevantes",
            str(['ALCOOLIS','BASMAIMP','DIAGANT','EXDIAG','HISTFAMC','LATERALI','MAISUMTU','ORIENC','TABAGISM'] )
        )
    )
    return df

def trata_ALCOOLIS(df):
    """Trata a coluna ALCOOLIS gerando a coluna _Gerada_ALCOOLIS .
    ALCOOLIS = 3 ==>> _Gerada_ALCOOLIS = True
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    a_var = 'ALCOOLIS'
    a_new_col = f'_Gerada_{a_var}'
    df[a_new_col] = np.where(df[a_var] == '3' , 1 , 0)

    return df

def trata_BASMAIMP(df):
    """Trata a coluna BASMAIMP gerando as colunas especificas para cada tipo.
        _Gerada_BASMAIMP_CLIN: 1.Clínica;
        _Gerada_BASMAIMP_PESQ: 2.Pesquisa clínica;
        _Gerada_BASMAIMP_IMG: 3.Exame por imagem;
        _Gerada_BASMAIMP_MARTUM: 4.Marcadores tumorais;
        _Gerada_BASMAIMP_CIT: 5.Citologia; 
        _Gerada_BASMAIMP_MET: 6.Histologia da metástase;
        _Gerada_BASMAIMP_TUMPRIM: 7.Histologia do tumor primário;
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    df['_Gerada_BASMAIMP_CLIN'] = np.where(df['_Gerada_BASMAIMP'] == '1' , 1 , 0)
    df['_Gerada_BASMAIMP_PESQ'] = np.where(df['_Gerada_BASMAIMP'] == '2' , 1 , 0)
    df['_Gerada_BASMAIMP_IMG'] = np.where(df['_Gerada_BASMAIMP'] == '3' , 1 , 0)
    df['_Gerada_BASMAIMP_MARTUM'] = np.where(df['_Gerada_BASMAIMP'] == '4' , 1 , 0)
    df['_Gerada_BASMAIMP_CIT'] = np.where(df['_Gerada_BASMAIMP'] == '5' , 1 , 0)
    df['_Gerada_BASMAIMP_MET'] = np.where(df['_Gerada_BASMAIMP'] == '6' , 1 , 0)
    df['_Gerada_BASMAIMP_TUMPRIM'] = np.where(df['_Gerada_BASMAIMP'] == '7' , 1 , 0)

    return df

def trata_DIAGANT(df):
    """Trata a coluna DIAGANT gerando as colunas especificas para cada tipo.
        _Gerada_DIAGANT_DIAG: 2  ou  3
        _Gerada_DIAGANT_TRAT: 3
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    df['_Gerada_DIAGANT_DIAG'] = np.where((df['DIAGANT'] == '2') | (df['DIAGANT'] == '3') , 1 , 0)
    df['_Gerada_DIAGANT_TRAT'] = np.where(df['DIAGANT'] == '3' , 1 , 0)

    return df


def trata_EXDIAG(df):
    """Trata a coluna EXDIAG gerando as colunas especificas para cada tipo.
        _Gerada_EXDIAG_EXCLIN: 1.Exame clínico e patologia clínica; 
        _Gerada_EXDIAG_IMG: 2.Exames por imagem;
        _Gerada_EXDIAG_END_CIR: 3.Endoscopia e cirurgia exploradora;
        _Gerada_EXDIAG_PAT; 4. Patologia
        _Gerada_EXDIAG_MARC: 5.Marcadores tumorais;
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    df['_Gerada_EXDIAG_EXCLIN'] = np.where(df['EXDIAG'] == '1' , 1 , 0)
    df['_Gerada_EXDIAG_IMG'] = np.where(df['EXDIAG'] == '2' , 1 , 0)
    df['_Gerada_EXDIAG_END_CIR'] = np.where(df['EXDIAG'] == '3' , 1 , 0)
    df['_Gerada_EXDIAG_PAT'] = np.where(df['EXDIAG'] == '4' , 1 , 0)
    df['_Gerada_EXDIAG_MARC'] = np.where(df['EXDIAG'] == '5' , 1 , 0)

    return df


def trata_HISTFAMC(df):
    """Trata a coluna HISTFAMC gerando a coluna _Gerada_ALCOOLIS .
    HISTFAMC = 1 ==>> _Gerada_HISTFAMC = True
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    a_var = 'HISTFAMC'
    a_new_col = f'_Gerada_{a_var}'
    df[a_new_col] = np.where(df[a_var] == '1' , 1 , 0)

    return df

def trata_TABAGISM(df):
    """Trata a coluna TABAGISM gerando a coluna _Gerada_TABAGISM .
    TABAGISM = 3 ==>> True
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    a_var = 'TABAGISM'
    a_new_col = f'_Gerada_{a_var}'
    df[a_new_col] = np.where(df[a_var] == '3' , 1 , 0)

    return df


def trata_LATERALI(df):
    """Trata a coluna LATERALI gerando a coluna _Gerada_LATERALI_ESQ ou _Gerada_LATERALI_DIR.
    LATERALI = 2 ou 3 ==>> _Gerada_LATERALI_ESQ
    LATERALI = 1 ou 3 ==>> _Gerada_LATERALI_DIR
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    df['_Gerada_LATERALI_ESQ'] = np.where((df['LATERALI'] == '2') | (df['LATERALI'] == '3') , 1 , 0)
    df['_Gerada_LATERALI_DIR'] = np.where((df['LATERALI'] == '1') | (df['LATERALI'] == '3') , 1 , 0)

    return df

def trata_MAISUMTU(df):
    """Trata a coluna MAISUMTU gerando a coluna _Gerada_MAISUMTU .
    MAISUMTU = 2 ==>> True
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    a_var = 'MAISUMTU'
    a_new_col = f'_Gerada_{a_var}'
    df[a_new_col] = np.where(df[a_var] == '2' , 1 , 0)

    return df


def trata_ORIENC(df):
    """Trata a coluna TABAGISM gerando a coluna _Gerada__ORIENC_SUS .
    ORIENC = 1 ==>> True
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    a_var = 'ORIENC'
    a_new_col = '_Gerada_ORIENC_SUS'
    df[a_new_col] = np.where(df[a_var] == '1' , 1 , 0)

    return df

#%% Principal
def main(df_unico):
    
    # rever funcao, juntar as duas 
    df_unico = gera_intervalos(df_unico)
 
    df_unico = gera_distancias(df_unico)
  
    df_unico = infere_BASMAIMP(df_unico)
    
    df_unico = infere_ESTDFIMT(df_unico)
        
    df_unico = define_valor_esperado(df_unico)
    df_unico = PRITRATH_dividir_tratamentos(df_unico)
    df_unico = TNM_PTNM_dividir(df_unico , a_var = 'TNM')
    df_unico = TNM_PTNM_dividir(df_unico , a_var = 'PTNM')
    df_unico = TIPOHIST_dividir(df_unico)
    
    df_unico = gera_informacao_relevante(df_unico)

    return df_unico


if __name__ == "__main__":
    log = Log()
    log.carregar_log("log_BaseIndicadores")
    df_unico = f.leitura_arquivo_parquet("BaseIndicadores")
    print(
        log.logar_acao_realizada(
            "Carga Dados",
            "Carregamento da base dos dados a serem transformados",
            df_unico.shape[0],
        )
    )

    df_unico = main(df_unico)

    log.salvar_log("log_BaseTransfor")
    f.salvar_parquet(df_unico, "BaseTransfor")
