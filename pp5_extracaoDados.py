# -*- coding: utf-8 -*-

"""Registros Hospitalares de Câncer (RHC) - Extracao e selecao dos dados para construcao dos modelos.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: Ulf Bergmann


#Roteiro de execucao:


"""

import pandas as pd
import numpy as np

import sys

sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")


import funcoes as f
from funcoes import Log
import funcoes_ulf as uf

#%% SELECAO
def seleciona_registros_consistentes_completos(df , lista_colunas):
    """Elimina os valores: inconsistentes e incompletos (nulos ou sem informacao) das colunas.

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado
        lista_colunas (list): variaveis cujos valores inconsistentes ou incompletos serão removidos

    Returns:
        (DataFrame): df modificado
        (Dictionary):  dict com indicadores
    """
#eliminar todos os registros inconsistentes das colunas
    inicial = df.shape[0]
    for a_var in lista_colunas:
        a_var_inconsistencia = f.get_nome_coluna_indicador_variavel(a_var)
        if a_var_inconsistencia in df.columns:
            
            # lista_valores = df[a_var_inconsistencia].value_counts(dropna=False, normalize=False)
            # print(f'---- {a_var_inconsistencia}')
            # print(lista_valores)
            
            df = df[df[a_var_inconsistencia].str.len() == 0]
            # log.logar_acao_realizada(
            #     "Selecao Dados",
            #     f"Eliminacao dos registros com {a_var} inconsistente",
            #     inicial - df.shape[0] ,
            # )
            
            log.logar_indicador(
                "Extracao",
                "Eliminar Dados",
                f"Eliminacao de registros com {a_var} inconsistente ou incompleto",
                a_var,
                inicial - df.shape[0],
            )
            
            inicial = df.shape[0]
            if inicial == 0:
                print('Todos os registros eliminados')
        else:
            print(f"Nao achou {a_var_inconsistencia} nas colunas")
        
    return df
    

def seleciona_nao_nulos(df, lista_colunas):
    """Elimina os valores nulos das variaveis passadas como parametros.
    Funcao a ser usada para variaveis onde nao foi feita a analise, ou seja, variaveis geradas durante as transformacoes.
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado
        lista_variaveis (list): variaveis cujos valores nulos serão removidos

    Returns:
        (DataFrame): df modificado
    """
    
    for a_var in lista_colunas:
        q_inicial = df.shape[0]
        df = df.dropna(subset=[a_var], inplace=False)
        log.logar_indicador(
            "Extracao",
            "Eliminar Dados",
            f"Eliminacao de registros com {a_var} nulo",
            a_var,
            q_inicial - df.shape[0],
        )

    print(
        log.logar_acao_realizada(
            "Selecao Dados",
            f"Eliminacao dos registros {lista_colunas} com valores nulos",
            "",
        )
    )

    return df

def seleciona_com_RESFINAL(df):
    count_antes = df.shape[0]
    df = df[(df['_Gerada_RESFINAL'] == 'com resposta') | (df['_Gerada_RESFINAL'] == 'sem resposta')]
    
    print(
        log.logar_acao_realizada(
            "Selecao Dados",
            "Eliminacao dos registros que _Gerada_RESFINAL (variável dependente do modelo) é nulo ou sem infor",
            count_antes - df.shape[0],
        ))
    return df

# def seleciona_com_tratamento(df):
#     """Seleciona os registros de quem fez tratamento (PRITRATH nulo) ou não possui resultado do tratamento (_RESFINAL != sem informacao).

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado


#     Returns:
#         (DataFrame): df modificado
#     """
#     inicial = df.shape[0]

#     df = df[(df['_AnalisePRITRATH'] != 'nulo') | ((df["_RESFINAL"] != 'sem informacao'))]
#     log.logar_indicador(
#         "Extracao",
#         "Eliminar Dados",
#         "Eliminacao de registros sem tratamento (PRIMTRATH nulo OU ESTDFIMT sem informacao)",
#         'PRIMTRATH',
#         inicial - df.shape[0],
#     )


#     print(
#         log.logar_acao_realizada(
#             "Selecao Dados",
#             "Eliminacao dos registros de quem nao fez tratamento (PRIMTRATH nulo)",
#             f"{inicial - df.shape[0]}",
#         )
#     )
 
    
    
    

#     return df

def remover_colunas_nao_significativas(df , lista_colunas):
    """Elimina as variaveis (colunas) que nao possuem significancia .

    Colunas: lista_colunas + colunas geradas nas analises

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    # remocao de variaveis nao significativas
    colunas = f.get_dict_names_colunas(df)

    colunas_Geradas_inconsistentes = colunas.get('inconsistente')
    colunas_Geradas_analise = colunas.get('analise')

    lista_colunas = lista_colunas + colunas_Geradas_inconsistentes + colunas_Geradas_analise

    df_aux = df.drop(columns=lista_colunas)
    print(
        log.logar_acao_realizada(
            "Selecao Dados",
            "Eliminacao de colunas com dados sem significancia",
            f"{lista_colunas}",
        )
    )

    return df_aux

def seleciona_sem_outliers(df):
    # selecionar os valores a partir do ano de 2000 pois tinham poucos validos antes (apenas 600)
    count_antes = df.shape[0]
    df = df[df['DTPRICON'] >= 2000]  
    print(
        log.logar_acao_realizada(
            "Selecao Dados",
            "Outliers - Eliminacao dos registros antes do ano de 2000 pois tinham poucos validos antes (apenas 600)",
            count_antes - df.shape[0],
        ))
    
    
    # remover os que levaram mais de 1 ano para iniciar o tratamento : _Gerada_tempo_para_inicio_tratamento
    count_antes = df.shape[0]
    df = uf.elimina_outliers_valores(df , '_Gerada_tempo_para_inicio_tratamento' , 0 , 365)
    print(
        log.logar_acao_realizada(
            "Selecao Dados",
            "Outliers - Eliminacao dos registros que levaram mais de 1 ano para iniciar o tratamento",
            count_antes - df.shape[0],
        ))
    
    # remover distancias elevadas
    count_antes = df.shape[0]
    q = df['_Gerada_distancia_tratamento'].quantile(0.99)
    df = uf.elimina_outliers_valores(df , '_Gerada_distancia_tratamento', 0 , q)
    print(
        log.logar_acao_realizada(
            "Selecao Dados",
            "Outliers - Eliminacao dos registros com distancias elevadas (percentil 99)",
            count_antes - df.shape[0],
        ))
    
    return df

#%% TRANSFORMACOES E GERAÇÕES DE DADOS

# def transforma_nulos_naoinformados(df):
#     """Transforma os valores nulos para nao informados. .

#     Nulo => 9
#         ALCOOLIS	Histórico de consumo de bebida alcoólica	1.Nunca; 2.Ex-consumidor; 3.Sim; 4.Não avaliado; 8.Não se aplica; 9.Sem informação
#         TABAGISM	Histórico de consumo de tabaco	1.Nunca; 2.Ex-consumidor; 3.Sim; 4.Não avaliado; 8.Não se aplica;  9.Sem informação
#         HISTFAMC	Histórico familiar de câncer	1.Sim; 2.Não; 9.Sem informação
#         ORIENC	Origem do encaminhamento	1.SUS; 2.Não SUS; 3.Veio por conta própria;8.Não se aplica; 9. Sem informação
#         ESTCONJ	Estado conjugal atual	1.Solteiro; 2.Casado; 3.Viúvo;4.Separado judicialmente; 5.União consensual; 9.Sem informação
#         DIAGANT	Diagnóstico e tratamento anteriores	1.Sem diag./Sem trat.; 2.Com diag./Sem trat.; 3.Com diag./Com trat.; 4.Outros; 9. Sem informação
#         BASMAIMP	Base mais importante para o diagnóstico do tumor	1.Clínica; 2.Pesquisa clínica; 3.Exame por imagem; 4.Marcadores tumorais; 5.Citologia; 6.Histologia da metástase; 7.Histologia do tumor primário; 9. Sem informação
#         EXDIAG	Exames relevantes para o diagnóstico e planejamento da terapêutica do tumor	1.Exame clínico e patologia clínica; 2.Exames por imagem; 3.Endoscopia e cirurgia exploradora; 4.Anatomia patológica; 5.Marcadores tumorais; 8.Não se aplica; 9. Sem informação
#         LATERALI	Lateralidade do tumor	1.Direita; 2. Esquerda; 3.Bilateral; 8.Não se aplica; 9.Sem informação

#     Nulo => 1
#         MAISUMTU	Ocorrência de mais um tumor primário	1.Não; 2.Sim; 3.Duvidoso

#     Parameters:
#         df (DataFrame): DataFrame a ser transformado / analisado


#     Returns:
#         (DataFrame): df modificado

#     """
#     values = {
#         "ALCOOLIS": "9",
#         "TABAGISM": "9",
#         "HISTFAMC": "9",
#         "ORIENC": "9",
#         "ESTCONJ": "9",
#         "MAISUMTU": "1",
#         "DIAGANT": "9",
#         "BASMAIMP": "9",
#         "EXDIAG": "9",
#         "LATERALI": "9",
#     }
#     df = df.fillna(value=values, inplace=False)

#     print(
#         log.logar_acao_realizada(
#             "Gerar / Transformar Dados",
#             "Setar valores nulos para sem informacao",
#             f"{values}",
#         )
#     )

#     return df





#%% PRINCIPAL

def main(df):
    """Funcao principal.

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado
    Returns:
        (DataFrame): df modificado
        (DataFrame): df indicadores
    """
    q_inicial = df.shape[0]
    df = f.filtrar_registros_validos(df)
    print(
        log.logar_acao_realizada(
            "Selecao Dados",
            "Eliminação dos casos não válidos para o modelo. Casos não analíticos ou sem tratamento",
            q_inicial - df.shape[0],
        )
    )
	    
    colunas = ['ESTADIAM' , 'ESTDFIMT' , 'IDADE' , 'LOCTUDET' , 'SEXO' , 'TIPOHIST' , 'TNM']

    
    #
    
    df = seleciona_registros_consistentes_completos(df, colunas )  
    
    df = seleciona_nao_nulos(df , [ '_Gerada_TNM_T'	, '_Gerada_TNM_N', '_Gerada_TNM_M'] )

    
    #Colunas que geraram novas colunas com informacoes relevantes (pp4_transf gera_informacao_relevante)
    colunas_a_remover = ['ALCOOLIS','BASMAIMP' , '_Gerada_BASMAIMP' , '_Gerada_ESTDFIMT', 'DIAGANT','EXDIAG','HISTFAMC','LATERALI','MAISUMTU','ORIENC','TABAGISM']   

    colunas_a_remover = colunas_a_remover + [
        'CNES',
        'DATAINITRT',
        'DATAOBITO',
        'DTDIAGNO',
        'MUUH',
        'ANOPRIDI',
        'ANTRI',
        'BASDIAGSP',
        'CLIATEN',
        'CLITRAT',
        'DATAINITRT',
        'DATAOBITO',
        'DATAPRICON',
        'DTINITRT',
        'ESTADIAG',
        'ESTADRES',
        'ESTDFIMT',
        'ESTCONJ',
        'INSTRUC',
        'LOCALNAS',
        'LOCTUPRI',
        'LOCTUPRO',
        'OCUPACAO',
        'OUTROESTA',
        'PROCEDEN',
        'RACACOR',
        'RZNTR',
        'DTTRIAGE',
        'VALOR_TOT',
        'TPCASO',
        'PRITRATH',
        'TNM',
        'PTNM',
        'TIPOHIST',
        '_Gerada_tempo_para_obito',
        '_Gerada_PRITRATH_7',
        '_Gerada_PRITRATH_6',
        '_Gerada_PRITRATH_5',
        '_Gerada_PRITRATH_4',	
        '_Gerada_PRITRATH_3',	
        '_Gerada_PRITRATH_2',
        '_Gerada_PTNM_T'	,'_Gerada_PTNM_N'	,'_Gerada_PTNM_M',
        '_TeveTratamento',
# =============================================================================
#        ANalise do motivo para eliminacao ... testar os modelos com a variavel ... '_Gerada_tempo_para_diagnostico',
        '_Gerada_tempo_para_diagnostico',
# =============================================================================
        ]

        
    df = remover_colunas_nao_significativas(df , colunas_a_remover) # tem que ser a utlima acao pois as colunas de analise precisam ser usadas

    # tornar colunas categoricas e zerar as categorias anteriores
    colunas_categoricas = ['SEXO', 'LOCTUDET', 'ESTADIAM', 'UFUH',
           '_Gerada_RESFINAL', 
           '_Gerada_Fez_Cirurgia', '_Gerada_Fez_Radioterapia', '_Gerada_Fez_Quimioterapia', '_Gerada_Fez_Hormonioterapia', '_Gerada_Fez_Transplante', '_Gerada_Fez_Imunoterapia', '_Gerada_Fez_OutroTrat', 
           '_Gerada_PRITRATH_1', 
           '_Gerada_TNM_T', '_Gerada_TNM_N', '_Gerada_TNM_M',
           '_Gerada_TIPOHIST_CELULAR', '_Gerada_TIPOHIST_BIOLOGICO',
           '_Gerada_ALCOOLIS', 
           '_Gerada_BASMAIMP_CLIN', '_Gerada_BASMAIMP_PESQ', '_Gerada_BASMAIMP_IMG', '_Gerada_BASMAIMP_MARTUM', '_Gerada_BASMAIMP_CIT', '_Gerada_BASMAIMP_MET', '_Gerada_BASMAIMP_TUMPRIM', 
           '_Gerada_DIAGANT_DIAG', '_Gerada_DIAGANT_TRAT',
           '_Gerada_EXDIAG_EXCLIN', '_Gerada_EXDIAG_IMG', '_Gerada_EXDIAG_END_CIR', '_Gerada_EXDIAG_PAT', '_Gerada_EXDIAG_MARC',
           '_Gerada_HISTFAMC', 
           '_Gerada_LATERALI_ESQ', '_Gerada_LATERALI_DIR',
           '_Gerada_MAISUMTU', 
           '_Gerada_ORIENC_SUS', 
           '_Gerada_TABAGISM']
    	   
    	   
    colunas_int = [ '_Gerada_tempo_para_inicio_tratamento', '_Gerada_distancia_tratamento', '_Gerada_PRITRATH_NrTratamentos']
    	
    for a_col in colunas_int:
        df[a_col] = pd.to_numeric(df[a_col], errors="coerce").fillna(0).astype(np.int64)
    for a_col in colunas_categoricas:
        df[a_col] = df[a_col].astype("category")
        
        
    df = seleciona_com_RESFINAL(df)

    df = seleciona_sem_outliers(df)

    return df


#%%INICIO

if __name__ == "__main__":
    log = Log()
    # log.carregar_log("log_BaseTransfor")
    # df = f.leitura_arquivo_parquet("BaseTransfor")
    
    log.carregar_log("log_BaseTransfor")
    df = f.leitura_arquivo_parquet("BaseTransfor")

    print(
        log.logar_acao_realizada(
            "Carga Dados",
            "Carregamento da base dos dados para seleção",
            df.shape[0],
        )
    )

    df = main(df)

    print(
        log.logar_acao_realizada(
            "Selecao dos Dados",
            "Selecionados os dados validos para a modelagem",
            df.shape[0],
        )
    )

    log.salvar_log("log_BaseModelagem")
    f.salvar_parquet(df, "BaseModelagem")