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

def trocar_valores_nulos(df):
    """Trocar os valores que representam brancos / nulos (99, 999, ...) por None.
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """ 
    lista_colunas = ['DTDIAGNO', '', 'DATAPRICON', 'DATAOBITO' , 'DATAINITRT']
    nan_values = ['88/88/8888' , '99/99/9999' , '00/00/0000' , '//'],
    
    
    nan_values = {
        'ALCOOLIS' : ['0' , '9' , '4' ],
        'BASDIAGSP' : ['9'],
        'BASMAIMP'	 : ['' , '9'],
        'CLIATEN'	 : ['99','0'],
        'CLITRAT'	 : ['99' , '0'],
        # 'CNES'	     : [''],
        'DIAGANT'	 : ['9' , '0'],
        
        # 'DTDIAGNO' : ['88/88/8888' , '99/99/9999' , '00/00/0000' , '//'],
        # 'DTTRIAGE' : ['88/88/8888' , '99/99/9999' , '00/00/0000' , '//'],
        # '' : ['88/88/8888' , '99/99/9999' , '00/00/0000' , '//'],
        # '' : ['88/88/8888' , '99/99/9999' , '00/00/0000' , '//'],
        # '' : ['88/88/8888' , '99/99/9999' , '00/00/0000' , '//'],
        
        'ESTADIAM'	 : ['99' , '88' , 'nan'],
        'ESTADRES'	 : ['99' , '77' , 'nan'],
        'ESTADIAG'   : ['0' , 'nan'],
        'ESTCONJ'	 : ['0','9'],
        'ESTDFIMT'	 : ['0' , '9'],
        'EXDIAG'	 : ['0' , '9' , '<NA>'],
        'HISTFAMC'	 : ['0' , '9'],
        # 'INSTRUC'	 : [''],
        'LATERALI'	 : ['0' , '9'],
        'LOCALNAS'	 : ['99' , 'nan'],
        'LOCTUDET'	 : ['nan' , 'D46' , 'E05' , 'N62' , 'C'],
        'LOCTUPRI'	 : ['nan' , 'D46' , 'E05' , 'N62' , 'C .' , 'C .0'],
        'LOCTUPRO'	 : ['' , ',' , '.' , '9' , '9.', 'nan'],
        'MAISUMTU'	 : ['0'],
        # 'MUUH'	 : [''],
        'OCUPACAO'	 : ['9999' , '999'],
        'ORIENC'	 : ['0'],
        'OUTROESTA'	 : ['','99','88','nan',',','.','[',']','+','\\',"'",'/','//','='],
        # 'PRITRATH'	 : [''],
        # 'PROCEDEN'	 : [''],
        'PTNM'	 : [''],
        'RACACOR'	 : [''],
        'RZNTR'	 : ['9'],
        'SEXO'	 : ['0' , '3'],
        'TABAGISM'	 : ['0'],
        'TIPOHIST'	 : ['nan' , '/' , '/3)' , '1  /' , '1   /' , '99999' , 'C090/3' ],
        # 'TNM'	 : [''],
        # 'TPCASO'	 : [''],
        # 'UFUH'	 : ['']
        }
    
    response_df = pd.DataFrame()
    
    for a_var in nan_values:
        df[a_var] = df[a_var].apply(lambda x: np.nan if (x in nan_values[a_var]) else x )
        
        nome_analise = 'Analise' + a_var
        b =  pd.Series({
            'Regra Nulo': nan_values[a_var] 
            } , name=nome_analise)
        a_df = pd.DataFrame(b)
        a_df.index.names = ['Atributo']
        a_df.columns = [a_var]
        
        response_df = pd.concat([response_df , a_df] , axis = 1)
        
    return df , response_df

def valida_idade(df):
    """Valida idade entre 0 e 110 anos. Torna o valor -1 se for diferente.
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    # a = df[a_var].value_counts(dropna=False, normalize=False)
    df['IDADE'] = df['IDADE'].apply(lambda x: -1 if x < 0 or x > 110 else -1 if np.isnan(x) else int(x))
    return df
    # a = df_unico[a_var].value_counts(dropna=False, normalize=False)
    # df_unico[a_var].info()

def analisa_LOCTUDET(df):
    """Analisa os valores da variavel LOCTUDET verificando se seguem os formatos padroes.
    
    LOCTUDET - Localização primária do tumor pela CID-O, 3 dígitos
    Insere uma nova coluna <AnaliseLOCTUDET> com os resultados: <exato | incompleto | nulo | demais>
    Insere uma nova coluna <AnaliseLOCTUDET_Tipo> com os resultados: <Hemato | demais>
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado com as colunas inseridas
    """
    nome_variavel = 'LOCTUDET'
    nome_analise = 'Analise' + nome_variavel
    
    r_exato = r'C\d{2}'
    r_incompleto = r'C\.?'
    
    df[nome_analise] =   np.where(df[nome_variavel].str.contains(r_exato, regex= True, na=False) , 'exato',   
                         np.where(df[nome_variavel].str.contains(r_incompleto, regex= True, na=False) , 'incompleto', 
                         np.where(df[nome_variavel].isnull(), 'nulo',
                        'demais')))
    
    a = df[nome_analise].value_counts(dropna=False, normalize=False)
    a_df = pd.DataFrame(a)
    a_df.index.names = ['Atributo']
    a_df.columns = [nome_variavel]
    
    b =  pd.Series({'Regra exata': r_exato , 'Regra incompleta': r_incompleto } , name=nome_analise)
    b_df = pd.DataFrame(b)
    b_df.index.names = ['Atributo']
    b_df.columns = [nome_variavel]
    
    hemato_values = {
        'C81',
        'C82',
        'C83',
        'C84',
        'C85',
        'C91',
        'C92',
        'C93',
        'C94',
        'C95',
        'C96',
        'C97'}

    df[nome_analise + '_tipo'] = df[nome_variavel].apply(lambda x: "Hemato" if (x in hemato_values) else 'demais')

    ab_df = pd.concat([a_df , b_df] , axis = 0)
    
    return df , ab_df


def analisa_LOCTUPRI(df):
    """Analisa os valores da variavel LOCTUPRI verificando se seguem os formatos padroes.
    
    LOCTUPRI - Localização primária do tumor pela CID-O, 4 dígitos
    Insere uma nova coluna <AnaliseLOCTUPRI> com os resultados: <exato | incompleto | nulo | demais>
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado com a coluna inserida
    """
    nome_variavel = 'LOCTUPRI'
    nome_analise = 'Analise' + nome_variavel
    
    r_exato = r'C\d{2}\.\d'
    r_incompleto = r'C\d{2}\.?'
    
    df[nome_analise] =   np.where(df[nome_variavel].str.contains(r_exato, regex= True, na=False) , 'exato',   
                         np.where(df[nome_variavel].str.contains(r_incompleto, regex= True, na=False) , 'incompleto',   
                         np.where(df[nome_variavel].isnull(), 'nulo',
                        'demais')))
    
    a = df[nome_analise].value_counts(dropna=False, normalize=False)
    a_df = pd.DataFrame(a)
    a_df.index.names = ['Atributo']
    a_df.columns = [nome_variavel]
    
    b =  pd.Series({'Regra exata': r_exato , 'Regra incompleta': r_incompleto } , name=nome_analise)
    b_df = pd.DataFrame(b)
    b_df.index.names = ['Atributo']
    b_df.columns = [nome_variavel]
    
    ab_df = pd.concat([a_df , b_df] , axis = 0)
    
    return df , ab_df

def analisa_LOCTUPRO(df):
    """Analisa os valores da variavel LOCTUPRO verificando se seguem os formatos padroes.
    
    Localização primária provável do tumor pela CID-O, 4 dígitos
    Insere uma nova coluna <AnaliseLOCTUPRO> com os resultados: <exato | incompleto | nulo | demais>
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado com a coluna inserida
    """    
    nome_variavel = 'LOCTUPRO'
    nome_analise = 'Analise' + nome_variavel
    
    r_exato = r'C\d{2}\.\d'
    r_incompleto = r'C\d{2}\.?'
    
    df[nome_analise] =   np.where(df[nome_variavel].str.contains(r_exato, regex= True, na=False) , 'exato',   
                         np.where(df[nome_variavel].str.contains(r_incompleto, regex= True, na=False) , 'incompleto',   
                         np.where(df[nome_variavel].isnull(), 'nulo',
                        'demais')))
    
    a = df[nome_analise].value_counts(dropna=False, normalize=False)
    a_df = pd.DataFrame(a)
    a_df.index.names = ['Atributo']
    a_df.columns = [nome_variavel]
    
    b =  pd.Series({'Regra exata': r_exato , 'Regra incompleta': r_incompleto } , name=nome_analise)
    b_df = pd.DataFrame(b)
    b_df.index.names = ['Atributo']
    b_df.columns = [nome_variavel]
    
    ab_df = pd.concat([a_df , b_df] , axis = 0)
    
    return df , ab_df

def analisa_TNM(df):
    """Analisa os valores da variavel TNM verificando se seguem os formatos padroes.
    
    TNM: Codificação do estádio clínico segundo classificação TNM
    T -a extensão do tumor primário
    N -a ausência ou presença e a extensão de metástase em linfonodos regionais ,
    M -a ausência ou presença de metástase à distância
   
    A adição de números a estes três componentes indica a extensão da doença maligna. Assim temos:
    T0, TI, T2, T3, T4 - N0, Nl, N2, N3 - M0, Ml
    
    Insere uma nova coluna <AnaliseTNM> com os resultados: <exato | incompleto | nao se aplica - Hemato | nao se aplica - Geral | nulo | demais>
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado com a coluna inserida
    """      
    nome_variavel = 'TNM'
    nome_analise = 'Analise' + nome_variavel
      
    r_invalido = r"^999|99$"
    r_nao_se_aplica_geral = r"^888|88|988|898|889$"
    r_exato = r"^[0-4IA][0-4][0-1]$"
    r_incompleto = r"^[0-4Xx][0-4Xx][0-1Xx]$"
    r_regrasp = r"^YYY|XXX$"
    r_nao_se_aplica_hemato = 'Hemato'
      
        
    df[nome_analise] =  np.where(df['AnaliseLOCTUDET_tipo'].str.contains(r_nao_se_aplica_hemato, regex= False, na=False), 'nao se aplica - Hemato',
                        np.where(df[nome_variavel].str.contains(r_invalido, regex= True, na=False), 'invalido',
                        np.where(df[nome_variavel].str.contains(r_nao_se_aplica_geral, regex= True, na=False), 'nao se aplica - Geral',
                        np.where(df[nome_variavel].str.contains(r_exato, regex= True, na=False) , 'exato',    
                        np.where(df[nome_variavel].str.contains(r_incompleto, regex= True, na=False), 'incompleto',          
                        np.where(df[nome_variavel].str.contains(r_regrasp, regex= True, na=False), 'nao se aplica - regra SP',
                        np.where(df[nome_variavel].isnull(), 'nulo',
                        'demais')))))))
        
    a = df[nome_analise].value_counts(dropna=False, normalize=False)
    a_df = pd.DataFrame(a)
    a_df.index.names = ['Atributo']
    a_df.columns = [nome_variavel]
      
    b =  pd.Series({
        'Regra exata': r_exato , 
        'Regra incompleta': r_incompleto, 
        'Regra não se aplica - Hemato ': 'CID-O de Hemato em LOCTUDET' ,
        'Regra não se aplica': r_nao_se_aplica_geral ,
        'Regra de SP ': r_regrasp ,
        'Regra invalido': r_invalido
        } , name=nome_analise)
    b_df = pd.DataFrame(b)
    b_df.index.names = ['Atributo']
    b_df.columns = [nome_variavel]
      
    ab_df = pd.concat([a_df , b_df] , axis = 0)
      
    return df , ab_df



def analisa_ESTADIAM(df):
    """Analisa os valores da variavel ESTADIAM verificando se seguem os formatos padroes.
    
    Localização primária provável do tumor pela CID-O, 4 dígitos
    Insere uma nova coluna <AnaliseESTADIAM> com os resultados: <exato | nulo | demais>
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado com a coluna inserida
    """
    nome_variavel = 'ESTADIAM'
    nome_analise = 'Analise' + nome_variavel
    
    r_exato = r'(^I|II|III|IV$)|(^[0-4]$)|(^[0-4][A|B|C]$)|(^0[0-4]$)'
    
    r_exato1 = r'^I|II|III|IV$' 
    r_exato2 = r'^[0-4]$'
    r_exato3 = r'^[0-4][A|B|C]$'
    r_exato4 = r'^0[0-4]$'

    df[nome_analise] = np.where(df[nome_variavel].str.contains(r_exato1, regex= True, na=False), 'exato',
                       np.where(df[nome_variavel].str.contains(r_exato2, regex= True, na=False), 'exato',        
                       np.where(df[nome_variavel].str.contains(r_exato3, regex= True, na=False), 'exato',
                       np.where(df[nome_variavel].str.contains(r_exato4, regex= True, na=False), 'exato',
                       np.where(df[nome_variavel].isnull(), 'nulo',
                         'demais')))))
    a = df[nome_analise].value_counts(dropna=False, normalize=False)
    a_df = pd.DataFrame(a)
    a_df.index.names = ['Atributo']
    a_df.columns = [nome_variavel]
    
    b =  pd.Series({'Regra exata': r_exato  } , name=nome_analise)
    b_df = pd.DataFrame(b)
    b_df.index.names = ['Atributo']
    b_df.columns = [nome_variavel]
    
    ab_df = pd.concat([a_df , b_df] , axis = 0)
    
    return df , ab_df


def main(df):
    """Funcao principal.
    
    Parameters:
        df (DataFrame) : df a ser modificado
    Returns:
        (DataFrame): df modificado       
    """     
    # TORNAR VALORES INVALIDOS COMO NULOS - NONE
    ind_antes=df.isnull().sum()
    
    df_unico , df_result = trocar_valores_nulos(df)
    
    ind_depois=df_unico.isnull().sum()
    
    df_aux = pd.DataFrame()
    df_aux['Nulos antes'] = ind_antes
    df_aux['Nulos depois'] = ind_depois
    df_aux['diferenca'] = ind_depois- ind_antes
    
    a_file_name = 'valores_tornados_null'
    f.salvar_excel_conclusao(df_aux , a_file_name)
    
    print(log.logar_acao_realizada('Valores Invalidos' , 'Corrigir valores invalidos para null. Ver arquivo valores_tornados_null.xslx' , ""))
    
    df_unico = valida_idade(df_unico)
    
    df_unico = tratar_variavel_municipio(df_unico)
    
    df_unico , df_result_aux = analisa_LOCTUDET(df_unico)
    df_result = pd.concat([df_result , df_result_aux] , axis = 1)
    print(log.logar_acao_realizada('Analise de valores' , f'Analisar o valor de LOCTUDET' , ''))
    
    df_unico , df_result_aux = analisa_LOCTUPRI(df_unico)
    df_result = pd.concat([df_result , df_result_aux] , axis = 1)
    print(log.logar_acao_realizada('Analise de valores' , f'Analisar o valor de LOCTUPRI' , ''))
    
    df_unico , df_result_aux = analisa_LOCTUPRO(df_unico)
    df_result = pd.concat([df_result , df_result_aux] , axis = 1)
    print(log.logar_acao_realizada('Analise de valores' , f'Analisar o valor de LOCTUPRO' , ''))
    
    df_unico , df_result_aux = analisa_TNM(df_unico)
    df_result = pd.concat([df_result , df_result_aux] , axis = 1)
    print(log.logar_acao_realizada('Analise de valores' , f'Analisar o valor de TNM' , ''))
    
    df_unico , df_result_aux = analisa_ESTADIAM(df_unico)
    df_result = pd.concat([df_result , df_result_aux] , axis = 1)
    print(log.logar_acao_realizada('Analise de valores' , f'Analisar o valor de ESTADIAM' , ''))
    
    df_result = df_result.fillna('')
    
    a_nome_arquivo = 'analiseValoresAtributos'
    f.salvar_excel_conclusao(df_result.T , a_nome_arquivo)
    
    print(log.logar_acao_realizada('Analise de valores' , 'Resultados consolidados da analise dos valores' , f'ver arquivo {a_nome_arquivo}'))
    
    return df_unico

    


if __name__ == "__main__":
    log = Log()
    log.carregar_log('log_BaseCompleta')
    
    # df_unico = f.leitura_arquivo_parquet('BaseAnaliticos')
    # print( log.logar_acao_realizada('Carga Dados' , 'Carregamento da base dos dados a serem analisados - Casos analiticos' , df_unico.shape[0]) )

    df_unico = f.leitura_arquivo_parquet('BaseCompleta')
    print( log.logar_acao_realizada('Carga Dados' , 'Carregamento da base dos dados a serem analisados - Casos completos' , df_unico.shape[0]) )

    df_unico = main(df_unico) 
    
    log.salvar_log('log_analise_valores') 
    f.salvar_parquet(df_unico , 'analise_valores')










