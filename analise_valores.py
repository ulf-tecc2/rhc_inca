# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Analise dos dados
MBA em Data Science e Analytics - USP/Esalq - 2025

@author: ulf Bergmann

"""

# =============================================================================
# Analisa os valores das variaveis inserindo uma coluna com o resultado
# Coluna: Analise<NOMEVARIAVEL> ou Analise<NOMEVARIAVEL>_Tipo
# Resultados: exato incompleto nulo demais
# =============================================================================



import dbfread as db
import pandas as pd
import numpy as np
import re 
import os
import glob
from datetime import *
import locale

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

import funcoes_ulf as ulfpp
import bib_graficos as ug

import funcoes as f
from funcoes import Log

from tabulate import tabulate

# ANALISE DADOS:  CARREGAR ANTERIOR PARA CONTINUIDADE
log = Log()
log.carregar_log('log_BaseCompleta')
a = log.asString()

df_unico = f.leitura_arquivo_parquet('BaseAnaliticos')
# df_unico = f.leitura_arquivo_parquet('BaseCompleta')

print( log.logar_acao_realizada('Carga Dados' , 'Carregamento da base dos dados a serem analisados - Casos analiticos' , df_unico.shape[0]) )


#%% Trocar os valores que representam brancos / nulos (99, 999, ...) por None
def trocar_valores_nulos(df):
    nan_values = {
        'ALCOOLIS' : ['0' , '9' , '4' ],
        'BASDIAGSP' : ['9'],
        'BASMAIMP'	 : ['' , '9'],
        'CLIATEN'	 : ['99','0'],
        'CLITRAT'	 : ['99' , '0'],
        # 'CNES'	     : [''],
        'DIAGANT'	 : ['9' , '0'],
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
        a_df.columns = [nome_analise]
        
        response_df = pd.concat([response_df , a_df] , axis = 1)
        
    return df , response_df

# ANALISE LOCAL TUMOR
# LOCTUDET	Localização primária (Categoria 3d)	Localização primária do tumor pela CID-O, 3 dígitos
# LOCTUPRI	Localização primária detalhada (Subcategoria 4d)	Localização primária do tumor pela CID-O, 4 dígitos
# LOCTUPRO	Localização provável do tumor primário	Localização primária provável do tumor pela CID-O, 4 dígitos

def analisa_LOCTUDET(df):
    
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
    a_df.columns = [nome_analise]
    
    b =  pd.Series({'Regra exata': r_exato , 'Regra incompleta': r_incompleto } , name=nome_analise)
    b_df = pd.DataFrame(b)
    b_df.index.names = ['Atributo']
    b_df.columns = [nome_analise]
    
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
    a_df.columns = [nome_analise]
    
    b =  pd.Series({'Regra exata': r_exato , 'Regra incompleta': r_incompleto } , name=nome_analise)
    b_df = pd.DataFrame(b)
    b_df.index.names = ['Atributo']
    b_df.columns = [nome_analise]
    
    ab_df = pd.concat([a_df , b_df] , axis = 0)
    
    return df , ab_df

def analisa_LOCTUPRO(df):
    
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
    a_df.columns = [nome_analise]
    
    b =  pd.Series({'Regra exata': r_exato , 'Regra incompleta': r_incompleto } , name=nome_analise)
    b_df = pd.DataFrame(b)
    b_df.index.names = ['Atributo']
    b_df.columns = [nome_analise]
    
    ab_df = pd.concat([a_df , b_df] , axis = 0)
    
    return df , ab_df

def analisa_TNM(df):
    
  # =============================================================================
  #     # TNM: Codificação do estádio clínico segundo classificação TNM
  #     # T -a extensão do tumor primário
  #     # N -a ausência ou presença e a extensão de metástase em linfonodos regionais ,
  #     # M -a ausência ou presença de metástase à distância
  # 
  #     # A adição de números a estes três componentes indica a extensão da doença maligna. Assim temos:
  #     # T0, TI, T2, T3, T4 - N0, Nl, N2, N3 - M0, Ml
  #     
  # =============================================================================
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
    a_df.columns = [nome_analise]
      
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
    b_df.columns = [nome_analise]
      
    ab_df = pd.concat([a_df , b_df] , axis = 0)
      
    return df , ab_df



def analisa_ESTADIAM(df):
    
    # df['AnaliseESTADIAM'] = np.where(df['ESTADIAM'].str.contains(r_est_exato1, regex= True, na=False), 'exato',
    #                         np.where(df['ESTADIAM'].str.contains(r_est_exato2, regex= True, na=False), 'exato',
    #                         np.where(df['ESTADIAM'].str.contains(r_est_exato3, regex= True, na=False), 'exato',
    #                         np.where(df['ESTADIAM'].str.contains(r_est_exato4, regex= True, na=False), 'exato',
    #                         np.where(df['ESTADIAM'].isnull(), 'nulo',

    #                                  'demais')))))
    
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
    a_df.columns = [nome_analise]
    
    b =  pd.Series({'Regra exata': r_exato  } , name=nome_analise)
    b_df = pd.DataFrame(b)
    b_df.index.names = ['Atributo']
    b_df.columns = [nome_analise]
    
    ab_df = pd.concat([a_df , b_df] , axis = 0)
    
    return df , ab_df




#%% TORNAR VALORES INVALIDOS COMO NULOS - NONE

ind_antes=df_unico.isnull().sum()

df_unico , df_result = trocar_valores_nulos(df_unico)

ind_depois=df_unico.isnull().sum()

df_aux = pd.DataFrame()
df_aux['Nulos antes'] = ind_antes
df_aux['Nulos depois'] = ind_depois
df_aux['diferenca'] = ind_depois- ind_antes

a_file_name = 'valores_tornados_null'
f.salvar_excel_conclusao(df_aux , a_file_name)

print(log.logar_acao_realizada('Valores Invalidos' , 'Corrigir valores invalidos para null. Ver arquivo valores_tornados_null.xslx' , ""))


#%% VALIDACOES E INFERENCIAS ENTRE VARIAVEIS

# Ver manual pagina 337


def infere_BASMAIMP():
    # =============================================================================
    # BASMAIMP e BASDIAGSP: informacoes coerentes entre eles. Preencho os valores 0 com os correspondentes preenchidos de BASDIAGSP
    # regras quando BASMAIMP for 0
    # '1' => '1'
    # '2' => '2' | '3' | '4'
    # '3' => '5' | '6' | '7'
    # APlicar apenas a que tem certeza
    # =============================================================================
    
    
    # a = df_unico['BASMAIMP'].value_counts(dropna=False, normalize=False)
    # b = df_unico['BASDIAGSP'].value_counts(dropna=False, normalize=False)
    # c = df_unico.groupby(['BASMAIMP' , 'BASDIAGSP'] , observed=True).agg({'TPCASO' : 'count'})
    aux_df = df_unico.loc[ (df_unico['BASMAIMP'] == '0') & (df_unico['BASDIAGSP'] == '1')  ]
    aux_quant = aux_df.shape[0]
    df_unico.loc[ (df_unico['BASMAIMP'] == '0') & (df_unico['BASDIAGSP'] == '1') , 'BASMAIMP' ] = '1'
    print(log.logar_acao_realizada('Inferir valor' , 'Inferir o valor de BASMAIMP a partir de BASDIAGSP. Regra 0 <= 1. Falta definir outras regras aplicaveis' , f'{aux_quant}'))
    

    
def infere_ESTDFIMT():
    dias_entre_DATAINITRT_DATAOBITO = 730
    
    aux_df = df_unico.loc[ (df_unico['ESTDFIMT'].isnull()) &  ~(df_unico['DATAOBITO'].isnull()) & 
                                      (df_unico['DATAOBITO'] < df_unico['DATAINITRT'] + timedelta(days = dias_entre_DATAINITRT_DATAOBITO)  ) ]
    aux_quant = aux_df.shape[0]
    
    df_unico.loc[ (df_unico['ESTDFIMT'].isnull()) &  ~(df_unico['DATAOBITO'].isnull()) & 
                                      (df_unico['DATAOBITO'] < df_unico['DATAINITRT'] + timedelta(days = dias_entre_DATAINITRT_DATAOBITO)  ) , 'ESTDFIMT'] = '6'
    
    print(log.logar_acao_realizada('Inferir valor' , f'Inferir o valor de ESTDFIMT a partir de DATAOBITO ate {dias_entre_DATAINITRT_DATAOBITO} dias apos o inicio do tratamento' , f'{aux_quant}'))
    
    
    
    

infere_BASMAIMP()


infere_ESTDFIMT() 




#%%



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
f.salvar_excel_conclusao(df_result , a_nome_arquivo)

print(log.logar_acao_realizada('Analise de valores' , 'Resultados consolidados da analise dos valores' , f'ver arquivo {a_nome_arquivo}'))


log.salvar_log('log_analise_valores') 
f.salvar_parquet(df_unico , 'analise_valores')
a = log.asString()

#%%

# df1 = df_unico[df_unico['AnaliseLOCTUPRI'] == 'demais']
# b = df1['LOCTUPRI'].value_counts(dropna=False, normalize=False)
# c = df_unico['AnaliseLOCTUDET_tipo'].value_counts(dropna=False, normalize=False)


df_unico['ESTDFIMT'].info()
#Resultado esperado. Variável dependente ==> ESTDFIMT
(6).info()
a = df_unico['ESTDFIMT'].value_counts(dropna=False, normalize=False)

quant_sem_resultado = df_unico.loc[ (df_unico['ESTDFIMT'].isnull()) ].shape[0] # resultado do tratamento eh nulo
df_sr_obitos = df_unico.loc[ (df_unico['ESTDFIMT'].isnull()) &  ~(df_unico['DATAOBITO'].isnull())]  # tem data de obito

aux = df_unico.loc[ (df_unico['ESTDFIMT'].isnull()) &  ~(df_unico['DATAOBITO'].isnull()) & (df_unico['DATAOBITO'] < df_unico['DATAINITRT'] + timedelta(days = 50)  ) ] 
aux.info()
df_sr_obitos_prazo[['DATAINITRT' , 'DATAOBITO' ]].head(100)

aux = df_sr_obitos_prazo['DATAINITRT'] - df_sr_obitos_prazo['DATAOBITO']
aux.head(20)

timedelta(days = 25)

PRITRATH


aux_quant = df_unico.loc[ (df_unico['ESTDFIMT'] is None) & (df_unico['BASDIAGSP'] == '1')  ].shape[0]
    
    
    







log.salvar_log('log_analise_valores') 
f.salvar_parquet(df_unico , 'analise_valores')

#%% Tratar datas

colunas_datas = ['DTDIAGNO', 'DTTRIAGE', 'DATAPRICON', 'DATAOBITO' , 'DATAINITRT']
colunas_anos = ['ANOPRIDI' , 'ANTRI' , 'DTPRICON' , 'DTINITRT' ]

# ver consistencia entre datas e anos
# DTDIAGNO e ANOPRIDI

b = df_unico[colunas_anos + colunas_datas]
#%%
ug.plot_null_values(df_unico)
#%%
a.isnull().sum()
#%% Local de Tratamento - Ver SP

# UFUH: UF da unidade hospitalar



aux_sp = df_unico[df_unico['UFUH'] == 'SP']
#ver a distribuicao do tempo

aux = aux_sp["DATAINITRT"].dt.year

plt.figure(figsize=(15,10))
hist1 = sns.histplot(data=aux)
plt.xlabel('Anos', fontsize=20)
plt.ylabel('Registros', fontsize=20)
plt.title(label = "Historico Registros de SP")




# =============================================================================

#%%


df_analitico.info()
#%%
colunas_datas = ['DTDIAGNO', 'DTTRIAGE', 'DATAPRICON', 'DATAOBITO' , 'DATAINITRT']
colunas_anos = ['ANOPRIDI' , 'ANTRI' , 'DTPRICON' , 'DTINITRT' ]

df_analitico.groupby('DTINITRT')['DTINITRT'].count()

ulfpp.print_count_cat_var_values(df_analitico , colunas_anos)
print(df_analitico['DTINITRT'].isnull().sum())

df_unico.info()
#%%
colunas_anos = ['ANOPRIDI' , 'ANTRI' , 'DTPRICON' , 'DTINITRT' ]
ulfpp.print_count_cat_var_values(df_unico , ['DTPRICON'])
    

df_unico['DATAOBITO_TRANSF'] = pd.to_datetime(df_unico['DATAOBITO'] , format="%d/%m/%Y" , errors= 'coerce').dt.date
a = df_unico.loc[df_unico['ESTDFIMT'] == str(6) and df_unico['DATAOBITO_TRANSF'].isnull() ]
a

df_unico.head()
print(ulfpp.tabela_frequencias(df_unico, 'ESTDFIMT'))
print(ulfpp.tabela_frequencias(df_unico, 'PRITRATH'))

ulfpp.print_count_cat_var_values(df_unico  , ['PRITRATH'] )




a_var = 'IDADE'  #nao consegui resolver a questao do nan e tipo. Tive que colocar -1 como nan
a = df_unico[a_var].value_counts(dropna=False, normalize=False)
df_unico[a_var] = df_unico[a_var].apply(lambda x: -1 if x < 0 or x > 110 else -1 if np.isnan(x) else int(x))
a = df_unico[a_var].value_counts(dropna=False, normalize=False)

df_unico[a_var].info()






