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

import pandas as pd
import numpy as np

import sys
sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

from funcoes import Log
import funcoes as f

def gera_intervalos(df):
    """Geracao de intervalos de tempo e selecao dos registros.
    
     Gerar intervalos:
        _tempo_diagnostico_tratamento = dias desde DTDIAGNO e DATAINITRT
        _tempo_consulta_tratamento = dias desde DATAPRICON e DATAINITRT
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """    
    df['_tempo_diagnostico_tratamento'] = (df['DATAINITRT'] - df['DTDIAGNO']).dt.days
    # df['_tempo_diagnostico_tratamento'].astype(int)
    
    df['_tempo_consulta_tratamento'] = (df['DATAINITRT'] - df['DATAPRICON']).dt.days
    # df['_tempo_consulta_tratamento'].astype(int)
    
    print(log.logar_acao_realizada('Gerar Dados' , 'Geracao das variaveis _tempo_diagnostico_tratamento e _tempo_consulta_tratamento ' ,''))
    
    return df
    

def  define_valor_esperado(df):
    """Analisa ESTDFIMT e cria a variavel booleana _RESFINAL com sucesso / insucesso. .
    
    Resultado esperado.  ESTDFIMT ==>> Variável dependente _RESFINAL
    1.Sem evidência da doença (remissão completa); 2.Remissão parcial; ===>>> SUCESSO
    3.Doença estável; 4.Doença em progressão; 5.Suporte terapêutico oncológico; 6. Óbito; 8. Não se aplica; 9. Sem informação ===>>> INSUCESSO
  
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    # gerar variavel dependente binaria => sucesso ou nao
    df['_RESFINAL'] =  np.where((df['ESTDFIMT'] == '1') | (df['ESTDFIMT'] == '2') , True , False)
    
    print(log.logar_acao_realizada('Gerar Valor' , 'Analise de ESTDFIMT e criacao da variavel dependente _RESFINAL ' , ''))
    
    return df

def transforma_nulos_naoinformados(df):
    """Transforma os valores nulos para nao informados. .
    
    Nulo => 9
        ALCOOLIS	Histórico de consumo de bebida alcoólica	1.Nunca; 2.Ex-consumidor; 3.Sim; 4.Não avaliado; 8.Não se aplica; 9.Sem informação
        TABAGISM	Histórico de consumo de tabaco	1.Nunca; 2.Ex-consumidor; 3.Sim; 4.Não avaliado; 8.Não se aplica;  9.Sem informação 
        HISTFAMC	Histórico familiar de câncer	1.Sim; 2.Não; 9.Sem informação
        ORIENC	Origem do encaminhamento	1.SUS; 2.Não SUS; 3.Veio por conta própria;8.Não se aplica; 9. Sem informação
        ESTCONJ	Estado conjugal atual	1.Solteiro; 2.Casado; 3.Viúvo;4.Separado judicialmente; 5.União consensual; 9.Sem informação
        DIAGANT	Diagnóstico e tratamento anteriores	1.Sem diag./Sem trat.; 2.Com diag./Sem trat.; 3.Com diag./Com trat.; 4.Outros; 9. Sem informação
        BASMAIMP	Base mais importante para o diagnóstico do tumor	1.Clínica; 2.Pesquisa clínica; 3.Exame por imagem; 4.Marcadores tumorais; 5.Citologia; 6.Histologia da metástase; 7.Histologia do tumor primário; 9. Sem informação
        EXDIAG	Exames relevantes para o diagnóstico e planejamento da terapêutica do tumor	1.Exame clínico e patologia clínica; 2.Exames por imagem; 3.Endoscopia e cirurgia exploradora; 4.Anatomia patológica; 5.Marcadores tumorais; 8.Não se aplica; 9. Sem informação
        LATERALI	Lateralidade do tumor	1.Direita; 2. Esquerda; 3.Bilateral; 8.Não se aplica; 9.Sem informação
        
    Nulo => 1
        MAISUMTU	Ocorrência de mais um tumor primário	1.Não; 2.Sim; 3.Duvidoso

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado
     

    Returns:
        (DataFrame): df modificado
       
    """    
    values = {'ALCOOLIS': '9' , 'TABAGISM': '9' , 'HISTFAMC': '9' , 'ORIENC': '9' ,
              'ESTCONJ': '9' , 'MAISUMTU' : '1' , 'DIAGANT' : '9' , 'BASMAIMP' : '9' ,
              'EXDIAG' : '9' , 'LATERALI' : '9'}
    df = df.fillna(value=values , inplace = False)
    
    print(log.logar_acao_realizada('Dados Nulos' , 'Setar valores nulos para sem informacao' ,f'{values}'))
    
    return df

def PRITRATH_dividir_tratamentos(df):
    """Analisa PRITRATH e desmembra a sequencia de tratamentos.
    
    Cria as colunas:
        PRITRATH  123 => PRITRATH_Primeiro 1  |  PRITRATH_Seguintes 23  | PRITRATH_NrTratamentos 3
  
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    df['PRITRATH_Primeiro'] = df['PRITRATH'].apply(lambda x: x[0]) 
    
    df['PRITRATH_Seguintes'] = df['PRITRATH'].apply(lambda x: x[1:])
    # df['PRITRATH_Seguintes'] = df['PRITRATH'].apply(lambda x: np.nan if (x == '') else x)
    
    df['PRITRATH_NrTratamentos'] = df['PRITRATH'].apply(lambda x: len(x)) 
    
    print(log.logar_acao_realizada('Gerar Valor' , 'Desmembramento de PRITRATH ' , ''))
    
    return df


#%%

def main(df_unico):
    df_unico = gera_intervalos(df_unico)
    df_unico = define_valor_esperado(df_unico)
    df_unico = transforma_nulos_naoinformados(df_unico)
    df_unico = PRITRATH_dividir_tratamentos(df_unico)

    return df_unico


# df_unico[['ALCOOLIS' , 'TABAGISM' , 'HISTFAMC' , 'ORIENC']].isnull().sum()

# df_unico.sample(20)[['PRITRATH','PRITRATH_Primeiro' , 'PRITRATH_Seguintes' , 'PRITRATH_NrTratamentos']]


if __name__ == "__main__":
    log = Log()
    log.carregar_log('log_analise_valores')
    df_unico = f.leitura_arquivo_parquet('analise_valores')
    print( log.logar_acao_realizada('Carga Dados' , 'Carregamento da base dos dados a serem transformados' , df_unico.shape[0]) )

    df_unico = main(df_unico) 
    
    log.salvar_log('log_transformacoes') 
    f.salvar_parquet(df_unico , 'transformacoes')









# data_inicio = busca_data_sp_iniciou_mais_que_1_trat(df_unico)
# print(log.logar_acao_realizada('Informacao', 'Data do inicio de envio de mais de um tratamento por SP', data_inicio))

#%%
# df = df_unico.sample(n=50000, random_state=1)
# df = df.dropna(subset=['DATAINITRT', 'PRITRATH_NrTratamentos'])


# c = df['DATAINITRT']
# c.unique()

# aux_ts = pd.Series(df['PRITRATH_NrTratamentos'].values , index=df['DATAINITRT'])


# aux_ts.tail(20)
# aux_ts.unique()

# df_unico['DATAINITRT'].describe()
# a = df_unico.groupby(['DATAINITRT'] , observed=True).size()

# pd.to_datetime(df_unico['DATAINITRT']).describe()

# df[a_col] = pd.to_datetime(df[a_col] , format="%d/%m/%Y" , errors= 'coerce')

# In[16]: Fazendo o grafico (Selecionar todos os comandos)
# plt.figure(figsize=(10, 6))
# plt.plot(aux_ts)
# plt.title("Total de Passageiros no Transporte Aereo BR")
# plt.xlabel("Jan/2011 a Mai/2024")
# plt.ylabel("Total de Passageiros Mensal")
# plt.show()



# c.head(100)

# b = df_unico.groupby(['PRITRATH_NrTratamentos'] , observed=True).size()
#  b = a['DATAINITRT']
#  b.iloc[0]
 
# aux_sp = df_unico[df_unico['UFUH'] == 'SP']

# df_unico.reset_index()
# df_unico.index
# aux_sp.reset_index
# a = df_unico.loc[aux_sp['PRITRATH'].str.len() > 1]
# a1 = df_unico.groupby(['PRITRATH_NrTratamentos'] , observed=True).size()
 
 
 
