# -*- coding: utf-8 -*-

"""Registros Hospitalares de Câncer (RHC) - Extracao e selecao dos dados para construcao dos modelos.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: ulf Bergmann

"""

import numpy as np

import sys
sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")


import funcoes as f
from funcoes import Log

def  define_valor_esperado(df):
    """Analisa ESTDFIMT e cria a variavel booleana _RESFINAL com sucesso / insucesso. Atualiza a variavel global df_base.
    
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
    
    return df

def seleciona_ESTDFIMT(df):
    """Elimina os valores invalidos ou nulos de ESTDFIMT. Atualiza a variavel global df_base.
    
    Regras:
        ESTDFIMT ==>> Variável ESTDFIMT = 8. Não se aplica; 9. Sem informação  ===>>> ELIMINAR
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    q_inicial = df.shape[0]
    #retirar valores 8 e 9
    df = df.drop(df[(df['ESTDFIMT'] == '8') | (df['ESTDFIMT'] == '9')].index, inplace = False)
    #retirar valores null
    df = df.dropna(subset = ['ESTDFIMT'], inplace=False)
    print(log.logar_acao_realizada('Selecao Dados' , 'Eliminacao de registros com ESTDFIMT invalido (8 , 9 e nulos)' ,f'{q_inicial - df.shape[0]}'))
    return df

def seleciona_ESTADIAM(df):
    """Elimina os valores invalidos ou nulos de ESTADIAM. Atualiza a variavel global df_base.
    
    ESTADIAM = codificação do grupamento do estádio clínico segundo classificação TNM
    
    Regras:
        AnaliseESTADIAM ==>> nulo | demais  ===>>> ELIMINAR
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    q_inicial = df.shape[0]
    #retirar valores que nao sao exatos
    df = df.drop(df[(df['AnaliseESTADIAM'] == 'nulo') | (df['AnaliseESTADIAM'] == 'demais')].index, inplace = False)
    print(log.logar_acao_realizada('Selecao Dados' , 'Eliminacao de registros com ESTADIAM que nao sao exatos' ,f'{q_inicial - df.shape[0]}'))
    return df


def seleciona_TNM(df):
    """Elimina os valores invalidos ou nulos de TNM. Atualiza a variavel global df_base.
    
    TNM   <AnaliseTNM>
 
    Regras:
        AnaliseTNM ==>> invalido | demais  ===>>> ELIMINAR
        Analisar o que deve ser feito com < nao se aplica - Hemato | nao se aplica - Geral>
        
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    q_inicial = df.shape[0]
    #retirar valores que nao sao exatos
    df = df.drop(df[(df['AnaliseTNM'] == 'invalido') | (df['AnaliseTNM'] == 'demais')].index, inplace = False)
    print(log.logar_acao_realizada('Selecao Dados' , 'Eliminacao de registros com TNM que nao sao exatos ou incompletos. \n  Analisar o que deve ser feito com < nao se aplica - Hemato | nao se aplica - Geral>' ,f'{q_inicial - df.shape[0]}'))

    return df

def seleciona_DATAS_gera_intervalos(df):
    """Geracao de intervalos de tempo e eliminacao de datas. Atualiza a variavel global df_base.
    
    ESTADIAM = codificação do grupamento do estádio clínico segundo classificação TNM
    
    Regras:
        DATAINITRT: nulo  ===>>> ELIMINAR
        DTDIAGNO: nulo  ===>>> ELIMINAR
        
    Gerar intervalos:
        _tempo_diagnostico_tratamento = dias desde DTDIAGNO e DATAINITRT
        _tempo_consulta_tratamento = dias desde DATAPRICON e DATAINITRT
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    q_inicial = df.shape[0]
    #retirar valores que nao sao exatos
    df = df.dropna(subset = ['DATAINITRT'], inplace=False)
    print(log.logar_acao_realizada('Dados Nulos' , 'Eliminacao de registros com DATAINITRT nulo' ,f'{q_inicial - df.shape[0]}'))
    
    # remover sem data de diagnostico
    q_inicial = df.shape[0]
    #retirar valores que nao sao exatos
    df = df.dropna(subset = ['DTDIAGNO'], inplace=False)
    print(log.logar_acao_realizada('Dados Nulos' , 'Eliminacao de registros com DTDIAGNO nulo' ,f'{q_inicial - df.shape[0]}'))
    
    df['_tempo_diagnostico_tratamento'] = (df['DATAINITRT'] - df['DTDIAGNO']).dt.days
    df['_tempo_diagnostico_tratamento'].astype(int)
    
    df['_tempo_consulta_tratamento'] = (df['DATAINITRT'] - df['DATAPRICON']).dt.days
    df['_tempo_consulta_tratamento'].astype(int)
    
    print(log.logar_acao_realizada('Gerar Dados' , 'Geracao das variaveis _tempo_diagnostico_tratamento e _tempo_consulta_tratamento ' ,''))
    
    return df
    
def transforma_nulos_naoinformados(df):
    """Transforma os valores nulos para nao informados. Atualiza a variavel global df_base.
    
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

def seleciona_naonulos(df , lista_variaveis):
    """Elimina os valores nulos das variaveis passadas como parametros. Atualiza a variavel global df_base.

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado
        lista_variaveis (list): variaveis cujos valores nulos serão removidos

    Returns:
        (DataFrame): df modificado
    """
    q_inicial = df.shape[0]
    df = df.dropna(subset = lista_variaveis, inplace=False)
    print(log.logar_acao_realizada('Dados Nulos' , f'Eliminacao dos registros {lista_variaveis} com valores nulos' ,f'{q_inicial - df.shape[0]}'))
    
    return df


def elimina_sem_tratamento(df):
    """Elimina os registros de quem nao fez tratamento (RZNTR nao nulo).

    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    df_aux = df[df['RZNTR'].notna()]

    q_inicial = df.shape[0]
   
    print(log.logar_acao_realizada('Dados Nulos' , f'Eliminacao dos registros de quem nao fez tratamento (RZNTR nao nulo)' ,f'{q_inicial - df_aux.shape[0]}'))
    return df_aux


def remover_colunas_naosignificativas(df):
    """Elimina as variaveis (colunas) que nao possuem significancia Atualiza a variavel global df_base.

    Colunas:
        'ESTDFIMT', 'RZNTR','DTDIAGNO' , 'DTTRIAGE' , 'DATAPRICON' , 'DATAINITRT' , 'DATAOBITO', 
        'TPCASO', 'LOCALNAS' , 'BASDIAGSP' , 'VALOR_TOT' , 
        'AnaliseLOCTUDET', 'AnaliseLOCTUDET_tipo', 'AnaliseLOCTUPRI', 'AnaliseLOCTUPRO','AnaliseTNM', 'AnaliseESTADIAM' ,
        'CLIATEN' , 'CLITRAT' , 'CNES' , 'DTINITRT' , 'LOCTUPRO' , 'ESTADRES', 'OUTROESTA' , 'OCUPACAO' , 'PROCEDEN' , 'ESTADIAG'
        
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """
    #remocao de variaveis nao significativas
    colunas_a_remover = ['ESTDFIMT', 'RZNTR','DTDIAGNO' , 'DTTRIAGE' , 'DATAPRICON' , 'DATAINITRT' , 'DATAOBITO', 
                         'TPCASO', 'LOCALNAS' , 'BASDIAGSP' , 'VALOR_TOT' , 
                         'AnaliseLOCTUDET', 'AnaliseLOCTUDET_tipo', 'AnaliseLOCTUPRI', 'AnaliseLOCTUPRO','AnaliseTNM', 'AnaliseESTADIAM' ,
                         'CLIATEN' , 'CLITRAT' , 'CNES' , 'DTINITRT' , 'LOCTUPRO' , 'ESTADRES', 'OUTROESTA' , 'OCUPACAO' , 'PROCEDEN' , 'ESTADIAG' ]
    
    df_aux = df.drop(columns=colunas_a_remover , axis=1)
    
    print(log.logar_acao_realizada('Remocao Registros' , 'Eliminacao de colunas com dados sem significancia' ,f'{colunas_a_remover}'))
    
    return df_aux


def coleta_sumario(df):
    """Insere no log as informacoes dos dados.
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado

    Returns:
        (DataFrame): df modificado
    """    
    print(log.logar_acao_realizada('Informacao' , 'Quantidade de registros com valores validos' ,f'{df.shape[0]}'))
    a = df.isnull().sum()
    print(log.logar_acao_realizada('Informacao' , 'Quantidade de registros com valores nulos' , a))


def main(df):
    """Funcao principal.
    
    Parameters:
        df (DataFrame): DataFrame a ser transformado / analisado
    Returns:
        (DataFrame): df modificado       
    """ 
    df = elimina_sem_tratamento(df)
    df = seleciona_ESTDFIMT(df)
    df = seleciona_ESTADIAM(df)
    df = seleciona_TNM(df)
    
    df = seleciona_DATAS_gera_intervalos(df)
    df = seleciona_naonulos(df , lista_variaveis = ['SEXO' , 'TIPOHIST'])    
    
    df = transforma_nulos_naoinformados(df)

    df = define_valor_esperado(df)
    
    df = remover_colunas_naosignificativas(df)
    
    coleta_sumario(df)
    return df

if __name__ == "__main__":
    log = Log()
    log.carregar_log('log_analise_valores')
    df_base = f.leitura_arquivo_parquet('analise_valores')
 
    print( log.logar_acao_realizada('Carga Dados' , 'Carregamento da base dos dados para seleção' , df_base.shape[0]) )

    df_base = main(df_base) 
    
    log.salvar_log('log_extracao_dados') 
    f.salvar_parquet(df_base , 'extracao_dados')
    
    a = log.asString()
