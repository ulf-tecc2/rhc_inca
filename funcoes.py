# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 07:46:21 2024

@author: ulf
"""
import dbfread as db
import pandas as pd
import numpy as np

from tabulate import tabulate

class Log:
    
    def __init__(self):
        self.acoes =  pd.DataFrame(columns=['tipo', 'descricao', 'indicador'])

    def logar_acao_realizada(self , tipo , descricao , indicador):
        """Registra no Log uma acao realizada.
        
        Parameters:
            tipo (String): tipo da acao
            descricao (String): descricao da acao
            indicador (Object): valores ou detalhamento da acao

        Returns:
            (String): acao no formato de tabela 
        """
        aNewDF = pd.DataFrame([{'tipo' : tipo, 'descricao' : descricao, 'indicador' : indicador}])
        self.acoes = pd.concat([self.acoes , aNewDF ], ignore_index=True)
        return  tabulate(aNewDF, headers='keys', tablefmt='simple_grid', numalign='right' , floatfmt=".2f" )
        
    def asString(self): 
        """Mostra o log como string.
        
        Parameters:
          
        Returns:
            (String): log no formato de tabela 
        """
        aTab = tabulate(self.acoes, headers='keys', tablefmt='simple_grid', numalign='right' , floatfmt=".2f" ) #gera um string com a tabela
        return aTab
    
    def salvar_log(self , file_name):
        """Salva o log em arquivo excel.
        
        Parameters:
            file_name (String): nome do arquivo sem extensao

        Returns:
            
        """
        with pd.ExcelWriter('dados/conclusoes/' + file_name + '.xlsx') as writer:
            self.acoes.to_excel(writer) 
            
    def carregar_log(self , file_name):
        self.acoes = pd.read_excel('dados/conclusoes/' + file_name + '.xlsx' , index_col=0)
   
    
def leitura_arquivo_csv(nome_arquivo):
    return pd.read_csv('dados/consolidado/' + nome_arquivo + '.csv' , dtype=str)

def leitura_arquivo_excel(nome_arquivo):
    return pd.read_excel('dados/consolidado/' + nome_arquivo + '.xlsx' , index_col=0)

def leitura_arquivo_parquet(nome_arquivo):
    return pd.read_parquet('dados/consolidado/' + nome_arquivo + '.parquet')     
    
def salvar_parquet(df , nome_arquivo):
    """Salva o df em arquivo parquet.
    
    Parameters:
        nome_arquivo (String): nome do arquivo sem extensao

    Returns:
        
    """
    df.to_parquet('dados/consolidado/' + nome_arquivo + '.parquet', compression='gzip')  

def salvar_csv(df , nome_arquivo):
    """Salva o df em arquivo CSV.
    
    Parameters:
        nome_arquivo (String): nome do arquivo sem extensao

    Returns:
        
    """
    df.to_csv('dados/consolidado/' + nome_arquivo + '.csv', index=False)
#   df_unico.to_parquet('dados/consolidado/parquet1', compression=None , index=False) 

def salvar_excel(df , nome_arquivo):
    """Salva o df em arquivo excel no diretorio padrao de dados consolidados.
    
    Parameters:
        nome_arquivo (String): nome do arquivo sem extensao

    Returns:
        
    """
    with pd.ExcelWriter('dados/consolidado/' + nome_arquivo + '.xlsx') as writer:
        df.to_excel(writer) 

def salvar_excel_conclusao(df , nome_arquivo):
    with pd.ExcelWriter('dados/Conclusoes/' + nome_arquivo + '.xlsx') as writer:
        df.to_excel(writer) 
        
def leitura_arquivo_excel_conclusao(nome_arquivo):
    return pd.read_excel('dados/Conclusoes/' + nome_arquivo + '.xlsx' , index_col=0)
