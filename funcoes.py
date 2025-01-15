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
        aNewDF = pd.DataFrame([{'tipo' : tipo, 'descricao' : descricao, 'indicador' : indicador}])
        self.acoes = pd.concat([self.acoes , aNewDF ], ignore_index=True)
        return  tabulate(aNewDF, headers='keys', tablefmt='simple_grid', numalign='right' , floatfmt=".2f" )
        
    def asString(self):   
        aTab = tabulate(self.acoes, headers='keys', tablefmt='simple_grid', numalign='right' , floatfmt=".2f" ) #gera um string com a tabela
        return aTab
    
    def salvar_log(self , file_name):
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
    df.to_parquet('dados/consolidado/' + nome_arquivo + '.parquet', compression='gzip')  

def salvar_csv(df , nome_arquivo):
    df.to_csv('dados/consolidado/' + nome_arquivo + '.csv', index=False)
#   df_unico.to_parquet('dados/consolidado/parquet1', compression=None , index=False) 

def salvar_excel(df , nome_arquivo):
    with pd.ExcelWriter('dados/consolidado/' + nome_arquivo + '.xlsx') as writer:
        df.to_excel(writer) 

def salvar_excel_conclusao(df , nome_arquivo):
    with pd.ExcelWriter('dados/Conclusoes/' + nome_arquivo + '.xlsx') as writer:
        df.to_excel(writer) 
        