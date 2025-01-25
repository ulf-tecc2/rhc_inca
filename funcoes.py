# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 07:46:21 2024

@author: ulf
"""
import dbfread as db
import pandas as pd
import numpy as np
from pathlib import Path
import os

from tabulate import tabulate

class Log:
    
    def __init__(self):
        self.__acoes =  pd.DataFrame(columns=['tipo', 'descricao', 'indicador'])
        self.__indicadores =  {} # key: tipo value: DataFrame
        

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
        self.__acoes = pd.concat([self.__acoes , aNewDF ], ignore_index=True)
        return  tabulate(aNewDF, headers='keys', tablefmt='simple_grid', numalign='right' , floatfmt=".2f" )
        
    def asString(self): 
        """Mostra o log como string.
        
        Parameters:
          
        Returns:
            (String): log no formato de tabela 
        """
        aTab = tabulate(self.__acoes, headers='keys', tablefmt='simple_grid', numalign='right' , floatfmt=".2f" ) #gera um string com a tabela
        return aTab
    
    def salvar_log(self , file_name):
        """Salva o log em arquivo excel.
        
        Parameters:
            file_name (String): nome do arquivo sem extensao

        Returns:
            
        """
        with pd.ExcelWriter('dados/conclusoes/' + file_name + '.xlsx') as writer:
            self.__acoes.to_excel(writer) 
            
        self.save_indicadores_excel(dir_name = 'dados/Conclusoes/' , new_folder_name = file_name)

            
    def carregar_log(self , file_name):
        self.__acoes = pd.read_excel('dados/conclusoes/' + file_name + '.xlsx' , index_col=0)
        
        self.carregar_indicadores('dados/conclusoes/' + file_name)
       
        
    def __get_indicador_df(self , tipo):
        if tipo not in list(self.__indicadores.keys()):
            df = pd.DataFrame()
            df.astype('str')
            self.__indicadores = self.__indicadores | { tipo : df }
        return self.__indicadores[tipo]
        
    def logar_indicador(self , tipo , indicador  , desc , var_col_name , valor ):
        a_df = self.__get_indicador_df(tipo)
        a_df.at[indicador , var_col_name] = str(valor)
        a_df.at[indicador + ' descricao' , var_col_name] = str(desc)
        self.logar_acao_realizada(indicador , var_col_name + ' -> ' + desc , valor )

        
    def get_indicador_asstring(self , tipo):
        df = self.__get_indicador_df(tipo)
        aTab = tabulate(df.fillna('') , headers='keys', tablefmt='simple_grid', numalign='right' , floatfmt=".2f" ) #gera um string com a tabela
        return aTab
    
    def save_indicadores_excel(self , dir_name = 'dados/Conclusoes/' , new_folder_name = 'log_indicadores'):
        
        root_dir_name = os.path.join( dir_name , new_folder_name)
        os.makedirs( root_dir_name , exist_ok=True)
        
        for tipo_ind in self.__indicadores:
            file_name = os.path.join(root_dir_name , tipo_ind) + '.xlsx'
            with pd.ExcelWriter(file_name) as writer:
                self.__indicadores[tipo_ind].to_excel(writer) 
            
    def carregar_indicadores(self , dir_name ):
        # ver se diretorio existe
        if not os.access(dir_name, os.F_OK):
            return
        
        # abrir cada arquivo do diretorio
        for a_dir , _, lista_arquivos in os.walk(dir_name): #percorre toda a estrutura olhando os subdiretorios
            for nome_arquivo in lista_arquivos:  #para cada subdiretorio
                arquivo = os.path.join( a_dir , nome_arquivo)
                ind_name = Path(arquivo).stem 
                # print("Processando arquivo: " + arquivo + ' para ind ' + ind_name  )
                a_df =  pd.read_excel(arquivo , index_col=0)
                self.__indicadores = self.__indicadores | { ind_name : a_df }
    
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

def lista_valores_unicos(df , lista_colunas):
    
    
    return 