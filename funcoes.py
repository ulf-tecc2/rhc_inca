# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Funções específicas do projeto.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: Ulf Bergmann

"""

import dbfread as db
import pandas as pd
import numpy as np
from pathlib import Path
import os

from tabulate import tabulate

import pickle


class Log:
    def __init__(self):
        self.__acoes = pd.DataFrame(columns=["tipo", "descricao", "indicador"])
        self.__indicadores = {}  # key: tipo value: DataFrame

    def logar_acao_realizada(self, tipo, descricao, indicador):
        """Registra no Log uma acao realizada.

        Parameters:
            tipo (String): tipo da acao
            descricao (String): descricao da acao
            indicador (Object): valores ou detalhamento da acao

        Returns:
            (String): acao no formato de tabela
        """
        aNewDF = pd.DataFrame(
            [{"tipo": tipo, "descricao": descricao, "indicador": indicador}]
        )
        self.__acoes = pd.concat([self.__acoes, aNewDF], ignore_index=True)
        return tabulate(
            aNewDF,
            headers="keys",
            tablefmt="simple_grid",
            numalign="right",
            floatfmt=".2f",
        )

    def asString(self):
        """Mostra o log como string.

        Parameters:

        Returns:
            (String): log no formato de tabela
        """
        aTab = tabulate(
            self.__acoes,
            headers="keys",
            tablefmt="simple_grid",
            numalign="right",
            floatfmt=".2f",
        )  # gera um string com a tabela
        return aTab

    def salvar_log(self, file_name):
        """Salva o log em arquivo excel.

        Parameters:
            file_name (String): nome do arquivo sem extensao

        Returns:

        """
        with pd.ExcelWriter("dados/conclusoes/" + file_name + ".xlsx") as writer:
            self.__acoes.to_excel(writer)

        self.save_indicadores_excel(
            dir_name="dados/Conclusoes/", new_folder_name=file_name
        )

    def carregar_log(self, file_name):
        self.__acoes = pd.read_excel(
            "dados/conclusoes/" + file_name + ".xlsx", index_col=0
        )

        self.carregar_indicadores("dados/conclusoes/" + file_name)

    def __get_indicador_df(self, tipo):
        if tipo not in list(self.__indicadores.keys()):
            lista_col = [
                "TPCASO",
                "SEXO",
                "IDADE",
                "LOCALNAS",
                "RACACOR",
                "INSTRUC",
                "CLIATEN",
                "CLITRAT",
                "HISTFAMC",
                "ALCOOLIS",
                "TABAGISM",
                "ESTADRES",
                "PROCEDEN",
                "ANOPRIDI",
                "ORIENC",
                "EXDIAG",
                "ESTCONJ",
                "ANTRI",
                "DTPRICON",
                "DIAGANT",
                "BASMAIMP",
                "_Gerada_BASMAIMP",
                "_Gerada_BASMAIMP",
                "LOCTUDET",
                "LOCTUPRI",
                "TIPOHIST",
                "LATERALI",
                "LOCTUPRO",
                "MAISUMTU",
                "TNM",
                "ESTADIAM",
                "OUTROESTA",
                "PTNM",
                "RZNTR",
                "DTINITRT",
                "PRITRATH",
                "ESTDFIMT",
                "_Gerada_ESTDFIMT",
                "CNES",
                "UFUH",
                "MUUH",
                "OCUPACAO",
                "DTDIAGNO",
                "DTTRIAGE",
                "DATAPRICON",
                "DATAINITRT",
                "DATAOBITO",
                "VALOR_TOT",
                "BASDIAGSP",
                "ESTADIAG",
            ]

            lista_col = list(lista_col)
            lista_col.sort()
            df = pd.DataFrame(columns=lista_col)
            df.astype("str")
            self.__indicadores = self.__indicadores | {tipo: df}
        return self.__indicadores[tipo]

    def logar_indicador(self, tipo, indicador, desc, var_col_name, valor):
        a_df = self.__get_indicador_df(tipo)
        a_df.at[indicador, var_col_name] = str(valor)
        a_df.at[indicador + " descricao", var_col_name] = str(desc)
        self.logar_acao_realizada(indicador, var_col_name + " -> " + desc, valor)

    def get_indicador_asstring(self, tipo):
        df = self.__get_indicador_df(tipo)
        aTab = tabulate(
            df.fillna(""),
            headers="keys",
            tablefmt="simple_grid",
            numalign="right",
            floatfmt=".2f",
        )  # gera um string com a tabela
        return aTab

    def save_indicadores_excel(
        self, dir_name="dados/Conclusoes/", new_folder_name="log_indicadores"
    ):
        root_dir_name = os.path.join(dir_name, new_folder_name)
        os.makedirs(root_dir_name, exist_ok=True)

        for tipo_ind in self.__indicadores:
            file_name = os.path.join(root_dir_name, tipo_ind) + ".xlsx"
            with pd.ExcelWriter(file_name) as writer:
                self.__indicadores[tipo_ind].to_excel(writer)

    def carregar_indicadores(self, dir_name):
        # ver se diretorio existe
        if not os.access(dir_name, os.F_OK):
            return

        # abrir cada arquivo do diretorio
        for a_dir, _, lista_arquivos in os.walk(
            dir_name
        ):  # percorre toda a estrutura olhando os subdiretorios
            for nome_arquivo in lista_arquivos:  # para cada subdiretorio
                arquivo = os.path.join(a_dir, nome_arquivo)
                ind_name = Path(arquivo).stem
                # print("Processando arquivo: " + arquivo + ' para ind ' + ind_name  )
                a_df = pd.read_excel(arquivo, index_col=0)
                self.__indicadores = self.__indicadores | {ind_name: a_df}


def leitura_arquivo_csv(nome_arquivo):
    return pd.read_csv("dados/consolidado/" + nome_arquivo + ".csv", dtype=str)


def leitura_arquivo_excel(nome_arquivo):
    return pd.read_excel("dados/consolidado/" + nome_arquivo + ".xlsx", index_col=0)


def leitura_arquivo_parquet(nome_arquivo):
    return pd.read_parquet("dados/consolidado/" + nome_arquivo + ".parquet")


def salvar_parquet(df, nome_arquivo):
    """Salva o df em arquivo parquet.

    Parameters:
        nome_arquivo (String): nome do arquivo sem extensao

    Returns:

    """
    df.to_parquet("dados/consolidado/" + nome_arquivo + ".parquet", compression="gzip")


def salvar_csv(df, nome_arquivo):
    """Salva o df em arquivo CSV.

    Parameters:
        nome_arquivo (String): nome do arquivo sem extensao

    Returns:

    """
    df.to_csv("dados/consolidado/" + nome_arquivo + ".csv", index=False)


#   df_unico.to_parquet('dados/consolidado/parquet1', compression=None , index=False)


def salvar_excel(df, nome_arquivo):
    """Salva o df em arquivo excel no diretorio padrao de dados consolidados.

    Parameters:
        nome_arquivo (String): nome do arquivo sem extensao

    Returns:

    """
    with pd.ExcelWriter("dados/consolidado/" + nome_arquivo + ".xlsx") as writer:
        df.to_excel(writer)


def salvar_excel_conclusao(df, nome_arquivo):
    with pd.ExcelWriter("dados/Conclusoes/" + nome_arquivo + ".xlsx") as writer:
        df.to_excel(writer)


def leitura_arquivo_excel_conclusao(nome_arquivo):
    return pd.read_excel("dados/Conclusoes/" + nome_arquivo + ".xlsx", index_col=0)


def lista_valores_unicos(df, lista_colunas):
    return

def filtrar_registros_validos(df):
    """Responde apenas os registros validos para a modelagem: Casos Analiticos e com Tratamento

    Parameters:
        df (DataFrame): dados a serem filtrados

    Returns:

    """
    if '_TeveTratamento' in df.columns:
        b = df[(df['_TeveTratamento'] == False) | (df['TPCASO'] != '1')]
    else:
        b = df[(~df['PRITRATH'].str.contains(r'^[1-8]+$', regex=True, na=False)) | (df['TPCASO'] != '1')]

    df_validos = df.drop(b.index, inplace = False)
    
    

    return df_validos

def get_nome_coluna_indicador_variavel(var):
    a_name = f'_Indicador{var}'
    return a_name

def get_dict_names_colunas(df):
    
    colunas = list(df.columns)
    colunas_geradas_inconsistentes = [s for s in colunas if s.startswith('_Indicador')]
    colunas_geradas_analise = [s for s in colunas if s.startswith('_Analise')]
    colunas_negocio = [s for s in colunas if not s.startswith('_')]
    colunas_geradas_negocio =  [s for s in colunas if s.startswith('_Gerada_')] 
    
    a_dict = {
        'inconsistente' : colunas_geradas_inconsistentes,
        'analise' : colunas_geradas_analise,
        'negocio' : colunas_negocio,
        'geradas' : colunas_geradas_negocio
        }
    
    return a_dict
    

def save_objects(a_dict , dir_name):
    """Salva os objetos definidos no dicionario.
    
    key: nome do objeto
    value: objeto a ser salvo
    nomes dos arquivos serao dir_padrao/dir_name/nome_objeto.pkl 

    Parameters:
        a_dict (Dictionary): dicionario com pares nome / objeto a serem salvos
        dir_name (String): diretorio de destino

    """
    dir_padrao = "dados/workdir/"
    root_dir_name = os.path.join(dir_padrao, dir_name)
    os.makedirs(root_dir_name, exist_ok=True)
    
    for a_key in a_dict:
        file_name = os.path.join(root_dir_name, a_key) + '.pkl'
        with open(file_name, 'wb') as file:  
            pickle.dump(a_dict[a_key], file)
            
          
def load_objects(dir_name):
    """carrega os objetos de dir_name em um dicionario
    key: nome do objeto
    value: objeto carregado

    Parameters:
        dir_name (String): diretorio de destino

    Returns:
        a_dict (Dictionary): dicionario com pares nome / objeto carregados
    """
    
    dir_padrao = "dados/workdir/"
    root_dir_name = os.path.join(dir_padrao, dir_name)
    
     # ver se diretorio existe
    if not os.access(root_dir_name, os.F_OK):
         return
    a_dict = {}
    for a_name in os.listdir(root_dir_name):
        a_file_name = os.path.join(root_dir_name, a_name)
        print(a_file_name)
        if os.path.isfile(a_file_name):
            with open(a_file_name, 'rb') as file:  
                an_obj = pickle.load(file)
            a_dict[a_name.removesuffix('.pkl')] = an_obj
                
    return a_dict    

def obter_modelo_salvo(a_name):
    a_dict = load_objects(a_name)
    return a_dict['modelo']

def save_model(model_name , model):
    """Salva o modelo.

    Parameters:
        model_name (String): nome do modelo para gerar o nome do arquivo pkl
        model (Modelo): modelo a ser salvo

    """
    dir_padrao = "dados/models/"

    file_name = os.path.join(dir_padrao, model_name) + '.pkl'
    with open(file_name, 'wb') as file:  
        pickle.dump(model, file)

def load_model(model_name):
    
    dir_padrao = "dados/models/"
    
    a_file_name = os.path.join(dir_padrao, model_name) + '.pkl'
    with open(a_file_name, 'rb') as file:  
        a_model = pickle.load(file)
        
    return a_model

def carrega_modelos(file_sufix = ''):
    mod_log = load_model('LogisticoBinario_Escolhido' + file_sufix)
    mod_rf = load_model('RandonForest_Model' + file_sufix)
    mod_xgb = load_model('XGBoost_Model' + file_sufix)
    
    return mod_log , mod_rf , mod_xgb
