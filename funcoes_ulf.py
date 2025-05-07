# -*- coding: utf-8 -*-

# lib/funcoes_ulf.py

"""Provide several general ML functions.

This module allows the user to reuse common functions allow ML projects.

@author: ulf Bergmann

"""

# =============================================================================
# Macetes em python / pandas
# =============================================================================


#%% Preparação da Variável categhorica => transforma em numeros por categoria
# y = y_train['label'].cat.codes


# b = a.groupby('teste' , observed = True).size()
# a = df[a_var].value_counts(dropna=False, normalize=False)

# df['floor'] = df.floor.str.replace('-','NaN').astype('float64')

# Carga de dados
# df = pd.read_excel('(1.2) dataset_principal.xls')
# dados_wdi = pd.read_excel('(2.2) WDI World Bank.xlsx', na_values='..')

# dados_tempo = dados_tempo.rename(columns={'Estudante':'estudante',
#                                           'Tempo para chegar à escola (minutos)':'tempo',
#                                           'Distância percorrida até a escola (quilômetros)': 'distancia',
#                                           'Quantidade de semáforos': 'semaforos',
#                                           'Período do dia': 'periodo',
#                                           'Perfil ao volante': 'perfil'})

# dados_novo = dados_tempo.rename(columns={dados_tempo.columns[0]: 'obs',
#                                          dados_tempo.columns[1]: 'temp',
#                                          dados_tempo.columns[5]: 'perf'})
#

# df = pd.read_csv('(3.3) CVM Dados Cadastrais.csv', 
#                         sep=';',
#                         encoding='latin1')



# df.iloc[3,] # linha 3 e todas as colunas
# df.iloc[:,4] # todas as linhas da coluna 4
# df.iloc[2:5,] # linhas 2 a 5, todas colunas
# df.iloc[:,3:5] # todas as linhas, colunas 3 a 5
# df.iloc[2:4,3:5] # linhas e colunas especificadas
# df.iloc[5,4] # posicao especifica

# Gerando novo dataframe com seleção das variáveis originais (colunas)
# df_aux = df.iloc[:, 0:6]


# df['tempo'] # pega a coluna inteira
# df.tempo

# # Se for mais de uma variável, inserir o argumento como uma lista
# df[['tempo', 'perfil']] # pega a lista de colunas

# Selecionando as colunas que tem o nome com inicio / fim especificado
# df1 = df.loc[:, df.columns.str.startswith('per')]
# df = df.loc[:, df.columns.str.endswith('o')]

# Seleção das observações com 'violations' menores ou iguais a 3
# df_aux = df[df['violations'] <= 3]

# Troca de valores especificos
# df.loc[df['desgaste'] != 0 , 'falha'] = 1 #cria/atualiza a coluna falha com o valor 1 se desgaste for diferente de 0
# df.loc[df['violations'] == -np.inf, 'violations'] = 0 #cria/atualiza a coluna violations com o valor 0 se for -np.inf 

# df = df[df['SETOR_ATIV'].notnull()] # elimina as linhas com valores missings

# df['sem_km'] = round((df['sem'] / df ['dist']), 2)  #Vamos criar uma variável em função de outras duas


# df['Nova Col'] = pd.Series([25,28,30,19,20,36,33,48,19,21])
# dados_novo['idade'] = idade
# nova_linha = pd.DataFrame({'per': ['Tarde'],
#                          'obs': ['Roberto'],
#                          'temp': [40]})
# df = pd.concat([df, nova_linha])




# Cria/atrualiza o valor da coluna em funcao de uma condicao 

# frequencia de uma variavel em porcentagem
# percent = (df_corruption['violations'].value_counts(dropna=False, normalize=True)*100).round(2)

# label_encoder = LabelEncoder()
# df_corrupcao['regiao_numerico'] = label_encoder.fit_transform(df_corrupcao['regiao'])


# Trocando textos por números usando um dicionario
# numeros = {'calmo': 1,
#            'moderado': 2,
#            'agressivo': 3}

# df_numeros = df.assign(novo_perfil = df.perfil.map(numeros))


# categorizar aplicando multiplas condições

# dados_tempo['faixa'] = np.where(dados_tempo['tempo']<=20, 'rápido',
#                        np.where((dados_tempo['tempo']>20) & (dados_tempo['tempo']<=40), 'médio',
#                        np.where(dados_tempo['tempo']>40, 'demorado',
#                                 'demais')))

# categorizar é por meio dos quartis de variáveis (q=4)
# dados_tempo['quartis'] = pd.qcut(dados_tempo['tempo'], q=4, labels=['1','2','3','4'])


# Organizando por mais de um critério
# df_aux = df.sort_values(by=['perfil', 'distancia'], 
#                                    ascending=[False, True]).reset_index(drop=True)

# Verificando se os valores das colunas 'coluna1' e 'coluna2' são nulos para a mesma linha 
# nulos_mesma_linha = df[['coluna1', 'coluna2']].isnull().all(axis=1)

# filtros
# df_aux = df[df['perfil'] == 'calmo']
# df_aux = df.query('perfil == "calmo"')
# df_aux = df[(df['perfil'] == 'calmo') & (df['periodo'] == 'Tarde')]
# df_aux = df.query('perfil == "calmo" & periodo == "Tarde"')

# nomes = pd.Series(["Gabriela", "Gustavo", "Leonor", "Ana", "Júlia"])
# df_aux = df[df['estudante'].isin(nomes)]
# df_aux = df.query('estudante.isin(@nomes)') # note o @ referenciando o objeto

#pegar o ano de uma coluna data
# ano = pd.to_datetime(df['DT_FIM_EXERC']).dt.year

# limpar um df exclkuindo as linhas que tenham observações extremas e nula em uma coluna
# df = df[~ df['VARIACAO'].isin([np.nan, np.inf, -np.inf])]

# eliminar linhas com valores extremos 
# df = df[df['VARIACAO'].between(-2, 2, inclusive='both')]

# gerar coluna com multiplas condicoes para o valor de outra coluna
# dados_jogos['venceu'] = np.where((dados_jogos['team_home_score'] - dados_jogos['team_away_score'] > 0), 'mandante',
#                         np.where((dados_jogos['team_home_score'] - dados_jogos['team_away_score'] < 0), 'visitante',
#                         np.where((dados_jogos['team_home_score'] - dados_jogos['team_away_score'] == 0), 'empate',
#                                 "Demais")))

# =============================================================================
# MERGE
# =============================================================================
# Parâmetros de configuração na função merge:
    # how: é a direção do merge (quais IDs restam na base final)
    # on: é a coluna com a chave para o merge


# # Left
# # Observações de dados_merge -> dados_tempo
# # Ficam os IDs de dados_tempo

# merge_1 = pd.merge(dados_tempo, dados_merge, how='left', on='estudante')

# # Right
# # Observações de dados_tempo -> dados_merge
# # Ficam os IDs de dados_merge

# merge_2 = pd.merge(dados_tempo, dados_merge, how='right', on='estudante')

# # Outer
# # Observações das duas bases de dados constam na base final 
# # Ficam todos os IDs presentes nas duas bases

# merge_3 = pd.merge(dados_tempo, dados_merge, how='outer', on='estudante')

# # Inner
# # Somente os IDs que constam nas duas bases ficam na base final 
# # É a interseção de IDs entre as duas bases de dados

# merge_4 = pd.merge(dados_tempo, dados_merge, how='inner', on='estudante')

# # Verificando apenas a diferença entre os bancos de dados (comparação)

# merge_5 = dados_tempo[~ dados_tempo.estudante.isin(dados_merge.estudante)]
# merge_6 = dados_merge[~ dados_merge.estudante.isin(dados_tempo.estudante)]

# # É importante analisar se há duplicidades de observações antes do merge

# #%% Analisando duplicidades de observações

# # Gerando o objeto após a remoção

# dados_tempo.drop_duplicates()
# # Interpretação: como retornou o mesmo DataFrame, não há duplicidades

# Contagem de linhas duplicadas

# len(dados_tempo) - len(dados_tempo.drop_duplicates())

# Se fosse para estabelecer uma remoção com base em algumas variáveis

# dados_tempo.drop_duplicates(subset=['estudante', 'perfil'])


# Os elementos do mesmo tópico iniciam com seu agregador
# Vamos selecionar as linhas com base em um critério
# dados_saude = dados_wdi[dados_wdi['topico'].str.startswith('Health')]


# Transformar os vaores de uma coluna em varias colunas
# Exemplo original
# 	pais	cod_pais	serie   ano_2021
# Afghanistan	AFG	          X        10
# Afghanistan	AFG	          Y        11
# Afghanistan	AFG	          Z        12

# 	pais	cod_pais      X         Y      Z
# Afghanistan	AFG	          10        11     12

# # As séries se tornam variáveis e as observações são os países
# df_aux = pd.pivot(df, index=['pais','cod_pais'], columns=['serie'], values='ano_2021')






# Tabela de frequências cruzadas para pares de variáveis qualitativas
# pd.crosstab(df['periodo'], df['perfil'])

# Criando uma tabela cruzada para comparar duas variaveis. Na logistica multinomial podemos usar para
# criar uma tabela parecida com a matriz de confusão
#      A    B      C  D  E
# 0  foo  one  small  1  2
# 1  foo  one  large  2  4
# 2  foo  one  large  2  5
# 3  foo  two  small  3  5
# 4  foo  two  small  3  6
# 5  bar  one  large  4  6
# 6  bar  one  small  5  8
# 7  bar  two  small  6  9
# 8  bar  two  large  7  9

# table = pd.pivot_table(df, values='D', index=['A', 'B'],
#                        columns=['C'], aggfunc="sum")
# table
# C        large  small
# A   B
# bar one    4.0    5.0
#     two    7.0    6.0
# foo one    4.0    1.0
#     two    NaN    6.0

# Criacao e Visualização de uma tabela 

#from tabulate import tabulate
# tabela = tabulate(table, headers='keys', tablefmt='grid', numalign='center') #gera um string com a tabela

# plt.figure(figsize=(8, 3))
# plt.text(0.1, 0.1, tabela, {'family': 'monospace', 'size': 15})
# plt.axis('off')
# plt.show()

#preencher Embarked com a cidade que mais passageiros entraram
# df['Embarked'].fillna((df['Embarked'].mode()[0]), inplace = True)
# agrupar e calcular a média
# median_ages = pd.Series(dados.groupby(by = 'Honorific')['Age'].median())



# Geracao de fórmulas quando temos muitas colunas
# lista_colunas = list(df.drop(columns=['id','fidelidade']).columns)
# a_formula = ' + '.join(lista_colunas)
# a_formula = "Y ~ " + a_formula
# print("Fórmula utilizada: ", a_formula)


#geracao de nr de acrdo com uma distribuição
# pois = np.random.poisson(lam=2, size=1000)

# from google_drive_downloader import GoogleDriveDownloader as gdd
# gdd.download_file_from_google_drive('1HBALGiUdzb-xyKMCorgMYdPkT4suvBkt' , './teste1')







#%%



import numpy as np
import pandas as pd
from scipy import stats # utilizado na definição da função 'breusch_pagan_test'
from statstests.tests import shapiro_francia
from scipy.stats import shapiro
    
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg

from sklearn.preprocessing import LabelEncoder # transformação de dados
import statsmodels.api as sm # estimação de modelos

from tabulate import tabulate

from statsmodels.iolib.summary2 import summary_col # comparação entre modelos

from sklearn.preprocessing import KBinsDiscretizer

import random

# import sys
# sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

# import funcoes_ulf as ulfpp
# import bib_graficos as ug
# from google_drive_downloader import GoogleDriveDownloader as gdd



# %%

def relatorio_missing(df):
    print(f'Número de linhas: {df.shape[0]} | Número de colunas: {df.shape[1]}')
    return pd.DataFrame({'Pct_missing': df.isna().mean().apply(lambda x: f"{x:.1%}"),
                          'Freq_missing': df.isna().sum().apply(lambda x: f"{x:,.0f}").replace(',','.')})

def comparar_modelos(lista_modelos , lista_nomes_modelos):
     summary = summary_col(results=lista_modelos, model_names=lista_nomes_modelos, stars=True,
                info_dict = {
                    'N':lambda x: "{0:d}".format(int(x.nobs))
            })
     return summary

def tabela_frequencias(df , var_name):
    contagem = df[var_name].value_counts(dropna=False)
    percent =  (df[var_name].value_counts(dropna=False, normalize=True)*100).round(2)
    aux=pd.concat([contagem, percent], axis=1, keys=['contagem', '%'], sort=False)

    tabela = tabulate(aux, headers='keys', tablefmt='grid', numalign='center') 
    return tabela

def print_count_cat_var_values(df  , lista_atributos ):
    """
    Print the attribute values for each categorical variable in lista_atributos.

    Parameters:
        df (DataFrame): DataFrame to be analysed

        lista_atributos (list): variable list 

    Returns:
        (None):

    """
    for i , j in enumerate(lista_atributos):
        # percent = df[j].value_counts( normalize=True)
        a = df[j].value_counts()
        a = a.sort_values(ascending=False)
        print(f"\n Count values for Variable {j}")
        for index, value in a.items():
            print(f"{index} ==> {value}")



def search_for_categorical_variables(df , lista_colunas = None):
    """Identify how many unique values exists in each column.

    Parameters:
        df (DataFrame): DataFrame to be analysed.

    Returns:
        cat_stats (DataFrame): Result DataFrame with
        
            - Coluna => the variable name
            - Valores => list os values
            - Contagem de Categorias => count of unique values

    """
    cat_stats = pd.DataFrame(
        columns=['Coluna', 'Valores', 'Contagem de Categorias'])
    tmp = pd.DataFrame()

    a_dict = {}
    if lista_colunas == None:
        lista_colunas = df.columns
    for c in lista_colunas:
        a_dict[c] = list(df[c].unique())
        tmp['Coluna'] = [c]
        tmp['Valores'] = list(df[c].unique().categories)
        tmp['Contagem de Categorias'] = f"{len(list(df[c].unique()))}"

        cat_stats = pd.concat([cat_stats, tmp], axis=0)
    return cat_stats , a_dict


def count_categorical_variables_values(a_serie):
    """Count how many instances of each values exists in the serie.

    Parameters:
        a_serie (Series): serie to be analysed.

    Returns:
        cat_stats (DataFrame): Result DataFrame with
        
            - Coluna => the variable name
            - Valores => list os values
            - Contagem de Categorias => count of unique values

    """
    cat_stats = pd.DataFrame(
        columns=['Category', 'Count',])
    tmp = pd.DataFrame()

    a_values = a_serie.unique()
    for c in a_values:
        print(c)
        tmp['Valores'] = [df[c].unique()]
        tmp['Contagem de Categorias'] = f"{len(list(df[c].unique()))}"

        cat_stats = pd.concat([cat_stats, tmp], axis=0)
    return cat_stats





def analyse_correlation_continuos_variables(df, lista_variaveis , quant_maximos):
    """
    Analyse and plot the correlation betwenn pairs of continuos variables.

    Parameters:
        df (DataFrame): DataFrame to be analysed

        lista_variaveis (list): variable list 
        
        quant_maximos : number of maximum values

    Returns:
        top_pairs_df (DataFrame): sorted DataFrame with Variable1 | Variable 2 | Correlation
        
        corr_matrix (Array): Correlation matrix with p-values on the   upper triangle 
    """
    cv_df = df[lista_variaveis]

    # metodos: 'pearson', 'kendall', 'spearman' correlations.
    corr_matrix = cv_df.corr(method='pearson')

    # Gera uma matriz de correlação onde a parte superior contem os p-valor
    # da correlação entre as variaveis considerando o nivel de significancia
    matriz_corr_with_pvalues = pg.rcorr(cv_df, method = 'pearson', upper = 'pval', decimals = 4, pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

    # Get the top n pairs with the highest correlation
    top_pairs = corr_matrix.unstack().sort_values(ascending=False)[
        :len(df.columns) + quant_maximos*2]

    # Create a list to store the top pairs without duplicates
    unique_pairs = []

    # Iterate over the top pairs and add only unique pairs to the list
    for pair in top_pairs.index:
        if pair[0] != pair[1] and (pair[1], pair[0]) not in unique_pairs:
            unique_pairs.append(pair)

    # Create a dataframe with the top pairs and their correlation coefficients
    top_pairs_df = pd.DataFrame(
        columns=['feature_1', 'feature_2', 'corr_coef'])
    for i, pair in enumerate(unique_pairs[:quant_maximos]):
        top_pairs_df.loc[i] = [pair[0], pair[1],
                               corr_matrix.loc[pair[0], pair[1]]]

    return top_pairs_df , matriz_corr_with_pvalues



def fill_categoric_field_with_value(serie, replace_nan = False):
    """Replace categorical value with int value.

    Parameters:
        serie (Series): data to be replace categorical with int
        replace_nan (Boolean): flag to replace nan with an index
        
    Returns:
        (Series): replaced values
        
    """
    # falta testar 

    label_encoder = LabelEncoder()
    a = label_encoder.fit_transform(serie)

    return a
    
    
    # names = serie.unique()
    # values = list(range(1, names.size + 1))
    # if not replace_nan:
    #     # a tabela de valores continha um float(nan) mapeado para um valor inteiro. Solução foi mudar na tabela de valores colocando o None
    #     nan_index = np.where(pd.isna(names))
    #     if len(nan_index) > 0 and len(nan_index[0]) > 0:
    #         nan_index = nan_index[0][0]
    #         values[nan_index] = None
    #     # else:
    #         # print("Não encontrou nan em " + str(names))

    # return serie.replace(names, values)


def teste_shapiro(modelo , alpha):
    # Teste de verificação da aderência dos resíduos à normalidade

    # Carregamento da função 'shapiro_francia' do pacote 'statstests.tests'

    # Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
    # https://stats-tests.github.io/statstests/


    if modelo.nobs < 30:     # Teste de Shapiro-Wilk (n < 30)
        
        #shapiro(modelo_linear.resid)
        print('Falta implementar este caso')
    else:  # Teste de Shapiro-Francia (n >= 30)
        # Teste de Shapiro-Francia: interpretação
        teste_sf = shapiro_francia(modelo.resid) #criação do objeto 'teste_sf'
        teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
        method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
        print('Teste de Shapiro-Francia - Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
        if p[1] > alpha:
            print('Teste de Shapiro-Francia - Não se rejeita H0 - Distribuição aderente à normalidade')
        else:
            print('Teste de Shapiro-Francia - Rejeita-se H0 - Distribuição não aderente à normalidade')
        return statistics_W[1], p[1]
    

def teste_breusch_pagan(modelo , alpha):
    # Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
    #relevante(s)
    
    # H0 do teste: ausência de heterocedasticidade.
    # H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
    # uma ou mais variáveis explicativas, o que indica omissão de variável relevante!

    df = pd.DataFrame({'yhat':modelo.fittedvalues, 'resid':modelo.resid})
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    chisq = anova_table['sum_sq'].iloc[0]
    p_value = stats.chi2.pdf(chisq, 1)*2
    # print(f"chisq: {chisq}")
    # print(f"p-value: {p_value}")
    print('Teste de BRESCH PAGAN - chisq=%.5f, p-value=%.6f' % (chisq , p_value))
    
    if p_value > alpha:
        print('Teste de BRESCH PAGAN - Não se rejeita H0 - Ausência de Heterocedasticidade')
    else:
    	print('Teste de BRESCH PAGAN - Rejeita-se H0 - Existência de Heterocedasticidade')
        
    return chisq, p_value

import os

def lista_arquivos(dir_name , funcao):
    for a_dir , _, lista_arquivos in os.walk(dir_name): #percorre toda a estrutura olhando os subdiretorios
        for nome_arquivo in lista_arquivos:  #para cada subdiretorio
            arquivo = os.path.join( a_dir , nome_arquivo)
            funcao(arquivo)
           
def imprime_nome(nome):
    print(nome)
    
import sys
import requests

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def get_arquivo_google_drive(url_link , destination_file):
    # file_name = 'https://drive.google.com/file/d/0Bxh_iVKkQbhVNk41ZTk3SU5vWEE/view?usp=sharing&resourcekey=0-EZB2hPBsXBeuQuFpzGzkRg'
   
    # obter o id a partir do link (texto entre /d e /view)
    
    i1 = url_link.find("/d/") + 3
    i2 = url_link.find("/view")

    file_id = file_id = url_link[i1:i2]
     
    download_file_from_google_drive(file_id, destination_file)
    
    # url_link = 'https://drive.google.com/file/d/0Bxh_iVKkQbhVNk41ZTk3SU5vWEE/view?usp=sharing&resourcekey=0-EZB2hPBsXBeuQuFpzGzkRg'
    # get_arquivo_google_drive(url_link, './teste1')
    
    
# Definição da função para realização do teste de razão de verossimilhança para verificar se 2 
# modelos são estatísticamente semelhantes, ou seja, se pode ser usado qualquer um ou não
# graus_liberdade = diferenca entre a quantidade de variaveis entre os modelos
def teste_ll_verossimilhanca(modelos , nivel_significancia = 0.05 , graus_liberdade = 1):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llnull
    llk_2 = modelo_1.llf
    
    if len(modelos)>1:
        llk_1 = modelo_1.llf
        llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, graus_liberdade) # 1 grau de liberdade
    
    print("Likelihood Ratio Test:")
    print(f"-2.(LL0-LLm): {round(LR_statistic, 2)}")
    print(f"p-value: {p_val:.3f}")
    print("")
    print("==================Result======================== \n")
    if p_val <= nivel_significancia:
        print("H1: Different models, favoring the one with the highest Log-Likelihood")
    else:
        print(f"H0: Models with log-likelihoods that are not statistically different at [{nivel_significancia}]% confidence level")
    
#Teste de Vuong
# VUONG, Q. H. Likelihood ratio tests for model selection and non-nested
#hypotheses. Econometrica, v. 57, n. 2, p. 307-333, 1989.
# Definição de função para elaboração do teste de Vuong

from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP,ZeroInflatedPoisson

def vuong_test(m1, m2 , sig_level = 0.05):

    from scipy.stats import norm    

    if m1.__class__.__name__ == "GLMResultsWrapper":
        
        glm_family = m1.model.family

        X = pd.DataFrame(data=m1.model.exog, columns=m1.model.exog_names)
        y = pd.Series(m1.model.endog, name=m1.model.endog_names)

        if glm_family.__class__.__name__ == "Poisson":
            m1 = Poisson(endog=y, exog=X).fit()
            
        if glm_family.__class__.__name__ == "NegativeBinomial":
            m1 = NegativeBinomial(endog=y, exog=X, loglike_method='nb2').fit()

    supported_models = [ZeroInflatedPoisson,ZeroInflatedNegativeBinomialP,Poisson,NegativeBinomial]
    
    if type(m1.model) not in supported_models:
        raise ValueError(f"Model type not supported for first parameter. List of supported models: (ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, Poisson, NegativeBinomial) from statsmodels discrete collection.")
        
    if type(m2.model) not in supported_models:
        raise ValueError(f"Model type not supported for second parameter. List of supported models: (ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, Poisson, NegativeBinomial) from statsmodels discrete collection.")
    
    # Extração das variáveis dependentes dos modelos
    m1_y = m1.model.endog
    m2_y = m2.model.endog

    m1_n = len(m1_y)
    m2_n = len(m2_y)

    if m1_n == 0 or m2_n == 0:
        raise ValueError("Could not extract dependent variables from models.")

    if m1_n != m2_n:
        raise ValueError("Models appear to have different numbers of observations.\n"
                         f"Model 1 has {m1_n} observations.\n"
                         f"Model 2 has {m2_n} observations.")

    if np.any(m1_y != m2_y):
        raise ValueError("Models appear to have different values on dependent variables.")
        
    m1_linpred = pd.DataFrame(m1.predict(which="prob"))
    m2_linpred = pd.DataFrame(m2.predict(which="prob"))        

    m1_probs = np.repeat(np.nan, m1_n)
    m2_probs = np.repeat(np.nan, m2_n)

    which_col_m1 = [list(m1_linpred.columns).index(x) if x in list(m1_linpred.columns) else None for x in m1_y]    
    which_col_m2 = [list(m2_linpred.columns).index(x) if x in list(m2_linpred.columns) else None for x in m2_y]

    for i, v in enumerate(m1_probs):
        m1_probs[i] = m1_linpred.iloc[i, which_col_m1[i]]

    for i, v in enumerate(m2_probs):
        m2_probs[i] = m2_linpred.iloc[i, which_col_m2[i]]

    lm1p = np.log(m1_probs)
    lm2p = np.log(m2_probs)

    m = lm1p - lm2p

    v = np.sum(m) / (np.std(m) * np.sqrt(len(m)))

    pval = 1 - norm.cdf(v) if v > 0 else norm.cdf(v)

    print("Vuong Non-Nested Hypothesis Test-Statistic (Raw):")
    print(f"Vuong z-statistic: {round(v, 3)}")
    print(f"p-value: {pval:.3f}")
    print("")
    print("==================Result======================== \n")
    if pval <= sig_level:
        print(f"H1: Indicates inflation of zeros at {1 - sig_level}% confidence level")
    else:
        print(f"H0: Indicates no inflation of zeros at {1 - sig_level}% confidence level")

from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, balanced_accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def avalia_clf(clf, y, X, rótulos_y=['falso', 'positivo'], base = 'treino'):
    
    # Calcular as classificações preditas
    pred = clf.predict(X)
    
    # Calcular a probabilidade de evento
    y_prob = clf.predict_proba(X)[:, -1]
    
    # Calculando acurácia e matriz de confusão
    cm = confusion_matrix(y, pred)
    ac = accuracy_score(y, pred)
    bac = balanced_accuracy_score(y, pred)

    print(f'\nBase de {base}:')
    print(f'A acurácia da árvore é: {ac:.1%}')
    print(f'A acurácia balanceada da árvore é: {bac:.1%}')
    
    # Calculando AUC
    auc_score = roc_auc_score(y, y_prob)
    print(f"AUC-ROC: {auc_score:.2%}")
    print(f"GINI: {(2*auc_score-1):.2%}")
    
    # Visualização gráfica
    sns.heatmap(cm, 
                annot=True, fmt='d', cmap='viridis', 
                xticklabels=rótulos_y, 
                yticklabels=rótulos_y)
    
    # Relatório de classificação do Scikit
    print('\n', classification_report(y, pred))
    
    # Gerar a Curva ROC
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    
    # Plotar a Curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Linha de referência (modelo aleatório)
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title(f"Curva ROC - base de {base}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def avalia_regressor(reg, y, X, rótulos_y=['falso', 'positivo'], base = 'treino'):
    
    # Calcular as classificações preditas
    y_pred = reg.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Exibindo as métricas
    print('Resultados ' + base)
    print(f'MSE: {mse:,.2}')
    print(f'RMSE: {rmse:,.2}')
    print(f'MAE: {mae:,.2}')
    print(f'R²: {r2:,.2%}')
 

def gerar_base_analise_credito():

    # Definir uma semente aleatória para reprodutibilidade
    np.random.seed(42)
    random.seed(42)

    # Gerar as variáveis simuladas com correlação
    idade = np.random.randint(18, 71, 10000)

    # Gerar variáveis correlacionadas usando a função multivariada normal
    mean_values = [5000, 2000, 0.5, 5]  # Médias das variáveis
    correlation_matrix = np.array([
        [1, 0.3, 0.2, -0.1],
        [0.3, 1, -0.1, 0.2],
        [0.2, -0.1, 1, 0.4],
        [-0.1, 0.2, 0.4, 1]
    ])  # Matriz de correlação

    # Gerar dados simulados
    simulated_data = np.random.multivariate_normal(mean_values, correlation_matrix, 10000)

    renda = simulated_data[:, 0]
    divida = simulated_data[:, 1]
    utilizacao_credito = np.clip(simulated_data[:, 2], 0, 1)  # Limita a utilização de crédito entre 0 e 1
    consultas_recentes = np.maximum(simulated_data[:, 3], 0)  # Garante que o número de consultas recentes seja não negativo

    # Gerar função linear das variáveis explicativas
    preditor_linear = -7 - 0.01 * idade - 0.0002 * renda + 0.003 * divida - 3 * utilizacao_credito + 0.5 * consultas_recentes

    # Calcular probabilidade de default (PD) usando a função de link logit
    prob_default = 1 / (1 + np.exp(-preditor_linear))

    # Gerar inadimplência como variável Bernoulli com base na probabilidade de default
    inadimplencia = np.random.binomial(1, prob_default, 10000)

    # Criar dataframe
    dados = pd.DataFrame({
        'idade': idade,
        'renda': renda,
        'divida': divida,
        'utilizacao_credito': utilizacao_credito,
        'consultas_recentes': consultas_recentes,
        'inadimplencia': inadimplencia
    })

    # Categorizar a idade
    kbin = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    dados['idade_cat'] = kbin.fit_transform(dados[['idade']])

    return dados


#%% Geracao de amostras randomicas
import random

# Definir uma semente aleatória para reprodutibilidade
np.random.seed(42)
random.seed(42)

# Gerar as variáveis simuladas com correlação
idade = np.random.randint(18, 71, 10000)

# Gerar variáveis correlacionadas usando a função multivariada normal
mean_values = [5000, 2000, 0.5, 5]  # Médias das variáveis
correlation_matrix = np.array([
    [1, 0.3, 0.2, -0.1],
    [0.3, 1, -0.1, 0.2],
    [0.2, -0.1, 1, 0.4],
    [-0.1, 0.2, 0.4, 1]
])  # Matriz de correlação

# Gerar dados simulados
simulated_data = np.random.multivariate_normal(mean_values, correlation_matrix, 10000)

#%% IBGE e GOOGLE MAPS: Distancias
def obter_nome_municipio(codigo_ibge):

    # URL da API de Localidades do IBGE
    url = f"https://servicodados.ibge.gov.br/api/v1/localidades/municipios/{codigo_ibge}"
    
    # Fazendo a requisição
    response = requests.get(url)
    data = response.json()
    
    # Extraindo o nome do município e a UF
    nome_municipio = data['nome']
    uf = data['microrregiao']['mesorregiao']['UF']['sigla']
    
    # Combinando o nome do município com a UF
    municipio_completo = f"{nome_municipio}, {uf}"
    return municipio_completo

def obter_distancia(cod_origem , cod_destino , gmaps):
    if cod_origem == cod_destino:
        return 0
    origem = obter_nome_municipio(cod_origem)
    destino = obter_nome_municipio(cod_destino)

    try:
        directions_result = gmaps.directions(origem, destino, mode="driving")
        
        if directions_result:
            if 'legs' in directions_result[0] and 'distance' in directions_result[0]['legs'][0]:
                distancia_km = directions_result[0]['legs'][0]['distance']['text']
                return distancia_km
            else:
                return "Sem informações de distância na resposta da API."
        else:
            return "Sem resultados da API para a rota desejada."
    except Exception as e:
        return f"Erro ao calcular distância: {str(e)}"
    
    
# google_maps_api_key = 'AIzaSyB1J_GG1WCrkDU0X2Z45Ao_Y6vXfYIz71w'
# gmaps = googlemaps.Client(key=google_maps_api_key)

# cod_origem = '3304557'
# cod_destino = '3550308'
# a = obter_distancia(cod_origem, cod_destino, gmaps)

def elimina_outliers_quartile(df , a_var , q1_porc=0.25 , q2_porc=0.75):

    # Cálculo do IQR
    Q1 = df[a_var].quantile(q1_porc)
    Q2 = df[a_var].quantile(q2_porc)
    IQR = Q2 - Q1
    
    # Limites para outliers
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q2 + 1.5 * IQR
    
    # Identificando outliers
    df = df[(df[a_var] >= limite_inferior) & (df[a_var] <= limite_superior)]

    return df

def elimina_outliers_valores(df , a_var , l1 , l2):
    
    # Identificando outliers
    # df = df[(df[a_var] >= l1) & (df[a_var] <= l2)]
    df = df[df[a_var].between(l1, l2, inclusive='both')]

    return df

def stepwise_ols(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()  
            included.append(new_pval.index[best_feature])
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(new_pval.index[best_feature], best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            
            included.remove(pvalues.index[worst_feature])
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(pvalues.index[worst_feature], worst_pval))
        if not changed:
            break
    return included

def stepwise_mod(model, pvalue_limit: float=0.05):
    
    r"""
    
    Stepwise process for Statsmodels regression models

    Usage example

    .. ipython:: python

        import statsmodels.api as sm
        from statstests.datasets import empresas
        from statstests.process import stepwise

        # import empresas dataset
        df = empresas.get_data()

        # Estimate and fit model
        model = sm.OLS.from_formula("retorno ~ disclosure + endividamento + ativos + liquidez", df).fit()

        # Print summary
        print(model.summary())

        # Stepwise process
        stepwise(model, pvalue_limit=0.05)

    Parameters
    ----------
    model : Statsmodels model

    pvalue_limit : float

    Returns
    -------
    model : Stepwised model

    References
    ----------
    .. [1] Reference

    """
    
    # dictionary that identifies the type of the inputed model
    models_types = {"<class 'statsmodels.regression.linear_model.OLS'>": "OLS",
                    "<class 'statsmodels.discrete.discrete_model.Logit'>": "Logit",
                    "<class 'statsmodels.genmod.generalized_linear_model.GLM'>": "GLM"}

    try:
        # identify model type
        model_type = models_types[str(type(model.model))]
    except:
        raise Exception("The model is not yet supported...",
                        "Suported types: ", list(models_types.values()))

    print("Regression type:", model_type, "\n")

    try:
#retirada a insercao de Q() nos nomes das variaveis
        # formula = model.model.data.ynames + " ~ " + \
        #     ' + '.join(["Q('" + name + "')" for name in  model.model.data.xnames[1:]])
        formula = model.model.data.ynames + " ~ " + \
            ' + '.join([name for name in  model.model.data.xnames[1:]])
            
        df = pd.concat([model.model.data.orig_endog,
                       model.model.data.orig_exog], axis=1)

        atributes_discarded = []

        while True:

            print("Estimating model...: \n", formula)

            if model_type == 'OLS':

                # return OLS model
                model = sm.OLS.from_formula(formula=formula, data=df).fit()

            elif model_type == 'Logit':

                # return Logit model
                model = sm.Logit.from_formula(formula=formula, data=df).fit()

            elif model_type == 'GLM':

                # dictionary that identifies the family type of the inputed glm model
                glm_families_types = {"<class 'statsmodels.genmod.families.family.Poisson'>": "Poisson",
                                      "<class 'statsmodels.genmod.families.family.NegativeBinomial'>": "Negative Binomial",
                                      "<class 'statsmodels.genmod.families.family.Binomial'>": "Binomial"}

                # identify family type
                family_type = glm_families_types[str(type(model.family))]

                print("\n Family type...: \n", family_type)

                if family_type == "Poisson":
                    model = smf.glm(formula=formula, data=df,
                                    family=sm.families.Poisson()).fit()
                elif family_type == "Negative Binomial":
                    model = smf.glm(formula=formula, data=df,
                                    family=sm.families.NegativeBinomial()).fit()
                elif family_type == "Binomial":
                    model = smf.glm(formula=formula, data=df,
                                    family=sm.families.Binomial()).fit()

            atributes = model.model.data.xnames[1:]

            # find atribute with the worst p-value
            worst_pvalue = (model.pvalues.iloc[1:]).max()
            worst_atribute = (model.pvalues.iloc[1:]).idxmax()

            # identify if the atribute with the worst p-value is higher than p-value limit
            if worst_pvalue > pvalue_limit:

                # exclude atribute higher than p-value limit from atributes list
                atributes = [
                    element for element in atributes if element is not worst_atribute]

                # declare the new formula without the atribute
                formula = model.model.data.ynames + \
                    " ~ " + ' + '.join(atributes)

                # append the atribute to the atributes_discarded list
                atributes_discarded.append(
                    {'atribute': worst_atribute, 'p-value': worst_pvalue})

                print(
                    '\n Discarding atribute "{}" with p-value equal to {} \n'.format(worst_atribute, worst_pvalue))

            else:

                # print that the loop is finished and there are not more atributes to discard
                print(
                    '\n No more atributes with p-value higher than {}'.format(pvalue_limit))

                break

        # print model summary after stepwised process
        print("\n Atributes discarded on the process...: \n")

        # print all the discarded atributes from the atributes_discarded list
        [print(item) for item in atributes_discarded]

        print("\n Model after stepwise process...: \n", formula, "\n")
        print(model.summary())

        # return stepwised model
        return model , atributes_discarded

    except Exception as e:
        raise e
  
import winsound

def alerta_sonoro():

    duration = 10000  # milliseconds
    freq = 1000  # Hz
    winsound.Beep(freq, duration)
    
def obter_mapeamento_TargetEncoder(encoder):
    dict_encoders = {}
    for a_dict in encoder.ordinal_encoder.mapping: 
        var_name = a_dict.get('col')
        ordinal_encoder = pd.DataFrame(a_dict.get('mapping'))
        media_encoder = pd.DataFrame(encoder.mapping.get(var_name))
        dict_encoders[var_name] = {'Ordinal' : ordinal_encoder , 'Media' : media_encoder }
    return dict_encoders