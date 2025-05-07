# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Aplicação do método PCA.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: Ulf Bergmann
"""

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import sympy as sy
import scipy as sp

import sys

sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

import funcoes as f
from funcoes import Log

import funcoes_ulf as uf
import bib_graficos as ug

#%% FUNCOES

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.05, point['y'], point['val'])
        
def plot_loading(df , var_list):
    '''Plot loading factors.
    Parameters
    ----------
    df(DataFrame) : data to be ploted.
    var_list(list) : variable column name in df.
    
    Returns
    -------
    None.

    '''

    plt.figure(figsize=(12,8))
    df_chart = df.reset_index()
    plt.scatter(df_chart[var_list[0]], df_chart[var_list[1]], s=50, color='red')



    label_point(x = df_chart[var_list[0]],
                y = df_chart[var_list[1]],
                val = df_chart['index'],
                ax = plt.gca()) 

    plt.axhline(y=0, color='grey', ls='--')
    plt.axvline(x=0, color='grey', ls='--')
    plt.ylim([-1.1,1.1])
    plt.xlim([-1.1,1.1])
    plt.title("Loading Plot", fontsize=16)
    plt.xlabel(var_list[0], fontsize=12)
    plt.ylabel(var_list[1], fontsize=12)
    plt.show()


#%% CARREGAMENTO DOS DADOS

log = Log()
log.carregar_log("log_BaseModelagemFinal")
df = f.leitura_arquivo_parquet("BaseModelagemFinal")

print(log.logar_acao_realizada("Carga Dados", "Carregamento dos dados para ANALISE DO PCA", df.shape[0]))

var_dep = "_Gerada_RESFINAL"

colunas_categoricas = [ '_Gerada_RESFINAL', '_Gerada_ALCOOLIS', '_Gerada_BASMAIMP_CLIN', '_Gerada_BASMAIMP_PESQ', '_Gerada_BASMAIMP_IMG', '_Gerada_BASMAIMP_MARTUM', '_Gerada_BASMAIMP_CIT', '_Gerada_BASMAIMP_MET', '_Gerada_BASMAIMP_TUMPRIM', '_Gerada_DIAGANT_DIAG', '_Gerada_DIAGANT_TRAT', '_Gerada_EXDIAG_EXCLIN', '_Gerada_EXDIAG_IMG', '_Gerada_EXDIAG_END_CIR', '_Gerada_EXDIAG_PAT', '_Gerada_EXDIAG_MARC', '_Gerada_HISTFAMC', '_Gerada_LATERALI_ESQ', '_Gerada_LATERALI_DIR', '_Gerada_MAISUMTU', '_Gerada_ORIENC_SUS', '_Gerada_TABAGISM', '_Gerada_TIPOHIST_BIOLOGICO_1', '_Gerada_TIPOHIST_BIOLOGICO_2', '_Gerada_TIPOHIST_BIOLOGICO_3', '_Gerada_TIPOHIST_BIOLOGICO_6', '_Gerada_TIPOHIST_BIOLOGICO_9', '_Gerada_TNM_M_1', '_Gerada_TNM_M_X', '_Gerada_TNM_N_1', '_Gerada_TNM_N_2', '_Gerada_TNM_N_3', '_Gerada_TNM_N_4', '_Gerada_TNM_N_X', '_Gerada_TNM_T_1', '_Gerada_TNM_T_2', '_Gerada_TNM_T_3', '_Gerada_TNM_T_4', '_Gerada_TNM_T_A', '_Gerada_TNM_T_I', '_Gerada_TNM_T_X', 'SEXO_2', '_Gerada_TIPOHIST_CELULAR_ENC']

colunas_metricas = [ 'IDADE' , '_Gerada_tempo_para_inicio_tratamento', '_Gerada_distancia_tratamento' , 'LOCTUDET_ENC', 'ESTADIAM_ENC']

# colunas = list(df.columns)
# for i in colunas_categoricas:
#     colunas.remove(i)
    


#%% Informações sobre as variáveis


# Informações gerais sobre o DataFrame

print(df.info())

# Estatísticas descritiva das variáveis

print(df.describe())

#%% Separando somente as variáveis quantitativas do banco de dados

df_metricas = df[colunas_metricas]

#%% Matriz de correlações de Pearson entre as variáveis

pg.rcorr(df_metricas, method = 'pearson', upper = 'pval', 
         decimals = 4, 
         pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

#%% Outra maneira de analisar as informações das correlações

# Matriz de correlações em um objeto "simples"

corr = df_metricas.corr()


#%% Teste de Esfericidade de Bartlett
# Executa um teste de hipoteses.
# H0: matriz de correlação igual a identidade
# H1: sao diferentes

# Ver o p-valor em relação ao nível de significancia.

bartlett, p_value = calculate_bartlett_sphericity(df_metricas)

print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')

#%% Definindo a PCA (procedimento inicial com todos os fatores possíveis)
#fatores maximos = nr de variaveis
fa = FactorAnalyzer(n_factors=5, method='principal', rotation=None).fit(df_metricas)

#%% Obtendo AutoValores e aplicando o Critério de Kaiser
# Obtendo os eigenvalues (autovalores): resultantes da função FactorAnalyzer e aplicando o Critério de Kaiser (raiz latente). Escolha da quantidade de fatores

autovalores = fa.get_eigenvalues()[0]

print(autovalores) # 


#Aplicando o Critério de Kaiser (raiz latente). Escolha da quantidade de fatores. 
# Verificar os autovalores com valores maiores que 1
# Existem 2 componentes maiores do que 1

#%% Parametrizando a PCA para dois fatores (autovalores > 1)

fa = FactorAnalyzer(n_factors=2, method='principal', rotation=None).fit(df_metricas)

#%% Eigenvalues, variâncias e variâncias acumuladas de 2 fatores

# Note que não há alterações nos valores, apenas ocorre a seleção dos fatores

autovalores_fatores = fa.get_factor_variance()

tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

#%% Determinando as cargas fatoriais

# Note que não há alterações nas cargas fatoriais nos 2 fatores!

cargas_fatoriais = fa.loadings_

tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = df_metricas.columns

print(tabela_cargas)

#%% Determinando as novas comunalidades

# As comunalidades são alteradas, pois há fatores retirados da análise!

comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = df_metricas.columns

print(tabela_comunalidades)

#%% Extração dos fatores para as observações do banco de dados

#  Vamos gerar novamente, agora para os 4 fatores extraídos
fatores = pd.DataFrame(fa.transform(df_metricas))
fatores.columns =  [f"_GeradaPCA_{i+1}" for i, v in enumerate(fatores.columns)]

# Adicionando os fatores ao banco de dados
df_pca_aplicado = pd.concat([df.reset_index(drop=True), fatores], axis=1)
df_pca_aplicado = df_pca_aplicado.drop(columns = colunas_metricas)

f.salvar_parquet(df_pca_aplicado, "BaseModelagemFinalAposPCA")
a = df_pca_aplicado.columns

#%% Analisando o impacto no VIF


from statsmodels.stats.outliers_influence import variance_inflation_factor

# # Cálculo do VIF para cada variável - ANTES DO PCA

vif_data = pd.DataFrame({
    "Variável": fatores.columns,
    "VIF": [variance_inflation_factor(fatores.values, i) for i in range(fatores.shape[1])]
})
print(vif_data)

vif_data1 = pd.DataFrame({
    "Variável": colunas_metricas,
    "VIF": [variance_inflation_factor(df[colunas_metricas].values, i) for i in range(df[colunas_metricas].shape[1])]
})
print(vif_data1)


# # # Cálculo do VIF para cada variável
# df_aux = df.drop(columns = [var_dep])
# vif_data = pd.DataFrame({
#     "Variável": df.columns,
#     "VIF": [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
# })

# df_pca_aplicado.columns


# Note que são os mesmos, apenas ocorre a seleção dos 2 primeiros fatores!

# #%% Identificando os scores fatoriais

# # Não há mudanças nos scores fatoriais!

# scores = fa.weights_

# tabela_scores = pd.DataFrame(scores)
# tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
# tabela_scores.index = df_metricas.columns

# print(tabela_scores)

# #%% Criando um ranking (soma ponderada e ordenamento)

# # O ranking irá considerar apenas os 2 fatores com autovalores > 1
# # A base de seleção é a tabela_eigen

# notas['Ranking'] = 0

# for index, item in enumerate(list(tabela_eigen.index)):
#     variancia = tabela_eigen.loc[item]['Variância']

#     df_pca_aplicado['Ranking'] = notas['Ranking'] + df_pca_aplicado[tabela_eigen.index[index]]*variancia
    
# print(notas)

# #%% Em certos casos, a "rotação de fatores" pode melhorar a interpretação

# # Analisando pelo loading plot, aplica-se a rotação dos eixos na origem (0,0)
# # O método mais comum é a 'varimax', que é a rotação ortogonal dos fatores
# # O objetivo é aumentar a carga fatorial em um fator e diminuir em outro
# # Em resumo, trata-se de uma redistribuição de cargas fatoriais

# #%% Adicionando a rotação: rotation='varimax'

# # Aplicando a rotação aos 2 fatores extraídos

# fa_1 = FactorAnalyzer(n_factors=2, method='principal', rotation='varimax').fit(df_metricas)

# cargas_fatoriais_1 = fa_1.loadings_

# tabela_cargas_1 = pd.DataFrame(cargas_fatoriais_1)
# tabela_cargas_1.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas_1.columns)]
# tabela_cargas_1.index = df_metricas.columns

# print(tabela_cargas_1)

# #%% Gráfico das cargas fatoriais (loading plot)

# plot_loading(tabela_cargas_1, ['Fator 1' , 'Fator 2'])



# #%% Fim!