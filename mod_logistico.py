# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Construcao do Modelo Logistico.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: Ulf Bergmann

"""

import pandas as pd
import numpy as np

import time

import sys

sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

import funcoes as f
from funcoes import Log

import funcoes_ulf as uf
import bib_graficos as ug

import statsmodels.api as sm # estimação de modelos
import statsmodels.formula.api as smf # estimação do modelo logístico binário
                                                        
                                                     
import warnings
warnings.filterwarnings('ignore')



#%% CARGA DADOS


log = Log()
log.carregar_log("log_BaseModelagemFinal")
df = f.leitura_arquivo_parquet("BaseModelagemFinal")
# BaseModelagemFinalAposPCA
print(log.logar_acao_realizada("Carga Dados", "Carregamento dos dados para modelagem", df.shape[0]))

var_dep = "_Gerada_RESFINAL"

#%% Criacao do Modelo

var_eliminar = [ '_Gerada_TIPOHIST_BIOLOGICO_2' ,   '_Gerada_TIPOHIST_BIOLOGICO_3' ,  '_Gerada_BASMAIMP_TUMPRIM' ]
df = df.drop(columns=var_eliminar)
print(log.logar_acao_realizada("Carga Dados", "Eliminadas as colunas", var_eliminar))

# Geracao de fórmulas quando temos muitas colunas

lista_colunas = list(df.columns)
lista_colunas.remove(var_dep)
# lista_colunas = ["Q('" + elemento + "')" for elemento in lista_colunas]
a_formula = ' + '.join(lista_colunas)
# a_formula = "Q('" + var_dep + "')" + " ~ " + a_formula
a_formula = var_dep + " ~ " + a_formula
print("Fórmula utilizada: ", a_formula)

# lista_colunas = list(df.columns)
# lista_colunas.remove(var_dep)
# a_formula = ' + '.join(lista_colunas)
# a_formula = var_dep + " ~ " + a_formula
# print("Fórmula utilizada: ", a_formula)

# from statsmodels.genmod.generalized_linear_model import L2

# import statsmodels.genmod.generalized_linear_model as glm


start_time = time.time()

# penalizacao = glm.L2(alpha=1.0)

# modelo_inicial = smf.glm(formula=a_formula, data=df, family=sm.families.Binomial() , penalizer=penalizacao ).fit()

modelo_inicial = smf.glm(formula=a_formula, data=df, family=sm.families.Binomial()).fit()



# X = df.drop([var_dep] , axis=1)  # Features (variáveis independentes)
# y = df[var_dep]       # Target (variável dependente)

# from sklearn.linear_model import LogisticRegression
# modelo_inicial = LogisticRegression(random_state=0).fit(X, y)
# model.coef_

end_time = time.time()
tempo_mod_inicial = end_time - start_time
print( log.logar_acao_realizada( "Modelagem", "criacao do modelo Logístico Binário", tempo_mod_inicial  ))



#%% Aplicação do stepwise

# =============================================================================
# POSSIBILIDADE: 
# em vez de usar stepwise tentar PCA , analise fatorial ou Deep learning não super (maquinas de boltzmann)
# =============================================================================

start_time = time.time()

modelo_step , att_disc = uf.stepwise_mod(modelo_inicial, pvalue_limit=0.05)



end_time = time.time()
tempo_mod_step = end_time - start_time

#%% Salvar dados do modelo

cutoff=0.45

# Calcular o AUC para a massa de testes de treino e teste
df['phat_inicial'] = modelo_inicial.predict()
df['phat_step'] = modelo_step.predict()

a_nome_coluna =  'LogisticoBinario_Escolhido'

resumo = pd.DataFrame()
resumo.at['Descricao' , a_nome_coluna] = 'Sem PCA e Remocao de _Gerada_TIPOHIST_BIOLOGICO_2 _Gerada_TIPOHIST_BIOLOGICO_3 _Gerada_BASMAIMP_TUMPRIM'
resumo.at['Variaveis modelo inicial' , a_nome_coluna] = str(list(modelo_inicial.params.index)) 
resumo.at['Variaveis descartadas step' , a_nome_coluna] = str(att_disc) 
resumo.at['Variaveis modelo Step' , a_nome_coluna] = str(list(modelo_inicial.params.index)) 
resumo.at['Tamanho Base Modelagem' , a_nome_coluna] = df.shape[0] 
resumo.at['Tempo de construcao do modelo inicial' , a_nome_coluna] = tempo_mod_inicial
resumo.at['Tempo de construcao do modelo stepwise' , a_nome_coluna] = tempo_mod_step
resumo.at['Formula Stepwise' , a_nome_coluna] = modelo_step.model.formula
resumo.at['Cutoff' , a_nome_coluna] = cutoff


indicadores = ug.calcula_indicadores_predicao_classificacao(f'{a_nome_coluna} Inicial'  , modelo_inicial , df[var_dep],df['phat_inicial'] , cutoff)
aux_ind = ug.calcula_indicadores_predicao_classificacao(f'{a_nome_coluna} Step'  , modelo_inicial , df[var_dep],df['phat_step'] , cutoff)

indicadores = pd.concat([indicadores, aux_ind], axis=0)

modelo_step_summary = modelo_step.summary()

an_obj_dict = {
    'modelo_inicial' : modelo_inicial,
    'resumo' : resumo,
    'summary' : modelo_step_summary,
    'modelo_step' : modelo_step,
    'cutoff' : cutoff,
    'indicadores' : indicadores,
    }

model = modelo_step
summary_df = pd.DataFrame({
    'Coeficiente': modelo_step.params,
    'Erro Padrão': modelo_step.bse,
    'Estatística z': modelo_step.tvalues,
    'p-valor': modelo_step.pvalues,
})

f.salvar_excel_conclusao(summary_df, f'{a_nome_coluna}_atributos_variaveis_modelo')  
f.save_model(a_nome_coluna + '_sem_pca' , modelo_step)
f.save_objects(an_obj_dict , a_nome_coluna)

#%% Analise dos resultados finais

# # # Plotagem de um gráfico que mostra a variação da especificidade e da sensitividade em função do cutoff
# # #Somente para entendimento e não para definir o cutoff

lista_medidas = ['Sensitividade', 'Precisao' , 'F1_Score']
tabela = ug.plota_analise_indicadores_predicao_classificacao('step' , modelo_step , df[var_dep] , df['phat_step'] , lista_medidas = lista_medidas)

cutoff=0.45
df['phat_step'] = modelo_step.predict()

#%%


ug.plot_matriz_confusao(df[var_dep],df['phat_step'] , cutoff)

# # Construção da curva ROC
ug.plot_curvas_roc(df , var_dep , ['phat_step'])

# from sklearn.metrics import classification_report
# a = np.where(df['phat_train'] >= cutoff , 1 , 0)
# print(classification_report(df[var_dep] ,a))

#%% Informaçoes das variaveis do modelo



modelo_step.summary()

