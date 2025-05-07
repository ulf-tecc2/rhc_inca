# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Analise da influencia das variaveis na predicao do modelo XGBoosting.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: Ulf Bergmann

"""
# !pip install shap --upgrade


import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import pickle
import sys

sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

import funcoes as f
from funcoes import Log

import funcoes_ulf as uf
import bib_graficos as ug

# import warnings
# warnings.filterwarnings('ignore')

#%% Carga dos dados


log = Log()
log.carregar_log("log_BaseModelagemFinal")
df = f.leitura_arquivo_parquet("BaseModelagemFinal")

print(log.logar_acao_realizada("Carga Dados", "Carregamento dos dados para modelagem", df.shape[0]))

var_dep = "_Gerada_RESFINAL"

mod_xgb = f.load_model('XGBoost_Model_sem_pca')

# Gerando as predicoes

# df_working['pred_xgb'] = mod_xgb.predict(X)
# X_test['pred_xgb'] = df_working['pred_xgb'] 
    
#%% Definir amostra

porcentagem_amostra_shap_values = 0.01
df_amostra = df.groupby(var_dep, group_keys=False).apply(lambda x: x.sample(frac=porcentagem_amostra_shap_values , random_state=1))
X_amostra = df_amostra.drop(columns=[var_dep])
y_amostra = df_amostra[var_dep]


#%% Calcular shap para XGB
   
explainer_xgb = shap.TreeExplainer(mod_xgb)

# =============================================================================
#  GERAR VALORES E SALVAR
# =============================================================================
shap_values_xgb = explainer_xgb.shap_values(X_amostra)

print('Termino de gerar shap_values para XGB')
# uf.alerta_sonoro()


# =============================================================================
# CARREGAR VALORES SALVOS
# =============================================================================

# with open("dados/workdir/shap_xgb5.pkl", 'rb') as file:  
#     shap_values_xgb = pickle.load(file)

# =============================================================================
# ANALISAR VALORES
# =============================================================================

#%%% Analise grafica
variaveis_principais = [
'ESTADIAM_ENC',
'LOCTUDET_ENC',
'_Gerada_TIPOHIST_CELULAR_ENC',
'_Gerada_distancia_tratamento',
'_Gerada_TNM_M_1',
'_Gerada_EXDIAG_PAT',
'_Gerada_ORIENC_SUS',
'_Gerada_tempo_para_inicio_tratamento',
'_Gerada_EXDIAG_MARC',
'IDADE',
]

## Gerar os gráficos de dependência
for i, var in enumerate(variaveis_principais):
    # plt.figure(figsize=(4, 3))
    shap.dependence_plot(var, shap_values_xgb, X_amostra , interaction_index=None)
    ## Ajustar layout
    # plt.tight_layout()
    plt.show()



shap.summary_plot(shap_values_xgb, X_amostra, feature_names=X_amostra.columns , max_display=36)
shap.summary_plot(shap_values_xgb, X_amostra, feature_names=X_amostra.columns , plot_type="bar" , max_display=36)

#%% Analise das variaveis TargetEncoder

mapping_df = f.leitura_arquivo_parquet("BaseMapeamentoTargetEncoder")
aux = mapping_df

Gerada_TIPOHIST_CELULAR_positivo = mapping_df[mapping_df['_Gerada_TIPOHIST_CELULAR_ENC'] > 0.8]
Gerada_TIPOHIST_CELULAR_positivo = Gerada_TIPOHIST_CELULAR_positivo.groupby(by='_Gerada_TIPOHIST_CELULAR' , observed = True).size()
print('Gerada_TIPOHIST_CELULAR_positivo')
print(Gerada_TIPOHIST_CELULAR_positivo.sort_values(ascending=False).head(2))
Gerada_TIPOHIST_CELULAR_negativo = mapping_df[mapping_df['_Gerada_TIPOHIST_CELULAR_ENC'] < 0.3]
Gerada_TIPOHIST_CELULAR_negativo = Gerada_TIPOHIST_CELULAR_negativo.groupby(by='_Gerada_TIPOHIST_CELULAR' , observed = True).size()
print('Gerada_TIPOHIST_CELULAR_negativo')
print(Gerada_TIPOHIST_CELULAR_negativo.sort_values(ascending=False).head(6))


LOCTUDET_ENC_positivo = mapping_df[mapping_df['LOCTUDET_ENC'] > 0.6]
LOCTUDET_ENC_positivo = LOCTUDET_ENC_positivo.groupby(by='LOCTUDET' , observed = True).size()
print('LOCTUDET_ENC_positivo')
print(LOCTUDET_ENC_positivo.sort_values(ascending=False).head(1))
LOCTUDET_ENC_negativo = mapping_df[mapping_df['LOCTUDET_ENC'] < 0.3]
LOCTUDET_ENC_negativo = LOCTUDET_ENC_negativo.groupby(by='LOCTUDET' , observed = True).size()
print('LOCTUDET_ENC_negativo')
print(LOCTUDET_ENC_negativo.sort_values(ascending=False).head(3))


ESTADIAM_ENC_positivo = mapping_df[mapping_df['ESTADIAM_ENC'] > 0.65]
ESTADIAM_ENC_positivo = ESTADIAM_ENC_positivo.groupby(by='ESTADIAM' , observed = True).size()
print('ESTADIAM_ENC_positivo')
print(ESTADIAM_ENC_positivo.sort_values(ascending=False).head(1))
ESTADIAM_ENC_negativo = mapping_df[mapping_df['ESTADIAM_ENC'] < 0.32]
ESTADIAM_ENC_negativo = ESTADIAM_ENC_negativo.groupby(by='ESTADIAM' , observed = True).size()
print('ESTADIAM_ENC_negativo')
print(ESTADIAM_ENC_negativo.sort_values(ascending=False).head(2))

#%%
## Converter os valores SHAP em um DataFrame
shap_df = pd.DataFrame(shap_values_xgb, columns=X_amostra.columns)

## Calcular a média e o desvio padrão dos valores SHAP para cada variável
# shap_df['Abs'] = shap_df.abs() 
shap_summary = pd.DataFrame({
    'Média': shap_df.mean(),
    'Média Absoluta': shap_df.abs().mean(),
    'Desvio Padrão da Média Absoluto': shap_df.abs().std(),
})
shap_summary = shap_summary.sort_values(by='Média Absoluta' , ascending=False)
f.salvar_excel_conclusao(shap_summary , 'ValoresSHAP_XGB_completa') 





#%%

## Calcular a média dos valores absolutos dos SHAP para cada variável
shap_df = pd.DataFrame(shap_values_xgb, columns=X_amostra.columns)
mean_abs_shap = shap_df.abs().mean()

## Plotar o gráfico de barras
plt.figure(figsize=(8, 15))
mean_abs_shap_sorted = mean_abs_shap.sort_values(ascending=True)
bars = plt.barh(mean_abs_shap_sorted.index, mean_abs_shap_sorted.values)

## Adicionar os valores às barras
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.2f}', 
             va='center', ha='left')

## Adicionar rótulos e título
plt.xlabel('Média Absoluta dos Valores SHAP')
plt.title('Importância das Variáveis com Valores SHAP')

## Mostrar o gráfico
plt.show()

#%% Gráfico de cascata no nível indivíduo
# shap.waterfall_plot(shap.Explanation(values=shap_values_sem_resposta[0,0], 
#                                      base_values=explainer.expected_value, 
#                                      data=amostra.iloc[0], 
#                                      feature_names=amostra.columns))


#%% Fazendo o waterfall 'na mão'
# df_shap = pd.DataFrame(shap_values_sem_resposta, columns = X.columns)
# df_shap.iloc[0].plot.bar()


# #%% Forceplot
# # Inicializar a visualização
# # shap.initjs()

# # Explicação no nível de indivíduo (force plot para a primeira amostra de teste)
# force_plot = shap.force_plot(explainer.expected_value, 
#                 shap_values[0], 
#                 amostra.iloc[0], 
#                 feature_names=amostra.columns)
# plt.show()

# # Este não mostra no console, vamos salvar em arquivo
# shap.save_html("force_plot.html", force_plot)

#%%
