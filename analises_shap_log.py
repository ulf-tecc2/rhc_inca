# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Analise da influencia das variaveis na predicao.

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

#%% Funcoes auxiliares

class CustomModel:
    intercept_ = None
    coef_ = None
    
    coef_names_ = None
    
    def __init__(self, model):
        self.model = model
        
        self.intercept_ = np.array([model.params['Intercept']])
        
        coef_series = model.params.drop(labels = ['Intercept'])
        a = np.array(coef_series)
        self.coef_ = a.reshape(1, len(a))
        
        self.coef_names_ = coef_series.index.values
        
    def predict(self, X):
        ## Aqui, chama o método predict do modelo interno
        return self.model.predict(X)




#%% Carga dos dados


log = Log()
log.carregar_log("log_BaseModelagemFinal")
df = f.leitura_arquivo_parquet("BaseModelagemFinal")

print(log.logar_acao_realizada("Carga Dados", "Carregamento dos dados para modelagem", df.shape[0]))

var_dep = "_Gerada_RESFINAL"
#%%

mod_log = f.load_model('LogisticoBinario_Escolhido' + '_sem_pca')

   
#%% Calcular e visualizar os 'shap values'

porcentagem_amostra_shap_values = 1
df_amostra = df.groupby(var_dep, group_keys=False).apply(lambda x: x.sample(frac=porcentagem_amostra_shap_values , random_state=1))
X_amostra = df_amostra.drop(columns=[var_dep])
y_amostra = df_amostra[var_dep]


#%%% Logistico
import numpy as np

mod_log_shap = CustomModel(mod_log)
a = np.append( mod_log_shap.coef_names_ , var_dep)

df_amostra_log = df_amostra[a]
X_amostra_log = df_amostra_log.drop(columns=[var_dep])
# var_eliminar_log = [ '_Gerada_TIPOHIST_BIOLOGICO_2' ,   '_Gerada_TIPOHIST_BIOLOGICO_3' ,  '_Gerada_BASMAIMP_TUMPRIM' ]
# X_amostra_log = df_amostra_log.drop(columns=var_eliminar_log)

y_amostra_log = df_amostra_log[var_dep]

a_file_name_log = f"dados/workdir/shap_log{int(porcentagem_amostra_shap_values*100)}.pkl"
explainer_log = shap.LinearExplainer(mod_log_shap , X_amostra_log)


# =============================================================================
#  GERAR VALORES E SALVAR
# =============================================================================
shap_values_log = explainer_log.shap_values(X_amostra_log)  

print('Termino de gerar shap_values para Log')
uf.alerta_sonoro()

with open(a_file_name_log, 'wb') as file:  
    pickle.dump(shap_values_log, file)    

# =============================================================================
# CARREGAR VALORES SALVOS
# =============================================================================

# with open(a_file_name_log, 'rb') as file:  
#     shap_values_log = pickle.load(file)

# =============================================================================
# ANALISAR VALORES
# =============================================================================
shap.summary_plot(shap_values_log, X_amostra_log, feature_names=X_amostra_log.columns , max_display=36)
shap.summary_plot(shap_values_log, X_amostra_log, feature_names=X_amostra_log.columns , plot_type="bar" , max_display=36)

## Converter os valores SHAP em um DataFrame
shap_df = pd.DataFrame(shap_values_log, columns=X_amostra_log.columns)

## Calcular a média e o desvio padrão dos valores SHAP para cada variável
# shap_df['Abs'] = shap_df.abs() 
shap_summary = pd.DataFrame({
    'Média': shap_df.mean(),
    'Média Absoluta': shap_df.abs().mean(),
    'Desvio Padrão da Média Absoluto': shap_df.abs().std(),
})
shap_summary = shap_summary.sort_values(by='Média Absoluta' , ascending=False)
f.salvar_excel_conclusao(shap_summary , 'ValoresSHAP_Logistico')  
#%%

## Calcular a média dos valores absolutos dos SHAP para cada variável
shap_df = pd.DataFrame(shap_values_log, columns=X_amostra_log.columns)
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
