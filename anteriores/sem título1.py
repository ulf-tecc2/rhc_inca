# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 08:05:57 2025

@author: ulf
"""


import pandas as pd

import sys
sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")


from funcoes import salvar_excel_conclusao
from funcoes import Log


import funcoes_ulf as uf
import bib_graficos as ug

# # df_base_analise = f.leitura_arquivo_parquet('BaseSanitizada')
# df_sem_tempo = f.leitura_arquivo_parquet('BaseModelagem')
# df_com_tempo = f.leitura_arquivo_parquet('BaseModelagem_com_tempo_diagnostico')

# coluna_tempo = df_com_tempo['_Gerada_tempo_para_diagnostico']
# aux = df_com_tempo.drop(columns = ['_Gerada_tempo_para_diagnostico'])

# a = df_com_tempo['_Gerada_tempo_para_diagnostico'].value_counts(dropna=False, normalize=False)

df =  f.leitura_arquivo_parquet('BaseTransfor')


#%%

var = '_AnaliseESTADIAM'
a = df[var].value_counts(dropna=False, normalize=False)
a.name = 'ESTADIAM'

var = '_AnaliseLOCTUDET'
b = df[var].value_counts(dropna=False, normalize=False)
b.name = 'LOCTUDET'

var = '_AnaliseLOCTUPRI'
c = df[var].value_counts(dropna=False, normalize=False)
c.name = 'LOCTUPRI'

var = '_AnaliseLOCTUPRO'
d = df[var].value_counts(dropna=False, normalize=False)
d.name = 'LOCTUPRO'

var = '_AnaliseMUUH'
e = df[var].value_counts(dropna=False, normalize=False)
e.name = 'MUUH'

var = '_AnalisePRITRATH'
f = df[var].value_counts(dropna=False, normalize=False)
f.name = 'PRITRATH'

var = '_AnalisePROCEDEN'
g = df[var].value_counts(dropna=False, normalize=False)
g.name = 'PROCEDEN'

var = '_AnalisePTNM'
h = df[var].value_counts(dropna=False, normalize=False)
h.name = 'PTNM'

var = '_AnaliseTIPOHIST'
i = df[var].value_counts(dropna=False, normalize=False)
i.name = 'TIPOHIST'

var = '_AnaliseTNM'
j = df[var].value_counts(dropna=False, normalize=False)
j.name = 'TNM'


a_tab = pd.concat([a , b,c,d,e,f,g,h,i,j] , axis=1)
a_tab.fillna(0, inplace=True)
a_tab.astype(int)

a_nome_arquivo = "TabResultadoRegrasCod"
salvar_excel_conclusao(a_tab, a_nome_arquivo)

#%%

aux_df = df.loc[df['_AnaliseTNM'] == 'nao se aplica - Geral' ]
x = df['_Gerada_TIPOHIST_CELULAR'].value_counts(dropna=False, normalize=False)
x = aux_df.groupby('TNM' , observed=True).size()

a = df.columns
#%%
import pandas as pd
df_a = aux
df_b =df_sem_tempo
mask = ~df_a.isin(df_b.to_dict(orient='list')).all(axis=1)
result = df_a[mask]

result1 = df_com_tempo[mask]


print(result)


3. Comparando apenas certas colunas
Se você precisa comparar apenas colunas específicas de ambos os DataFrames:
# Especificar as colunas para comparação
mask = ~df_a[['col1', 'col2']].isin(df_b[['col1', 'col2']].to_dict(orient='list')).all(axis=1)
result = df_a[mask]
result.isnull().sum()
print(result)



df = df_base_transf
# df_base_model =  f.leitura_arquivo_parquet('BaseModelagem')

a_dict = f.get_dict_names_colunas(df)

df_aux = df.drop(columns = a_dict['inconsistente'])
df_aux = df_aux.drop(columns = a_dict['analise'])

from statsmodels.stats.outliers_influence import variance_inflation_factor


vif_data = pd.DataFrame({
    "Variável": df_aux.columns,
    "VIF": [variance_inflation_factor(df_aux.values, i) for i in range(df_aux.shape[1])]
})
print(vif_data)

#%%

df_encoded.info()

a = df.columns
df.isnull().sum()
df.info()
df = df_unico

a_var = 'ESTADRES'
a = df_com_tempo[a_var].value_counts(dropna=False, normalize=False)
b = df_encoded['_Gerada_ESTDFIMT'].value_counts(dropna=False, normalize=False)
c = df['_Gerada_DIAGANT_TRAT'].value_counts(dropna=False, normalize=False)



ug.plot_frequencias_valores_atributos(df , ['DTPRICON'] , title = 'Registros por UF')

com = df.loc[df['_Gerada_RESFINAL'] == 'com resposta'].shape[0] / df.shape[0]
sem = df.loc[df['_Gerada_RESFINAL'] == 'sem resposta'].shape[0] / df.shape[0]

print(f'Distribuição dos registros pelo resultado (_Gerada_RESFINAL) com resposta:{com} sem resposta:{sem}')

#%%

colunas_higt_card = ['_Gerada_TIPOHIST_CELULAR' , 'LOCTUDET' , 'ESTADIAM']

a = pd.get_dummies(df, columns=colunas_higt_card, dtype=int, drop_first=True)
a.shape[1]

a_dict = {}
lista_colunas = colunas_higt_card
for c in lista_colunas:
    a_dict[c] = list(df[c].unique())
