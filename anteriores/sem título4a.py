# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 13:55:59 2025

@author: ulf
"""




import pandas as pd
import numpy as np

import time
import os
import sys

import seaborn as sns

import matplotlib.pyplot as plt

sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

import funcoes as f
from funcoes import Log

import funcoes_ulf as uf
import bib_graficos as ug

import statsmodels.api as sm # estimação de modelos
import statsmodels.formula.api as smf # estimação do modelo logístico binário
from statstests.process import stepwise # procedimento Stepwise
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos

import statsmodels.formula.api as smf # estimação do modelo logístico binário

                                                        
import category_encoders as ce 
                                                        
from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score , precision_score , f1_score,  roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

df = f.leitura_arquivo_parquet("BaseModelagemFinal")
var_dep = "_Gerada_RESFINAL"
var_eliminar = [ '_Gerada_TIPOHIST_BIOLOGICO_2' ,   '_Gerada_TIPOHIST_BIOLOGICO_3' ,  '_Gerada_BASMAIMP_TUMPRIM' ]
df = df.drop(columns=var_eliminar)

modelo_step = f.load_model('LogisticoBinario_Escolhido')

df['phat_step'] = modelo_step.predict()


lista_medidas = ['Sensitividade', 'Precisao' , 'F1_Score']
tabela = ug.plota_analise_indicadores_predicao_classificacao('step' , modelo_step , df[var_dep] , df['phat_step'] , lista_medidas = lista_medidas)

coef_series = modelo_step.params.drop(labels = ['Intercept'])
    
a = modelo_step.summary()

model = modelo_step
summary_df = pd.DataFrame({
    'Coeficiente': model.params,
    'Erro Padrão': model.bse,
    'Estatística z': model.tvalues,
    'p-valor': model.pvalues,
    # 'Intervalo Inferior': model.conf_int()[:, 0],
    # 'Intervalo Superior': model.conf_int()[:, 1]
})

