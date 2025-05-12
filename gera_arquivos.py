# -*- coding: utf-8 -*-
"""Registros Hospitalares de Câncer (RHC) - Analise dos modelos construidos.

MBA em Data Science e Analytics - USP/Esalq - 2025

@author: Ulf Bergmann

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


#%% Carga dos dados
a_list = ['BaseIndicadores' ]
for a_name in a_list:
    df = f.leitura_arquivo_parquet(a_name)
    
    # a_new_name = "dados/consolidado/" + a_name
    f.salvar_csv(df, a_name)
    
    print('Arquivo gerado: ' + a_name)


