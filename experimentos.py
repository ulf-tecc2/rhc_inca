# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:50:51 2024

@author: ulf
"""

df_unico = leitura_arquivo_csv("analitico")
df_unico.info()

df_unico = definir_tipos_variaveis(df_unico)

df_unico = trocar_valores_nulos(df_unico)


df_analitico.groupby("DTINITRT")["DTINITRT"].count()

ulfpp.print_count_cat_var_values(df_unico, ["LOCTUDET"])
ulfpp.print_count_cat_var_values(df_unico, ["LOCTUPRI"])
ulfpp.print_count_cat_var_values(df_unico, ["LOCTUPRO"])

print(df_unico["LOCTUPRI"].isnull().sum())
LOCTUPRO
