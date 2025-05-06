# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 07:10:33 2024

@author: ulf
"""

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
        
from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score , precision_score , f1_score,  roc_auc_score, roc_curve, auc
from sklearn.feature_selection import chi2   

# cmap=plt.cm.viridis_r
# Paletas de cores ('_r' reverte a sequência de cores):
# viridis
# inferno
# magma
# cividis
# coolwarm
# Blues
# Greens
# Reds



# import sys
# sys.path.append("C:/Users/ulf/OneDrive/Python/ia_ml/templates/lib")

# import funcoes_ulf as ulfpp
# import bib_graficos as ug


def plot_correlation_heatmap(df, lista_variaveis ):
    """
    Plot the correlation betwenn pairs of continuos variables.

    Parameters:
        df (DataFrame): DataFrame to be analysed

        lista_variaveis (list): continuos variable list 

    Returns:
        (None):

    """
    cv_df = df[lista_variaveis]

    # metodos: 'pearson', 'kendall', 'spearman' correlations.
    corr_matrix = cv_df.corr(method='pearson')

    fig = plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, annot_kws={'size': 15} , cmap="Blues")
    plt.title("Correlation Heatmap")
    #fig.tight_layout()

    fig.show()

def plot_boxplot_for_variables(df , variables_list):
    """Plot a boxplot for all variables in variables_list.
    
    Can be used to verify if the variables are in the same scale

    Examples:
        >>> plot_boxplot_for_variables(df, ['va1' , 'var2' , 'var3'])
        return None

    Parameters:
        df (DataFrame): DataFrame to be analysed.
        variables_list (list): variable list.

    Returns:
        (None):

    """   
    df_filtered = df[variables_list]

    plt.figure(figsize=(10,7))
    sns.boxplot(x='variable', y='value', data=pd.melt(df_filtered))
    plt.ylabel('Values', fontsize=16)
    plt.xlabel('Variables', fontsize=16)
    plt.show()
    
    return

def plot_null_values(df: pd.DataFrame):
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100      
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %' :na_df})
        ax = missing_data.plot.barh()
        ax.bar_label(ax.containers[0] , labels=missing_data['Missing Ratio %'].astype('int') , fontsize=10)    
        plt.show()
    else:
        print('No NAs found')

def plot_frequencias_valores_atributos(df, lista_atributos , bins = 10 , title = 'Histograma'):
    """
    Plot the frequency graphic for the attribute values for each variable in lista_atributos.

    Parameters:
        df (DataFrame): DataFrame to be analysed

        lista_atributos (list): variable list 

    Returns:
        (None):

    """
    
    for i , j in enumerate(lista_atributos):
        plt.figure(figsize=(10,7))
        sns.histplot(data=df, x=lista_atributos[i] , bins=bins)
        
        ax = plt.gca()

        ## Definir título e personalizar a fonte
        ax.set_xlabel(f'Variavel {lista_atributos[i]}', fontsize=11, fontfamily='arial')  ## Tamanho e tipo de fonte
        ax.set_ylabel('Quantidade', fontsize=11, fontfamily='arial')  ## Tamanho e tipo de fonte

        ax.spines['bottom'].set_linewidth(1.5)  ## Eixo X
        ax.spines['left'].set_linewidth(1.5)    ## Eixo Y
        ax.spines['top'].set_linewidth(0)  ## Eixo X
        ax.spines['right'].set_linewidth(0)    ## Eixo Y
        
        # plt.title(title)
        plt.show()



def analyse_plot_correlation_categorical_variables(df, lista_variaveis):
    """
    Analyse and plot the correlation betwenn pairs of categorical variables. Variables must be not continuos (not float).

    Use the qui-quadrad and p-value for:
        H0: dependent variables
        H1: independent variables

    Parameters:
        df (DataFrame): DataFrame to be analysed

        lista_variaveis (list): variable list 

    Returns:
        resultant (DataFrame): Dataframe with all p-values
        
        lista_resultado_analise (DataFrame):  with Variable1 | Variable 2 | p-value
    """
    resultant = pd.DataFrame(data=[(0 for i in range(len(lista_variaveis))) for i in range(len(lista_variaveis))],
                             columns=list(lista_variaveis), dtype=float)
    resultant.set_index(pd.Index(list(lista_variaveis)), inplace=True)

    # Encontrando os p-valores para as variáveis e formatando em matriz de p-valor
    lista_resultado_analise = []
    for i in list(lista_variaveis):
        for j in list(lista_variaveis):
            if i != j:
                try:
                    chi2_val, p_val = chi2(
                        np.array(df[i]).reshape(-1, 1), np.array(df[j]).reshape(-1, 1))
                    p_val = round(p_val[0], 4)
                    resultant.loc[i, j] = p_val
                    lista_resultado_analise.append([i, j,  p_val])
                except ValueError:
                    print(f"Variavel {j} não é categórica ")
                    return

    fig = plt.figure(figsize=(25, 20))
    sns.heatmap(resultant, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Resultados do teste Qui-quadrado (p-valor)')
    plt.show()
    df_lista_resultado_analise =  pd.DataFrame(lista_resultado_analise , columns=['Var 1' , 'Var 2' , 'p-value'])
    return resultant, df_lista_resultado_analise



def plot_ajuste_entre_variaveis(df , var1 , var2 , log=True , fit_reg = True):
    """ Plota o ajuste logístico/linear entre as variáveis.
    
    Examples:
        >>> plot_ajuste_entre_variaveis(df , 'variavel x' , 'variavel y' , log=True)
        return None

    Parameters:
        df (DataFrame): DataFrame to be analysed.
        var1 (string): variable name for X.
        var2 (string): variable name for Y.
        log (boolean): True for logistic / False for linear.
        fit_reg (boolean): True estimate and plot a regression model relating the x and y
        
    Returns:
        (None):

    """ 
    plt.figure(figsize=(15,10))
    sns.regplot(x=df[var1], y=df[var2],
                ci=None, marker='o', logistic=log, fit_reg=fit_reg,
                scatter_kws={'color':'orange', 's':250, 'alpha':0.7},
                line_kws={'color':'darkorchid', 'linewidth':7})
   
    plt.xlabel('X: ' + var1, fontsize=20)
    plt.ylabel('Y: ' + var2, fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Gráfico de Dispersão', fontsize=22)
    plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24, loc='upper left')
    plt.show
    
    
    

    
     
    
def calcula_indicadores_predicao_classificacao(label , model , observado , predicts, cutoff):
    
    predicao_binaria = []   
    if cutoff == None:
        predicao_binaria = predicts
    else:
        values = predicts.values 
        for item in values:
            if item < cutoff:
                predicao_binaria.append(0)
            else:
                predicao_binaria.append(1)    
    
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)
    precisao = precision_score(observado, predicao_binaria)
    f1_scor = f1_score(observado, predicao_binaria)
    auc = roc_auc_score(observado , predicao_binaria)
    
    if hasattr(model, 'llf'):
        llf = model.llf
    else:
        llf = None
        
    if hasattr(model, 'pearson_chi2'):
        chi2 = model.pearson_chi2
    else:
        chi2 = None

    if hasattr(model, 'pseudo_rsquared'):
        pr2 = model.pseudo_rsquared()
    else:
        pr2 = None
        
    # Visualização dos principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Label' : [label],
                                'Cutoff' : [cutoff],
                                'tamanho' : [observado.shape[0]],
                                'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acuracia':[acuracia],
                                'Precisao':[precisao],
                                'F1_Score':[f1_scor],
                                'AUC' : [auc],
                                'LLF' : [llf],
                                'Pearson_chi2':[chi2],
                                'Pseudo R2':[pr2],
                                })
    return indicadores

def plot_matriz_confusao(observado , predicts , cutoff):
    """Plot a confusion matrix.

    Parameters:
        observado (Series): real value.
        predicts (Series): predicted values.
        cutoff (float): if none the predicts are the class

    Returns:
        (None):

    """  
    predicao_binaria = []   
    if cutoff == None:
        predicao_binaria = predicts
    else:
        values = predicts.values 
        for item in values:
            if item < cutoff:
                predicao_binaria.append(0)
            else:
                predicao_binaria.append(1)
           
    cm = confusion_matrix(observado , predicao_binaria)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('Predição')
    plt.ylabel('Real')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
    
    
        
def plota_analise_indicadores_predicao_classificacao(label , model , observado , predicts , intervalo=0.01 , lista_medidas = None):
    
    if lista_medidas == None:
        lista_medidas = ['Sensitividade', 'Especificidade' , 'Acuracia' , 'Precisao' , 'F1_Score']
    
    cutoffs = np.arange(0, 1.00 , 0.01)
    # cutoffs = np.arange(0, 1.00 , 0.2)
    dados = pd.DataFrame()
    for cutoff in cutoffs:
        aux = calcula_indicadores_predicao_classificacao(label , model , observado , predicts, cutoff)
        dados = pd.concat([dados , aux])

    # plt.figure(figsize=(15,10))
    fig, ax = plt.subplots(figsize=(15,10))
    with plt.style.context('seaborn-v0_8-whitegrid'):
        if 'Sensitividade' in lista_medidas:
            plt.plot(dados.Cutoff,dados.Sensitividade, marker='o', color='indigo', markersize=6)
        if 'Especificidade' in lista_medidas:
            plt.plot(dados.Cutoff,dados.Especificidade, marker='o',color='limegreen', markersize=6)
        if 'Acuracia' in lista_medidas:
            plt.plot(dados.Cutoff,dados.Acuracia, marker='o', color='red', markersize=6)
        if 'Precisao' in lista_medidas:
            plt.plot(dados.Cutoff,dados.Precisao, marker='o', color='cyan', markersize=6)
        if 'F1_Score' in lista_medidas:
            plt.plot(dados.Cutoff,dados.F1_Score, marker='o', color='magenta', markersize=6)

    ax.set_xlabel('Cutoff', fontsize=16, fontfamily='arial')  ## Tamanho e tipo de fonte
    ax.set_ylabel('Indicadores', fontsize=16, fontfamily='arial') 
    ax.spines['bottom'].set_linewidth(1.5)  ## Eixo X
    ax.spines['left'].set_linewidth(1.5)    ## Eixo Y
    ax.spines['top'].set_linewidth(0)  ## Eixo X
    ax.spines['right'].set_linewidth(0)    ## Eixo Y

    # plt.xlabel('Cutoff', fontsize=20)
    # plt.ylabel('Indicadores', fontsize=20)
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=16 , fontfamily='arial')
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=16 , fontfamily='arial')
    plt.legend(lista_medidas, fontsize=16)
    plt.show()
    
    return dados

import plotly.graph_objects as go # gráficos 3D

def plot_dispersao_3d(df , x,y,z):
    # Gráfico 3D com scatter gerado em HTML e aberto no browser
    # (figura 'EXEMPLO2_scatter3D.html' salva na pasta do curso)

    trace = go.Scatter3d(
        x=df[x], 
        y=df[y], 
        z=df[z], 
        mode='markers',
        marker={
            'size': 10,
            'color': 'darkorchid',
            'opacity': 0.7,
        },
    )

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        width=800,
        height=800,
        plot_bgcolor='white',
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(200, 200, 200)',
                backgroundcolor='whitesmoke'
            ),
            yaxis=dict(
                gridcolor='rgb(200, 200, 200)',
                backgroundcolor='whitesmoke'
            ),
            zaxis=dict(
                gridcolor='rgb(200, 200, 200)',
                backgroundcolor='whitesmoke'
            )
        )
    )

    data = [trace]

    plot_figure = go.Figure(data=data, layout=layout)
    plot_figure.update_layout(scene=dict(
        xaxis_title=x,
        yaxis_title=y,
        zaxis_title=z
    ))

    plot_figure.write_html('EXEMPLO_3D.html')

    # Abre o arquivo HTML no browser
    import webbrowser
    webbrowser.open('EXEMPLO_3D.html')
    
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
    
def plota_grafo_correlacao(correlation_matrix):
    
    # Diagrama interessante (grafo) que mostra a inter-relação entre as
    # variáveis e a magnitude das correlações entre elas

    plt.figure(figsize=(15,10))

    # Criação de um grafo direcionado
    G = nx.DiGraph()

    # Adição das variáveis como nós do grafo
    for variable in correlation_matrix.columns:
        G.add_node(variable)

    # Adição das arestas com espessuras proporcionais às correlações
    for i, variable1 in enumerate(correlation_matrix.columns):
        for j, variable2 in enumerate(correlation_matrix.columns):
            if i != j:
                correlation = correlation_matrix.iloc[i, j]
                if abs(correlation) > 0:
                    G.add_edge(variable1, variable2, weight=correlation)

    # Obtenção da lista de correlações das arestas
    correlations = [d["weight"] for _, _, d in G.edges(data=True)]

    # Definição da dimensão dos nós
    node_size = 2700

    # Definição da cor dos nós
    node_color = 'black'

    # Definição da escala de cores das retas (correspondência com as correlações)
    cmap = plt.colormaps.get_cmap('coolwarm_r')

    # Criação de uma lista de espessuras das arestas proporcional às correlações
    edge_widths = [abs(d["weight"]) * 10 for _, _, d in G.edges(data=True)]
    

    # Criação do layout do grafo com maior distância entre os nós
    pos = nx.spring_layout(G, k=0.75)  # k para controlar a distância entre os nós

    # Desenho dos nós e das arestas com base nas correlações e espessuras
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=correlations,
                           edge_cmap=cmap, alpha=0.7)

    # Adição dos rótulos dos nós
    labels = {node: node for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='white')

    # Ajuste dos limites dos eixos
    ax = plt.gca()
    ax.margins(0.1)
    plt.axis("off")

    # Criação da legenda com a escala de cores definida
    smp = cm.ScalarMappable(cmap=cmap)
    smp.set_array([min(correlations), max(correlations)])
    cbar = plt.colorbar(smp, ax=ax, label='Correlação')

    # Exibição do gráfico
    plt.show()
    
    
from scipy import stats
from scipy.stats import norm

def plota_histograma_distribuicao(df , var):
    plt.figure(figsize=(15,10))
    hist1 = sns.histplot(data=df[var], kde=True,  stat="density",
                         color = 'darkorange', alpha=0.4, edgecolor='silver',
                         line_kws={'linewidth': 3})
    hist1.get_lines()[0].set_color('orangered')
    plt.xlabel(var, fontsize=20)
    plt.ylabel('Frequência', fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    
    # Calcula os valores de ajuste da distribuição normal
    (mu, sigma) = norm.fit(df[var])
    x = np.linspace(df[var].min(), df[var].max(), 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=2)
    
    plt.legend(['KDE', 'Dist Normal' , 'Dist Frequencia - ' + var ], fontsize=20)
    plt.show()


def plot_curvas_roc(df , true_var_name , predict_var_names):
    """
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    true_var_name: nome da variavel com o valor 
    predict_var_names : lista com os nomes das variaveis preditas
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(10,6))
    for var_name in predict_var_names:
        fpr, tpr, thresholds =roc_curve(df[true_var_name], df[var_name])
        roc_auc = auc(fpr, tpr)
        roc_gini = (roc_auc - 0.5)/(0.5)
        plt.plot(fpr, tpr, marker='o' , markersize=2, linewidth=2,
                 label=f'{var_name} AUC: {roc_auc:.4f} | GINI: {roc_gini:.4f}')

    plt.plot(fpr, fpr, color='gray', linestyle='dashed')
    
    plt.title('', fontsize=22)
    ax.set_xlabel('Especificidade', fontsize=11, fontfamily='arial')  ## Tamanho e tipo de fonte
    ax.set_ylabel('Sensitividade', fontsize=11, fontfamily='arial') 
       
    ax.spines['bottom'].set_linewidth(1.5)  ## Eixo X
    ax.spines['left'].set_linewidth(1.5)    ## Eixo Y
    ax.spines['top'].set_linewidth(0)  ## Eixo X
    ax.spines['right'].set_linewidth(0)    ## Eixo Y

    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=11)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=11)
    plt.legend(fontsize = 11)
    plt.show()
          
def plot_curvas_roc_1(modelo, X_train, y_train, X_test, y_test):
    p_train = modelo.predict_proba(X_train)[:, 1]
    # c_train = modelo.predict(X_train)
    
    p_test = modelo.predict_proba(X_test)[:, 1]
    # c_test = modelo.predict(X_test)

    auc_train = roc_auc_score(y_train, p_train)
    auc_test = roc_auc_score(y_test, p_test)
    
    print(f'Avaliação base de treino: AUC = {auc_train:.2f}')
    print(f'Avaliação base de teste: AUC = {auc_test:.2f}')
    
    fpr_train, tpr_train, _ = roc_curve(y_train, p_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, p_test)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_train, tpr_train, color='red', label=f'Treino AUC = {auc_train:.2f}')
    plt.plot(fpr_test, tpr_test, color='blue', label=f'Teste AUC = {auc_test:.2f}')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('Falso Positivo')
    plt.ylabel('Verdadeiro Positivo')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()
    
def plota_predicao_3d(df , var_x , var_y , var_z1 , var_z2, var_z3):
    # Visualização das sigmoides tridimensionais em um único gráfico

    pio.renderers.default = 'browser'

    trace = go.Mesh3d(
        x=df[var_x], 
        y=df[var_y],
        z=df[var_z1],
        opacity=1,
        color='indigo')

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        width=800,
        height=800
    )

    data = [trace]

    plot_figure = go.Figure(data=data, layout=layout)

    trace_1 = go.Mesh3d(
                x=df[var_x], 
                y=df[var_y],
                z=df[var_z2],
                opacity=1,
                color='darkgreen')

    plot_figure.add_trace(trace_1)

    trace_2 = go.Mesh3d(
                x=df[var_x], 
                y=df[var_y],
                z=df[var_z3],
                opacity=1,
                color='darkorange')


    plot_figure.add_trace(trace_2)

    plot_figure.update_layout(
        template='plotly_dark',
        scene = dict(
            xaxis_title=var_x,
            yaxis_title=var_y,
            zaxis_title='valores')
        )

    plot_figure.show()
    
    
# plota_comparacao_predicoes_modelos(df , 'staff' , 'violations' , ['fitted_poisson' , 'fitted_bneg'] , 'Number of Diplomats (staff)'  , 'Unpaid Parking Violations (violations)' )

def plota_comparacao_predicoes_modelos(df , x_var_name , y_var_name , list_fitted_values_names , x_label , y_label):
    plt.figure(figsize=(15,10))
    with plt.style.context('seaborn-v0_8-whitegrid'):
        sns.scatterplot(x=x_var_name, y=y_var_name, data=df, color='darkgrey',
                    s=200, label='Valores Reais', alpha=0.8)
        
        for a_name in list_fitted_values_names:
            sns.regplot(data=df, x=x_var_name, y= a_name, order=3, ci=False,
                    scatter=False, 
                    label=a_name,
                    line_kws={'linewidth': 4})
        

    plt.xlabel(x_label , fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper center', fontsize=17)
    plt.show

def plota_dispersao_com_ajustes(df , x , y):
   # Gráfico de dispersão com ajustes (fits) linear e não linear
    # com argumento 'lowess=True' (locally weighted scatterplot smoothing)

    plt.figure(figsize=(15,10))
    sns.scatterplot(x=x, y=y, data=df, color='grey',
                    s=300, label='Valores Reais', alpha=0.7)
    sns.regplot(x=x, y=y, data=df, lowess=True,
                color='darkviolet', ci=False, scatter=False, label='Ajuste Não Linear',
                line_kws={'linewidth': 2.5})
    sns.regplot(x=x, y=y, data=df,
                color='darkorange', ci=False, scatter=False, label='OLS Linear',
                line_kws={'linewidth': 2.5})
    plt.title('Dispersão dos dados e ajustes linear e não linear', fontsize=20)
    plt.xlabel(x, fontsize=17)
    plt.ylabel(y, fontsize=17)
    plt.legend(loc='lower right', fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

def plota_tabela(df):

    from tabulate import tabulate
    tabela = tabulate(df, headers='keys', tablefmt='grid', numalign='center')
    
    plt.figure(figsize=(8, 3))
    plt.text(0.1, 0.1, tabela, {'family': 'monospace', 'size': 15})
    plt.axis('off')
    plt.show()
    
def plota_mapa_mundi(df):
    import pandas as pd
    import plotly.express as px
    import webbrowser

    # Mapa-múndi
    fig = px.choropleth(df, 
                        locations='code',  # Código ISO dos países
                        color='corruption',
                        hover_name='country',  # Informações que aparecerão ao passar o mouse
                        color_continuous_scale=px.colors.sequential.RdBu_r,  # Escala de cores
                        projection="natural earth",  # Projeção do mapa
                        title="Mapa de Corrupção por País")

    fig.write_html("mapa_corrupcao.html")

    webbrowser.open("mapa_corrupcao.html")


#Ajustes dos modelos: valores previstos (fitted values) X valores reais
def plota_dispersao_previsao(df , real_name , fitted_name):
    # Valores preditos pelo modelo para as observações da amostra
    # dados['fitted_bc'] = modelo_stepwise_bc.predict()
    
    sns.regplot(df, x=real_name , y=fitted_name, color='blue', ci=False, line_kws={'color': 'red'})
    plt.title('Analisando o Ajuste das Previsões', fontsize=10)
    plt.xlabel(f'Variavel Original ({real_name})', fontsize=10)
    plt.ylabel(f'Variavel Prevista ({fitted_name})', fontsize=10)
    plt.axline((5.95, 5.95), (max(df[real_name]), max(df[real_name])), linewidth=1, color='grey')
    plt.show()
    
# plota_dispersao_previsao(df, 'preco', 'fitted_bc')


def plota_variacoes_pos_neg(df , var_x , var_y):
    colors = ['springgreen' if x>0 else 'red' for x in df[var_x]]

    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(0, point['y'], str(round(point['x'],4)), fontsize=17,
                    verticalalignment='center')

    plt.figure(figsize=(15,10))
    plt.barh(df[var_y], df[var_x], color=colors)

    label_point(x = df[var_x],
                y = df[var_y],
                val = df[var_x],
                ax = plt.gca()) 
    plt.ylabel('Escola', fontsize=20)
    plt.xlabel(r'$\nu_{0j}$', fontsize=20)
    plt.tick_params(axis='x', labelsize=17)
    plt.tick_params(axis='y', labelsize=17)
    plt.yticks(np.arange(0, 11, 1))
    plt.show()

# plota_variacoes_pos_neg(efeitos_aleatorios , 'v0j' , 'escola')
    
def descritiva(df_, var, vresp, max_classes=5):
    descritiva_categorica(df_, var, vresp , max_classes)


def descritiva_categorica(df_, var, vresp, max_classes=10):
    """
    Gera um gráfico descritivo  por categoria da variável especificada em relacao a variavel resposta.
    
    Parâmetros:
    df : DataFrame - Base de dados a ser analisada.
    var : str - Nome da variável categórica a ser analisada.
    vresp: variavel resposta
    """
    
    df = df_.copy()
    
    # if df[var].nunique()>max_classes:
    #     df[var] = pd.qcut(df[var], max_classes, duplicates='drop')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.pointplot(data=df, y=vresp, x=var, ax=ax1)
    
    # Criar o segundo eixo y para a taxa de sobreviventes
    ax2 = ax1.twinx()
    sns.countplot(data=df, x=var, palette='viridis', alpha=0.5, ax=ax2)
    ax2.set_ylabel('Frequência', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)  # Tornar o fundo do eixo 1 transparente
    
    # Exibir o gráfico
    plt.show()
    
def descritiva_metrica(df, var, vresp):
    """
    Gera um gráfico boxplot mostrando a distribuicao da variavel metrica em relacao a variavel resposta.
    
    Parâmetros:
    df : DataFrame - Base de dados a ser analisada.
    var : str - Nome da variável metrica a ser analisada.
    vresp: variavel resposta
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=df, x=vresp , y=var, ax=ax)
    ax.set_xlabel("Variavel Resposta - " + vresp)
    ax.set_ylabel("Variavel " + var)
    ax.set_title("Analise da Influencia da Variavel")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    
# descritiva_metrica(df , var = df.columns[4] ,vresp='label')

def diagnóstico(df_, var, vresp='survived', pred = 'pred', max_classes=5):
    """
    Gera um gráfico descritivo da taxa de sobreviventes por categoria da variável especificada.
    
    Parâmetros:
    df : DataFrame - Base de dados a ser analisada.
    var : str - Nome da variável categórica a ser analisada.
    """
    
    df = df_.copy()
    
    if df[var].nunique()>max_classes:
        df[var] = pd.qcut(df[var], max_classes, duplicates='drop')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.pointplot(data=df, y=vresp, x=var, ax=ax1)
    sns.pointplot(data=df, y=pred, x=var, ax=ax1, color='red', linestyles='--', ci=None)
    
    # Criar o segundo eixo y para a taxa de sobreviventes
    ax2 = ax1.twinx()
    sns.countplot(data=df, x=var, palette='viridis', alpha=0.5, ax=ax2)
    ax2.set_ylabel('Frequência', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)  # Tornar o fundo do eixo 1 transparente
    
    # Exibir o gráfico
    plt.show()

#%% Execução dos exemplos

# df_aux = pd.read_csv('templates/dados/base_exemplos_graficos.csv',delimiter=',')

# plot_dispersao_3d(df_aux , 'cpi' , 'idade' , 'horas')

# plota_histograma_distribuicao(df_aux , 'horas')

# df_aux1 = df_aux.iloc[:,12:15]
# df_aux1['x1'] = df_aux['x1']

# correlation_matrix = df_aux1.corr()
# plota_grafo_correlacao(correlation_matrix)

# plot_boxplot_for_variables(df_aux , ['idade' , 'horas'])
# plot_frequencias_valores_atributos(df_aux , ['x1' , 'x2'])
# plot_correlation_heatmap(df_aux , ['x1' , 'x2' , 'predicao'])

# plot_matriz_confusao(df_aux['true'], df_aux['predicao'], cutoff=0.5)
# indicadores = plota_analise_indicadores_predicao_logistica(df_aux['true'], df_aux['predicao'])
# print(indicadores)

# plot_ajuste_entre_variaveis(df_aux , 'x2' , 'quali_1' , log=True)

# plot_curvas_roc(df_aux , 'true' , 'predicao')
# plot_curvas_roc(df_aux , 'fidelidade' , ['phat' , 'phat2'])

#%%
# Esta dando erro nesta chamada, variavel não categorica
 # analyse_plot_correlation_categorical_variables(df_aux , ['quali_2' , 'quali_5' , 'quali_3'])
 
# TENTATIVA DE ARRUMAR
# df_aux.columns
# df_aux['quali_1'] = df_aux['quali_1'].astype("object")
# df_aux['quali_2'] = df_aux['quali_2'].astype("object")
# df_aux['quali_3'] = df_aux['quali_3'].astype("object")
# df_aux['quali_4'] = df_aux['quali_4'].astype("object")
# df_aux['quali_5'] = df_aux['quali_5'].astype("object")
# df_aux['quali_6'] = df_aux['quali_6'].astype("object")

# df_aux['quali_1'] = df_aux['quali_1'].astype("category")
# df_aux['quali_2'] = df_aux['quali_2'].astype("category")
# df_aux['quali_3'] = df_aux['quali_3'].astype("category")
# df_aux['quali_4'] = df_aux['quali_4'].astype("category")
# df_aux['quali_5'] = df_aux['quali_5'].astype("category")
# df_aux['quali_6'] = df_aux['quali_6'].astype("category")

# df_aux.dtypes




