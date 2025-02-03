# Registros Hospitalares de Câncer (RHC) - INCA
Tratamento dos Registros Hospitalares de Câncer (RHC) - INCA, do período de 1985 até 2022 

# Autores
Ulf Bergmann - Aluno do MBA USP/Esalq - ulf@tecc2.com.br

Patrícia Belfiore Fávero - Orientadora - patricia.belfavero@gmail.com 

Dr Anke Bergmann - Pesquisadora do INCA - abergmann@inca.gov.br

Dr Luiz Claudio Santos Thuler - Pesquisador do INCA - lthuler@inca.gov.br.


## Contexto
Este projeto foi desenvolvido no contexto do MBA em Data Science e Analytics - USP/Esalq - 2025 e se baseia nos dados disponibilizados pelo IntegradorRHC, um sistema desenvolvido pelo INCA – Instituto Nacional de Câncer, para consolidação de casos assistidos nas Unidades Hospitalares após eliminação de multiplicidades. Os dados são provenientes dos Registros Hospitalares de Câncer (RHC) de todo o Brasil e são de acesso público no Instituto Nacional de Câncer [INCA] (2023a), abrangendo o período de 1985 até 2022 (Instituto Nacional de Câncer [INCA], 2011).

## Objetivos
Este projeto foi desenvolvido a partir dos dados disponíveis no IntegradorRHC com dois objetivos principais:

a. Identificar transformações nos dados que possam completar informações ausentes, melhorando o indicador de Completude, e transformações que possam identificar e resolver inconsistências, todas com o objetivo de contribuir para que pesquisadores da área tenham acesso a melhores e mais completas informações sobre o tratamento oncológico no Brasil. 

b. Utilização da base na construção de modelos de Machine Learning para a predição da resposta ao primeiro tratamento identificando oportunidades de melhoria nas ações de saúde pública no tratamento oncológico.

## Processo Utilizado
A figura abaixo mostra as etapas e artefatos gerados no processo de preparação dos dados obtidos a partir do RHC/INCA

![Processo Utilizado](imagens/metodo.png)


## Documentação

[**Descriçao das Funções do Código Fonte**](https://ulf-tecc2.github.io/rhc_inca/site)

## Conjunto de Dados

**Arquivos com os conjntos de dados gerados - Período de 1985 a 2022** 


#### Dados brutos consolidados em um arquivo único (CSV)
[**Base Consolidada**](https://drive.google.com/uc?export=download&id=1Zt2Kv9DtM7IBdAGDdFwvMogPvceG7KYA) 


#### Dados com tipos definidos (arquivo parquet)
[**Base Inicial  (parquet)**](https://drive.google.com/uc?export=download&id=1oNnt1K2yJhk3FzuK1ELUgUlFT-uf0d9x) 

 

#### Dados com resultado das análises e sanitização (arquivo parquet)
[**Base Sanitizada  (parquet)**](https://drive.google.com/uc?export=download&id=1P61hUWlMjr53jvomKmxHuLJkVR8XOuAf) 



#### Dados com resultado das transformações e inferências (arquivo parquet)
[**Base Transformada  (parquet)**](https://drive.google.com/uc?export=download&id=1P61hUWlMjr53jvomKmxHuLJkVR8XOuAf) 



#### Dados de Casos Analíticos pré-processados - Prontos para uso na construção de modelos (arquivo parquet)
[**Base Modelagem (parquet)**](https://drive.google.com/uc?export=download&id=1Bj0FcA6lO6PJfJs5-i003Ldgag4q_jP-) 




## Analises e Conclusões
**Arquivo com os resultados detalhados das analises e conclusoes**

#### Indicadores 
[**Indicadores**](dados/Publicos/Resultados.xlsx) 




