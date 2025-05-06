##### Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

#alguns params
niv_significancia = 0.05 # nível de significância
vif_threshold = 10 # threshold para multicolinearidade (VIF)


##### DADOS
dados = pd.read_csv('dataset_5.csv', sep=',', encoding='latin1') # LEITURA
# colunas = dados.columns.values.tolist() #['cpu_cores', 'ram_gb', 'latencia_ms', 'armazenamento_tb', 'sistema_operacional', 'tipo_hd', 'tipo_processador', 'tempo_resposta']

#########################################################
################## INICIO DAS QUESTÕES ##################
#########################################################
# Parte I - Análise Estatística:
print('------Parte 1: Análise Estatística Inicial-------')
# 1.Realize uma análise estatística inicial do conjunto de dados, obtendo medidas como média, mediana, mínimo, máximo, dentre outras medidas que julgue adequadas. Em seguida, interprete os resultados, comentando sobre a distribuição e as tendências centrais das variáveis.
def p1_q1():
    print(dados.describe())
    #TODO: interpretação dos resultados, comentando sobre a distribuição e as tendências centrais das variáveis.
# parte1_q1() #TODO: descomentar uma vez que finalizado

# Parte II - Modelo e Diagnóstico
print('\n\n\n------Parte 2: Modelo e Diagnóstico-------')
#  2. Ajuste um modelo de regressão linear múltipla considerando:
def p2_q2():
    ### Dataframes isolados
    # Variável dependente: tempo_resposta
    y_df = dados[['tempo_resposta']].copy()
    # Variáveis explicativas: demais variáveis ('cpu_cores', 'ram_gb', 'latencia_ms', 'armazenamento_tb')
    X_df = dados[['cpu_cores', 'ram_gb', 'latencia_ms', 'armazenamento_tb','sistema_operacional', 'tipo_hd', 'tipo_processador']].copy()



    #variaveis dummy
    X_df = pd.get_dummies(X_df, columns=['sistema_operacional', 'tipo_hd', 'tipo_processador'], drop_first=True)

    #remove NAs
    X_df = X_df.dropna()
    y_df = y_df.dropna()

    #alinha índices de y_df (endog) e X_df (exog)
    y_df, X_df = y_df.align(X_df, join='inner', axis=0)

    #intercepto pra X
    X_df = sm.add_constant(X_df)

    print(np.asarray(X_df).dtype) #verifica se o X_df é um array numpy


    #cria e treina modelo
    modelo = sm.OLS(y_df, X_df).fit()

    modelo.summary() #mostra resultados
    return modelo

p2_q2() #TODO: descomentar uma vez que finalizado

# 3. Informe (de acordo com as técnicas, abordagens e testes vistos em sala de aula):
#  ● O valor do intercepto e dos coeficientes estimados.
#  ● O valor de R² e R² ajustado.
#  ● Os valores de testes para interpretação dos coeficientes e do modelo de forma global.
# 4. Sobre as variáveis categóricas:
#  ● Como você tratou as variáveis categóricas do seu dataset (Mencione também quais foram as variáveis categóricas)?
#  ● Qual categoria base foi considerada para cada uma?
#  ● Interprete os coeficientes associados a essas categorias. 
# 5. Faça o diagnóstico de multicolinearidade:
#  ● Calcule o(s) fator(es) que auxiliam no diagnóstico de multicolinearidade, de acordo com o que foi visto em sala de aula.
#  ● Interprete: existe multicolinearidade? Alguma ação seria necessária? 
# 6. Faça o diagnóstico de heterocedasticidade:
#  ● Elabore os gráficos e testes pertinentes para o diagnóstico da heterocedasticidade.
#  ● Interprete os resultados.


# Parte III - Análise Crítica
# 7. Compare dois modelos:
#  ● Modelo 1: com todas as variáveis.
#  ● Modelo 2: excluindo uma variável (ou variáveis). Explique o motivo da exclusão dessa variável.
#  ● Compare o R² ajustado e o teste F entre os modelos.
#  ● Justifique qual modelo você recomendaria utilizar.
#  ● Quais ações práticas você sugeriria para melhorar o tempo de resposta do sistema? Considere aspectos como escolha de hardware e possíveis gargalos identificados