# bibliotecas
import pandas as pd
import statsmodels.api as sm
#minhas funcoes
from funcoes.parte1.estatisticas_iniciais import calc_estatisticas_iniciais

#LEITURA DOS DADOS
dados = pd.read_csv('dataset_5.csv', sep=',', encoding='latin1')

# obtem labels das colunas
labels = dados.columns.values.tolist()


########### PARTE 1 - Análise Estatística ###########

# estatisticas iniciais
estatisticas = dados.describe() #calcula média, mediana, desvio padrão, min, max, e quartis
print(estatisticas)
