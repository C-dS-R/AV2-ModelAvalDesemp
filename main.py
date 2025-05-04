# bibliotecas
import pandas as pd
import statsmodels.api as sm
#minhas funcoes
# from funcoes.parte1.estatisticas_iniciais import calc_estatisticas_iniciais

#LEITURA DOS DADOS
dados = pd.read_csv('dataset_5.csv', sep=',', encoding='latin1')

# obtem labels das colunas
# labels = dados.columns.values.tolist()
labels = ['cpu_cores', 'ram_gb', 'latencia_ms', 'armazenamento_tb', 'sistema_operacional', 'tipo_hd', 'tipo_processador', 'tempo_resposta']


########### PARTE 1 - Análise Estatística ###########
print('------------------ PARTE 1 - Análise Estatística ------------------')
print(dados.describe())
#TODO: Interpretar resultados



########### PARTE 2 - Modelo e Diagnóstico ###########
