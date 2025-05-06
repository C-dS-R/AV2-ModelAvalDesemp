# Bibliotecas
from pathlib import Path
import warnings
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# Minhas funções
from funcoes.grafico_influencia import grafico_influencia
from funcoes.run_log_model import run_log_model


warnings.filterwarnings("ignore") # para lidar com o erro do tipo do pandas

# caminho de output
pasta_output = Path("output")
pasta_output.mkdir(exist_ok=True)
arquivo_relatorio_output = pasta_output / "relatorio_output.txt"

#################################### INICIO ####################################

# LEITURA DE DADOS
df = pd.read_csv('dataset_5.csv', sep=",")

#quantidade original de linhas
quant_linhas_og = len(df)

#infos iniciais
result_str = (f'--- Infos do Dataset\n{df.isnull().sum().to_string()} dados faltantes ({df.isnull().sum() / quant_linhas_og * 100}% do total)\n')

#pre processamento
df_limpo = df.dropna().copy()
quant_linhas_limpo = len(df_limpo) #quantidade de linhas depois do dropna
linhas_a_menos = quant_linhas_og - quant_linhas_limpo

result_str += (f"---Tratamento de dados NA\n{linhas_a_menos} linhas a menos ({(linhas_a_menos / quant_linhas_og) * 100}% do total)\nNova shape: {df_limpo.shape}")

#identifica colunas numericas/categóricas
cols_num = df_limpo.select_dtypes(include=["number", "bool"]).columns.tolist()
cols_cat = df_limpo.select_dtypes(include="object").columns.tolist()

result_str += (f"\n--- Vars numéricas ---\n{cols_num}\n--- Vars categóricas ---\n{cols_cat}")


#####PARTE 1
result_str += (f"\n\n--- PARTE 1 ---\n\n--- Estatística descritiva (N={quant_linhas_limpo}) ---\n")
result_str += df_limpo.describe().to_markdown()

######PARTE 2
result_str += ("\n\n--- PARTE 2 ---\n\n--- Modelo 1 ---\n")

# one-hot encoding
X_categ_com_ohe = pd.DataFrame(index=df_limpo.index)

X_numericas = df_limpo[cols_num]
X_todas = pd.concat([X_numericas, X_categ_com_ohe], axis=1) #junta as numericas com as categoricas apos one hot encoding
X_todas_com_intercepto = sm.add_constant(X_todas, has_constant="add")
y = df_limpo['tempo_resposta'] # variavel dependente


#cria e treina o modelo
modelo = sm.OLS(y, X_todas_com_intercepto).fit(cov_type="HC3")

#mostra resultados do modelo
result_str += (modelo.summary().as_text()) + "\n"

#gera grafico de influencia
grafico_influencia(modelo, 'graf_influencia.png', pasta_output)


# diagnostico VIF
result_str += ("\n\n--- Diagnóstico VIF (Modelo 1)\n")
vif_df = pd.DataFrame(
    {
        "Variavel": X_todas.columns,
        "VIF": [variance_inflation_factor(X_todas.values, i) for i in range(X_todas.shape[1])],
    }
)
result_str += (vif_df.to_markdown(index=False)) + "\n"

# Diagnóstico Breusch-Pagan
result_str += ("\n\n--- Teste de Heterocedasticidade de Breusch-Pagan\n")
bp = het_breuschpagan(modelo.resid, X_todas_com_intercepto) #realiza teste

#Mostra resultado do teste
result_str += (
    f'Estatística Multiplicador Lagrange: {bp[0]:.4f}\n'
    f'p-valor LM: {bp[1]:.4f}\n'
    f'Estatística F: {bp[2]:.4f}\n'
    f'p-valor F: {bp[3]:.4f}\n'
    f'\nResultado: H0 {"rejeitada (Heterocedasticidade)" if bp[3] < .05 else "aceita (Homocedasticidade)"}\n\n'
)


# Modelo 2
result_str += ("\n\n--- Modelo 2 ---\n")
candidatos = modelo.pvalues.drop("const", errors="ignore")
vars_insignificantes   = candidatos[candidatos > .05] # .05 sendo o nivel de significancia

if not vars_insignificantes.empty: #se houver variavel insignificante
    #obtem a variável com maior p-valor para remover
    var_a_remov = vars_insignificantes.idxmax()

    result_str += (f"\n- Variavel a remover: {var_a_remov} (p-valor={vars_insignificantes.max()})\n")
    X_red = X_todas_com_intercepto.drop(columns=[var_a_remov])

    #cria e treina modelo
    modelo2 = sm.OLS(y, X_red).fit(cov_type="HC3")

    #mostra resultados do modelo
    result_str += (modelo2.summary().as_text()) + "\n"
else: #caso não haja variáveis insignificantes
    result_str += ("\nNão há variáveis insignificante.\n")

# -----------------------------------------------------------------------
# Parte III - Modelo log(y)
# -----------------------------------------------------------------------
result_str += ("\n--- Modelo com log(tempo_resposta) ---\n")
try:
    modelo_log = run_log_model(df_limpo)
    result_str += (modelo_log.summary().as_text())
    result_str += ("\n")
except Exception as exc:
    result_str += (f"Erro ao ajustar modelo log: {exc}\n")

result_str += ("\n--- Análise concluída ---\n")

# ----------------------------------------------------------------------------
print(result_str) #mostra resultados
#escreve resultados em arquivo

with arquivo_relatorio_output.open("w", encoding="utf-8") as f:
    f.write(result_str)