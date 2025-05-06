# Bibliotecas
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

from funcoes.grafico_influencia import grafico_influencia


warnings.filterwarnings("ignore") # para lidar com o erro do tipo do pandas

# caminho de output
pasta_output = Path("output")
pasta_output.mkdir(exist_ok=True)
arquivo_saida = pasta_output / "relatorio_output.txt"

#funcao aux
def run_log_model(df_limpo: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Ajusta regressão OLS robusta: log(tempo_resposta) ~ Vars numéricas + dummies
    Retorna o objeto de modelo já ajustado.
    """
    # Seleção de colunas
    cols_categ = df_limpo.select_dtypes(include="object").columns.tolist()
    cols_num   = (
        df_limpo
        .select_dtypes(include=["number", "bool"])
        .columns.difference(['tempo_resposta'])
        .tolist()
    )

    # Observações com tempo_resposta positivo
    df_valid = df_limpo[df_limpo['tempo_resposta'] > 0].copy()

    # features
    X_num = df_valid[cols_num]
    X_dummies = pd.get_dummies(df_valid[cols_categ], drop_first=True)
    X = pd.concat([X_num, X_dummies], axis=1)

    #target
    y_log = np.log(df_valid['tempo_resposta'])

    #intercepto pra X
    X_com_intercepto = sm.add_constant(X, has_constant="add")

    #treina modelo
    modelo = sm.OLS(y_log, X_com_intercepto).fit(cov_type="HC3")

    #saidas
    print("MODELO LOG")
    print(modelo.summary()) #mostra resultados
    grafico_influencia(modelo, "influence_log.png", pasta_output)

    return modelo


log_str = '' #para os logs

#####INICIO

# LEITURA DE DADOS
df = pd.read_csv("dataset_5.csv", sep=",")

quant_linhas_og = len(df) #quantidade original de linhas

#infos iniciais
log_str += (f'--- Infos do Dataset\n{df.isnull().sum().to_string()} dados faltantes ({df.isnull().sum() / quant_linhas_og * 100}% do total)\n')

#pre processamento
df_limpo = df.dropna().copy()
quant_linhas_limpo = len(df_limpo) #quantidade de linhas depois do dropna
linhas_a_menos = quant_linhas_og - quant_linhas_limpo

log_str += (f"---Tratamento de dados NA\n{linhas_a_menos} linhas a menos ({(linhas_a_menos / quant_linhas_og) * 100}% do total)\nNova shape: {df_limpo.shape}")

#identificar vars
num_cols = df_limpo.select_dtypes(include=["number", "bool"]).columns.tolist()
cat_cols = df_limpo.select_dtypes(include="object").columns.tolist()

log_str += (f"\n--- Vars numéricas ---\n{num_cols}\n--- Vars categóricas ---\n{cat_cols}")


#####PARTE 1
log_str += (f"\n\n--- PARTE 1 ---\n\n--- Estatística descritiva (N={quant_linhas_limpo}) ---\n")
desc = df_limpo[num_cols+['tempo_resposta']].describe()
log_str += (desc.to_markdown())

######PARTE 2
log_str += ("\n\n--- PARTE 2 ---\n\n--- Modelo ---\n")

# one-hot encoding
X_cat = pd.DataFrame(index=df_limpo.index)

X_num = df_limpo[num_cols]
X_full = pd.concat([X_num, X_cat], axis=1)
X_full_com_intercepto = sm.add_constant(X_full, has_constant="add")
y = df_limpo['tempo_resposta']


#cria e treina o modelo
modelo = sm.OLS(y, X_full_com_intercepto).fit(cov_type="HC3")

#mostra resultados
log_str += (modelo.summary().as_text()) + "\n"

#gera grafico de
grafico_influencia(modelo, "influence.png", pasta_output)


# diagnostico VIF
log_str += ("\n\n--- Diagnóstico VIF (Modelo 1)---\n")
vif_df = pd.DataFrame(
    {
        "Variavel": X_full.columns,
        "VIF": [variance_inflation_factor(X_full.values, i) for i in range(X_full.shape[1])],
    }
)
log_str += (vif_df.to_markdown(index=False)) + "\n"

# -----------------------------------------------------------------------
# Diagnóstico - Breusch-Pagan
# -----------------------------------------------------------------------
log_str += ("--- Heterocedasticidade (Breusch-Pagan) ---\n")
bp = het_breuschpagan(modelo.resid, X_full_com_intercepto)
bp_labels = [
    "LM estatística", "LM p-valor", "F estatística", "F p-valor"
]
for label, val in zip(bp_labels, bp):
    log_str += (f"{label}: {val:.4f}\n")
log_str += (
    "Resultado: "
    + (
        "H0 rejeitada (heterocedasticidade)"
        if bp[3] < .05 # .05 sendo o nivel de significancia
        else "H0 aceita (homocedasticidade)"
    )
)


# -----------------------------------------------------------------------
# Modelo 2 (reduzido) - regra simples: remove variável com maior p>0.05
# -----------------------------------------------------------------------
candidates = modelo.pvalues.drop("const", errors="ignore")
insignif   = candidates[candidates > .05] # .05 sendo o nivel de significancia

if not insignif.empty:
    var_to_remove = insignif.idxmax()
    log_str += (f"\n--- Modelo 2: removendo '{var_to_remove}' (p={insignif.max():.3f}) ---\n")
    X_red = X_full_com_intercepto.drop(columns=[var_to_remove])
    model2 = sm.OLS(y, X_red).fit(cov_type="HC3")
    log_str += (model2.summary().as_text())
    log_str += ("\n")
else:
    log_str += ("\nNenhuma variável insignificante para remover; Modelo 2 não gerado.\n")

# -----------------------------------------------------------------------
# Parte III - Modelo log(y)
# -----------------------------------------------------------------------
log_str += ("\n--- Modelo com log(tempo_resposta) ---\n")
try:
    model_log = run_log_model(df_limpo)
    log_str += (model_log.summary().as_text())
    log_str += ("\n")
except Exception as exc:
    log_str += (f"Erro ao ajustar modelo log: {exc}\n")

log_str += ("\n--- Análise concluída ---\n")

# ----------------------------------------------------------------------------
print(log_str) #mostra resultados
#escreve resultados em arquivo

with arquivo_saida.open("w", encoding="utf-8") as f:
    f.write(log_str)