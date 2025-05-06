# Bibliotecas
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.regressionplots import influence_plot

warnings.filterwarnings("ignore")

# Configuração
arquivo_dados = "dataset_5.csv"
pasta_output = Path("output")
pasta_output.mkdir(exist_ok=True)
arquivo_saida = pasta_output / "relatorio_output.txt"

DEPENDENT_VAR    = "tempo_resposta"
niv_significancia = 0.05 # nível de significância
vif_threshold = 10 # threshold para multicolinearidade

#funcoes aux
def plot_influence(modelo, nome_imagem):
    """Gera e salva influence plot (Cook's Distance)."""
    caminho = pasta_output / nome_imagem

    # Gera o gráfico de influência
    fig = influence_plot(modelo, criterion="cooks")
    fig.tight_layout()
    fig.savefig(caminho, dpi=300)
    plt.close(fig)

def run_log_model(df_processado: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Ajusta regressão OLS robusta: log(tempo_resposta) ~ variáveis numéricas + dummies
    Retorna o objeto de modelo já ajustado.
    """
    # Seleção de colunas
    colunas_categoricas = df_processado.select_dtypes(include="object").columns.tolist()
    colunas_numericas   = (
        df_processado
        .select_dtypes(include=["number", "bool"])
        .columns.difference(['tempo_resposta'])
        .tolist()
    )

    # Observações com tempo_resposta positivo
    df_valid = df_processado[df_processado['tempo_resposta'] > 0].copy()

    # features
    X_num = df_valid[colunas_numericas]
    X_cat = pd.get_dummies(df_valid[colunas_categoricas], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)

    #target
    y_log = np.log(df_valid['tempo_resposta'])

    #intercepto pra X
    X_const = sm.add_constant(X, has_constant="add")

    #treina modelo
    modelo_log = sm.OLS(y_log, X_const).fit(cov_type="HC3")

    #saidas
    print("MODELO LOG")
    print(modelo_log.summary()) #mostra resultados
    plot_influence(modelo_log, "influence_log.png")

    return modelo_log


#inicio script principal

with arquivo_saida.open("w",encoding="utf-8") as f:
    # Lê o arquivo CSV
    df = pd.read_csv(arquivo_dados, sep=",")

    total_rows_initial = len(df)

    #infos iniciais
    f.write("--- Infos do Dataset ---\n")
    df.info(buf=f)

    f.write(f'{df.isnull().sum().to_string()} dados faltantes ({df.isnull().sum() / total_rows_initial * 100}% do total)')

    #pre processamento
    f.write("Tratamento de dados NA\n")
    df_limpo = df.dropna().copy()
    total_rows_final = len(df_limpo)
    linhas_a_menos = total_rows_initial - total_rows_final

    f.write(
        f"\n--- Remoção de linhas com NaN ---\n"
        f"Nova shape: {df_limpo.shape}\n"
        f"{linhas_a_menos} linhas a menos ({(linhas_a_menos / total_rows_initial) * 100}% do total)"
    )

    #identificar vars
    num_cols = df_limpo.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = df_limpo.select_dtypes(include="object").columns.tolist()

    f.write(f"\n--- Variáveis numéricas ---\n{num_cols}")
    f.write(f"\n--- Variáveis categóricas ---\n{cat_cols}")


    #####PARTE 1
    f.write("\n\n--- PARTE 1 ---\n")
    f.write(f"\n--- Estatística descritiva (N={total_rows_final}) ---\n")
    desc = df_limpo[num_cols+[DEPENDENT_VAR]].describe()
    f.write(desc.to_markdown())

    ######PARTE 2
    f.write("\n\n--- PARTE 2 ---\n")
    f.write("\n--- Modelo ---\n")

    # One-hot encoding
    if cat_cols:
        X_cat = pd.get_dummies(df_limpo[cat_cols], drop_first=True, dtype=int)
    else:
        X_cat = pd.DataFrame(index=df_limpo.index)

    X_num = df_limpo[num_cols]
    X_full = pd.concat([X_num, X_cat], axis=1)
    X_full_const = sm.add_constant(X_full, has_constant="add")
    y = df_limpo['tempo_resposta']

    modelo1 = sm.OLS(y, X_full_const).fit(cov_type="HC3")

    f.write(modelo1.summary().as_text()) #mostra resultados
    f.write("\n")
    plot_influence(modelo1, "influence.png")


    # diagnostico VIF
    f.write("\n\n--- Diagnóstico VIF (Modelo 1)---\n")
    vif_df = pd.DataFrame(
        {
            "Variavel": X_full.columns,
            "VIF": [
                variance_inflation_factor(X_full.values, i)
                for i in range(X_full.shape[1])
            ],
        }
    )
    f.write(vif_df.to_markdown(index=False))
    f.write("\n")

    # -----------------------------------------------------------------------
    # Diagnóstico - Breusch-Pagan
    # -----------------------------------------------------------------------
    f.write("--- Heterocedasticidade (Breusch-Pagan) ---\n")
    bp = het_breuschpagan(modelo1.resid, X_full_const)
    bp_labels = [
        "LM estatística", "LM p-valor", "F estatística", "F p-valor"
    ]
    for label, val in zip(bp_labels, bp):
        f.write(f"{label}: {val:.4f}\n")
    f.write(
        "Resultado: "
        + (
            "H0 rejeitada (heterocedasticidade)"
            if bp[3] < niv_significancia
            else "H0 aceita (homocedasticidade)"
        )
    )


    # -----------------------------------------------------------------------
    # Modelo 2 (reduzido) - regra simples: remove variável com maior p>0.05
    # -----------------------------------------------------------------------
    candidates = modelo1.pvalues.drop("const", errors="ignore")
    insignif   = candidates[candidates > niv_significancia]

    if not insignif.empty:
        var_to_remove = insignif.idxmax()
        f.write(f"\n--- Modelo 2: removendo '{var_to_remove}' (p={insignif.max():.3f}) ---\n")
        X_red = X_full_const.drop(columns=[var_to_remove])
        model2 = sm.OLS(y, X_red).fit(cov_type="HC3")
        f.write(model2.summary().as_text())
        f.write("\n")
    else:
        f.write("\nNenhuma variável insignificante para remover; Modelo 2 não gerado.\n")

    # -----------------------------------------------------------------------
    # Parte III - Modelo log(y)
    # -----------------------------------------------------------------------
    f.write("\n--- Modelo com log(tempo_resposta) ---\n")
    try:
        model_log = run_log_model(df_limpo)
        f.write(model_log.summary().as_text())
        f.write("\n")
    except Exception as exc:
        f.write(f"Erro ao ajustar modelo log: {exc}\n")

    f.write("\n--- Fim da análise ---\n")

# ----------------------------------------------------------------------------
print(f"\nAnálise concluída. Relatório salvo em '{arquivo_saida}'.")
print(f"Gráficos e influence plots no diretório '{pasta_output}/'.")
