#funcao aux
from funcoes.grafico_influencia import grafico_influencia

import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_log_model(df_limpo: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Ajusta regressão OLS robusta: log(tempo_resposta) ~ Vars numéricas + dummies
    Retorna o objeto de modelo já ajustado.
    """
    # Seleção de colunas
    cols_categ = df_limpo.select_dtypes(include="object").columns.tolist()
    cols_num   = (df_limpo.select_dtypes(include=["number", "bool"]).columns.difference(['tempo_resposta']).tolist())

    # Observações com tempo_resposta positivo
    df_valid = df_limpo[df_limpo['tempo_resposta'] > 0].copy()

    # features e target
    X_num = df_valid[cols_num]
    X_dummies = pd.get_dummies(df_valid[cols_categ], drop_first=True)
    X = pd.concat([X_num, X_dummies], axis=1)
    y_log = np.log(df_valid['tempo_resposta'])

    #adiciona intercepto no X
    X = sm.add_constant(X, has_constant="add")

    #treina modelo
    modelo = sm.OLS(y_log, X).fit(cov_type="HC3")

    #saidas
    print("MODELO LOG")
    print(modelo.summary()) #mostra resultados
    grafico_influencia(modelo, "log_influencia.png")

    return modelo