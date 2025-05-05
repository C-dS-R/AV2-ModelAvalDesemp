import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


##### DADOS
df = pd.read_csv('dataset_5.csv', sep=',', encoding='latin1') # LEITURA

#trata vars categ
cat_cols = df.select_dtypes(include="object").columns
df_dummies = pd.get_dummies(df, columns=cat_cols, drop_first=True)
base_categories = {col: df[col].unique()[0] for col in cat_cols}
print(base_categories)


#ajust modelo
import statsmodels.api as sm

y = df_dummies["tempo_resposta"]
X = df_dummies.drop(columns=["tempo_resposta"])
X = sm.add_constant(X)                  # inclui intercepto

model1 = sm.OLS(y, X).fit()
print(model1.summary())
