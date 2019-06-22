import pandas as pd
from sklearn import linear_model

FILE = "data/FF-Factors.csv"

def load_annual_factors():
    df_factors = (pd.read_csv(FILE, skiprows=1120, skipfooter=1, engine='python')
        .rename({'Unnamed: 0': "Year"}, axis=1))
    df_factors[['Mkt-RF','SMB','HML','RF']] = df_factors[['Mkt-RF','SMB','HML','RF']] / 100.
    return df_factors

def load_monthly_factors():
    df_factors = (pd.read_csv(FILE, skiprows=3, skipfooter=97, engine='python')
        .rename({'Unnamed: 0': "YearMonth"}, axis=1))
    df_factors[['Mkt-RF','SMB','HML','RF']] = df_factors[['Mkt-RF','SMB','HML','RF']] / 100.
    df_factors['Year'] = df_factors['YearMonth'].astype(str).str[:4].astype(int)
    df_factors['Month'] = df_factors['YearMonth'].astype(str).str[4:].astype(int)
    return df_factors

FACTORS=['Mkt-RF','SMB','HML']

def ff_decomposition(df, returns, factors=FACTORS):
    model = linear_model.LinearRegression()
    model.fit(df[factors], returns - df['RF'])
    r2 = model.score(df[factors], returns - df['RF'])
    return pd.Series(
        list(model.coef_) + [model.intercept_, r2],
        index=factors + ['Alpha', "R^2"]
    )

def ff_weights(df, index_cols, factors=FACTORS):
    return pd.DataFrame({
        col: ff_decomposition(df, df[col], factors=FACTORS)
        for col in index_cols
    })

def ff_importances(df, ff_weights, factors=FACTORS):
    importances = (ff_weights
        .loc[factors]
        .abs()
        .multiply(
            df[factors].std(),
            axis=0
        )
    )

    return importances / importances.sum()
