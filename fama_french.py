import pandas as pd
from sklearn import linear_model
from IPython.display import display, HTML
import waterfall_chart

FILE = "data/FF-Factors.csv"

def load_annual_factors():
    df_factors = (pd.read_csv(FILE, skiprows=1120, skipfooter=1, engine='python')
        .rename({'Unnamed: 0': "Year"}, axis=1))
    df_factors[['Mkt-RF','SMB','HML','RF']] = df_factors[['Mkt-RF','SMB','HML','RF']] / 100.
    df_factors['Alpha'] = 1.
    return df_factors

def load_monthly_factors():
    df_factors = (pd.read_csv(FILE, skiprows=3, skipfooter=97, engine='python')
        .rename({'Unnamed: 0': "YearMonth"}, axis=1))
    df_factors[['Mkt-RF','SMB','HML','RF']] = df_factors[['Mkt-RF','SMB','HML','RF']] / 100.
    df_factors['Alpha'] = 1.
    df_factors['Year'] = df_factors['YearMonth'].astype(str).str[:4].astype(int)
    df_factors['Month'] = df_factors['YearMonth'].astype(str).str[4:].astype(int)
    return df_factors

FACTORS=['Mkt-RF','SMB','HML','Alpha']

def ff_decomposition(df, returns, factors=FACTORS):
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(df[factors], returns - df['RF'])
    r2 = model.score(df[factors], returns - df['RF'])
    assert(model.intercept_ == 0.0)
    return pd.Series(
        list(model.coef_) + [r2],
        index=factors + ['R^2']
    )

def ff_weights(df, index_cols, factors=FACTORS):
    return pd.DataFrame({
        col: ff_decomposition(df, df[col], factors=FACTORS)
        for col in index_cols
    })

def ff_importances(df, ff_weights, factors=FACTORS, monthly=False):
    importances = (ff_weights
        .loc[factors]
        .multiply(
            df[factors].mean(),
            axis=0
        )
    )

    return importances

def ff_display(df, index_cols, waterfall_cols=None, monthly=False):
    display(HTML("<b>Fama French factors:</b>"))
    ff_weights_ = ff_weights(df, index_cols)
    display(ff_weights_)

    print("")
    display(HTML("<b>Contributions to return:</b>"))
    ff_importances_ = ff_importances(df, ff_weights_) * (12. if monthly else 1.)
    display(ff_importances_)

    if waterfall_cols is None:
        waterfall_cols = index_cols

    for col in waterfall_cols:
        waterfall_chart.plot(
            ff_importances_.index,
            ff_importances_[col] * 100,
            formatting = "{:,.2f}%",
            Title=col);