import pandas as pd

def discretizar_columna(df, columna, valores=None, nombres=None):
    df[columna] = pd.cut(df[columna], bins=valores, labels=nombres)
    return df