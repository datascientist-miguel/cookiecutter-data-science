import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def detect_multicolinealidad(data, exclude_vars=[]):

    features = [col for col in data.columns if col not in exclude_vars]
    vif_df = pd.DataFrame({'feature': features})
    vif_df['VIF'] = [variance_inflation_factor(data[features].values, i) for i in range(len(features))]
    vif_df['Multicolinealidad'] = ['Si' if vif > 10 else 'No' for vif in vif_df['VIF']]
    return vif_df