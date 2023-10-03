import pandas as pd
from imblearn.over_sampling import SMOTE

def aplicar_smote(X, y):
    # Aplicar SMOTE al conjunto de datos
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    # Convertir X_resampled en DataFrame
    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    # Agregar la variable objetivo al DataFrame
    X_resampled_df[y.name] = y_resampled
    
    return X_resampled_df