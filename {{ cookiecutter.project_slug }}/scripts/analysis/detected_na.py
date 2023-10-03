import pandas as pd

def detect_valores_nulos(data):
    cantidad_valores_nulos = data.isnull().sum()
    porcentaje_valores_nulos = (cantidad_valores_nulos / len(data)) * 100
    valores_nulos = pd.concat([cantidad_valores_nulos, porcentaje_valores_nulos], axis=1)
    valores_nulos.columns = ['Total Valores Nulos', 'Porcentaje de Valores Nulos']
    
    return valores_nulos[valores_nulos['Total Valores Nulos'] > 0].sort_values(by='Total Valores Nulos', ascending=False)