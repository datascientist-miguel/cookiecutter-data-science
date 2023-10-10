import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

def correlations(data, target):
    """
    This function returns a dataframe with the correlations separated by correlation method.
    Pearson: Trust when dealing with linear relationships and normally distributed data.
    Spearman: Trust for monotonic (not necessarily linear) relationships.
    Kendall: Trust for associations in ordinal or non-normally distributed data.
    """
    # Check if the target variable is numeric
    if target not in data.select_dtypes(include=[np.number]).columns:
        print("Target variable is not numeric. Skipping correlation calculation.")
        return
    
    # Check if all variables in data are numeric
    non_numeric_vars = [col for col in data.columns if col not in data.select_dtypes(include=[np.number]).columns]
    if non_numeric_vars:
        print("Non-numeric variables found. Skipping correlation calculation for those variables.")
        data = data.drop(non_numeric_vars, axis=1)
    
    try:
        # Correlations matrix
        corr = data.corr(method='pearson')
        corr_2 = data.corr(method='spearman')
        corr_3 = data.corr(method='kendall')

        # Create dataframes with the correlations separated by correlation method
        corr_df = pd.DataFrame(corr[target].sort_values(ascending=False)).reset_index()
        corr_2_df = pd.DataFrame(corr_2[target].sort_values(ascending=False)).reset_index()
        corr_3_df = pd.DataFrame(corr_3[target].sort_values(ascending=False)).reset_index()

        # Rename columns
        corr_df.columns = ['Variable_Pearson', 'Pearson']
        corr_2_df.columns = ['Variable_Spearman', 'Spearman']
        corr_3_df.columns = ['Variable_Kendall', 'Kendall']

        # Merge dataframes
        corr_merged = pd.merge(corr_df, corr_2_df, left_index=True, right_index=True)
        corr_merged = pd.merge(corr_merged, corr_3_df, left_index=True, right_index=True)

        # Plot correlations using Plotly
        fig = px.bar(corr_merged, x='Variable_Pearson', y=['Pearson', 'Spearman', 'Kendall'],
                     title='Correlations by Method',
                     labels={'Variable_Pearson': 'Variable'})
        fig.update_layout(barmode='group', xaxis_tickangle=-45, width=1000, height=600)
        fig.show()
        
        # Save the plot in folder report and in the format png
        fig.write_image("../../reports/figures/correlations.png") 

        return corr_merged
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")