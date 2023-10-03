import numpy as np
import pandas as pd
from scipy.stats import kstest, shapiro, anderson

def norm_distribution(data, significance_level, column_name=None):
    # Initialize an empty list to store the results
    results = []

    if column_name is None:
        # Process the entire DataFrame
        for col_name, column_data in data.items():
            # Kolmogorov-Smirnov test
            kolmogorov_smirnov = kstest(column_data, 'norm')
            kolmogorov_result = 'Normal' if kolmogorov_smirnov.pvalue > significance_level else 'Not Normal'
            results.append({
                'Column': col_name,
                'Test': 'Kolmogorov-Smirnov',
                'P-value': f'{kolmogorov_smirnov.pvalue:.3f}',
                'Significance Level': significance_level,
                'Result': kolmogorov_result
            })

            # Shapiro-Wilk test
            shapiro_wilk = shapiro(column_data)
            shapiro_result = 'Normal' if shapiro_wilk.pvalue > significance_level else 'Not Normal'
            results.append({
                'Column': col_name,
                'Test': 'Shapiro-Wilk',
                'P-value': f'{shapiro_wilk.pvalue:.3f}',
                'Significance Level': significance_level,
                'Result': shapiro_result
            })

            # Anderson-Darling test
            anderson_darling = anderson(column_data)
            anderson_result = 'Normal' if anderson_darling.statistic < anderson_darling.critical_values[2] else 'Not Normal'
            results.append({
                'Column': col_name,
                'Test': 'Anderson-Darling',
                'P-value': f'{anderson_darling.statistic:.3f}',
                'Significance Level': significance_level,
                'Result': anderson_result
            })
    else:
        # Process a specific column
        column_data = data[column_name]

        # Kolmogorov-Smirnov test
        kolmogorov_smirnov = kstest(column_data, 'norm')
        kolmogorov_result = 'Normal' if kolmogorov_smirnov.pvalue > significance_level else 'Not Normal'
        results.append({
            'Column': column_name,
            'Test': 'Kolmogorov-Smirnov',
            'P-value': f'{kolmogorov_smirnov.pvalue:.3f}',
            'Significance Level': significance_level,
            'Result': kolmogorov_result
        })

        # Shapiro-Wilk test
        shapiro_wilk = shapiro(column_data)
        shapiro_result = 'Normal' if shapiro_wilk.pvalue > significance_level else 'Not Normal'
        results.append({
            'Column': column_name,
            'Test': 'Shapiro-Wilk',
            'P-value': f'{shapiro_wilk.pvalue:.3f}',
            'Significance Level': significance_level,
            'Result': shapiro_result
        })

        # Anderson-Darling test
        anderson_darling = anderson(column_data)
        anderson_result = 'Normal' if anderson_darling.statistic < anderson_darling.critical_values[2] else 'Not Normal'
        results.append({
            'Column': column_name,
            'Test': 'Anderson-Darling',
            'P-value': f'{anderson_darling.statistic:.3f}',
            'Significance Level': significance_level,
            'Result': anderson_result
        })

    # Create a DataFrame from the results
    final_df = pd.DataFrame(results)

    return final_df