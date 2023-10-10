import pandas as pd

def undersample(data, target):
    """
    This function undersamples the majority class.
    """
    # Calculate the minority class
    minority_class = min(data[target].value_counts())
    # Calculate the majority class
    majority_class = max(data[target].value_counts())
    
    # Separate majority and minority classes
    df_majority = data[data[target]==majority_class]
    df_minority = data[data[target]==minority_class]
    
    # Undersample majority class
    df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=1)
    
    # Concatenate minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    
    # Return downsampled dataframe
    return df_downsampled