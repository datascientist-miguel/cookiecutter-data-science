import pandas as pd
def undersample(data, target):
    """
    This function undersamples the majority class.
    """
    # Calculate the class distribution
    class_counts = data[target].value_counts()
    
    # Find the minority and majority class
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    # Separate majority and minority classes
    df_majority = data[data[target] == majority_class]
    df_minority = data[data[target] == minority_class]
    
    # Undersample majority class
    df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=1)
    
    # Concatenate minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    
    # Return downsampled dataframe
    return df_downsampled