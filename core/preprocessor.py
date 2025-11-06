import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df: pd.DataFrame):
    # """Handle missing values and basic preprocessing"""
    # df = df.dropna()
    
    # for col in df.select_dtypes(include=['object']).columns:
    #     df[col] = LabelEncoder().fit_transform(df[col])
    
    # print(" Data cleaned and encoded successfully.")
    return df
