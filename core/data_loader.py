import pandas as pd

def load_data(path: str , target):
    """Load CSV or Excel file"""
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format! Use CSV or XLSX.")
    
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df
