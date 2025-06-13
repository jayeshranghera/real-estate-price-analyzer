import pandas as pd 
import os

def load_data(file_path='data/USA_Housing.csv'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'file not found: {file_path}')
    
    df = pd.read_csv(file_path)
    print(f'Data Loaded Successfully, shape: {df.shape} ')
    return df