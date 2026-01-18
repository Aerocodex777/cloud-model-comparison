
import pandas as pd
import numpy as np

def analyze_dataset():
    df = pd.read_csv('autoscaling_dataset.csv')
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    
    users = df['user'].unique()
    items = df['item'].unique()
    all_nodes = np.union1d(users, items)
    
    print(f"Unique Source Nodes (user): {len(users)}")
    print(f"Unique Target Nodes (item): {len(items)}")
    print(f"Total Unique Nodes: {len(all_nodes)}")
    
    print("Value counts for state_label:")
    print(df['state_label'].value_counts())
    
    # Check features
    feature_cols = ['req_count', 'cpu_src', 'mem_src', 'cpu_tgt', 'mem_tgt', 'pod_src_count', 'pod_tgt_count']
    print(f"Feature columns: {len(feature_cols)}")
    
    print("First few rows:")
    print(df[feature_cols].head())

if __name__ == "__main__":
    analyze_dataset()
