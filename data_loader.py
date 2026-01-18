
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_autoscaling_data(file_path):
    """
    Loads autoscaling_dataset.csv and processes it for TGN/Transformer models.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Encode Users and Items to Node IDs
    le = LabelEncoder()
    # Combine user and item to find all unique nodes
    all_nodes = pd.concat([df['user'], df['item']]).unique()
    le.fit(all_nodes)
    
    df['src'] = le.transform(df['user'])
    df['dst'] = le.transform(df['item'])
    
    num_nodes = len(all_nodes)
    
    # 2. Extract Timestamps
    # Normalize timestamps to start from 0
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    timestamps = torch.tensor(df['timestamp'].values, dtype=torch.float)
    
    # 3. Extract Edge Features and Labels
    # Edge features: req_count
    # Node features (dynamic): cpu, mem, pod_count
    # For TGN, we often pass interaction features.
    
    # Feature columns
    feature_cols = ['req_count', 'cpu_src', 'mem_src', 'cpu_tgt', 'mem_tgt', 'pod_src_count', 'pod_tgt_count']
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols].values)
    edge_attr = torch.tensor(features, dtype=torch.float)
    
    # Labels
    y = torch.tensor(df['state_label'].values, dtype=torch.long)
    
    # Edge Index
    src = torch.tensor(df['src'].values, dtype=torch.long)
    dst = torch.tensor(df['dst'].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    
    # 4. Construct Node Features (Static X)
    # We aggregate cpu/mem/pod counts for each node from the interactions
    # Source nodes
    src_df = df[['src', 'cpu_src', 'mem_src', 'pod_src_count']].rename(
        columns={'src': 'node_id', 'cpu_src': 'cpu', 'mem_src': 'mem', 'pod_src_count': 'pod_count'}
    )
    # Target nodes
    tgt_df = df[['dst', 'cpu_tgt', 'mem_tgt', 'pod_tgt_count']].rename(
        columns={'dst': 'node_id', 'cpu_tgt': 'cpu', 'mem_tgt': 'mem', 'pod_tgt_count': 'pod_count'}
    )
    
    # Concatenate and group by node_id to get mean features
    node_features_df = pd.concat([src_df, tgt_df])
    node_features_grouped = node_features_df.groupby('node_id').mean()
    
    # Reindex to ensure order matches node identifiers 0..num_nodes-1
    # Fill missing (if any node appears only in one role or never? (unlikely from unique))
    node_features_grouped = node_features_grouped.reindex(range(num_nodes), fill_value=0)
    
    # Normalize node features
    x_np = scaler.fit_transform(node_features_grouped.values)
    x = torch.tensor(x_np, dtype=torch.float)
    
    data = {
        'x': x,
        'edge_index': edge_index,
        'timestamps': timestamps,
        'edge_attr': edge_attr,
        'y': y,
        'num_nodes': num_nodes,
        'node_feature_dim': x.shape[1],
        'edge_feature_dim': edge_attr.shape[1]
    }
    
    print(f"Data Loaded: {num_nodes} nodes, {len(df)} edges")
    print(f" - Node Features (X): {x.shape} (cpu, mem, pod)")
    print(f" - Edge Features (Attr): {edge_attr.shape} (req_count + others)")
    return data

if __name__ == "__main__":
    data = load_autoscaling_data('autoscaling_dataset.csv')
    print("Keys:", data.keys())
