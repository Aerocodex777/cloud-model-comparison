"""
EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs
Implementation of EvolveGCN-H (using GRU to evolve GCN parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class EvolveGCNLayer(nn.Module):
    """Single EvolveGCN layer that evolves GCN weights over time"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # GRU to evolve weight matrix
        # Input is x_mean (size in_channels)
        # Hidden state (weight row) is size in_channels
        self.weight_gru = nn.GRUCell(in_channels, in_channels)
        
        # Initial weight state
        self.weight_state = nn.Parameter(torch.Tensor(out_channels, in_channels))
        nn.init.xavier_uniform_(self.weight_state)
        
    def forward(self, x, edge_index, weight_state=None):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            weight_state: Current weight state [out_channels, in_channels]
        """
        if weight_state is None:
            weight_state = self.weight_state
        
        # Evolve weights using GRU
        # Use mean of node features as input to GRU
        x_mean = x.mean(dim=0)  # [in_channels]
        
        # Update each row of weight matrix
        new_weight_state = []
        for i in range(self.out_channels):
            new_row = self.weight_gru(x_mean.unsqueeze(0), weight_state[i].unsqueeze(0))
            new_weight_state.append(new_row)
        
        new_weight_state = torch.cat(new_weight_state, dim=0)
        
        # Apply GCN with evolved weights
        # Normalize adjacency matrix
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Message passing
        out = torch.matmul(x, new_weight_state.t())
        
        # Aggregate from neighbors
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Simple aggregation
        aggregated = torch.zeros_like(out)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            aggregated[dst] += out[src] * edge_weight[i]
        
        return aggregated, new_weight_state


class EvolveGCN(nn.Module):
    """
    EvolveGCN: Evolving Graph Convolutional Networks
    
    Uses GRU to evolve GCN parameters over time for dynamic graphs
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of EvolveGCN layers
        dropout: Dropout rate
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(EvolveGCNLayer(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(EvolveGCNLayer(hidden_channels, hidden_channels))
        
        # Last layer
        if num_layers > 1:
            self.layers.append(EvolveGCNLayer(hidden_channels, out_channels))
        else:
            self.layers[0] = EvolveGCNLayer(in_channels, out_channels)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Store weight states for each layer
        self.weight_states = None
        
    def forward(self, x, edge_index, timestamps):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            timestamps: Edge timestamps [num_edges] (used for temporal ordering)
        
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        # Initialize weight states for this forward pass (Stateless across epochs)
        current_weight_states = [None] * len(self.layers)
        
        h = x
        new_weight_states = []
        
        for i, layer in enumerate(self.layers):
            h, weight_state = layer(h, edge_index, current_weight_states[i])
            new_weight_states.append(weight_state)
            
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout_layer(h)
        
        # We don't store weight states persistently to avoid backward graph retention issues
        # self.weight_states = new_weight_states
        
        return h
    
    def reset_parameters(self):
        """Reset all learnable parameters"""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight_state)
        self.weight_states = None
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'num_layers={self.num_layers})')
