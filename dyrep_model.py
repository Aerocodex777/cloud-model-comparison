"""
DyRep: Dynamic Representation Learning for Dynamic Graphs
Implementation of the DyRep model for temporal graph learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DyRepAttention(nn.Module):
    """Temporal attention mechanism for DyRep"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_emb, neighbor_emb, time_delta):
        """
        Args:
            node_emb: [batch_size, hidden_dim]
            neighbor_emb: [batch_size, num_neighbors, hidden_dim]
            time_delta: [batch_size, num_neighbors]
        """
        Q = self.query(node_emb).unsqueeze(1)  # [batch, 1, hidden]
        K = self.key(neighbor_emb)  # [batch, num_neighbors, hidden]
        V = self.value(neighbor_emb)  # [batch, num_neighbors, hidden]
        
        # Attention scores with time decay
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        
        # Apply time decay
        time_decay = torch.exp(-time_delta.unsqueeze(-1))
        scores = scores * time_decay
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V).squeeze(1)
        
        return output, attn_weights


class DyRepLayer(nn.Module):
    """Single DyRep layer with temporal attention"""
    
    def __init__(self, in_channels, hidden_channels, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        self.attention = DyRepAttention(hidden_channels)
        self.node_update = nn.GRUCell(hidden_channels, hidden_channels)
        self.feature_transform = nn.Linear(in_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, timestamps, node_states=None):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            timestamps: Edge timestamps [num_edges]
            node_states: Previous node states [num_nodes, hidden_channels]
        """
        num_nodes = x.size(0)
        
        # Initialize node states if not provided
        if node_states is None:
            node_states = torch.zeros(num_nodes, self.hidden_channels, device=x.device)
        
        # Transform features
        x_transformed = self.feature_transform(x)
        
        # Update node states based on edges
        new_states = node_states.clone()
        
        for i in range(num_nodes):
            # Find neighbors
            mask = edge_index[0] == i
            if mask.sum() > 0:
                neighbors = edge_index[1, mask]
                neighbor_states = node_states[neighbors]
                
                # Calculate time deltas (simplified)
                time_delta = torch.ones(neighbors.size(0), device=x.device)
                
                # Apply attention
                attended, _ = self.attention(
                    node_states[i].unsqueeze(0),
                    neighbor_states.unsqueeze(0),
                    time_delta.unsqueeze(0)
                )
                
                # Ensure 2D for GRU
                attended = attended.view(1, -1)
                hidden_state = node_states[i].view(1, -1)
                
                # Update state with GRU
                new_states[i] = self.node_update(
                    attended,
                    hidden_state
                )
        
        new_states = self.dropout(new_states)
        return new_states


class DyRep(nn.Module):
    """
    DyRep: Dynamic Representation Learning
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of DyRep layers
        dropout: Dropout rate
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # DyRep layers
        self.layers = nn.ModuleList([
            DyRepLayer(hidden_channels if i > 0 else in_channels, 
                      hidden_channels, dropout)
            for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, timestamps):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            timestamps: Edge timestamps [num_edges]
        
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        # Initialize node states
        node_states = self.input_proj(x)
        
        # Apply DyRep layers
        for layer in self.layers:
            node_states = layer(x, edge_index, timestamps, node_states)
        
        # Project to output dimension
        output = self.output_proj(node_states)
        
        return output
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'num_layers={self.num_layers})')
