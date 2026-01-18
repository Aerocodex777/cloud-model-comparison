"""
JODIE: Joint Dynamic User-Item Embedding
Implementation for temporal recommendation and dynamic graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JODIEEmbedding(nn.Module):
    """Dynamic embedding update module for JODIE"""
    
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # RNN for updating embeddings
        self.update_rnn = nn.RNNCell(embedding_dim * 2, embedding_dim)
        
    def forward(self, user_emb, item_emb, time_delta):
        """
        Update embeddings based on interaction
        
        Args:
            user_emb: User embedding [batch_size, embedding_dim]
            item_emb: Item embedding [batch_size, embedding_dim]
            time_delta: Time since last interaction [batch_size, 1]
        """
        # Concatenate user and item embeddings
        combined = torch.cat([user_emb, item_emb], dim=-1)
        
        # Update user embedding
        new_user_emb = self.update_rnn(combined, user_emb)
        
        return new_user_emb


class JODIEProjection(nn.Module):
    """Time-aware projection for JODIE"""
    
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Time projection
        self.time_proj = nn.Linear(1, embedding_dim)
        
    def forward(self, embedding, time_delta):
        """
        Project embedding with time information
        
        Args:
            embedding: Node embedding [batch_size, embedding_dim]
            time_delta: Time delta [batch_size, 1]
        """
        time_encoding = self.time_proj(time_delta)
        projected = embedding + time_encoding
        return projected


class JODIELayer(nn.Module):
    """Single JODIE layer for dynamic graph learning"""
    
    def __init__(self, in_channels, hidden_channels, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Embedding update
        self.embedding_update = JODIEEmbedding(hidden_channels)
        
        # Time projection
        self.time_projection = JODIEProjection(hidden_channels)
        
        # Feature transformation
        self.feature_transform = nn.Linear(in_channels, hidden_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, timestamps, node_embeddings=None):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            timestamps: Edge timestamps [num_edges]
            node_embeddings: Current node embeddings [num_nodes, hidden_channels]
        """
        num_nodes = x.size(0)
        
        # Initialize embeddings if not provided
        if node_embeddings is None:
            node_embeddings = self.feature_transform(x)
        
        # Update embeddings based on interactions
        new_embeddings = node_embeddings.clone()
        
        # Process each edge (interaction)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            
            # Calculate time delta (simplified)
            time_delta = torch.tensor([[1.0]], device=x.device)
            
            # Update source node embedding
            updated_src = self.embedding_update(
                node_embeddings[src].unsqueeze(0),
                node_embeddings[dst].unsqueeze(0),
                time_delta
            )
            
            # Apply time projection
            new_embeddings[src] = self.time_projection(
                updated_src.squeeze(0).unsqueeze(0),
                time_delta
            ).squeeze(0)
        
        new_embeddings = self.dropout(new_embeddings)
        return new_embeddings


class JODIE(nn.Module):
    """
    JODIE: Joint Dynamic User-Item Embedding
    
    Predicts dynamic embeddings for nodes in temporal graphs
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of JODIE layers
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
        
        # JODIE layers
        self.layers = nn.ModuleList([
            JODIELayer(hidden_channels if i > 0 else in_channels,
                      hidden_channels, dropout)
            for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        
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
        # Initialize embeddings
        node_embeddings = self.input_proj(x)
        
        # Apply JODIE layers
        for layer in self.layers:
            node_embeddings = layer(x, edge_index, timestamps, node_embeddings)
        
        # Batch normalization
        node_embeddings = self.batch_norm(node_embeddings)
        
        # Project to output dimension
        output = self.output_proj(node_embeddings)
        
        return output
    
    def predict_interaction(self, user_emb, item_emb):
        """
        Predict interaction score between user and item
        
        Args:
            user_emb: User embedding [batch_size, hidden_channels]
            item_emb: Item embedding [batch_size, hidden_channels]
        
        Returns:
            Interaction scores [batch_size]
        """
        scores = (user_emb * item_emb).sum(dim=-1)
        return torch.sigmoid(scores)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'num_layers={self.num_layers})')
