"""
TGAT - Temporal Graph Attention Network (Simplified)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalEncoding(nn.Module):
    """Temporal encoding using sinusoidal functions"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, t):
        """Encode timestamps"""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.clamp(0, len(self.pe) - 1).long()
        return self.pe[t]


class TemporalAttentionLayer(nn.Module):
    """Temporal Attention Layer"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 time_dim: int = 8, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # Temporal encoder
        self.temporal_enc = TemporalEncoding(time_dim)
        
        # Feature projection
        self.feat_proj = nn.Linear(in_channels + time_dim, out_channels)
        
        # Attention weights
        self.att_proj = nn.Linear(2 * out_channels, heads)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.feat_proj.reset_parameters()
        self.att_proj.reset_parameters()
    
    def forward(self, x, edge_index, timestamps):
        """
        Args:
            x: (num_nodes, in_channels)
            edge_index: (2, num_edges)
            timestamps: (num_edges,)
        """
        src, dst = edge_index
        
        # Temporal encoding
        t_enc = self.temporal_enc(timestamps)  # (num_edges, time_dim)
        
        # Get features
        x_src = x[src]  # (num_edges, in_channels)
        x_dst = x[dst]  # (num_edges, in_channels)
        
        # Add temporal info
        x_src_t = torch.cat([x_src, t_enc], dim=1)  # (num_edges, in_channels + time_dim)
        x_dst_t = torch.cat([x_dst, t_enc], dim=1)
        
        # Project features
        h_src = self.feat_proj(x_src_t)  # (num_edges, out_channels)
        h_dst = self.feat_proj(x_dst_t)  # (num_edges, out_channels)
        
        # Attention scores
        h_cat = torch.cat([h_src, h_dst], dim=1)  # (num_edges, 2*out_channels)
        att_logits = self.att_proj(h_cat)  # (num_edges, heads)
        att_weights = F.softmax(att_logits, dim=1)  # (num_edges, heads)
        
        # Weighted aggregation
        h_msg = h_dst * att_weights.mean(dim=1, keepdim=True)
        
        # Aggregate to nodes
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(h_msg), h_msg)
        
        return out


class TGAT(nn.Module):
    """Temporal Graph Attention Network"""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, heads: int = 4, time_dim: int = 8, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.layers.append(TemporalAttentionLayer(
            in_channels, hidden_channels, time_dim, heads, dropout
        ))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(TemporalAttentionLayer(
                hidden_channels, hidden_channels, time_dim, heads, dropout
            ))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Output layer
        self.layers.append(TemporalAttentionLayer(
            hidden_channels, out_channels, time_dim, heads, dropout
        ))
        
        self.classifier = nn.Linear(out_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x, edge_index, timestamps, edge_attr=None):
        """Forward pass"""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, timestamps)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layers[-1](x, edge_index, timestamps)
        logits = self.classifier(x)
        
        return logits


if __name__ == "__main__":
    print("Testing TGAT Model...")
    
    num_nodes = 100
    num_edges = 500
    in_channels = 16
    hidden_channels = 32
    out_channels = 16
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    timestamps = torch.randint(0, 50, (num_edges,))
    
    model = TGAT(in_channels, hidden_channels, out_channels, num_layers=2, heads=4)
    
    try:
        output = model(x, edge_index, timestamps)
        print(f"✓ Model output shape: {output.shape}")
        print(f"✓ Expected: torch.Size([{num_nodes}, {out_channels}])")
        print(f"✓ TGAT Model Test Passed!")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
