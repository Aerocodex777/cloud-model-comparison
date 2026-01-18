"""
TGA - Temporal Graph Attention (Simplified variant)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional temporal encoding"""
    
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


class GATAttentionHead(nn.Module):
    """Single GAT attention head with temporal info"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 time_dim: int = 8, dropout: float = 0.0, negative_slope: float = 0.2):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        
        self.lin = nn.Linear(in_channels + time_dim, out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, 2 * out_channels))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
    
    def forward(self, x, edge_index, t_enc):
        """
        Args:
            x: (num_nodes, in_channels)
            edge_index: (2, num_edges)
            t_enc: (num_edges, time_dim)
        """
        src, dst = edge_index
        
        # Add temporal info
        x_src = x[src]  # (num_edges, in_channels)
        x_dst = x[dst]  # (num_edges, in_channels)
        
        x_src_t = torch.cat([x_src, t_enc], dim=1)  # (num_edges, in_channels + time_dim)
        x_dst_t = torch.cat([x_dst, t_enc], dim=1)
        
        # Project
        h_src = self.lin(x_src_t)  # (num_edges, out_channels)
        h_dst = self.lin(x_dst_t)  # (num_edges, out_channels)
        
        # Attention
        h_cat = torch.cat([h_src, h_dst], dim=1)  # (num_edges, 2*out_channels)
        att_logits = (h_cat @ self.att.t()).squeeze(-1)  # (num_edges,)
        att_logits = F.leaky_relu(att_logits, self.negative_slope)
        
        # Softmax normalization
        att_logits_max = torch.full((x.size(0),), float('-inf'), device=x.device)
        att_logits_max.scatter_(0, dst, att_logits)
        
        att_weights = att_logits - att_logits_max[dst]
        att_weights = torch.exp(att_weights)
        
        att_sum = torch.zeros(x.size(0), device=x.device)
        att_sum.scatter_add_(0, dst, att_weights)
        att_weights = att_weights / (att_sum[dst] + 1e-8)
        
        att_weights = self.dropout(att_weights)
        
        # Aggregate
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        h_weighted = h_dst * att_weights.unsqueeze(1)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(h_weighted), h_weighted)
        
        return out


class GATLayer(nn.Module):
    """Multi-head GAT layer"""
    
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 4,
                 time_dim: int = 8, dropout: float = 0.0, concat: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat
        
        self.heads = nn.ModuleList([
            GATAttentionHead(in_channels, out_channels, time_dim, dropout)
            for _ in range(num_heads)
        ])
    
    def forward(self, x, edge_index, t_enc):
        outputs = []
        for head in self.heads:
            out = head(x, edge_index, t_enc)
            outputs.append(out)
        
        if self.concat:
            out = torch.cat(outputs, dim=-1)
        else:
            out = torch.mean(torch.stack(outputs), dim=0)
        
        return out


class TGN(nn.Module):
    """Temporal Graph Network"""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, heads: int = 4, time_dim: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.temporal_enc = PositionalEncoding(time_dim)
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.layers.append(GATLayer(
            in_channels, hidden_channels, heads, time_dim, dropout, concat=True
        ))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(
                hidden_channels * heads, hidden_channels, heads, time_dim, dropout, concat=True
            ))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Output layer
        final_in = hidden_channels * heads if num_layers > 1 else in_channels
        self.layers.append(GATLayer(
            final_in, out_channels, heads, time_dim, dropout, concat=False
        ))
        
        self.classifier = nn.Linear(out_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x, edge_index, timestamps, edge_attr=None):
        """Forward pass"""
        t_enc = self.temporal_enc(timestamps)
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, t_enc)
            if i < len(self.norms):
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layers[-1](x, edge_index, t_enc)
        logits = self.classifier(x)
        
        return logits


if __name__ == "__main__":
    print("Testing TGN Model...")
    
    num_nodes = 100
    num_edges = 500
    in_channels = 16
    hidden_channels = 32
    out_channels = 16
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    timestamps = torch.randint(0, 50, (num_edges,))
    
    model = TGN(in_channels, hidden_channels, out_channels, num_layers=2, heads=4)
    
    try:
        output = model(x, edge_index, timestamps)
        print(f"✓ Model output shape: {output.shape}")
        print(f"✓ Expected: torch.Size([{num_nodes}, {out_channels}])")
        print(f"✓ TGN Model Test Passed!")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
