#!/usr/bin/env python3
"""
Simplified HRM model for inference
Based on the checkpoint structure we inspected
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class HRM(nn.Module):
    """
    Hierarchical Reasoning Model
    L-module: Low-level pattern recognition
    H-module: High-level reasoning
    """
    
    def __init__(self, num_classes=10, d_model=256, n_heads=8, n_layers=4, 
                 dropout=0.1, max_seq_len=784, vocab_size=12):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # L-module (Low-level)
        self.l_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        # H-module (High-level)
        self.h_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        # Output heads
        self.h_output = nn.Linear(d_model, num_classes)
        self.l_output = nn.Linear(d_model, num_classes)
        
        # Fusion layer
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_h, x_l=None):
        """
        Forward pass through HRM
        
        Args:
            x_h: High-level input (e.g., strategic view)
            x_l: Low-level input (e.g., tactical view)
        
        Returns:
            h_out: High-level predictions
            l_out: Low-level predictions
        """
        # If no separate L input, use same as H
        if x_l is None:
            x_l = x_h
        
        # Embed tokens
        h_emb = self.token_embedding(x_h) * math.sqrt(self.d_model)
        l_emb = self.token_embedding(x_l) * math.sqrt(self.d_model)
        
        # Add positional encoding
        h_emb = self.pos_encoding(h_emb)
        l_emb = self.pos_encoding(l_emb)
        
        h_emb = self.dropout(h_emb)
        l_emb = self.dropout(l_emb)
        
        # Process through L-module
        l_features = self.l_layers(l_emb)
        
        # Process through H-module with L-module context
        # Fuse L features into H processing
        l_context = l_features.mean(dim=1, keepdim=True)  # Global L context
        h_with_context = torch.cat([h_emb, l_context.expand(-1, h_emb.size(1), -1)], dim=-1)
        h_with_context = self.fusion(h_with_context)
        
        h_features = self.h_layers(h_with_context)
        
        # Generate outputs
        h_out = self.h_output(h_features)
        l_out = self.l_output(l_features)
        
        return h_out, l_out

class SimpleHRM(nn.Module):
    """
    Simplified HRM for quick testing
    """
    def __init__(self, num_classes=10, d_model=256, vocab_size=12):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=1024,
                batch_first=True
            ),
            num_layers=4
        )
        self.output = nn.Linear(d_model, num_classes)
        
    def forward(self, x, *args):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.output(x)
        return x, None  # Return None for L output