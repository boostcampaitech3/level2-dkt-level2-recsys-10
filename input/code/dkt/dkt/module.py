import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import copy
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self,ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))

class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.args.n_heads)

        self.ffn1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.ffn2 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)   

        if self.args.layer_norm:
            self.ln1 = nn.LayerNorm(self.hidden_dim)
            self.ln2 = nn.LayerNorm(self.hidden_dim)


    def forward(self, embed, mask):
        q = self.query(embed).permute(1, 0, 2)
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        out, _ = self.attn(q, k, v, attn_mask=mask)
        
        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        
        if self.args.layer_norm:
            out = self.ln1(out)

        ## feed forward network
        out = self.ffn1(out)
        out = F.relu(out)
        out = self.ffn2(out)

        ## residual + layer norm
        out = embed + out

        if self.args.layer_norm:
            out = self.ln2(out)

        return out