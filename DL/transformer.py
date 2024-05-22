from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d
import math

class FixedPositionalEncoding(nn.Module):
    
    def __init__(self, segment_length, d_model, dropout, device, scale_factor=1.0):
        
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.pe = torch.zeros(segment_length, d_model).to(device)
        position = torch.arange(0, segment_length, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = scale_factor * self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class LearnablePositionalEncoding(nn.Module):

    def __init__(self, segment_length, d_model, dropout):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(segment_length, 1, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MyEncoderLayer(nn.modules.Module):
    
    def __init__(self, d_model, dim_feedforward, num_heads, dropout=0.1):
        super(MyEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.gelu

    def forward(self, data: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, is_causal: Optional[bool] = None) -> Tensor:
        
        # Self-attention
        data2 = self.self_attn(data, data, data, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, is_causal=is_causal)[0]
        # Add & Norm
        data = data + self.dropout1(data2)  # [segment_length, batch_size, d_model]
        data = data.permute(1, 2, 0)  # [batch_size, d_model, segment_length]
        data = self.norm1(data)
        data = data.permute(2, 0, 1)  # [segment_length, batch_size, d_model]
        # Feed-forward
        data2 = self.linear2(self.dropout(self.activation(self.linear1(data))))
        # Add & Norm
        data = data + self.dropout2(data2)  # [segment_length, batch_size, d_model]
        data = data.permute(1, 2, 0)  # [batch_size, d_model, segment_length]
        data = self.norm2(data)
        data = data.permute(2, 0, 1)  # [segment_length, batch_size, d_model]

        return data

class TSTransformerClassifier(nn.Module):
    def __init__(self, config, device):
        super(TSTransformerClassifier, self).__init__()

        self.num_signals = config['num_signals']
        self.segment_length = config['segment_length']
        self.d_model = config['d_model']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.num_classes = config['num_classes']
        self.pe_type = config['pe_type']
        self.device = device

        self.project_inp = nn.Linear(self.num_signals, self.d_model)
        if self.pe_type == 'learnable':
            self.pos_enc = LearnablePositionalEncoding(self.segment_length, self.d_model, self.dropout)
        elif self.pe_type == 'fixed':
            self.pos_enc = FixedPositionalEncoding(self.segment_length, self.d_model, self.dropout, self.device)
        encoder_layer = MyEncoderLayer(self.d_model, self.dim_feedforward, self.num_heads, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers, mask_check=False, enable_nested_tensor=False)
        self.output_layer = nn.Linear(self.d_model * self.segment_length, self.num_classes)
        self.act = F.gelu
        self.dropout1 = nn.Dropout(self.dropout)

    def forward(self, X):

        # Dimensione data input: [batch_size, num_signals, segment_length]
        inp = X.permute(2, 0, 1) # [segment_length, batch_size, num_signals]
        # Moltiplicazione per la redice quadrata di d_model per scalare l'input
        # come descritto nel paper "Attention is All You Need"
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        inp = self.pos_enc(inp) # Aggiunta del positional encoding
        output = self.transformer_encoder(inp) # Layer di encoder
        # [segment_length, batch_size, d_model]
        output = self.act(output) # Funzione di attivazione
        output = output.permute(1, 0, 2)  # [batch_size, segment_length, d_model]
        output = self.dropout1(output) # Dropout
        
        output = output.reshape(output.shape[0], -1) # [batch_size, segment_length * d_model]
        output = self.output_layer(output)  # Layer di output
        
        return output # [batch_size, num_classes]