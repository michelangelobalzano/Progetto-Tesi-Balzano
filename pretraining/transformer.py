import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=240):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Channel embedding
class ChannelEmbedding(nn.Module):
    def __init__(self, hidden_dim, kernel_size=4, stride=1, padding=1):
        super(ChannelEmbedding, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.GELU()
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.max_pooling = nn.MaxPool1d(kernel_size=4, stride=4)

    def forward(self, x):

        x = self.conv1d(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.max_pooling(x)
        
        x = x.permute(0, 2, 1)
        
        return x

# Representation module
class RepresentationModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(RepresentationModule, self).__init__()

        self.num_signals = 3
    
        self.linear_projection = nn.Linear(hidden_dim, hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.add_norm_1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.add_norm_2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        x = self.linear_projection(x)
        x = x.permute(1, 0, 2)
        x, _ = self.self_attention(x, x, x)
        x = x.permute(1, 0, 2)
        x = self.add_norm_1(x)
        residual = x
        x = self.feed_forward(x)
        x = self.add_norm_2(x + residual)

        return x
		
# Transformation head
class TransformationHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformationHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, output_dim)
        )

    def forward(self, x):

        x = torch.mean(x, dim=1)
        x = self.mlp(x)
        x = x.squeeze(0)
        x = x.unsqueeze(1)
        return x

# Transformer
class Transformer(nn.Module):
    #def __init__(self, input_dim, hidden_dim, num_heads, output_dim):
    def __init__(self, signals, segment_length, d_model, num_heads, dropout, output_dim):
        super(Transformer, self).__init__()
        self.channel_embeddings = nn.ModuleDict({signal: ChannelEmbedding(d_model) for signal in signals})
        self.representation_module = RepresentationModule(segment_length, d_model, num_heads)
        self.transformation_head = TransformationHead(d_model, segment_length)
        
    def forward(self, data):
        
        embeddings = [embed(x) for embed, x in zip(self.channel_embeddings, data)]
        combined_embeddings = torch.cat(embeddings, dim=-1)
        representations = self.representation_module(combined_embeddings)
        output = self.transformation_head(representations)
        return output
    

        outputs = {}
        for signal, segment in data.items():

            segment = segment.permute(0, 2, 1)

            ce_output = self.channel_embeddings[signal](segment)
            rep_output = self.representation_module(ce_output)
            pred_output = self.transformation_head(rep_output)

            pred_output = pred_output.permute(0, 2, 1)

            outputs[signal] = pred_output

        return outputs
