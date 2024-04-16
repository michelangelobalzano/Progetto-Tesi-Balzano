import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
import math
'''
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
        
    def forward(self, data, masks):
        print('transformer data input size: ', data.size())
        # data = (batch_size, num_signals, segment_length)
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
    
'''

class PositionalEncoding(nn.Module):
    
    def __init__(self, segment_length, d_model, dropout, device, scale_factor=1.0):
        
        super(PositionalEncoding, self).__init__()
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

class MyEncoderLayer(nn.modules.Module):
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(MyEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.gelu

    def forward(self, data, src_mask=None, is_causal=False, src_key_padding_mask=None) -> Tensor:
        # Self-attention
        data2 = self.self_attn(data, data, data)[0]
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

class TSTransformer(nn.Module):
    def __init__(self, num_signals, segment_length, iperparametri, device):
        super(TSTransformer, self).__init__()

        self.num_signals = num_signals
        self.segment_length = segment_length
        self.d_model = iperparametri['d_model']
        self.dropout = iperparametri['dropout']
        self.num_layers = iperparametri['num_layers']
        self.num_heads = iperparametri['num_heads']
        self.device = device

        self.project_inp = nn.Linear(self.num_signals, self.d_model)
        self.pos_enc = PositionalEncoding(segment_length, self.d_model, self.dropout, self.device)
        encoder_layer = MyEncoderLayer(self.d_model, self.num_heads, self.d_model, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers, mask_check=False, enable_nested_tensor=False)
        self.output_layer = nn.Linear(self.d_model, self.num_signals)
        self.act = F.gelu
        self.dropout1 = nn.Dropout(self.dropout)
        
    def forward(self, data):

        # Dimensione data input: [batch_size, num_signals, segment_length]
        inp = data.permute(2, 0, 1) # [segment_length, batch_size, num_signals]
        # Moltiplicazione per la redice quadrata di d_model per scalare l'input
        # come descritto nel paper "Attention is All You Need"
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        inp = self.pos_enc(inp) # Aggiunta del positional encoding
        output = self.transformer_encoder(inp) # Layer di encoder
        # [segment_length, batch_size, d_model]
        output = self.act(output) # Funzione di attivazione
        output = output.permute(1, 0, 2)  # [batch_size, segment_length, d_model]
        output = self.dropout1(output) # Dropout
        
        
        output = self.output_layer(output) # Layer di output
        # [batch_size, segment_length, num_signals]
        output = output.permute(0, 2, 1) # [batch_size, num_signals, segment_length]

        return output # [batch_size, num_signals, segment_length]

class TSTransformerClassifier(nn.Module):
    def __init__(self, num_signals, segment_length, iperparametri, num_classes, device):
        super(TSTransformerClassifier, self).__init__()

        self.num_signals = num_signals
        self.segment_length = segment_length
        self.d_model = iperparametri['d_model']
        self.dropout = iperparametri['dropout']
        self.num_layers = iperparametri['num_layers']
        self.num_heads = iperparametri['num_heads']
        self.device = device
        self.num_classes = num_classes

        self.project_inp = nn.Linear(self.num_signals, self.d_model)
        self.pos_enc = PositionalEncoding(segment_length, self.d_model, self.dropout, self.device)
        encoder_layer = MyEncoderLayer(self.d_model, self.num_heads, self.d_model, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers, mask_check=False, enable_nested_tensor=False)
        self.output_layer = nn.Linear(self.d_model * segment_length, num_classes)
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