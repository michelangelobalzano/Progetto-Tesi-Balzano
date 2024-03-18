import torch
import torch.nn as nn
import torch.nn.functional as F

# CHANNEL EMBEDDINGS

class ChannelEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=4, stride=1, padding=1):
        super(ChannelEmbedding, self).__init__()
        self.conv1d = nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.GELU()
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.max_pooling = nn.MaxPool1d(kernel_size=4, stride=4)

    def forward(self, x):

        x = self.conv1d(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.max_pooling(x)
        
        batch_size = x.size(0)
        time_steps = x.size(2)
        filters = x.size(1)
        x = x.view(batch_size, time_steps, filters)
        
        return x



# REPRESENTATION MODULE
    
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
		


# TRANSFORMATION HEAD
    
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

# ASSEMBLAGGIO DEL TRANSFORMER

class Transformer(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(Transformer, self).__init__()
        self.channel_embeddings = nn.ModuleDict({
            'bvp': ChannelEmbedding(input_dim, hidden_dim, hidden_dim),
            'eda': ChannelEmbedding(input_dim, hidden_dim, hidden_dim),
            'hr': ChannelEmbedding(input_dim, hidden_dim, hidden_dim)
        })
        self.representation_module = RepresentationModule(input_dim, hidden_dim, num_heads)
        self.transformation_head = TransformationHead(hidden_dim, input_dim)

    def forward(self, input_dict):
        
        outputs = {}
        for signal, segment in input_dict.items():
            #segment = segment.squeeze(1)
            #segment = segment.unsqueeze(0).unsqueeze(1)

            segment = segment.permute(0, 2, 1)

            ce_output = self.channel_embeddings[signal](segment)
            rep_output = self.representation_module(ce_output)
            pred_output = self.transformation_head(rep_output)

            pred_output = pred_output.permute(0, 2, 1)

            outputs[signal] = pred_output

        return outputs


