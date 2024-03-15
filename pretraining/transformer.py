import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################
# COMPONENTI DEL TRANSFORMER
###############################################################

# CHANNEL EMBEDDINGS

class ChannelEmbedding(nn.Module):
    def __init__(self, input_size, output_size, sampling_frequency):
        super(ChannelEmbedding, self).__init__()
        self.conv1d = nn.Conv1d(input_size, output_size, kernel_size=sampling_frequency, stride=sampling_frequency, padding=0)
        self.activation = nn.GELU()
        self.batch_norm = nn.BatchNorm1d(output_size)
        self.max_pooling = nn.MaxPool1d(kernel_size=sampling_frequency, stride=sampling_frequency)

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

# REPRESENTATIO MODULE

class RepresentationModule(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, dropout=0.1):
        super(RepresentationModule, self).__init__()
        self.linear_projection = nn.Linear(input_size, hidden_size)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.add_norm_1 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.add_norm_2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        residual = x
        x = self.linear_projection(x)
        x = x.permute(1, 0, 2)  # Change to (seq_len, batch_size, hidden_size) for Multihead Attention
        x, _ = self.self_attention(x, x, x)
        x = x.permute(1, 0, 2)  # Change back to (batch_size, seq_len, hidden_size)
        x = self.add_norm_1(x + residual)
        residual = x
        x = self.feed_forward(x)
        x = self.add_norm_2(x + residual)
        return x
		
# TRANSFORMATION HEAD

class TransformationHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformationHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.GELU(),
            nn.Linear(input_size, output_size)
        )
        self.avg_pooling = nn.AvgPool1d(kernel_size=2)

    def forward(self, x):
        x = self.mlp(x)
        x = self.avg_pooling(x)
        return x












###############################################################
# ASSEMBLAGGIO DEL TRANSFORMER
###############################################################

class Transformer(nn.Module):
    def __init__(self, input_sizes, sampling_frequency, channel_embedding_output_size, representation_hidden_size, representation_num_heads, transformation_output_size):
        super(Transformer, self).__init__()
        self.channel_embeddings = nn.ModuleDict({
            'bvp': ChannelEmbedding(input_sizes['bvp'], channel_embedding_output_size, sampling_frequency),
            'eda': ChannelEmbedding(input_sizes['eda'], channel_embedding_output_size, sampling_frequency),
            'hr': ChannelEmbedding(input_sizes['hr'], channel_embedding_output_size, sampling_frequency),
        })
        self.representation_module = RepresentationModule(len(input_sizes) * channel_embedding_output_size, representation_num_heads, representation_hidden_size)
        self.transformation_head = TransformationHead(representation_hidden_size, transformation_output_size)

    def forward(self, x):
        embedded_data = []
        for signal, segment in x.items():
            embedded_data.append(self.channel_embeddings[signal](segment.unsqueeze(0)))
        concatenated_data = torch.cat(embedded_data, dim=2)
        representation_output = self.representation_module(concatenated_data)
        transformation_output = self.transformation_head(representation_output)
        return transformation_output