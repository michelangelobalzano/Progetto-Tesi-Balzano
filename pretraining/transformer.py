import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################
# COMPONENTI DEL TRANSFORMER
###############################################################

# CHANNEL EMBEDDINGS

class ChannelEmbedding(nn.Module):
    def __init__(self, output_size, sampling_frequency):
        super(ChannelEmbedding, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=output_size, kernel_size=sampling_frequency)
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
        x = self.linear_projection(x)
        residual = x
        x, _ = self.self_attention(x, x, x)
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
        self.avg_pooling = nn.AvgPool1d(kernel_size=1)
        self.final_projection = nn.Linear(in_features=output_size, out_features=input_size)

    def forward(self, x):
        x = self.mlp(x)
        x = self.avg_pooling(x)
        output1 = self.final_projection(x)
        output2 = self.final_projection(x)
        output3 = self.final_projection(x)

        output1 = torch.mean(output1, dim=1)
        output2 = torch.mean(output2, dim=1)
        output3 = torch.mean(output3, dim=1)

        # Trasponi i tensori risultanti
        output1 = output1.transpose(0, 1)
        output2 = output2.transpose(0, 1)
        output3 = output3.transpose(0, 1)

        return output1, output2, output3












###############################################################
# ASSEMBLAGGIO DEL TRANSFORMER
###############################################################

class Transformer(nn.Module):
    def __init__(self, segment_lenght, sampling_frequency, channel_embedding_output_size, representation_hidden_size, representation_num_heads, transformation_output_size):
        super(Transformer, self).__init__()

        # Creazione di un channel embedding per ogni segnale
        self.channel_embeddings = nn.ModuleDict({
            'bvp': ChannelEmbedding(channel_embedding_output_size, sampling_frequency),
            'eda': ChannelEmbedding(channel_embedding_output_size, sampling_frequency),
            'hr': ChannelEmbedding(channel_embedding_output_size, sampling_frequency)
        })

        # Inizializzazione representation module
        self.representation_module = RepresentationModule(channel_embedding_output_size * 3, representation_num_heads, representation_hidden_size)
        
        # Inizializzazione transformation head
        self.transformation_head = TransformationHead(representation_hidden_size, transformation_output_size)

    def forward(self, x):

        embedded_data = []
        for signal, tensor in x.items():

            tensor = tensor.squeeze(1)
            tensor = tensor.unsqueeze(0).unsqueeze(1)
            embedded_data.append(self.channel_embeddings[signal](tensor))
        
        # Concatenazione dell'output dei channel embeddings lungo la dimensione F
        concatenated_data = torch.cat(embedded_data, dim=2)

        representation_output = self.representation_module(concatenated_data)
        
        output1, output2, output3 = self.transformation_head(representation_output)

        output = {
            'bvp': output1,
            'eda': output2,
            'hr': output3
        }

        return output