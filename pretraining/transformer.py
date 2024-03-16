import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################
# COMPONENTI DEL TRANSFORMER
###############################################################

# CHANNEL EMBEDDINGS
class ChannelEmbedding(nn.Module):
    #def __init__(self, output_size, sampling_frequency):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=4, stride=1, padding=1):
        super(ChannelEmbedding, self).__init__()
        #self.conv1d = nn.Conv1d(in_channels=1, out_channels=output_size, kernel_size=sampling_frequency)
        self.conv1d = nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.GELU()
        #self.batch_norm = nn.BatchNorm1d(output_size)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        #self.max_pooling = nn.MaxPool1d(kernel_size=sampling_frequency, stride=sampling_frequency)
        self.max_pooling = nn.MaxPool1d(kernel_size=4, stride=4)

    def forward(self, x):

        #print('channel embedding...')

        #print('input dim: ', x.size())

        # Convoluzione 1D
        x = self.conv1d(x)
        #print('conv1d dim: ', x.size())
        # Funzione di attivazione
        x = self.activation(x)
        #print('gelu dim: ', x.size())
        # Normalizzazione
        x = self.batch_norm(x)
        #print('normalization dim: ', x.size())
        # Max Pooling
        x = self.max_pooling(x)
        #print('maxpool dim: ', x.size())

        # Riorganizzazione in tripla (B, N, F)
        
        batch_size = x.size(0)
        time_steps = x.size(2)
        filters = x.size(1)
        x = x.view(batch_size, time_steps, filters)
        
        return x

# REPRESENTATION MODULE

class RepresentationModule(nn.Module):
    #def __init__(self, input_size, num_heads, hidden_size, dropout=0.1):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(RepresentationModule, self).__init__()

        self.num_signals = 3
    
        #self.linear_projection = nn.Linear(input_size, hidden_size)
        self.linear_projection = nn.Linear(hidden_dim * self.num_signals, hidden_dim)
        #self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads)
        #self.add_norm_1 = nn.LayerNorm(hidden_size)
        self.add_norm_1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            #nn.Linear(hidden_size, hidden_size),
            #nn.GELU(),
            #nn.Linear(hidden_size, hidden_size)
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        #self.add_norm_2 = nn.LayerNorm(hidden_size)
        self.add_norm_2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        print('representation module...')
        '''
        x = self.linear_projection(x)
        residual = x
        x, _ = self.self_attention(x, x, x)
        x = self.add_norm_1(x + residual)
        residual = x
        x = self.feed_forward(x)
        x = self.add_norm_2(x + residual)
        '''

        print('representation input size: ', x.size())
        x = self.linear_projection(x)
        print('after linear size: ', x.size())
        x = x.permute(1, 0, 2)  # Cambia a (seq_len, batch_size, hidden_dim)
        print('permutation: ', x.size())
        x, _ = self.self_attention(x, x, x)
        print('self attention size: ', x.size())
        x = x.permute(1, 0, 2)  # Cambia a (batch_size, seq_len, hidden_dim)
        print('permutation size: ', x.size())
        x = self.add_norm_1(x)
        print('normalization size: ', x.size())
        residual = x
        x = self.feed_forward(x)
        print('feed-forward size: ', x.size())
        x = self.add_norm_2(x + residual)
        print('normalization size: ', x.size())

        return x
		
# TRANSFORMATION HEAD

class TransformationHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformationHead, self).__init__()
        self.mlp = nn.Sequential(
            #nn.Linear(input_dim, input_dim),
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            #nn.Linear(input_dim, output_dim)
            nn.Linear(input_dim * 4, output_dim)
        )
        #self.avg_pooling = nn.AvgPool1d(kernel_dim=4)
        #self.final_projection = nn.Linear(in_features=output_dim, out_features=input_dim)

    def forward(self, x):

        print('transformation head')

        '''x = self.mlp(x)
        x = self.avg_pooling(x)
        print(x.size())
        output1 = self.final_projection(x)
        output2 = self.final_projection(x)
        output3 = self.final_projection(x)

        output1 = torch.mean(output1, dim=1)
        output2 = torch.mean(output2, dim=1)
        output3 = torch.mean(output3, dim=1)

        output1 = output1.transpose(0, 1)
        output2 = output2.transpose(0, 1)
        output3 = output3.transpose(0, 1)

        return output1, output2, output3
        '''

        x = torch.mean(x, dim=1)  # Pooling media su seq_len
        x = self.mlp(x)
        return x

# ASSEMBLAGGIO DEL TRANSFORMER

class Transformer(nn.Module):
    '''
    def __init__(self, sampling_frequency, ce_output_size, representation_hidden_size, num_heads, transformation_output_size):
        super(Transformer, self).__init__()

        # Creazione di un channel embedding per ogni segnale
        self.channel_embeddings = nn.ModuleDict({
            'bvp': ChannelEmbedding(ce_output_size, sampling_frequency),
            'eda': ChannelEmbedding(ce_output_size, sampling_frequency),
            'hr': ChannelEmbedding(ce_output_size, sampling_frequency)
        })

        # Inizializzazione representation module
        self.representation_module = RepresentationModule(ce_output_size * 3, num_heads, representation_hidden_size)
        
        # Inizializzazione transformation head
        self.transformation_head = TransformationHead(representation_hidden_size, transformation_output_size)
    '''

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(Transformer, self).__init__()
        self.channel_embeddings = nn.ModuleDict({
            'bvp': ChannelEmbedding(input_dim, hidden_dim, hidden_dim),
            'eda': ChannelEmbedding(input_dim, hidden_dim, hidden_dim),
            'hr': ChannelEmbedding(input_dim, hidden_dim, hidden_dim)
        })
        self.representation_module = RepresentationModule(input_dim, hidden_dim, num_heads)
        self.transformation_head = TransformationHead(hidden_dim, output_dim)
    '''
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
    '''

    def forward(self, input_dict):

        ce_outputs = {}

        for signal, segment in input_dict.items():

            segment = segment.squeeze(1)
            segment = segment.unsqueeze(0).unsqueeze(1)
            ce_output = self.channel_embeddings[signal](segment)
            ce_outputs[signal] = ce_output

        concatenated_data = torch.cat(list(ce_outputs.values()), dim=2)

        rep_output = self.representation_module(concatenated_data)
        pred_output = self.transformation_head(rep_output)
        
        return pred_output

