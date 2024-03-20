import pandas as pd
import torch

# Caricamento dei dati
def load_data(file_path, signals):

    data = {}
    for signal in signals:
        data[signal] = pd.read_csv(file_path[signal])
    return data

# Conversione dei dati in un tensore di dimensioni (num_signals, num_segments, segment_length)
def prepare_data(data, num_signals, num_segments, segment_length):
    
    prepared_data = torch.zeros(num_signals, num_segments, segment_length)
    
    for i, (key, df) in enumerate(data.items()):

        for k, (segment_id, segment_data) in enumerate(df.groupby('segment_id')):
            segment_tensor = torch.tensor(segment_data.iloc[:, :-1].values, dtype=torch.float32)
            prepared_data[i, k] = segment_tensor.squeeze()
    
    return prepared_data

# Collate Function per il DataLoader
def my_collate_fn(batch):
    return torch.stack([item[0] for item in batch])