import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Caricamento dei dati
def load_data(data_directory, signals, task):

    data = {}
    for signal in signals:
        if task == 'pretraining':
            file_path = data_directory + signal + '.csv'
        elif task == 'classification':
            file_path = data_directory + signal + '_LABELED.csv'
        data[signal] = pd.read_csv(file_path[signal])
    return data

# Suddivisione dei segmenti in train e val per la task pretraining
def pretrain_data_split(data, split_ratios, signals):
    segment_ids = data['bvp']['segment_id'].unique()
    train_segment_ids, val_segment_ids = train_test_split(segment_ids, train_size=split_ratios[0], random_state=42)

    train_data = {}
    val_data = {}
    for signal in signals:
        train_data[signal] = data[signal][data[signal]['segment_id'].isin(train_segment_ids)].reset_index(drop=True)
        val_data[signal] = data[signal][data[signal]['segment_id'].isin(val_segment_ids)].reset_index(drop=True)

    return train_data, val_data

# Suddivisione dei segmenti in train val e test per la task classification
def classification_data_split(data, split_ratios, signals):
    segment_ids = data['bvp']['segment_id'].unique()
    train_segment_ids, remaining = train_test_split(segment_ids, train_size=split_ratios[0], random_state=42)
    val_segment_ids, test_segment_ids = train_test_split(remaining, train_size=split_ratios[1] / (split_ratios[1] + split_ratios[2]), random_state=42)

    train_data = {}
    val_data = {}
    test_data = {}
    for signal in signals:
        train_data[signal] = data[signal][data[signal]['segment_id'].isin(train_segment_ids)].reset_index(drop=True)
        val_data[signal] = data[signal][data[signal]['segment_id'].isin(val_segment_ids)].reset_index(drop=True)
        test_data[signal] = data[signal][data[signal]['segment_id'].isin(test_segment_ids)].reset_index(drop=True)

    return train_data, val_data, test_data

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