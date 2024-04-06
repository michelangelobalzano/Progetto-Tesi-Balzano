import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Caricamento dei dati
def load_unlabeled_data(data_directory, signals):

    data = {}
    for signal in signals:
        file_path = data_directory + signal + '.csv'
        data[signal] = pd.read_csv(file_path, low_memory=False)
    return data

# Caricamento dei dati
def load_labeled_data(data_directory, signals, label):

    data = {}
    for signal in signals:
        file_path = data_directory + signal + '_LABELED.csv'
        data[signal] = pd.read_csv(file_path, low_memory=False)
    if label == 'valence':
        labels = pd.read_csv(data_directory + 'VALENCE.csv')
    elif label == 'arousal':
        labels = pd.read_csv(data_directory + 'AROUSAL.csv')
    return data, labels

# Suddivisione dei segmenti in train e val per la task pretraining
def pretrain_data_split(data, split_ratios, signals):
    segment_ids = data['BVP']['segment_id'].unique()
    train_segment_ids, val_segment_ids = train_test_split(segment_ids, train_size=split_ratios[0]/100, random_state=42)

    train_data = {}
    val_data = {}
    for signal in signals:
        train_data[signal] = data[signal][data[signal]['segment_id'].isin(train_segment_ids)].reset_index(drop=True)
        val_data[signal] = data[signal][data[signal]['segment_id'].isin(val_segment_ids)].reset_index(drop=True)

    return train_data, val_data

# Suddivisione dei segmenti in train val e test per la task classification
def classification_data_split(data, labels, split_ratios, signals):
    segment_ids = data['BVP']['segment_id'].unique()
    train_segment_ids, remaining = train_test_split(segment_ids, train_size=split_ratios[0]/100, random_state=42)
    val_segment_ids, test_segment_ids = train_test_split(remaining, train_size=split_ratios[1] / (split_ratios[1] + split_ratios[2]), random_state=42)

    train_data = {}
    val_data = {}
    test_data = {}
    for signal in signals:
        train_data[signal] = data[signal][data[signal]['segment_id'].isin(train_segment_ids)].reset_index(drop=True)
        val_data[signal] = data[signal][data[signal]['segment_id'].isin(val_segment_ids)].reset_index(drop=True)
        test_data[signal] = data[signal][data[signal]['segment_id'].isin(test_segment_ids)].reset_index(drop=True)
    train_labels = labels[labels['segment_id'].isin(train_segment_ids)].reset_index(drop=True)
    val_labels = labels[labels['segment_id'].isin(val_segment_ids)].reset_index(drop=True)
    test_labels = labels[labels['segment_id'].isin(test_segment_ids)].reset_index(drop=True)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels

# Conversione dei dati in un tensore di dimensioni (num_signals, num_segments, segment_length)
def prepare_data(data, num_signals, num_segments, segment_length):
    
    prepared_data = torch.zeros(num_signals, num_segments, segment_length)
    
    for i, (key, df) in enumerate(data.items()):

        for k, (segment_id, segment_data) in enumerate(df.groupby('segment_id')):
            segment_tensor = torch.tensor(segment_data.iloc[:, :-1].values, dtype=torch.float32)
            prepared_data[i, k] = segment_tensor.squeeze()
    
    return prepared_data

# Conversione dei dati in un tensore di dimensioni (num_signals, num_segments, segment_length)
def prepare_classification_data(data, labels, num_signals, num_segments, segment_length, label):
    
    # Conversione dati in tensore
    prepared_data = torch.zeros(num_signals, num_segments, segment_length)
    for i, (key, df) in enumerate(data.items()):
        for k, (segment_id, segment_data) in enumerate(df.groupby('segment_id')):
            segment_tensor = torch.tensor(segment_data.iloc[:, :-1].values, dtype=torch.float32)
            prepared_data[i, k] = segment_tensor.squeeze()

    # Conversione etichette valence in tensore
    prepared_labels = []
    for index, row in labels.iterrows():
        # Memorizzazione del valore della colonna nella lista del tensore
        if row[label] == 'negative':
            prepared_labels.append(-1)
        elif row[label] == 'positive':
            prepared_labels.append(1)
        elif row[label] == 'neutral':
            prepared_labels.append(0)
    prepared_labels = torch.tensor(prepared_labels)
    
    return prepared_data, prepared_labels


# Collate Function per il DataLoader
def pretrain_collate_fn(batch):

    return torch.stack([item[0] for item in batch])

def classification_collate_fn(batch):

    return torch.stack([item[0] for item in batch]), torch.tensor([item[1] for item in batch])