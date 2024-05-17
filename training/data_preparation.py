import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import math

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
    users = pd.read_csv(data_directory + 'labeled_user_ids.csv')
    return data, labels, users

# Suddivisione dei segmenti in train e val per la task pretraining
def pretrain_data_split(data, val_ratio, signals):
    segment_ids = data[signals[0]]['segment_id'].unique()
    train_segment_ids, val_segment_ids = train_test_split(segment_ids, train_size=(100-val_ratio)/100, random_state=42)

    train_data = {}
    val_data = {}
    for signal in signals:
        train_data[signal] = data[signal][data[signal]['segment_id'].isin(train_segment_ids)].reset_index(drop=True)
        val_data[signal] = data[signal][data[signal]['segment_id'].isin(val_segment_ids)].reset_index(drop=True)

    return train_data, val_data

# Suddivisione dei segmenti in train val e test per la task classification
def classification_data_split(data, labels, users, val_ratio, test_ratio, signals, split_per_subject):

    if split_per_subject:
        user_ids = users['user_id'].unique()
        train_user_ids, remaining = train_test_split(user_ids, train_size=(100-val_ratio-test_ratio+5)/100, random_state=42)
        val_user_ids, test_user_ids = train_test_split(remaining, train_size=val_ratio / (val_ratio + test_ratio), random_state=42)

        train_segment_ids = users[users['user_id'].isin(train_user_ids)]['segment_id'].tolist()
        val_segment_ids = users[users['user_id'].isin(val_user_ids)]['segment_id'].tolist()
        test_segment_ids = users[users['user_id'].isin(test_user_ids)]['segment_id'].tolist()

    else:
        segment_ids = data[signals[0]]['segment_id'].unique()
        train_segment_ids, remaining = train_test_split(segment_ids, train_size=(100-val_ratio-test_ratio)/100, random_state=42)
        val_segment_ids, test_segment_ids = train_test_split(remaining, train_size=val_ratio / (val_ratio + test_ratio), random_state=42)

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
def pretraining_data_to_tensor(data, num_signals, num_segments, segment_length):
    
    prepared_data = torch.zeros(num_signals, num_segments, segment_length)
    
    for i, (key, df) in enumerate(data.items()):

        for k, (segment_id, segment_data) in enumerate(df.groupby('segment_id')):
            segment_tensor = torch.tensor(segment_data.iloc[:, :-1].values, dtype=torch.float32)
            prepared_data[i, k] = segment_tensor.squeeze()
    
    return prepared_data

# Conversione dei dati in un tensore di dimensioni (num_signals, num_segments, segment_length)
def classification_data_to_tensor(data, labels, num_signals, num_segments, segment_length, label):
    
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
            prepared_labels.append(0)
        elif row[label] == 'positive':
            prepared_labels.append(2)
        elif row[label] == 'neutral':
            prepared_labels.append(1)
    prepared_labels = torch.tensor(prepared_labels)
    
    return prepared_data, prepared_labels


# Collate Function per i batch del pretraining
def pretrain_collate_fn(batch):

    return torch.stack([item[0] for item in batch])

# Collate Function per i batch della classificazione
def classification_collate_fn(batch):

    return torch.stack([item[0] for item in batch]), torch.tensor([item[1] for item in batch])

# Caricamento dati e creazione dataloaders per pretraining
def get_pretraining_dataloaders(config, device):
    # Caricamento e preparazione dei dati
    data = load_unlabeled_data(config['data_path'], config['signals'])
    # Split dei dati
    train, val = pretrain_data_split(data, config['val_ratio'], config['signals'])
    num_train_segments = len(train[next(iter(train))].groupby('segment_id'))
    num_val_segments = len(val[next(iter(val))].groupby('segment_id'))
    # Conversione dati in tensori
    train_data = pretraining_data_to_tensor(train, config['num_signals'], num_train_segments, config['segment_length'])
    val_data = pretraining_data_to_tensor(val, config['num_signals'], num_val_segments, config['segment_length'])
    # Creazione del DataLoader
    train_data = train_data.permute(1, 0, 2).to(device)
    val_data = val_data.permute(1, 0, 2).to(device)
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, collate_fn=pretrain_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, collate_fn=pretrain_collate_fn)    

    return train_dataloader, val_dataloader

# Caricamento dati e creazione dataloaders per classificazione
def get_classification_dataloaders(config, device):
    # Caricamento e preparazione dei dati
    data, labels, users = load_labeled_data(config['data_path'], config['signals'], config['label'])
    # Split dei dati
    train, train_labels, val, val_labels, test, test_labels = classification_data_split(data, labels, users, config['val_ratio'], config['test_ratio'], config['signals'], config['split_per_subject'])
    num_train_segments = len(train[config['signals'][0]].groupby('segment_id'))
    num_val_segments = len(val[config['signals'][0]].groupby('segment_id'))
    num_test_segments = len(test[config['signals'][0]].groupby('segment_id'))
    # Preparazione dati
    train_data, train_labels = classification_data_to_tensor(train, train_labels, config['num_signals'], num_train_segments, config['segment_length'], config['label'])
    val_data, val_labels = classification_data_to_tensor(val, val_labels, config['num_signals'], num_val_segments, config['segment_length'], config['label'])
    test_data, test_labels = classification_data_to_tensor(test, test_labels, config['num_signals'], num_test_segments, config['segment_length'], config['label'])
    # Creazione del DataLoader
    train_data = train_data.permute(1, 0, 2).to(device)
    val_data = val_data.permute(1, 0, 2).to(device)
    test_data = test_data.permute(1, 0, 2).to(device)
    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)
    test_labels = test_labels.to(device)
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, collate_fn=classification_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, collate_fn=classification_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, collate_fn=classification_collate_fn)

    return train_dataloader, val_dataloader, test_dataloader