import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Rimozione delle etichette neutral in base all'etichetta da classificare
def remove_neutrals(data, labels, users, config):
    labels_filtered = labels[labels[config['label']] != 'neutral']
    valid_segment_ids = labels_filtered['segment_id']

    data_filtered = {}
    for signal in config['signals']:
        data_filtered[signal] = data[signal][data[signal]['segment_id'].isin(valid_segment_ids)]
    users_filtered = users[users['segment_id'].isin(valid_segment_ids)]
    
    return data_filtered, labels_filtered, users_filtered

# Caricamento dei dati
def load_data(config):

    data = {}
    for signal in config['signals']:
        file_path = config['data_path'] + signal + '.csv'
        data[signal] = pd.read_csv(file_path, low_memory=False)
    if config['label'] == 'valence':
        labels = pd.read_csv(config['data_path'] + 'VALENCE.csv')
    elif config['label'] == 'arousal':
        labels = pd.read_csv(config['data_path'] + 'AROUSAL.csv')
    users = pd.read_csv(config['data_path'] + 'labeled_user_ids.csv')

    data, labels, users = remove_neutrals(data, labels, users, config)

    return data, labels, users

# Suddivisione dei segmenti in train val e test per la task classification
def segment_data_split(data, labels, val_ratio, test_ratio, signals):

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

def subject_data_split(data, labels, users, val_subjects, test_subjects, signals):

    val_segment_ids = users[users['user_id'].isin(val_subjects)]['segment_id'].tolist()
    test_segment_ids = users[users['user_id'].isin(test_subjects)]['segment_id'].tolist()
    all_segment_ids = users['segment_id'].tolist()
    train_segment_ids = list(set(all_segment_ids) - set(val_segment_ids) - set(test_segment_ids))

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
def data_to_tensor(data, labels, num_signals, num_segments, segment_length, label):
    
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
            prepared_labels.append(1)
    prepared_labels = torch.tensor(prepared_labels)
    
    return prepared_data, prepared_labels

# Collate Function per i batch della classificazione
def my_collate_fn(batch):

    return torch.stack([item[0] for item in batch]), torch.tensor([item[1] for item in batch])

# Caricamento dati e creazione dataloaders per classificazione
def get_subject_dataloaders(data, labels, users, config, device, val_subjects, test_subjects):
    # Split dei dati
    train, train_labels, val, val_labels, test, test_labels = subject_data_split(data, labels, users, val_subjects, test_subjects, config['signals'])
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train, train_labels, val, val_labels, test, test_labels, config, device)

    print(len(train_dataloader))
    print(len(val_dataloader))
    print(len(test_dataloader))

    return train_dataloader, val_dataloader, test_dataloader

# Caricamento dati e creazione dataloaders per classificazione
def get_segment_dataloaders(data, labels, config, device):
    # Split dei dati
    train, train_labels, val, val_labels, test, test_labels = segment_data_split(data, labels, config['val_ratio'], config['test_ratio'], config['signals'])
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train, train_labels, val, val_labels, test, test_labels, config, device)

    print(len(train_dataloader))
    print(len(val_dataloader))
    print(len(test_dataloader))

    return train_dataloader, val_dataloader, test_dataloader

def get_dataloaders(train, train_labels, val, val_labels, test, test_labels, config, device):
    num_train_segments = len(train[config['signals'][0]].groupby('segment_id'))
    num_val_segments = len(val[config['signals'][0]].groupby('segment_id'))
    num_test_segments = len(test[config['signals'][0]].groupby('segment_id'))
    # Preparazione dati
    train_data, train_labels = data_to_tensor(train, train_labels, config['num_signals'], num_train_segments, config['segment_length'], config['label'])
    val_data, val_labels = data_to_tensor(val, val_labels, config['num_signals'], num_val_segments, config['segment_length'], config['label'])
    test_data, test_labels = data_to_tensor(test, test_labels, config['num_signals'], num_test_segments, config['segment_length'], config['label'])
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
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, collate_fn=my_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, collate_fn=my_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, collate_fn=my_collate_fn)

    return train_dataloader, val_dataloader, test_dataloader
