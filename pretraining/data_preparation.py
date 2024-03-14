import pandas as pd
import numpy as np
import torch

# Caricamento dei dati
def load_data(file_path, signals):

    data = {}
    for signal in signals:
        data[signal] = pd.read_csv(file_path[signal])
    return data

# Preparazione dei dati
# Conversione di ogni segmento di dati in un tensore
def prepare_data(data):
    
    prepared_data = {}
    for signal, df in data.items():
        segments = []
        for segment in df.groupby('segment_id'):
            segments.append(torch.tensor(segment[1].iloc[:, :-1].values, dtype=torch.float32))
        prepared_data[signal] = segments
    return prepared_data

# Generazione della maschera booleana
def generate_mask(segment_length, masking_ratio, lm):
    
    lu = (1 - masking_ratio) / masking_ratio * lm
    
    masks = []
    for _ in range(segment_length):
        mask = np.random.geometric(1 / lu, segment_length[_]) - 1
        mask = np.where(mask >= lm, 1, 0)
        masks.append(mask)
    return masks