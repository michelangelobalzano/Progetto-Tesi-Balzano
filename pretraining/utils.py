from sklearn.model_selection import train_test_split
import numpy as np
import torch

# Suddivisione dei segmenti in train e val
def data_split(data, split_ratio, signals):
    segment_ids = data['bvp']['segment_id'].unique()
    train_segment_ids, val_segment_ids = train_test_split(segment_ids, train_size=split_ratio, random_state=42)

    train_data = {}
    val_data = {}
    for signal in signals:
        train_data[signal] = data[signal][data[signal]['segment_id'].isin(train_segment_ids)].reset_index(drop=True)
        val_data[signal] = data[signal][data[signal]['segment_id'].isin(val_segment_ids)].reset_index(drop=True)

    return train_data, val_data

# Generazione e applicazione di una stessa maschera ai dati di ogni segnale di un segmento
def apply_mask(segment_data, masking_ratio=0.15, mean_lenght=12):

    segment_length = segment_data['bvp'].size(0)
    print(segment_length)
    mask = torch.ones(segment_length)

    # Probabilità che la sequenza di valori mascherati si stoppi
    p_m = 1 / mean_lenght
    # Probabilità che la sequenza di valori non mascherati si stoppi
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    p = [p_m, p_u]

    state = int(np.random.rand() > masking_ratio)
    for i in range(segment_length):
        mask[i] = state
        if np.random.rand() < p[state]:
            state = 1 - state

    masked_data = {}

    for signal, data in segment_data.items():
        masked_data[signal] = data * mask.unsqueeze(1)

    return masked_data, mask


