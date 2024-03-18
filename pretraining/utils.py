from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, index):
        sample = {}
        for signal, segments in self.data.items():
            sample[signal] = segments[index]
        return sample

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
def apply_mask(batch, batch_size, device, masking_ratio=0.15, mean_length=12):

    segment_length = next(iter(batch.values())).size(1)
    masks = torch.ones(batch_size, segment_length, device=device)

    for b in range(batch_size):
        mask = torch.ones(segment_length).to(device)
        p_m = 1 / mean_length
        p_u = p_m * masking_ratio / (1 - masking_ratio)
        p = [p_m, p_u]

        state = int(np.random.rand() > masking_ratio)
        for i in range(segment_length):
            mask[i] = state
            if np.random.rand() < p[state]:
                state = 1 - state

        masks[b] = mask

    masked_batch = {signal: data * masks.unsqueeze(2) for signal, data in batch.items()}
    
    return masked_batch, masks

