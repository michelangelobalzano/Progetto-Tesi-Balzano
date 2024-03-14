import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import random

# Criterio di minimizzazione dell'errore
criterion = nn.MSELoss()

# Calcolo dell'RMSE
def masked_prediction_loss(outputs, targets, masks):
    loss = 0
    for output, target, mask in zip(outputs, targets, masks):
        loss += torch.sqrt(criterion(output * mask, target * mask))
    return loss / len(outputs)

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

# Generazione della maschera booleana
def generate_mask(segment_length, masking_ratio, lm):
    
    num_zeros = int(segment_length * masking_ratio)    
    num_sequences = num_zeros // lm
    idx = generate_random_start_idx(num_sequences, 0, segment_length - lm - 1, lm)

    mask = np.ones(segment_length, dtype=int)
    for id in idx:
        mask[id:id + lm] = 0
    
    return mask

# Generazione indici casuali per l'azzeramento
def generate_random_start_idx(num_numbers, range_start, range_end, distance):
    
    numbers = []
    
    while len(numbers) < num_numbers:
        # Genera un nuovo numero casuale nell'intervallo
        new_number = random.uniform(range_start, range_end)
        
        # Verifica la distanza tra il nuovo numero e gli altri numeri generati
        if all(abs(new_number - num) >= distance for num in numbers):
            numbers.append(int(new_number))
    
    return numbers

# Metodo che riceve i dati e li restituisce mascherati
def apply_mask_to_tensor(tensor, masking_ratio, lm):
    segment_length = tensor.size(0)
    mask = generate_mask(segment_length, masking_ratio, lm)
    masked_tensor = tensor * mask.unsqueeze(1)
    return masked_tensor

# Funzione per applicare la maschera a ciascun tensore in un dizionario di tensori
def apply_mask(data, masking_ratio, lm):
    masked_data = {}
    
    for signal_name, tensor_list in data.items():

        masked_tensors = []
        for tensor in tensor_list:

            masked_tensor = apply_mask_to_tensor(tensor, masking_ratio, lm)
            masked_tensors.append(masked_tensor)
        masked_data[signal_name] = masked_tensors
    
    return masked_data