from sklearn.model_selection import train_test_split
import torch
import random

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
'''
# Generazione di una maschera per ogni segnale e segmento
def generate_masks(batch_size, masking_ratio, lm, num_signals, segment_length, device):

    masks = torch.ones(batch_size, num_signals, segment_length, device=device, dtype=torch.bool)
    
    for b in range(batch_size):
        for f in range(num_signals):
            mask = torch.ones(segment_length, dtype=bool).to(device)
            p_m = 1 / lm
            p_u = p_m * masking_ratio / (1 - masking_ratio)
            p = [p_m, p_u]

            state = int(np.random.rand() > masking_ratio)
            for i in range(segment_length):
                mask[i] = state
                if np.random.rand() < p[state]:
                    state = 1 - state

            masks[b][f] = mask
    
    return masks

'''
# Generazione di una maschera per ogni segnale e segmento
def generate_masks(batch_size, masking_ratio, lm, num_signals, segment_length, device):

    # Creazione maschere impostate a 1
    masks = torch.ones(batch_size, num_signals, segment_length, device=device, dtype=torch.bool)
    # Calcolo del numero degli zeri per maschera
    num_zeros = int(segment_length * masking_ratio)    
    # Calcolo del numero di sequenze di zeri
    num_sequences = num_zeros // lm

    for b in range(batch_size):
        for f in range(num_signals):
            # Creazione maschera per singolo segmento del batch e per singolo segnale impostata a 1
            mask = torch.ones(segment_length, dtype=bool).to(device)
            # Generazione di indici casuali per gli inizi delle sequenze di zeri
            idx = generate_random_start_idx(num_sequences, 0, segment_length - lm - 1, lm)
            # Azzeramento delle sequenze
            for id in idx:
                mask[id:id + lm] = 0

            masks[b][f] = mask

    return masks

# Calcolo di indici random per gli inizi delle sequenze di zeri in una singola maschera
def generate_random_start_idx(num_numbers, range_start, range_end, distance):

    numbers = []

    while len(numbers) < num_numbers:
        # Genera un nuovo numero casuale nell'intervallo
        new_number = random.uniform(range_start, range_end)
        
        # Verifica la distanza tra il nuovo numero e gli altri numeri generati
        if all(abs(new_number - num) >= distance for num in numbers):
            numbers.append(int(new_number))

    return numbers