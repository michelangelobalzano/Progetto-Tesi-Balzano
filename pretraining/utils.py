import torch
import random
import pandas as pd
import csv

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

# Caricamento del modello
def load_model(model, model_path, model_name, task, info_path=None, old_task=None):
    iperparametri = {}
    if task == 'classification': # Caricamento di un modello preaddestrato per classificazione
        stato_modello = torch.load(model_path + old_task + '_' + model_name + '.pth')
        # Eliminazione output layer
        if old_task == 'pretraining':
            output_layer_keys = ['output_layer.weight', 'output_layer.bias']
            stato_modello = {key: value for key, value in stato_modello.items() if key not in output_layer_keys}
        # Caricamento del modello senza output layer
        model.load_state_dict(stato_modello, strict=False)
    elif task == 'pretraining': # Caricamento per continuare pretraining
        # Caricamento del modello
        model.load_state_dict(torch.load(model_path + old_task + '_' + model_name + '.pth'))
        # Caricamento degli iperparametri
        with open(info_path + 'pretraining_' + model_name + '.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row_number, row in enumerate(reader):
                if 2 <= row_number <= 8:
                    chiave = row[0]
                    valore = int(row[1]) if row[1].isdigit() else float(row[1])
                    iperparametri[chiave] = valore
                elif row_number > 8:
                    break
    
    return model, iperparametri

# Salvataggio del modello
def save_model(model, model_path, name, task):
    # Salvataggio pickle modello
    torch.save(model.state_dict(), model_path + task + '_' + name + '.pth')

def save_session_info(name, info_path, iperparametri, 
                      epoch_info, num_epochs, elapsed_time, 
                      task, label=None, test_info=None, old_session_info=None):
    csv_filename = info_path + task + '_' + name + '.csv'
    # Salvataggio info sessione
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Riscrittura salvataggio info sessione vecchia
        if old_session_info is not None:
            writer.writerows(old_session_info)
            writer.writerow([])
        else:
            # Nome task
            if label is not None:
                writer.writerow([f'TASK: {task} of {label}'])
            else:
                writer.writerow([f'TASK: {task}'])
            # Iperparametri
            writer.writerow(['IPERPARAMETRI:'])
            for key, value in iperparametri.items():
                writer.writerow([key, value])
        # Loss training e validation
        writer.writerow(epoch_info.keys())
        for i in range(num_epochs):
            values = []
            for _, vettore in epoch_info.items():
                values.append(vettore[i])
            writer.writerow(values)
        # Loss testing (solo per classificazione)
        if test_info is not None:
            writer.writerow(test_info.keys())
            writer.writerow(test_info.values())
        # Tempi sessione
        writer.writerow(["Numero epoche", num_epochs])
        writer.writerow(["Tempo tot", elapsed_time])
        writer.writerow(["Tempo per epoca", elapsed_time / num_epochs])

# Determina la riga di partenza per scrivere il salvataggio di un modello pre-caricato
def start_row_idx(name, info_path, task):
    csv_filename = info_path + task + '_' + name + '.csv'
    with open(csv_filename, 'r', newline='') as file:
        reader = csv.reader(file)
        return len(list(reader))
    
# Legge il contenuto di un salvataggio
def read_old_session_info(name, path, task):
    csv_filename = path + task + '_' + name + '.csv'
    with open(csv_filename, 'r', newline='') as file:
        reader = csv.reader(file)
        return list(reader)
