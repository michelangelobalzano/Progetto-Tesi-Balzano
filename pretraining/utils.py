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

# Caricamento modello preaddestrato per continuare pretraining o per classificare
def load_pretrained_model(model, model_path, model_name, task, freeze=False):
    if task == 'pretraining': # Si continua a effettuare pretraining
        # Caricamento del modello
        model.load_state_dict(torch.load(model_path + 'pretraining_' + model_name + '.pth'))
    elif task == 'classification': # Si usa il modello preaddestrato per classificare
        stato_modello = torch.load(model_path + 'pretraining_' + model_name + '.pth')
        # Eliminazione output layer transformer
        output_layer_keys = ['output_layer.weight', 'output_layer.bias']
        stato_modello = {key: value for key, value in stato_modello.items() if key not in output_layer_keys}
        # Caricamento del modello senza output layer
        model.load_state_dict(stato_modello, strict=False)
        # Freeze o fine-tuning dei pesi
        if freeze:
            for name, param in model.named_parameters():
                if name.startswith('output_layer'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    return model

# Caricamento parametri modello preaddestrato per classificare
def load_pretrained_model_params(model_name, info_path):
    model_parameters = {}
    with open(info_path + 'pretraining_' + model_name + '.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row_number, row in enumerate(reader):
            if 1 <= row_number <= 6: # Lettura parametri modello
                chiave = row[0]
                if row[0] in ['batch_size', 'd_model', 'num_heads', 'num_layers']:
                    valore = int(row[1])
                elif row[0] == 'dropout':
                    valore = float(row[1])
                elif row[0] == 'pe_type':
                    valore = row[1]
                model_parameters[chiave] = valore
            elif row_number > 6:
                break
    return model_parameters

# Salvataggio del modello
def save_model(model, model_path, name, task):
    # Salvataggio pickle modello
    torch.save(model.state_dict(), model_path + task + '_' + name + '.pth')

# Salvataggio info pretraining
def save_pretraining_info(model_name, 
                          info_path, 
                          model_parameters, 
                          masking_parameters,
                          epoch_info,
                          num_epochs,
                          elapsed_time):
    with open(info_path + 'pretraining_' + model_name + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['task', 'pretraining'])
        # Parametri modello
        for key, value in model_parameters.items():
            writer.writerow([key, value])
        # Parametri mascheramento
        for key, value in masking_parameters.items():
            writer.writerow([key, value])
        writer.writerow(['num. epochs', num_epochs])
        writer.writerow(['elapsed time', elapsed_time])
        writer.writerow(['time per epoch', elapsed_time / num_epochs])
        # Loss training e validation
        writer.writerow(epoch_info.keys())
        for i in range(num_epochs):
            values = []
            for _, vettore in epoch_info.items():
                values.append(vettore[i])
            writer.writerow(values)

# Salvataggio info classificazione
def save_classification_info(model_name, 
                             info_path, 
                             model_parameters,
                             label,
                             epoch_info,
                             num_epochs,
                             elapsed_time,
                             test_info=None):
    with open(info_path + 'classification_' + model_name + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['task', 'classification'])
        writer.writerow(['label', label])
        # Parametri modello
        for key, value in model_parameters.items():
            writer.writerow([key, value])
        writer.writerow(['num. epochs', num_epochs])
        writer.writerow(['elapsed time', elapsed_time])
        writer.writerow(['time per epoch', elapsed_time / num_epochs])
        # Loss training e validation
        writer.writerow(epoch_info.keys())
        for i in range(num_epochs):
            values = []
            for _, vettore in epoch_info.items():
                values.append(vettore[i])
            writer.writerow(values)
        if test_info is not None:
            writer.writerow(test_info.keys())
            writer.writerow(test_info.values())
        else: 
            writer.writerow(['early stopped'])
