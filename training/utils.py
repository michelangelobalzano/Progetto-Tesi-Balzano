import torch
import random
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
def generate_masks(config, device):

    # Creazione maschere impostate a 1
    masks = torch.ones(config['batch_size'], config['num_signals'], config['segment_length'], device=device, dtype=torch.bool)
    # Calcolo del numero degli zeri per maschera
    num_zeros = int(config['segment_length'] * config['masking_ratio'])    
    # Calcolo del numero di sequenze di zeri
    num_sequences = num_zeros // config['lm']

    for b in range(config['batch_size']):
        for f in range(config['num_signals']):
            # Creazione maschera per singolo segmento del batch e per singolo segnale impostata a 1
            mask = torch.ones(config['segment_length'], dtype=bool).to(device)
            # Generazione di indici casuali per gli inizi delle sequenze di zeri
            idx = generate_random_start_idx(num_sequences, 0, config['segment_length'] - config['lm'] - 1, config['lm'])
            # Azzeramento delle sequenze
            for id in idx:
                mask[id:id + config['lm']] = 0

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
def load_pretrained_model(model, config):
    stato_modello = torch.load(config['model_path'] + 'pretrained_' + config['model_to_load'] + '.pth')
    # Eliminazione output layer transformer
    output_layer_keys = ['output_layer.weight', 'output_layer.bias']
    stato_modello = {key: value for key, value in stato_modello.items() if key not in output_layer_keys}
    # Caricamento del modello senza output layer
    model.load_state_dict(stato_modello, strict=False)
    # Freeze o fine-tuning dei pesi
    if config['freeze']:
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False
    return model

# Caricamento parametri modello preaddestrato per classificare
def load_pretrained_model_params(config):
    with open(config['info_path'] + 'pretraining_' + config['model_to_load'] + '.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row_number, row in enumerate(reader):
            if 1 <= row_number <= 6: # Lettura parametri modello
                chiave = row[0]
                if row[0] in ['batch_size', 'd_model', 'dim_feedforward', 'num_heads', 'num_layers']:
                    valore = int(row[1])
                elif row[0] == 'dropout':
                    valore = float(row[1])
                elif row[0] == 'pe_type':
                    valore = row[1]
                config[chiave] = valore
            elif row_number > 6:
                break
    return config

# Salvataggio del modello
def save_model(model, model_path, name):
    # Salvataggio pickle modello
    torch.save(model.state_dict(), model_path + 'pretrained_' + name + '.pth')

# Salvataggio info pretraining
def save_pretraining_info(model_name, 
                          config,
                          epoch_info,
                          num_epochs,
                          elapsed_time):
    with open(config['info_path'] + 'pretraining_' + model_name + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['task', 'pretraining'])
        # Parametri modello
        model_params = ['batch_size', 'd_model', 'dim_feedforward', 'dropout', 'num_heads', 'num_layers', 'pe_type']
        for param in model_params:
            writer.writerow([param, config[param]])
        # Parametri mascheramento
        masking_params = ['masking_ratio', 'lm']
        for param in masking_params:
            writer.writerow([param, config[param]])
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
                             config,
                             epoch_info,
                             num_epochs,
                             elapsed_time,
                             test_info=None):
    with open(config['info_path'] + 'classification_' + model_name + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['task', 'classification'])
        writer.writerow(['label', config['label']])
        # Parametri modello
        model_params = ['batch_size', 'd_model', 'dim_feedforward', 'dropout', 'num_heads', 'num_layers', 'pe_type']
        for param in model_params:
            writer.writerow([param, config[param]])
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
