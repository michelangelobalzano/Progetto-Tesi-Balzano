import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from transformer import TSTransformer
import torch
import time
from datetime import datetime
from data_preparation import load_unlabeled_data, prepare_data, pretrain_collate_fn, pretrain_data_split
from training_methods import train_pretrain_model, validate_pretrain_model
from graphs_methods import losses_graph
from utils import save_pretraining_info, save_model

# Variabili dipendenti dal preprocessing
signals = ['BVP', 'EDA', 'HR'] # Segnali considerati
num_signals = len(signals) # Numero dei segnali
segment_length = 240 # Lunghezza dei segmenti in time steps

# Percorsi per caricamento e salvataggio dati
data_directory = 'processed_data\\' # Percorso dei dati
info_path = 'sessions\\' # Percorso per esportazione info training
model_path = 'pretraining\\models\\' # Percorso salvataggio modello

# Variabili del training
split_ratios = [85, 15] # Split ratio dei segmenti (train/val)
num_epochs = 100 # Numero epoche task pretraining
num_epochs_to_save = 2 # Ogni tot epoche effettua un salvataggio del modello (oppure None)
improvement_patience = 10 # Numero di epoche di pazienza senza miglioramenti
max_num_lr_reductions = 2 # Numero di riduzioni massimo del learning rate

# Iperparametri nuovo modello (se non se ne carica uno)
model_parameters = {
    'batch_size': 256, # Dimensione di un batch di dati (in numero di segmenti)
    'd_model': 256, # Dimensione interna del modello
    'dropout': 0.1, # Percentuale spegnimento neuroni
    'num_heads': 4, # Numero di teste del modulo di auto-attenzione 
    'num_layers': 3, # Numero di layer dell'encoder
    'pe_type': 'learnable' # Tipo di positional encoder ('learnable' / 'fixed')
}
# Parametri mascheramento
masking_parameters = {
    'masking_ratio': 0.15, # Rapporto di valori mascherati
    'lm': 3, # Lunghezza delle sequenze mascherate all'interno di una singola maschera
}

# MAIN
# Check disponibilità GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"GPU disponibili: {device_count}")
    current_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"GPU in uso: {current_device_name}")
else:
    print("GPU non disponibile. Si sta utilizzando la CPU.")

# Caricamento e preparazione dei dati
print('Caricamento dei dati...')
data = load_unlabeled_data(data_directory, signals)

# Split dei dati
print('Split dei dati in train/val...')
train, val = pretrain_data_split(data, split_ratios, signals)
num_train_segments = len(train[next(iter(train))].groupby('segment_id'))
num_val_segments = len(val[next(iter(val))].groupby('segment_id'))
print('Numero di segmenti per training: ', num_train_segments)
print('Numero di segmenti per validation: ', num_val_segments)

# Preparazione dati
print('Conversione dati in tensori...')
train_data = prepare_data(train, num_signals, num_train_segments, segment_length)
val_data = prepare_data(val, num_signals, num_val_segments, segment_length)

# Creazione del DataLoader
print('Suddivisione dati in batch...')
train_data = train_data.permute(1, 0, 2).to(device)
val_data = val_data.permute(1, 0, 2).to(device)
train_dataset = TensorDataset(train_data)
val_dataset = TensorDataset(val_data)
train_dataloader = DataLoader(train_dataset, batch_size=model_parameters['batch_size'], shuffle=True, drop_last=True, collate_fn=pretrain_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=model_parameters['batch_size'], shuffle=True, drop_last=True, collate_fn=pretrain_collate_fn)

# Definizione transformer
print('Creazione del modello...')
model = TSTransformer(num_signals, segment_length, model_parameters, device)
model = model.to(device)
val_losses = []
epoch_info = {'train_losses': [], 'val_losses': []}
current_datetime = datetime.now()
model_name = current_datetime.strftime("%m-%d_%H-%M")
# Definizione dell'ottimizzatore (AdamW)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
# Definizione dello scheduler di apprendimento
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=improvement_patience, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

# Ciclo di training
start_time = time.time()
num_lr_reductions = 0
best_val_loss = np.inf
epochs_without_improvements = 0
early_stopped = False

for epoch in range(num_epochs):
    
    print(f'\nEPOCA: {epoch + 1}')

    # Training
    train_loss = train_pretrain_model(model, train_dataloader, num_signals, segment_length, 
                                             model_parameters['batch_size'], masking_parameters, 
                                             optimizer, device)
    # Validation
    val_loss = validate_pretrain_model(model, val_dataloader, num_signals, segment_length, 
                                              model_parameters['batch_size'], masking_parameters, 
                                              device)

    val_losses.append(round(val_loss, 2))

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Salvataggio delle informazioni dell'epoca
    epoch_info['train_losses'].append(train_loss)
    epoch_info['val_losses'].append(val_loss)

    # Aggiorna lo scheduler della velocità di apprendimento in base alla loss di validazione
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= improvement_patience:
        num_lr_reductions += 1
        epochs_without_improvement = 0
        if num_lr_reductions > max_num_lr_reductions:
            print('Arresto anticipato')
            break

    # Ogni tot epoche effettua un salvataggio del modello
    if num_epochs_to_save is not None:
        if (epoch + 1) % num_epochs_to_save == 0 and epoch > 0:
            now = time.time()
            elapsed_time = now - start_time
            save_pretraining_info(model_name, 
                                  info_path, 
                                  model_parameters, 
                                  masking_parameters, 
                                  epoch_info, 
                                  epoch+1, 
                                  elapsed_time)
            save_model(model, 
                       model_path, 
                       model_name, 
                       task='pretraining')

end_time = time.time()
elapsed_time = end_time - start_time
save_pretraining_info(model_name, 
                        info_path, 
                        model_parameters, 
                        masking_parameters, 
                        epoch_info, 
                        epoch+1, 
                        elapsed_time)
save_model(model, 
            model_path, 
            model_name, 
            task='pretraining')
    
losses_graph(epoch_info, save_path=f'graphs\\losses_plot_{model_name}.png')