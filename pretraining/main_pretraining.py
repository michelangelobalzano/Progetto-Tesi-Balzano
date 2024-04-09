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
from utils import load_model
from utils import save_model, save_partial_model

data_directory = 'processed_data\\' # Percorso dei dati
model_path = 'pretraining\\models\\' # Percorso del modello da caricare
info_path = 'sessions\\' # Percorso per esportazione info training
signals = ['BVP', 'EDA', 'HR'] # Segnali considerati
num_signals = len(signals) # Numero dei segnali
segment_length = 240 # Lunghezza dei segmenti in time steps
split_ratios = [85, 15] # Split ratio dei segmenti per la task pretraining
num_epochs = 2 # Numero epoche task pre-training
model_to_load = None # Modello da caricare (oppure None)
num_epochs_to_save = 3 # Ogni tot epoche effettua un salvataggio del modello (oppure None)

# Iperparametri nuovo modello (se non se ne carica uno)
iperparametri = {
    'batch_size' : 256, # Dimensione di un batch di dati (in numero di segmenti)
    'masking_ratio' : 0.15, # Rapporto di valori mascherati
    'lm' : 12, # Lunghezza delle sequenze mascherate all'interno di una singola maschera
    'd_model' : 256, #
    'dropout' : 0.1, #
    'num_heads' : 4, # Numero di teste del modulo di auto-attenzione 
    'num_layers' : 3 # Numero di layer dell'encoder
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
train_dataloader = DataLoader(train_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=pretrain_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=pretrain_collate_fn)

# Definizione transformer
print('Creazione del modello...')
model = TSTransformer(num_signals, segment_length, iperparametri, device)
model = model.to(device)
if model_to_load is not None:
    model, iperparametri = load_model(model, model_path, model_to_load, info_path)
    model_name = model_to_load
    write_mode = 'a'
    new_model = False
else:
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%m-%d_%H-%M")
    model_name = formatted_datetime
    write_mode = 'w'
    new_model = True

# Definizione dell'ottimizzatore (AdamW)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Definizione dello scheduler di apprendimento
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

# Ciclo di training
val_losses = []
epoch_info = {'train_losses': [], 'val_losses': []}
start_time = time.time()

for epoch in range(num_epochs):
    
    print(f'\nEPOCA: {epoch + 1}')

    # Training
    train_loss, model = train_pretrain_model(model, train_dataloader, num_signals, segment_length, iperparametri, optimizer, device)
    # Validation
    val_loss, model = validate_pretrain_model(model, val_dataloader, num_signals, segment_length, iperparametri, device)

    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Salvataggio delle informazioni dell'epoca
    epoch_info['train_losses'].append(train_loss)
    epoch_info['val_losses'].append(val_loss)

    # Aggiorna lo scheduler della velocità di apprendimento in base alla loss di validazione
    scheduler.step(val_loss)

    # Ogni tot epoche effettua un salvataggio del modello
    if num_epochs_to_save is not None:
        if (epoch + 1) % num_epochs_to_save == 0 and epoch > 0:
            save_partial_model(model, model_path, model_name)

end_time = time.time()
elapsed_time = end_time - start_time

# Salvataggio modello e info training
print('Salvataggio informazioni pre-training su file...')

# Salvataggio modello e info training
save_model(model, model_path, model_name, info_path, iperparametri, epoch_info, num_epochs, elapsed_time, task='pre-training', write_mode=write_mode, new_model=new_model)

# Stampa grafico delle loss
losses_graph(epoch_info, save_path=f'graphs\\losses_plot_{model_name}.png')