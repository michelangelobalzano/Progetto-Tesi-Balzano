import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from transformer import TSTransformer
from utils import data_split
import torch
import csv
from datetime import datetime
from data_preparation import load_data, prepare_data, my_collate_fn
from training_methods import train_model, validate_model, early_stopping, try_model
from graphs_methods import losses_graph

data_path = {'bvp': 'processed_data/bvp.csv', 'eda': 'processed_data/eda.csv', 'hr': 'processed_data/hr.csv'} # Percorsi dei dati
signals = ['bvp', 'eda', 'hr'] # Segnali considerati
num_signals = len(signals) # Numero dei segnali
segment_length = 240 # Lunghezza dei segmenti in time steps
train_data_ratio = 0.85 # Proporzione dati per la validazione sul totale

# Iperparametri modello
iperparametri = {
    'batch_size' : 256, # Dimensione di un batch di dati (in numero di segmenti)
    'masking_ratio' : 0.15, # Rapporto di valori mascherati
    'lm' : 12, # Media della lunghezza delle sequenze mascherate
    'd_model' : 128, #
    'dropout' : 0.1, #
    'num_heads' : 2, # 
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
data = load_data(data_path, signals)

# Suddivisione dei dati in train e val
print('Split dei dati in train/val...')
train, val = data_split(data, train_data_ratio, signals)
num_train_segments = len(train['bvp'].groupby('segment_id'))
num_val_segments = len(val['bvp'].groupby('segment_id'))
print('Numero di segmenti per training: ', num_train_segments)
print('Numero di segmenti per validation: ', num_val_segments)

# Preparazione dati
print('Conversione dati in tensori')
train_data = prepare_data(train, num_signals, num_train_segments, segment_length)
val_data = prepare_data(val, num_signals, num_val_segments, segment_length)

# Creazione del DataLoader
print('Suddivisione dati in batch...')
train_data = train_data.permute(1, 0, 2).to(device)
val_data = val_data.permute(1, 0, 2).to(device)
train_dataset = TensorDataset(train_data)
val_dataset = TensorDataset(val_data)
train_dataloader = DataLoader(train_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=my_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=my_collate_fn)

# Definizione transformer
print('Creazione del modello...')
model = TSTransformer(num_signals, segment_length, iperparametri, device)
model = model.to(device)

# Definizione dell'ottimizzatore (AdamW)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Definizione dello scheduler di apprendimento
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

# Ciclo di training
num_epochs = 10
val_losses = []
epoch_info = {'train_losses': [], 'val_losses': []}

for epoch in range(num_epochs):
    
    print(f'\nEPOCA: {epoch + 1}')

    # Training
    train_loss, model = train_model(model, train_dataloader, num_signals, segment_length, iperparametri, optimizer, device)
    # Validation
    val_loss, model = validate_model(model, val_dataloader, num_signals, segment_length, iperparametri, device)

    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Salvataggio delle informazioni dell'epoca
    epoch_info['train_losses'].append(train_loss)
    epoch_info['val_losses'].append(val_loss)

    # Controllo dell'arresto anticipato
    if early_stopping(val_losses):
        print("Arresto anticipato.")
        break

    # Aggiorna lo scheduler della velocità di apprendimento in base alla loss di validazione
    scheduler.step(val_loss)

    try_model(model, val_dataloader, num_signals, segment_length, iperparametri, device)

# Salvataggio delle loss su file
print('Salvataggio informazioni training su file...')
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%m-%d_%H-%M")
csv_filename = f"training_sessions\\training_info_{formatted_datetime}.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    for key, value in iperparametri.items():
        writer.writerow([key, value])
    writer.writerow(["Epoch", "Train Loss", "Val Loss"])
    for epoch, (train_loss, val_loss) in enumerate(zip(epoch_info['train_losses'], epoch_info['val_losses']), start=1):
        writer.writerow([epoch, train_loss, val_loss])

# Stampa grafico delle loss
losses_graph(epoch_info, save_path=f'graphs\\losses_plot_{formatted_datetime}.png')