import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from transformer import TSTransformerClassifier
import torch
import csv
import time
from datetime import datetime
from data_preparation import load_labeled_data, prepare_data, prepare_classification_data, my_classification_collate_fn, classification_data_split
from training_methods import train_pretrain_model, validate_pretrain_model#, early_stopping
from graphs_methods import losses_graph

data_directory = 'processed_data\\' # Percorso dei dati
model_path = 'pretraining\\models\\model_'
signals = ['BVP', 'EDA', 'HR'] # Segnali considerati
num_signals = len(signals) # Numero dei segnali
segment_length = 240 # Lunghezza dei segmenti in time steps
split_ratios = [70, 15, 15] # Split ratio dei segmenti per la task classification
num_epochs = 15 # Numero epoche task classification

label = 'valence' # oppure 'arousal'

# Iperparametri modello
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
data, labels = load_labeled_data(data_directory, signals, 'classification')

# Split dei dati
print('Split dei dati in train/val/test...')
train, train_labels, val, val_labels, test, test_labels = classification_data_split(data, labels, split_ratios, signals)
num_train_segments = len(train['BVP'].groupby('segment_id'))
num_val_segments = len(val['BVP'].groupby('segment_id'))
num_test_segments = len(test['BVP'].groupby('segment_id'))
print('Numero di segmenti per training: ', num_train_segments)
print('Numero di segmenti per validation: ', num_val_segments)
print('Numero di segmenti per test: ', num_test_segments)

# Preparazione dati
print('Conversione dati in tensori...')
train_data, train_valence, train_arousal = prepare_classification_data(train, train_labels, num_signals, num_train_segments, segment_length)
val_data, val_valence, val_arousal = prepare_classification_data(val, val_labels, num_signals, num_val_segments, segment_length)
test_data, test_valence, test_arousal = prepare_classification_data(test, test_labels, num_signals, num_test_segments, segment_length)

# Creazione del DataLoader
print('Suddivisione dati in batch...')
train_data = train_data.permute(1, 0, 2).to(device)
val_data = val_data.permute(1, 0, 2).to(device)
test_data = test_data.permute(1, 0, 2).to(device)
train_dataset = TensorDataset(train_data, train_valence, train_arousal)
val_dataset = TensorDataset(val_data, val_valence, val_arousal)
test_dataset = TensorDataset(test_data, test_valence, test_arousal)
train_dataloader = DataLoader(train_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=my_classification_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=my_classification_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=my_classification_collate_fn)

# Definizione transformer
print('Creazione del modello...')
model = TSTransformerClassifier(num_signals, segment_length, iperparametri, 3, device)
model.load_state_dict(torch.load('pretraining\\models\\model_03-28_12-47.pth'))
model = model.to(device)

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

    '''# Controllo dell'arresto anticipato
    if early_stopping(val_losses):
        print("Arresto anticipato.")
        break'''

    # Aggiorna lo scheduler della velocità di apprendimento in base alla loss di validazione
    scheduler.step(val_loss)

    #try_model(model, val_dataloader, num_signals, segment_length, iperparametri, device)

end_time = time.time()
elapsed_time = end_time - start_time

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%m-%d_%H-%M")
torch.save(model.state_dict(), model_path + formatted_datetime + '.pth') # Salvataggio del modello pre-addestrato

# Salvataggio delle loss su file
print('Salvataggio informazioni pre-training su file...')
csv_filename = f"training_sessions\\training_info_{formatted_datetime}.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    for key, value in iperparametri.items():
        writer.writerow([key, value])
    writer.writerow(["Epoch", "Train Loss", "Val Loss"])
    for epoch, (train_loss, val_loss) in enumerate(zip(epoch_info['train_losses'], epoch_info['val_losses']), start=1):
        writer.writerow([epoch, train_loss, val_loss])
    writer.writerow(["Numero epoche", num_epochs])
    writer.writerow(["Tempo tot", elapsed_time])
    writer.writerow(["Tempo per epoca", elapsed_time / num_epochs])


# Stampa grafico delle loss
losses_graph(epoch_info, save_path=f'graphs\\losses_plot_{formatted_datetime}.png')