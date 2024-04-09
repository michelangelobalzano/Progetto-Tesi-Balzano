import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from transformer import TSTransformerClassifier
import torch
import time
from datetime import datetime
from data_preparation import load_labeled_data, prepare_classification_data, classification_collate_fn, classification_data_split
from training_methods import train_classification_model, val_classification_model
from graphs_methods import losses_graph
from utils import load_model, save_model, save_partial_model

# Variabili dipendenti dal preprocessing
signals = ['BVP', 'EDA', 'HR'] # Segnali considerati
num_signals = len(signals) # Numero dei segnali
segment_length = 240 # Lunghezza dei segmenti in time steps
num_classes = 3 # 'negative', 'positive', 'neutral'

# Percorsi per caricamento e salvataggio dati
data_directory = 'processed_data\\' # Percorso dei dati
info_path = 'sessions\\' # Percorso per esportazione info training
model_path = 'pretraining\\models\\' # Percorso del modello da caricare

# Variabili per il caricamento del modello pre-addestrato
model_to_load = '04-09_12-04' # Nome del modello da caricare (oppure None)

# Variabili del training
split_ratios = [70, 15, 15] # Split ratio dei segmenti (train/val/test)
num_epochs = 2 # Numero epoche task classification
num_epochs_to_save = 3 # Ogni tot epoche effettua un salvataggio del modello (oppure None)
label = 'arousal' # Etichetta da classificare ('valence'/'arousal')

# Iperparametri modello (se non se ne carica uno)
iperparametri = {
    'batch_size' : 256, # Dimensione di un batch di dati (in numero di segmenti)
    'masking_ratio' : 0.15, # Rapporto di valori mascherati
    'lm' : 12, # Lunghezza delle sequenze mascherate all'interno di una singola maschera
    'd_model' : 256, # Dimensione interna del modello
    'dropout' : 0.1, # Percentuale spegnimento neuroni
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
data, labels = load_labeled_data(data_directory, signals, label)

# Split dei dati
print('Split dei dati in train/val/test...')
train, train_labels, val, val_labels, test, test_labels = classification_data_split(data, labels, split_ratios, signals)
num_train_segments = len(train[signals[0]].groupby('segment_id'))
num_val_segments = len(val[signals[0]].groupby('segment_id'))
num_test_segments = len(test[signals[0]].groupby('segment_id'))
print('Numero di segmenti per training: ', num_train_segments)
print('Numero di segmenti per validation: ', num_val_segments)
print('Numero di segmenti per test: ', num_test_segments)

# Preparazione dati
print('Conversione dati in tensori...')
train_data, train_labels = prepare_classification_data(train, train_labels, num_signals, num_train_segments, segment_length, label)
val_data, val_labels = prepare_classification_data(val, val_labels, num_signals, num_val_segments, segment_length, label)
test_data, test_labels = prepare_classification_data(test, test_labels, num_signals, num_test_segments, segment_length, label)

# Creazione del DataLoader
print('Suddivisione dati in batch...')
train_data = train_data.permute(1, 0, 2).to(device)
val_data = val_data.permute(1, 0, 2).to(device)
test_data = test_data.permute(1, 0, 2).to(device)
train_labels = train_labels.to(device)
val_labels = val_labels.to(device)
test_labels = test_labels.to(device)
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
test_dataset = TensorDataset(test_data, test_labels)
train_dataloader = DataLoader(train_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=classification_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=classification_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=classification_collate_fn)

# Definizione transformer
print('Creazione del modello...')
model = TSTransformerClassifier(num_signals, segment_length, iperparametri, num_classes, device)
if model_to_load is not None:
    model, _ = load_model(model, model_path, model_to_load, task='classification')
    model_name = model_to_load
else:
    current_datetime = datetime.now()
    model_name = current_datetime.strftime("%m-%d_%H-%M")
model = model.to(device)

# Definizione dell'ottimizzatore (AdamW)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Definizione dello scheduler di apprendimento
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

# Ciclo di training
val_losses = []
accuracy = []
epoch_info = {'train_losses': [], 'val_losses': [], 'accuracy': []}
test_info = {}
start_time = time.time()

for epoch in range(num_epochs):
    
    print(f'\nEPOCA: {epoch + 1}')

    # Training
    train_loss, model = train_classification_model(model, train_dataloader, optimizer, device)
    # Validation
    val_loss, val_accuracy, model = val_classification_model(model, val_dataloader, device, task='Validation')

    val_losses.append(val_loss)
    accuracy.append(val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {val_accuracy}")

    # Salvataggio delle informazioni dell'epoca
    epoch_info['train_losses'].append(train_loss)
    epoch_info['val_losses'].append(val_loss)
    epoch_info['accuracy'].append(val_accuracy)

    # Aggiorna lo scheduler della velocità di apprendimento in base alla loss di validazione
    scheduler.step(val_loss)

    # Ogni tot epoche effettua un salvataggio del modello
    if num_epochs_to_save is not None:
        if epoch + 1 % num_epochs_to_save == 0 and epoch > 0:
            save_partial_model(model, model_path, model_name)

# Test
test_loss, test_accuracy, model = val_classification_model(model, test_dataloader, device, task='Test')
test_info['test_loss'] = test_loss
test_info['test_accuracy'] = test_accuracy
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

end_time = time.time()
elapsed_time = end_time - start_time

# Salvataggio modello e info training
print('Salvataggio informazioni training su file...')
save_model(model, model_path, model_name, info_path, iperparametri, epoch_info, num_epochs, elapsed_time, task='classification', label=label, write_mode = 'w', test_info=test_info)

# Stampa grafico delle loss
losses_graph(epoch_info, save_path=f'graphs\\losses_plot_{model_name}.png')