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
from utils import save_model, save_partial_model

data_directory = 'processed_data\\' # Percorso dei dati
load_model = False # Se caricare un modello pre-addestrato o classificare direttamente
model_path = 'pretraining\\models\\model_' # Percorso del modello da caricare
info_path = 'training_sessions\\training_info_' # Percorso per esportazione info training
signals = ['BVP', 'EDA', 'HR'] # Segnali considerati
num_signals = len(signals) # Numero dei segnali
segment_length = 240 # Lunghezza dei segmenti in time steps
split_ratios = [70, 15, 15] # Split ratio dei segmenti per la task classification
num_epochs = 15 # Numero epoche task classification
num_classes = 3 # 'negative', 'positive', 'neutral'
num_epochs_to_save = 3 # Ogni tot epoche effettua un salvataggio del modello

label = 'valence' # oppure 'arousal'

# Iperparametri modello
iperparametri = {
    'batch_size' : 256, # Dimensione di un batch di dati (in numero di segmenti)
    'masking_ratio' : 0.15, # Rapporto di valori mascherati
    'lm' : 12, # Lunghezza delle sequenze mascherate all'interno di una singola maschera
    'd_model' : 256, # Dimensione interna del modello
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
data, labels = load_labeled_data(data_directory, signals, label)

# Split dei dati
print('Split dei dati in train/val/test...')
train, train_labels, val, val_labels, test, test_labels = classification_data_split(data, labels, split_ratios, signals)
num_train_segments = len(train[0].groupby('segment_id'))
num_val_segments = len(val[0].groupby('segment_id'))
num_test_segments = len(test[0].groupby('segment_id'))
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
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
test_dataset = TensorDataset(test_data, test_labels)
train_dataloader = DataLoader(train_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=classification_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=classification_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=classification_collate_fn)

# Definizione transformer
print('Creazione del modello...')
model = TSTransformerClassifier(num_signals, segment_length, iperparametri, num_classes, device)
if load_model:
    model.load_state_dict(torch.load('pretraining\\models\\model_03-28_12-47.pth'))
model = model.to(device)

# Definizione dell'ottimizzatore (AdamW)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Definizione dello scheduler di apprendimento
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

# Ciclo di training
val_losses = []
accuracy = []
epoch_info = {'train_losses': [], 'val_losses': [], 'accucary': []}
start_time = time.time()

# Recupero data e ora da usare come nome per il salvataggio del modello
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%m-%d_%H-%M")

for epoch in range(num_epochs):
    
    print(f'\nEPOCA: {epoch + 1}')

    # Training
    train_loss, model = train_classification_model(model, train_dataloader, num_signals, segment_length, iperparametri, optimizer, device)
    # Validation
    val_loss, val_accuracy, model = val_classification_model(model, val_dataloader)

    val_losses.append(val_loss)
    accuracy.append(val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {val_accuracy}")

    # Salvataggio delle informazioni dell'epoca
    epoch_info['train_losses'].append(train_loss)
    epoch_info['val_losses'].append(val_loss)
    epoch_info['accuracy'].append(val_accuracy)

    # Aggiorna lo scheduler della velocità di apprendimento in base alla loss di validazione
    scheduler.step(val_loss)

    # Ogni tre epoche effettua un salvataggio del modello
    if epoch + 1 % num_epochs_to_save == 0 and epoch > 0:
        save_partial_model(model, model_path, formatted_datetime)

end_time = time.time()
elapsed_time = end_time - start_time

# Salvataggio modello e info training
print('Salvataggio informazioni training su file...')
save_model(model, model_path, formatted_datetime, info_path, iperparametri, epoch_info, num_epochs, elapsed_time, write_mode = 'w')

# Stampa grafico delle loss
losses_graph(epoch_info, save_path=f'graphs\\losses_plot_{formatted_datetime}.png')