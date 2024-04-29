import numpy as np
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
from utils import load_pretrained_model, load_pretrained_model_params, save_classification_info, save_model

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
model_to_load = '04-27_12-45' # Nome del modello preaddestrato da caricare (oppure None)
freeze = False # True = freeze dei pesi, False = Fine-tuning dei pesi

# Variabili del training
split_ratios = [70, 15, 15] # Split ratio dei segmenti (train/val/test)
num_epochs = 3 # Numero epoche task classification
num_epochs_to_save = 2 # Ogni tot epoche effettua un salvataggio del modello (oppure None)
improvement_patience = 10 # Numero di epoche di pazienza senza miglioramenti
max_num_lr_reductions = 2 # Numero di riduzioni massimo del learning rate
label = 'valence' # Etichetta da classificare ('valence'/'arousal')

# Iperparametri nuovo modello (se non se ne carica uno)
model_parameters = {
    'batch_size': 256, # Dimensione di un batch di dati (in numero di segmenti)
    'd_model': 256, # Dimensione interna del modello
    'dim_feedforward': 512, # Dimensione feedforward network
    'dropout': 0.1, # Percentuale spegnimento neuroni
    'num_heads': 4, # Numero di teste del modulo di auto-attenzione 
    'num_layers': 3, # Numero di layer dell'encoder
    'pe_type': 'learnable' # Tipo di positional encoder ('learnable' / 'fixed')
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
train_dataloader = DataLoader(train_dataset, batch_size=model_parameters['batch_size'], shuffle=True, drop_last=True, collate_fn=classification_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=model_parameters['batch_size'], shuffle=True, drop_last=True, collate_fn=classification_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=model_parameters['batch_size'], shuffle=True, drop_last=True, collate_fn=classification_collate_fn)

# Definizione transformer
print('Creazione del modello...')
model = TSTransformerClassifier(num_signals, 
                                segment_length, 
                                model_parameters, 
                                num_classes, 
                                device)
model = model.to(device)
if model_to_load is not None:
    model_name = model_to_load
    # Caricamento modello
    model = load_pretrained_model(model, 
                                  model_path, 
                                  model_name, 
                                  task='classification', 
                                  freeze=freeze)
    # Caricamento parametri modello ed etichetta da classificare
    model_parameters = load_pretrained_model_params(model_name, info_path)
else:
    current_datetime = datetime.now()
    model_name = current_datetime.strftime("%m-%d_%H-%M")

# Definizione dell'ottimizzatore (AdamW)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Definizione dello scheduler di apprendimento
scheduler = ReduceLROnPlateau(optimizer, 
                              mode='min', 
                              factor=0.3, 
                              patience=10, 
                              threshold=1e-4, 
                              threshold_mode='rel', 
                              cooldown=0, 
                              min_lr=0, 
                              eps=1e-8)

# Ciclo di training
epoch_info = {'train_losses': [], 'val_losses': [], 'accuracy': []}
test_info = {}
start_time = time.time()
num_lr_reductions = 0
best_val_loss = np.inf
epochs_without_improvements = 0

for epoch in range(num_epochs):
    
    print(f'\nEPOCA: {epoch + 1}')

    # Training
    train_loss = train_classification_model(model, 
                                            train_dataloader, 
                                            optimizer, 
                                            device)
    # Validation
    val_loss, val_accuracy = val_classification_model(model, 
                                                      val_dataloader, 
                                                      device, 
                                                      task='validation')

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {val_accuracy}")

    # Salvataggio delle informazioni dell'epoca
    epoch_info['train_losses'].append(train_loss)
    epoch_info['val_losses'].append(val_loss)
    epoch_info['accuracy'].append(val_accuracy)

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
            save_classification_info(model_name, 
                                     info_path, 
                                     model_parameters, 
                                     label, 
                                     epoch_info, epoch+1, 
                                     elapsed_time)
            save_model(model, 
                       model_path, 
                       model_name, 
                       task='classification')

# Test
test_loss, test_accuracy = val_classification_model(model, 
                                                    test_dataloader, 
                                                    device, 
                                                    task='testing')
test_info['test_loss'] = test_loss
test_info['test_accuracy'] = test_accuracy
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

end_time = time.time()
elapsed_time = end_time - start_time
save_classification_info(model_name, 
                         info_path, 
                         model_parameters, 
                         label,
                         epoch_info, 
                         epoch+1, 
                         elapsed_time, 
                         test_info)
save_model(model, 
           model_path, 
           model_name, 
           task='classification')
    
losses_graph(epoch_info, save_path=f'graphs\\losses_plot_{model_name}.png')