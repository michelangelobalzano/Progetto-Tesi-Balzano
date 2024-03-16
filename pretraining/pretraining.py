import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_preparation import load_data, prepare_data
from training_methods import train_model, validate_model, early_stopping
from transformer import Transformer
from utils import data_split

data_path = {'bvp': 'processed_data/bvp.csv', 'eda': 'processed_data/eda.csv', 'hr': 'processed_data/hr.csv'} # Percorsi dei dati
signals = ['bvp', 'eda', 'hr'] # Segnali considerati
sampling_frequency = 4 # Frequenza di campionamento unica in Hz
segment_length = 240 # Lunghezza dei segmenti in time steps
masking_ratio = 0.15 # Rapporto di valori mascherati
lm = 12 # Media della lunghezza delle sequenze mascherate
train_data_ratio = 0.85 # Proporzione dati per la validazione sul totale

# Model data
'''
ce_output_size = 16 # Dimensione output dei channel embeddings
hidden_size = 240
num_heads = 2 # Numero teste auto attenzione multi testa
t_output_size = 240 #
'''
hidden_dim = 60
output_dim = 16
num_heads = 2

# MAIN

# Caricamento e preparazione dei dati
data = load_data(data_path, signals)

# Suddivisione dei dati in train e val
train_data, val_data = data_split(data, train_data_ratio, signals)

# Preparazione dati
train_data = prepare_data(train_data)
val_data = prepare_data(val_data)

'''
# Definizione del modello
model = Transformer(sampling_frequency, ce_output_size, hidden_size, num_heads, t_output_size)
'''
model = Transformer(segment_length, hidden_dim, output_dim, num_heads)

# Definizione dell'ottimizzatore (AdamW)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Definizione dello scheduler di apprendimento
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

# Ciclo di training
num_epochs = 10
val_losses = []  # Lista per memorizzare le loss di validazione per il controllo dell'arresto anticipato
for epoch in range(num_epochs):
    
    print(f'###########################\nEPOCA: {epoch}...')

    # Training
    train_loss = train_model(model, train_data, optimizer)
    # Validation
    val_loss = validate_model(model, val_data)

    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Controllo dell'arresto anticipato
    if early_stopping(val_losses):
        print("Arresto anticipato.")
        break

    # Aggiorna lo scheduler della velocità di apprendimento in base alla loss di validazione
    scheduler.step(val_loss)
