import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_preparation import load_data, prepare_data
from training_methods import train_model, validate_model, early_stopping
from transformer import Transformer
from utils import data_split, generate_mask, apply_mask

# Percorsi dei dati
data_path = {
    'bvp': 'processed_data/bvp.csv',
    'eda': 'processed_data/eda.csv',
    'hr': 'processed_data/hr.csv'
}
# Segnali considerati
signals = ['bvp', 'eda', 'hr']
sampling_frequency = 4 # Hz
# Variabili per il mascheramento
segment_length = 240 # Lunghezza dei segmenti in time steps
masking_ratio = 0.15 # Rapporto di valori mascherati
lm = 12 # Media della lunghezza dei segmenti mascherati in secondi

# Proporzione dati per la validazione sul totale 
train_data_ratio = 0.85










# MAIN

# Caricamento e preparazione dei dati
data = load_data(data_path, signals)

# Suddivisione dei dati in train e val
train_data, val_data = data_split(data, train_data_ratio, signals)

# Preparazione dati
prepared_train_data = prepare_data(train_data)
prepared_val_data = prepare_data(val_data)

# Definizione del modello
ce_output_size = 16
hidden_size = 240
num_heads = 2
t_output_size = 240
model = Transformer(sampling_frequency, ce_output_size, hidden_size, num_heads, t_output_size)

# Definizione dell'ottimizzatore (AdamW)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Definizione dello scheduler di apprendimento
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)









##################################################################################################################
# Ciclo di training

num_epochs = 10
val_losses = []  # Lista per memorizzare le loss di validazione per il controllo dell'arresto anticipato
for epoch in range(num_epochs):
    
    print(f'###########################\nEPOCA: {epoch}...')

    # Mascheramento train data
    masked_train_data, train_masks = apply_mask(prepared_train_data, masking_ratio, lm)
    # Training
    train_loss = train_model(model, prepared_train_data, masked_train_data, train_masks, optimizer)

    # Mascheramento val data
    masked_val_data, val_masks = apply_mask(prepared_val_data, masking_ratio, lm)
    # Validation
    val_loss = validate_model(model, prepared_val_data, masked_val_data, val_masks)

    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Controllo dell'arresto anticipato
    if early_stopping(val_losses):
        print("Arresto anticipato.")
        break

    # Aggiorna lo scheduler della velocità di apprendimento in base alla loss di validazione
    scheduler.step(val_loss)

##################################################################################################################