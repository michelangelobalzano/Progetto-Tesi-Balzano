import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformer import Transformer
from utils import data_split
import torch
from data_preparation import load_data, prepare_data
from training_methods import train_model, validate_model, early_stopping
from utils import CustomDataset

data_path = {'bvp': 'processed_data/bvp.csv', 'eda': 'processed_data/eda.csv', 'hr': 'processed_data/hr.csv'} # Percorsi dei dati
signals = ['bvp', 'eda', 'hr'] # Segnali considerati
sampling_frequency = 4 # Frequenza di campionamento unica in Hz
segment_length = 240 # Lunghezza dei segmenti in time steps
masking_ratio = 0.15 # Rapporto di valori mascherati
lm = 12 # Media della lunghezza delle sequenze mascherate
train_data_ratio = 0.85 # Proporzione dati per la validazione sul totale
batch_size = 32 # Dimensione di un batch di dati (in numero di segmenti)

# Model data
hidden_dim = 60
output_dim = 16
num_heads = 2

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
data = load_data(data_path, signals)

# Suddivisione dei dati in train e val
train, val = data_split(data, train_data_ratio, signals)

# Preparazione dati
train_data = prepare_data(train)
val_data = prepare_data(val)

# Creazione del DataLoader
train_data = CustomDataset(train_data)
val_data = CustomDataset(val_data)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)

# Definizione transformer
model = Transformer(segment_length, hidden_dim, output_dim, num_heads)
model = model.to(device)

# Definizione dell'ottimizzatore (AdamW)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Definizione dello scheduler di apprendimento
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

# Ciclo di training
num_epochs = 10
val_losses = []
for epoch in range(num_epochs):
    
    print(f'EPOCA: {epoch}...')

    # Training
    train_loss, model = train_model(model, train_dataloader, batch_size, optimizer, device)
    # Validation
    val_loss, model = validate_model(model, val_dataloader, batch_size, device)

    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Controllo dell'arresto anticipato
    if early_stopping(val_losses):
        print("Arresto anticipato.")
        break

    # Aggiorna lo scheduler della velocità di apprendimento in base alla loss di validazione
    scheduler.step(val_loss)
