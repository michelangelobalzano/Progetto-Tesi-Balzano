import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformer import TSTransformer
import torch
import optuna
from data_preparation import get_pretraining_dataloaders
from training_methods import train_pretrain_model, validate_pretrain_model
from options import Options

def objective(trial, config, device):
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    d_model = trial.suggest_categorical('d_model', [32, 64, 128, 256])
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512, 1024, 2048])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    num_layers = trial.suggest_int('num_layers', 2, 6)
    pe_type = trial.suggest_categorical('pe_type', ['fixed', 'learnable'])
    
    config['batch_size'] = batch_size
    config['d_model'] = d_model
    config['dim_feedforward'] = dim_feedforward
    config['dropout'] = dropout
    config['num_heads'] = num_heads
    config['num_layers'] = num_layers
    config['pe_type'] = pe_type
    
    train_dataloader, val_dataloader = get_pretraining_dataloaders(config, device)

    # Costruzione del modello con i parametri suggeriti
    model = TSTransformer(config, device)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, 
                                mode='min', 
                                factor=config['factor'], 
                                patience=config['patience'], 
                                threshold=config['threshold'], 
                                threshold_mode='rel', 
                                cooldown=0, 
                                min_lr=0, 
                                eps=1e-8)
    
    # Addestramento del modello per un numero fisso di epoche
    for _ in range(config['num_optimization_epochs']):
        _ = train_pretrain_model(model, train_dataloader, optimizer)
        val_loss = validate_pretrain_model(model, val_dataloader)
        
        scheduler.step(val_loss)
    
    # Restituisci l'ultimo valore di perdita di validazione
    return val_loss

def main (config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, 
                                        config, 
                                        device), 
                                        n_trials=config['num_optimization_trials'])
    print('Migliori iperparametri:', study.best_params)
    print('Miglior valore di perdita:', study.best_value)

args = Options().parse()
config = args.__dict__
config['signals'] = ['BVP', 'EDA', 'HR']
config['num_signals'] = 3
config['segment_length'] = 240
main(config)