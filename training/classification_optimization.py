import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformer import TSTransformerClassifier
import torch
import optuna
from datetime import datetime
import csv
from data_preparation import get_classification_dataloaders
from training_methods import train_classification_model, val_classification_model
from options import Options

def objective(trial, config, device, run_name):
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    d_model = trial.suggest_categorical('d_model', [32, 64, 128, 256])
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512, 1024, 2048])
    dropout = round(trial.suggest_float('dropout', 0.1, 0.5), 2)
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
    
    train_dataloader, val_dataloader, _ = get_classification_dataloaders(config, device)

    model = TSTransformerClassifier(config, device)
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

    for _ in range(config['num_optimization_epochs']):
        _ = train_classification_model(model, train_dataloader, optimizer, device)
        val_loss, val_accuracy = val_classification_model(model, val_dataloader, device, task='Validation')
        
        scheduler.step(val_loss)

    with open('sessions\\classification_optimization' + run_name + '.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([trial.number, val_accuracy, batch_size, d_model, dim_feedforward, dropout, num_heads, num_layers, pe_type])

    print(f'trial {trial.number + 1}/{config["num_optimization_trials"]} conclusa con accuracy {val_accuracy}.')
    
    return val_accuracy

def main(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Creazione file salvataggio sessione
    current_datetime = datetime.now()
    run_name = current_datetime.strftime("%m-%d_%H-%M")
    with open('sessions\\classification_optimization' + run_name + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['trial num.', 'accuracy', 'batch_size', 'd_model', 'dim_feedforward', 'dropout', 'num_heads', 'num_layers', 'pe_type'])
    
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Eliminazione stampe studio
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, 
                                        config, 
                                        device,
                                        run_name), 
                                        n_trials=config['num_optimization_trials'])

    # Scrittura migliori iperparametri su file salvataggio
    with open('sessions\\classification_optimization' + run_name + '.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(['Best trial:'])
        writer.writerow([study.best_trial.number,
                         study.best_value, 
                         study.best_params['batch_size'], 
                         study.best_params['d_model'], 
                         study.best_params['dim_feedforward'], 
                         study.best_params['dropout'], 
                         study.best_params['num_heads'], 
                         study.best_params['num_layers'], 
                         study.best_params['pe_type']])

args = Options().parse()
config = args.__dict__
config['signals'] = ['BVP', 'EDA', 'HR']
config['num_signals'] = 3
config['segment_length'] = 240
config['num_classes'] = 3
main(config)