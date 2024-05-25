import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from datetime import datetime
from itertools import combinations
import random
import csv

from data_preparation import get_segment_dataloaders, get_subject_dataloaders, load_data
from training_methods import train_model, val_model
from transformer import TSTransformerClassifier
from options import Options
from utils import save_session_info, save_test_info, save_final_info

def main(config):

    # Caricamento dati ed etichette
    data, labels, users = load_data(config)

    test_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    # Utilizzo scheda grafica se disponibile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Definizione del modello
    model = TSTransformerClassifier(config, device)
    current_datetime = datetime.now()
    model_name = current_datetime.strftime("%m-%d_%H-%M")
    model = model.to(device)

    # Creazione dataloaders se lo split è per segmento
    if config['split_type'] == 'segment':

        train_dataloader, val_dataloader, test_dataloader = get_segment_dataloaders(data, labels, config, device)
        num_run = 1
    
    # Determinazione soggetti di training, validazione e test se lo split è per soggetto
    else:

        num_run = 15
        subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        if config['batch_size'] > 16:
                config['batch_size'] = 16

        if config['split_type'] == 'LOSO':
            val_subjects =  [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
            test_subjects = [[2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [1]]
        
        elif config['split_type'] == 'L2SO':
            subject_combinations = list(combinations(subjects, 2))
            val_subjects = random.sample(subject_combinations, num_run)
            test_subjects = []

            for combo1 in val_subjects:
                valid_combinations = [combo2 for combo2 in subject_combinations if not any(subject in combo1 for subject in combo2)]
                combo2 = random.choice(valid_combinations)
                test_subjects.append(combo2)
        
        elif config['split_type'] == 'L3SO':
            subject_combinations = list(combinations(subjects, 3))
            val_subjects = random.sample(subject_combinations, num_run)
            test_subjects = []

            for combo1 in val_subjects:
                valid_combinations = [combo2 for combo2 in subject_combinations if not any(subject in combo1 for subject in combo2)]
                combo2 = random.choice(valid_combinations)
                test_subjects.append(combo2)

    save_session_info(model_name, config)
    for i in range(num_run):
        print(f'iterazione {i + 1}')

        # Definizione dell'ottimizzatore e scheduler
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'], patience=config['patience'], threshold=config['threshold'], threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

        if config['split_type'] != 'segment':
            train_dataloader, val_dataloader, test_dataloader = get_subject_dataloaders(data, labels, users, config, device, val_subjects[i], test_subjects[i])

        # Ciclo di training
        num_lr_reductions = 0
        current_lr = [config['learning_rate']]

        for epoch in range(config['num_epochs']):
            # Training
            _ = train_model(model, train_dataloader, optimizer, device, epoch)
            # Validation
            val_loss, _, _, _, _ = val_model(model, val_dataloader, device, epoch=epoch, task='validation')

            # Aggiorna lo scheduler della velocità di apprendimento in base alla loss di validazione
            scheduler.step(val_loss)
            if scheduler._last_lr != current_lr:
                num_lr_reductions += 1
                current_lr = scheduler._last_lr
            if num_lr_reductions > config['max_lr_reductions']:
                break

        # Test
        _, test_accuracy, test_precision, test_recall, test_f1 = val_model(model, test_dataloader, device, task='testing')
        test_results['accuracy'].append(test_accuracy)
        test_results['precision'].append(test_precision)
        test_results['recall'].append(test_recall)
        test_results['f1'].append(test_f1)
        # Salvataggio risultato test i-esimo su file
        save_test_info(model_name, config, i+1, test_accuracy, test_precision, test_recall, test_f1)

    mean_accuracy = round(sum(test_results['accuracy'])/len(test_results['accuracy']), 4)
    mean_precision = round(sum(test_results['precision'])/len(test_results['precision']), 4)
    mean_recall = round(sum(test_results['recall'])/len(test_results['recall']), 4)
    mean_f1 = round(sum(test_results['f1'])/len(test_results['f1']), 4)

    # Salvataggio risultato finale su file
    save_final_info(model_name, config, mean_accuracy, mean_precision, mean_recall, mean_f1)

args = Options().parse()
config = args.__dict__
config['data_path'] = 'processed_data\\'
config['info_path'] = 'DL\\sessions\\'
config['signals'] = ['BVP', 'EDA', 'HR']
config['num_signals'] = 3
config['segment_length'] = 240
config['num_classes'] = 3
main(config)