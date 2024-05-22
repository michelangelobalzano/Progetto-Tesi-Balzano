import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformer import TSTransformerClassifier
import torch
import time
from datetime import datetime
from options import Options
from itertools import combinations
import random

from data_preparation import get_segment_dataloaders, get_subject_dataloaders, load_data
from training_methods import train_model, val_model
from utils import save_session_info

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
        if config['batch_size'] > 64:
                config['batch_size'] = 64

        if config['split_type'] == 'LOSO':
            val_subjects =  [[1]]
            test_subjects = [[3]]
        
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
    
    for i in range(num_run):

        # Definizione dell'ottimizzatore e scheduler
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'], patience=config['patience'], threshold=config['threshold'], threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

        if config['split_type'] != 'segment':
            train_dataloader, val_dataloader, test_dataloader = get_subject_dataloaders(data, labels, users, config, device, val_subjects[i], test_subjects[i])

        # Ciclo di training
        epoch_info = {'train_losses': [], 'val_losses': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1score': []}
        test_info = {}
        start_time = time.time()
        num_lr_reductions = 0
        current_lr = [config['learning_rate']]

        for epoch in range(config['num_epochs']):
            # Training
            train_loss = train_model(model, train_dataloader, optimizer, device, epoch)
            # Validation
            val_loss, accuracy, precision, recall, f1 = val_model(model, val_dataloader, device, epoch=epoch, task='validation')

            print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1Score: {f1}")

            # Salvataggio delle informazioni dell'epoca
            epoch_info['train_losses'].append(train_loss)
            epoch_info['val_losses'].append(val_loss)
            epoch_info['accuracy'].append(accuracy)
            epoch_info['precision'].append(precision)
            epoch_info['recall'].append(recall)
            epoch_info['f1score'].append(f1)

            # Aggiorna lo scheduler della velocità di apprendimento in base alla loss di validazione
            scheduler.step(val_loss)
            if scheduler._last_lr != current_lr:
                print(f'learning rate reduced: {current_lr} -> {scheduler._last_lr}')
                num_lr_reductions += 1
                current_lr = scheduler._last_lr
            if num_lr_reductions > config['max_lr_reductions']:
                print('Early stopped')
                break

            # Ogni tot epoche effettua un salvataggio del modello
            if config['num_epochs_to_save'] is not None:
                if (epoch + 1) % config['num_epochs_to_save'] == 0 and epoch > 0:
                    now = time.time()
                    elapsed_time = now - start_time
                    save_session_info(model_name, config, epoch_info, epoch+1, elapsed_time)

        # Test
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = val_model(model, test_dataloader, device, task='testing')
        test_info['test_loss'] = test_loss
        test_info['accuracy'] = test_accuracy
        test_info['precision'] = test_precision
        test_info['recall'] = test_recall
        test_info['f1score'] = test_f1
        print(f"Test Loss: {test_loss}, Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1Score: {test_f1}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        save_session_info(model_name, config, epoch_info, epoch+1, elapsed_time, test_info)
        
        test_results['accuracy'].append(test_accuracy)
        test_results['precision'].append(test_precision)
        test_results['recall'].append(test_recall)
        test_results['f1'].append(test_f1)

    mean_accuracy = round(sum(test_results['accuracy'])/len(test_results['accuracy']), 4)
    mean_precision = round(sum(test_results['precision'])/len(test_results['precision']), 4)
    mean_recall = round(sum(test_results['recall'])/len(test_results['recall']), 4)
    mean_f1 = round(sum(test_results['f1'])/len(test_results['f1']), 4)

    print(f'FINAL: accuracy: {mean_accuracy}, precision: {mean_precision}, recall: {mean_recall}, f1: {mean_f1}')

args = Options().parse()
config = args.__dict__
config['signals'] = ['BVP', 'EDA', 'HR']
config['num_signals'] = 3
config['segment_length'] = 240
config['num_classes'] = 3
main(config)