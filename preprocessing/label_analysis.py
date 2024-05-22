import pandas as pd
import matplotlib.pyplot as plt

from preprocessing_methods import get_users

user_protocol = {
    '1': 2,
    '2': 2,
    '3': 1,
    '4': 1,
    '5': 2,
    '6': 1,
    '7': 1,
    '8': 2,
    '9': 1,
    '10': 2,
    '11': 1,
    '12': 2,
    '13': 1,
    '14': 2,
    '15': 1
}
data_directory = 'data\\WESAD\\'

users = get_users(data_directory)

# Processing delle etichette
valence_labels = pd.DataFrame()
arousal_labels = pd.DataFrame()
for user_id in users:
    valence = []
    arousal = []
    label_df = pd.read_csv(f'{data_directory}{user_id}\\{user_id}_quest.csv', sep=';', header=None)
    valence = label_df.iloc[17:22, 1].tolist()
    arousal = label_df.iloc[17:22, 2].tolist()
    valence = [int(i) for i in valence]
    arousal = [int(i) for i in arousal]

    if user_protocol[user_id] == 1:
        valence_row = {
            'user_id': user_id,
            'neutr': valence[0],
            'divert': valence[1],
            'medi1': valence[2],
            'stress': valence[3],
            'medi2': valence[4]
        }
        arousal_row = {
            'user_id': user_id,
            'neutr': arousal[0],
            'divert': arousal[1],
            'medi1': arousal[2],
            'stress': arousal[3],
            'medi2': arousal[4]
        }
    elif user_protocol[user_id] == 2:
        valence_row = {
            'user_id': user_id,
            'neutr': valence[0],
            'divert': valence[3],
            'medi1': valence[2],
            'stress': valence[1],
            'medi2': valence[4]
        }
        arousal_row = {
            'user_id': user_id,
            'neutr': arousal[0],
            'divert': arousal[3],
            'medi1': arousal[2],
            'stress': arousal[1],
            'medi2': arousal[4]
        }
    valence_row_df = pd.DataFrame([valence_row])
    arousal_row_df = pd.DataFrame([arousal_row])
    valence_labels = pd.concat([valence_labels, valence_row_df])
    arousal_labels = pd.concat([arousal_labels, arousal_row_df])

valence_labels['user_id'] = valence_labels['user_id'].astype(int)
valence_labels = valence_labels.sort_values(by='user_id')
valence_long = valence_labels.melt(id_vars='user_id', var_name='stato_emotivo', value_name='valence')
arousal_labels['user_id'] = arousal_labels['user_id'].astype(int)
arousal_labels = arousal_labels.sort_values(by='user_id')
arousal_long = arousal_labels.melt(id_vars='user_id', var_name='stato_emotivo', value_name='arousal')
user_ids = valence_long['user_id'].unique()
stati_emotivi = valence_long['stato_emotivo'].unique()

# Grafico valenza-attivazione
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
for user_id in user_ids:
    user_data = valence_long[valence_long['user_id'] == user_id]
    axes[0].plot(user_data['stato_emotivo'], user_data['valence'], marker='o', label=user_id)
axes[0].set_ylabel('Valenza')
axes[0].legend(title='Soggetto', bbox_to_anchor=(1.05, 1), loc='upper left')
for user_id in user_ids:
    user_data = arousal_long[arousal_long['user_id'] == user_id]
    axes[1].plot(user_data['stato_emotivo'], user_data['arousal'], marker='o', label=user_id)
axes[1].set_ylabel('Attivazione')
plt.tight_layout()
plt.show()

valence = pd.read_csv('processed_data\\VALENCE.csv', header='infer')
num_pos_val = valence['valence'].value_counts()['positive']
num_neg_val = valence['valence'].value_counts()['negative']
arousal = pd.read_csv('processed_data\\AROUSAL.csv', header='infer')
num_pos_aro = arousal['arousal'].value_counts()['positive']
num_neg_aro = arousal['arousal'].value_counts()['negative']

# Grafico numero segmenti valenza-attivazione
labels = ['Valenza\nNegativa', 'Valenza\nPositiva', 'Attivazione\nNegativa', 'Attivazione\nPositiva']
occorrenze = [num_neg_val, num_pos_val, num_neg_aro, num_pos_aro]
plt.figure(figsize=(10, 6))
plt.bar(labels, occorrenze, color=['red', 'green', 'red', 'green'])
plt.ylabel('Numero di segmenti')
plt.show()