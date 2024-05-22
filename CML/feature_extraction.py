import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

def remove_neutrals(df, label):
    
    df = df[df[label] != 1]
    df[label] = df[label].replace(2, 1)
    return df

def extract_features(segment, segment_id):
    features = {}
    
    features['segment_id'] = segment_id

    columns_to_process = ['bvp', 'eda', 'hr']

    for column in columns_to_process:
        features[f'{column}_mean'] = round(segment[column].mean(), 2)
        features[f'{column}_median'] = round(segment[column].median(), 2)
        features[f'{column}_std'] = round(segment[column].std(), 2)
        features[f'{column}_min'] = round(segment[column].min(), 2)
        features[f'{column}_max'] = round(segment[column].max(), 2)
        features[f'{column}_ptp'] = round(features[f'{column}_max'] - features[f'{column}_min'], 2)
        features[f'{column}_skewness'] = round(segment[column].skew(), 2)
        features[f'{column}_kurtosis'] = round(segment[column].kurtosis(), 2)
        features[f'{column}_n_above_mean'] = (segment[column] > segment[column].mean()).sum()
        features[f'{column}_n_below_mean'] = (segment[column] < segment[column].mean()).sum()
        features[f'{column}_iqr'] = round(segment[column].quantile(0.75) - segment[column].quantile(0.25), 2)
        features[f'{column}_iqr_5_95'] = round(segment[column].quantile(0.95) - segment[column].quantile(0.05), 2)
        features[f'{column}_pct_5'] = round(segment[column].quantile(0.05), 2)
        features[f'{column}_pct_95'] = round(segment[column].quantile(0.95), 2)
        prob = segment[column].value_counts(normalize=True)
        features[f'{column}_entropy'] = round(-np.sum(prob * np.log2(prob)), 2)
        
    return pd.DataFrame(features, index=[0])

def do_pca(num_components):

    features_df = pd.read_csv('classic_ml\\features.csv', header='infer')

    features = features_df.drop(['segment_id', 'valence', 'arousal'], axis=1)
    labels = features_df[['valence', 'arousal']]
    segment_ids = features_df[['segment_id']]
    users = features_df[['user_id']]

    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(features)
    features = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

    result_df = pd.concat([segment_ids, features, labels, users], axis=1)

    return result_df

def process_data():
    
    users = pd.read_csv('processed_data\\labeled_user_ids.csv', header='infer')

    # Lettura dei df
    bvp_df = pd.read_csv('processed_data\\BVP_LABELED.csv')
    eda_df = pd.read_csv('processed_data\\EDA_LABELED.csv')
    hr_df = pd.read_csv('processed_data\\HR_LABELED.csv')
    valence_df = pd.read_csv('processed_data\\VALENCE.csv')
    arousal_df = pd.read_csv('processed_data\\AROUSAL.csv')

    # Mappatura delle etichette in valori interi
    valence_df['valence'] = valence_df['valence'].map(label_map).astype(int)
    arousal_df['arousal'] = arousal_df['arousal'].map(label_map).astype(int)

    # Unione dei singoli df in un unico df
    merged_df = pd.DataFrame()
    merged_df['segment_id'] = bvp_df['segment_id']
    merged_df['bvp'] = bvp_df['bvp'].astype(float)
    merged_df['eda'] = eda_df['eda'].astype(float)
    merged_df['hr'] = hr_df['hr'].astype(float)

    features_df = pd.DataFrame()

    for segment_id, segment_data in tqdm(merged_df.groupby('segment_id'), desc='feature extraction', leave=False):
        segment_features = extract_features(segment_data, segment_id)
        features_df = pd.concat([features_df, segment_features], axis=0, ignore_index=True)

    features_df = pd.merge(features_df, valence_df, on='segment_id')
    features_df = pd.merge(features_df, arousal_df, on='segment_id')

    features = features_df.drop(['segment_id', 'valence', 'arousal'], axis=1)
    labels = features_df[['valence', 'arousal']]
    segment_ids = features_df[['segment_id']]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

    result_df = pd.concat([segment_ids, scaled_features_df, labels], axis=1)
    result_df = pd.merge(result_df, users, on='segment_id')

    result_df.to_csv('classic_ml\\features.csv', index=False)