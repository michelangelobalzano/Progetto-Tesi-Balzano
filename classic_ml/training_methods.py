from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import combinations
import numpy as np
from tqdm import tqdm
import random

def LOSO(model, X, y, groups):
    
    logo = LeaveOneGroupOut()
    accuracy = []
    precision = []
    recall = []
    f1 = []

    for train_index, test_index in tqdm(logo.split(X, y, groups), desc='LOSO', leave=False):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        recall.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    mean_accuracy = round(sum(accuracy) / len(accuracy), 4)
    mean_precision = round(sum(precision) / len(precision), 4)
    mean_recall = round(sum(recall) / len(recall), 4)
    mean_f1 = round(sum(f1) / len(f1), 4)

    return mean_accuracy, mean_precision, mean_recall, mean_f1

def KF(model, X, y, num_folds):

    kf = KFold(n_splits=num_folds)
    accuracy = []
    precision = []
    recall = []
    f1 = []

    for train_index, test_index in tqdm(kf.split(X), desc=f'KF({num_folds})', leave=False):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        recall.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    mean_accuracy = round(sum(accuracy) / len(accuracy), 4)
    mean_precision = round(sum(precision) / len(precision), 4)
    mean_recall = round(sum(recall) / len(recall), 4)
    mean_f1 = round(sum(f1) / len(f1), 4)

    return mean_accuracy, mean_precision, mean_recall, mean_f1

def LNSO(model, X, y, groups, num_subject_out):
    
    users = np.unique(groups)
    SO_combinations = list(combinations(users, num_subject_out))
    random_combinations = random.sample(SO_combinations, 10)
    accuracy = []
    precision = []
    recall = []
    f1 = []

    for combination in tqdm(random_combinations, desc=f'L{num_subject_out}SO', leave=False):
        test_indices = np.where(np.isin(groups, combination))
        train_indices = np.delete(np.arange(len(X)), test_indices)

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        recall.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    mean_accuracy = round(sum(accuracy) / len(accuracy), 4)
    mean_precision = round(sum(precision) / len(precision), 4)
    mean_recall = round(sum(recall) / len(recall), 4)
    mean_f1 = round(sum(f1) / len(f1), 4)

    return mean_accuracy, mean_precision, mean_recall, mean_f1