# %% Imports
import pickle
import argparse

from ray import tune
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.covariance import LedoitWolf, OAS
from xgboost.sklearn import XGBClassifier
from tqdm import tqdm
from pathlib import Path
from hyperopt import hp
import shap

from modules.model_selection import *
from modules.visualize import *
from modules.utils import *
from modules.feature_extraction import *
from modules.tracking import *

DATA_DIR = Path('./dataset')
FIGURE_DIR = Path('./figures')
CACHE_DIR = Path('./cache')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %% Parse body features
gait_features = pd.read_csv(CACHE_DIR/'gait_features.csv', index_col=0)
gait_scores = pd.read_csv(CACHE_DIR/'gait_scores.csv', index_col=0)

gait_features['id'] = np.arange(len(gait_features))
gait_scores['id'] = np.arange(len(gait_scores))

mask = ~gait_scores.isna().any(axis=1)
mask &= ~gait_features.isna().any(axis=1)
gait_features = gait_features[mask]
gait_scores = gait_scores[mask]

gait_labels = gait_scores[['3.total', 'id']]
low = gait_labels['3.total'] <= 32
gait_labels.loc[low, '3.total'] = 0
gait_labels.loc[~low, '3.total'] = 1
gait_labels = gait_labels.rename(columns={'3.total': 'label'}).astype(int)

tmp = gait_features.reset_index().set_index(['pidn'])
print(f'Samping from {len(np.unique(tmp.index))} valid patient entries:')
print(
    f'    Feature Matrix: {gait_features.shape[0]} x {gait_features.shape[1]}')
print()
print(f'Total Score Labels [Cutoff = 32]:')
print(f'    # Mild: {gait_labels.value_counts(subset="label")[0]}')
print(f'    # Severe: {gait_labels.value_counts(subset="label")[1]}')
print('─' * 25)

# %% Parse hand features
ft_features = pd.read_csv(CACHE_DIR/'fingertap_features.csv', index_col=0)
ft_scores = pd.read_csv(CACHE_DIR/'fingertap_scores.csv', index_col=0)

ft_features['id'] = np.arange(len(ft_features))
ft_scores['id'] = np.arange(len(ft_scores))

mask = ~ft_scores.isna().any(axis=1)
mask &= ~ft_features.isna().any(axis=1)
ft_features = ft_features[mask]
ft_scores = ft_scores[mask]

ft_labels = ft_scores[['3.total', 'id']]
low = ft_labels['3.total'] <= 32
ft_labels.loc[low, '3.total'] = 0
ft_labels.loc[~low, '3.total'] = 1
ft_labels = ft_labels.rename(columns={'3.total': 'label'}).astype(int)

tmp = ft_features.reset_index().set_index(['pidn'])
print(f'Samping from {len(np.unique(tmp.index))} valid patient entries:')
print(f'    Feature Matrix: {ft_features.shape[0]} x {ft_features.shape[1]}')
print()
print(f'Total Score Labels [Cutoff = 32]:')
print(f'    # Mild: {ft_labels.value_counts(subset="label")[0]}')
print(f'    # Severe: {ft_labels.value_counts(subset="label")[1]}')
print('─' * 25)

# %% Parse combined features
int_features = gait_features.reset_index().merge(
    ft_features.drop('id', axis=1).reset_index(), on='pidn')
int_features = int_features.set_index('pidn')

int_scores = []
for id in int_features['id']:
    int_scores.append(gait_scores[gait_scores['id'] == id])
int_scores = pd.concat(int_scores)

int_labels = int_scores[['3.total', 'id']]
low = int_labels['3.total'] <= 32
int_labels.loc[low, '3.total'] = 0
int_labels.loc[~low, '3.total'] = 1
int_labels = int_labels.rename(columns={'3.total': 'label'}).astype(int)

tmp = int_features.reset_index().set_index(['pidn'])
tmp2 = int_features.drop_duplicates(subset="id")
print(f'Samping from {len(np.unique(tmp.index))} valid patient entries:')
print(
    f'    Feature Matrix with Duplicates: {int_features.shape[0]} x {int_features.shape[1]}')
print(
    f'    Feature Matrix without Duplicates: {tmp2.shape[0]} x {tmp2.shape[1]}')
print()

tmp = int_labels.drop_duplicates(subset='id')
print(f'Total Score Labels with Duplicates [Cutoff = 32]:')
print(f'    # Mild: {int_labels.value_counts(subset="label")[0]}')
print(f'    # Severe: {int_labels.value_counts(subset="label")[1]}')
print()
print(f'Total Score Labels without Duplicates [Cutoff = 32]:')
print(f'    # Mild: {tmp.value_counts(subset="label")[0]}')
print(f'    # Severe: {tmp.value_counts(subset="label")[1]}')
print('─' * 25)

# %% eval models
models = {
    'SVM': LinearSVC(max_iter=5000, dual=False, fit_intercept=False),
    'LR': LogisticRegression(solver='liblinear', fit_intercept=False),
    'LDA': LinearDiscriminantAnalysis(solver='lsqr'),
    'RF': RandomForestClassifier(),
    'XGB': XGBClassifier(),
    'KNN': KNeighborsClassifier(),
    'GNB': GaussianNB(),
}

params = defaultdict(dict)
params['SVM'] = {
    'C': tune.loguniform(1e-4, 1e4),
    'penalty': tune.choice(['l1', 'l2']),
}
params['LR'] = {
    'C': tune.loguniform(1e-4, 1e4),
    'penalty': tune.choice(['l1', 'l2']),
}
params['LDA'] = {
    'shrinkage': tune.uniform(0, 1),
}
params['RF'] = {
    'n_estimators': tune.uniform(100, 500),
    'max_depth': tune.uniform(5, 20),
    'min_samples_leaf': tune.uniform(1, 5),
    'min_samples_split': tune.uniform(2, 6),
}
params['XGB'] = {
    'n_estimators': tune.uniform(100, 500),
    'max_depth': tune.uniform(5, 20),
    'learning_rate': tune.uniform(0.01, 0.1),
}
params['KNN'] = {
    'n_neighbors': tune.randint(2, 10),
    'weights': tune.choice(['uniform', 'distance']),
}

# %%
results = eval_all(models, params, gait_features, gait_labels,
                   n_repeats=10, alpha=0.01, split=True, upsample=True)
with open(CACHE_DIR/'gait.dat', 'wb') as f:
    pickle.dump(results, f)

# %%
results = eval_all(models, params, ft_features, ft_labels,
                   n_repeats=10, alpha=0.05, split=False, upsample=True)
with open(CACHE_DIR/'fingertap.dat', 'wb') as f:
    pickle.dump(results, f)

# %%
results = eval_all(models, params, int_features, int_labels,
                   n_repeats=10, alpha=0.001, split=True, upsample=True)
with open(CACHE_DIR/'integrated.dat', 'wb') as f:
    pickle.dump(results, f)
