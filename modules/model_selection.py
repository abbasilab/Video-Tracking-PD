from typing import Callable, Dict
from collections import defaultdict

import shap
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

from modules.cross_validation import *
from modules.utils import splitter


def eval_all(models: Dict,
             params: Dict,
             features: pd.DataFrame,
             labels: pd.DataFrame,
             alphas: list = None,
             n_repeats: int = 10,
             selected: list = None) -> Dict:
    if alphas is None:
        alphas = compute_alphas(features, labels)
        print(alphas)
    results = defaultdict(list)
    for name, clf in models.items():
        result = eval_model(
            name, clf, params[name], features, labels, alphas, n_repeats, selected)
        results[name].append(result)
    processed = defaultdict(lambda: defaultdict(list))
    for model_name, result_list in results.items():
        for d in result_list:
            for result_name, result in d.items():
                processed[result_name][model_name].extend(result)
    results = dict(processed)
    return results


def compute_alphas(X: pd.DataFrame, y: pd.DataFrame):
    XX, yy = X.drop('id', axis=1), y.drop('id', axis=1)
    if 'id2' in X:
        XX = XX.drop('id2', axis=1)
    loo = LeaveOneSubjectOut()
    n = loo.get_n_splits(XX)
    alphas = []
    pbar = tqdm(loo.split(XX, yy), total=n, desc='LASSO')
    for train, _ in pbar:
        X_train, y_train = XX.iloc[train], yy.iloc[train].squeeze()

        # # split features
        # if True:
        #     X_train, y_train = splitter(X_train, y_train)
            
        alpha = alpha_loocv(X_train, y_train, alphas=np.logspace(-4, 0, 128))
        alphas.append(alpha)
        pbar.set_postfix({'Alpha': alpha})
    return alphas


def eval_model(name: str,
               clf: Callable,
               params: Dict,
               X: pd.DataFrame,
               y: pd.DataFrame,
               alphas: list,
               n_repeats: int = 100,
               selected: list = None) -> Dict:

    results = defaultdict(list)

    pbar = tqdm(range(n_repeats), desc=name)
    for _ in pbar:
        XX, yy = X.drop('id', axis=1), y.drop('id', axis=1)
        if 'id2' in X:
            XX = XX.drop('id2', axis=1)

        # init
        pred_prob, pred, gt = [], [], []

        # LOOCV
        loo = LeaveOneSubjectOut()
        n = loo.get_n_splits(XX)
        for i, (train, test) in enumerate(loo.split(XX, yy)):
            # split
            X_train, y_train = XX.iloc[train], yy.iloc[train]
            X_test, y_test = XX.iloc[test], yy.iloc[test]

            # # split features
            # if True:
            #     X_train, y_train = splitter(X_train, y_train)
            #     X_test, y_test = splitter(X_test, y_test)

            y_train, y_test = y_train.squeeze(), y_test.squeeze()

            # normalize
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), X_train.index, X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), X_test.index, X_test.columns)

            # feature selection
            if selected is not None:
                X_train = X_train[selected]
                X_test = X_test[selected]
            else:
                lasso = LassoLars(alpha=alphas[i])
                selector = SelectFromModel(lasso)
                selector = selector.fit(X_train, y_train)
                to_keep = selector.get_support(True)
                to_keep = X_train.columns[to_keep]

                results['selected'].append(to_keep)
                X_train = X_train[to_keep]
                X_test = X_test[to_keep]

            # hyperparam tuning
            best_params = tune_loocv(X_train, y_train, clf, params)
            clf.set_params(**best_params)

            # get results
            clf = clf.fit(X_train, y_train)
            try:
                p = clf.decision_function(X_test)
            except:
                p = clf.predict_proba(X_test)[:, 1]
            pred_prob.extend(np.array(p).reshape(-1,))
            pred.extend(clf.predict(X_test).reshape(-1,))
            gt.extend(np.array(y_test).reshape(-1,))

            # update info
            pbar.set_postfix({'CV': f'{i}/{n}', '# Feats': len(to_keep),
                             't_Acc': np.round(clf.score(X_train, y_train), 2)})

        # store results
        fpr, tpr, _ = roc_curve(gt, pred_prob)
        roc_auc = auc(fpr, tpr)
        acc = accuracy_score(gt, pred)
        conf_mat = confusion_matrix(gt, pred)
        pbar.set_description(f'{name} [Acc={acc:.2f}, AUC={roc_auc:.2f}]')

        results['acc'].append(acc)
        results['auc'].append(roc_auc)
        results['fpr'].append(fpr)
        results['tpr'].append(tpr)
        results['conf_mats'].append(conf_mat)

    return results


def model_shap(clf: Callable,
               X: pd.DataFrame,
               y: pd.DataFrame,
               selected: list = None) -> Dict:
    XX, yy = X.drop('id', axis=1), y.drop('id', axis=1)
    if 'id2' in X:
        XX = XX.drop('id2', axis=1)

    shap_values, X_test_array = [], []
    loo = LeaveOneSubjectOut()
    for train, test in loo.split(XX, yy):
        X_train, y_train = XX.iloc[train], yy.iloc[train]
        X_test, y_test = XX.iloc[test], yy.iloc[test]
        y_train, y_test = y_train.squeeze(), y_test.squeeze()

        if selected is not None:
            X_train = X_train[selected]
            X_test = X_test[selected]

        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), X_train.index, X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), X_test.index, X_test.columns)

        clf.fit(X_train, y_train)
        explainer = shap.LinearExplainer(clf, X_train)

        shap_values.append(explainer.shap_values(X_test))
        X_test_array.append(X_test)

    shap_values = np.vstack(shap_values)
    X_test_array = np.vstack(X_test_array)

    return shap_values, X_test_array