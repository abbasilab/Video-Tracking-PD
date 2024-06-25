from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score


class LeaveOneSubjectOut:
    def split(self, X, y=None, groups=None):
        for id in X.index.unique():
            match = X.index.isin([id])
            yield np.nonzero(~match)[0], np.nonzero(match)[0]

    def get_n_splits(self, X, y=None, groups=None):
        return len(X.index.unique())


def _alpha_helper(X, y, alpha, weighted=False):
    pred, gt = [], []
    if weighted:
        sample_weights = []
    clf = LassoLars(alpha=alpha)
    for train, test in LeaveOneSubjectOut().split(X, y):
        X_train, y_train = X.iloc[train], y.iloc[train].squeeze()
        X_test, y_test = X.iloc[test], y.iloc[test].squeeze()

        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(
            X_train), X_train.index, X_train.columns)
        X_test = pd.DataFrame(scaler.transform(
            X_test), X_test.index, X_test.columns)

        clf.fit(X_train, y_train)
        pred.extend(clf.predict(X_test).reshape(-1,))
        gt.extend(np.array(y_test).reshape(-1,))
        if weighted:
            sample_weights.extend([1 / len(test)] * len(test))
    pred_alt = [0 if p < 0.5 else 1 for p in pred]
    # if weighted:
    #     return accuracy_score(gt, pred_alt, sample_weight=sample_weights)
    # return accuracy_score(gt, pred_alt)
    if weighted:
        return f1_score(gt, pred_alt, sample_weight=sample_weights)
    return f1_score(gt, pred_alt)


def alpha_loocv(X, y, alphas, weighted=False):
    with Pool() as pool:
        results = [pool.apply_async(
            _alpha_helper, (X.copy(), y.copy(), a, weighted)) for a in alphas]
        scores = [res.get() for res in results]
    return alphas[np.argmax(scores)]


def _tune_helper(X, y, clf, param, weighted=False):
    pred, gt, pred_prob = [], [], []
    if weighted:
        sample_weights = []
    clf.set_params(**param)
    for train, test in LeaveOneSubjectOut().split(X, y):
        X_train, y_train = X.iloc[train], y.iloc[train].squeeze()
        X_test, y_test = X.iloc[test], y.iloc[test].squeeze()

        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(
            X_train), X_train.index, X_train.columns)
        X_test = pd.DataFrame(scaler.transform(
            X_test), X_test.index, X_test.columns)

        clf = clf.fit(X_train, y_train)
        try:
            p = clf.decision_function(X_test)
        except:
            p = clf.predict_proba(X_test)[:, 1]
        pred_prob.extend(np.array(p).reshape(-1,))
        pred.extend(clf.predict(X_test).reshape(-1,))
        gt.extend(np.array(y_test).reshape(-1,))
        if weighted:
            sample_weights.extend([1 / len(test)] * len(test))
    # acc = accuracy_score(pred, gt)
    # fpr, tpr, _ = roc_curve(gt, pred_prob)
    # roc_auc = auc(fpr, tpr)
    # return acc + roc_auc
    if weighted:
        return f1_score(gt, pred, sample_weight=sample_weights)
    return f1_score(gt, pred)


def tune_loocv(X, y, clf, params, n_iter=32, weighted=False):
    if not params:
        return params

    params = list(ParameterSampler(params, n_iter))

    with Pool() as pool:
        results = [pool.apply_async(
            _tune_helper, (X.copy(), y.copy(), clone(clf), p, weighted)) for p in params]
        scores = [res.get() for res in results]
    return params[np.argmax(scores)]
