from typing import Callable, Dict
from collections import defaultdict

import shap
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoLars, LassoCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE

from modules.utils import splitter


class LeaveOneSubjectOut:
    def split(self, X, y=None, groups=None):
        for id in X.index.unique():
            match = X.index.isin([id])
            yield np.nonzero(~match)[0], np.nonzero(match)[0]

    def get_n_splits(self, X, y=None, groups=None):
        return len(X.index.unique())


def eval_all(models: Dict,
             params: Dict,
             features: pd.DataFrame,
             labels: pd.DataFrame,
             n_repeats: int = 10,
             alpha: int = 0,
             selected: list = None,
             split: bool = False,
             upsample: bool = False) -> Dict:
    results = defaultdict(list)
    for name, clf in models.items():
        results[name].append(eval_model(name, clf, params[name], features, labels,
                             n_repeats=n_repeats, alpha=alpha, selected=selected, split=split, upsample=upsample))
    processed = defaultdict(lambda: defaultdict(list))
    for model_name, result_list in results.items():
        for d in result_list:
            for result_name, result in d.items():
                processed[result_name][model_name].extend(result)
    results = dict(processed)
    return results


def eval_model(name: str,
               clf: Callable,
               params: Dict,
               X: pd.DataFrame,
               y: pd.DataFrame,
               n_repeats: int = 100,
               alpha: int = 0,
               selected: list = None,
               split: bool = False,
               upsample: bool = False) -> Dict:

    results = defaultdict(list)

    pbar = tqdm(range(n_repeats), desc=name)
    for _ in pbar:
        # drop duplicate columns
        XX = X.copy().drop('id', axis=1)
        yy = y.copy().drop('id', axis=1)

        # init
        pred_prob = []
        pred = []
        gt = []

        # LOOCV
        loo = LeaveOneSubjectOut()
        n = loo.get_n_splits(XX)
        for i, (train, test) in enumerate(loo.split(XX, yy)):
            # split
            X_train, y_train = XX.iloc[train].copy(), yy.iloc[train].copy()
            X_test, y_test = XX.iloc[test].copy(), yy.iloc[test].copy()

            # # split features
            # if split:
            #     X_train, y_train = splitter(X_train, y_train)
            #     X_test, y_test = splitter(X_test, y_test)

            # keep select features
            if selected is not None:
                X_train = X_train[selected]
                X_test = X_test[selected]

            y_train = y_train.squeeze()
            y_test = y_test.squeeze()

            # # define search space for Hyperopt
            # search_space = params.copy()
            # search_space['sel__a'] = tune.uniform(1e-5, 1e-1)

            # # define objective function for Hyperopt
            # def objective(space):
            #     scaler = StandardScaler()
            #     opt_gt, opt_p, opt_pred = [], [], []
            #     for train, test in LeaveOneSubjectOut().split(X_train, y_train):
            #         # split
            #         X_train2, y_train2 = X_train.iloc[train].copy(
            #         ), y_train.iloc[train].copy()
            #         X_test2, y_test2 = X_train.iloc[test].copy(
            #         ), y_train.iloc[test].copy()

            #         # preprocess
            #         X_train2.loc[:, :] = scaler.fit_transform(X_train2)
            #         X_test2.loc[:, :] = scaler.transform(X_test2)

            #         # feature selection
            #         lasso = LassoLars(
            #             alpha=space['sel__a'], fit_intercept=False)
            #         selector = SelectFromModel(lasso).fit(X_train2, y_train2)
            #         to_keep = selector.get_support(True)
            #         to_keep = X_train.columns[to_keep]
            #         X_train2 = X_train2[to_keep]
            #         X_test2 = X_test2[to_keep]

            #         # upsample
            #         if upsample:
            #             smote = SMOTE()
            #             X_train2, y_train2 = smote.fit_resample(
            #                 X_train2, y_train2)

            #         # get results
            #         clf_params = {k: v for k,
            #                       v in space.items() if k != 'sel__a'}
            #         opt_clf = clf.set_params(
            #             **clf_params).fit(X_train2, y_train2)
            #         try:
            #             p = opt_clf.decision_function(X_test2)
            #         except:
            #             p = opt_clf.predict_proba(X_test2)[:, 1]
            #         pred = opt_clf.predict(X_test2)
            #         opt_gt.extend(np.array(y_test2).reshape(-1,))
            #         opt_p.extend(np.array(p).reshape(-1,))
            #         opt_pred.extend(np.array(pred).reshape(-1,))

            #     fpr, tpr, _ = roc_curve(opt_gt, opt_p)
            #     auc = auc(fpr, tpr)
            #     acc = balanced_accuracy_score(opt_gt, opt_pred)
            #     return {'auc': auc, 'acc': acc}

            # # optimize
            # tuner = tune.Tuner(
            #     objective,
            #     tune_config=tune.TuneConfig(
            #         mode='max',
            #         metric='acc',
            #         search_alg=HyperOptSearch(),
            #         num_samples=100,
            #     ),
            #     run_config=RunConfig(verbose=0),
            #     param_space=search_space,
            # )
            # tune_results = tuner.fit()
            # best_params = tune_results.get_best_result().config

            # normalize
            scaler = StandardScaler()
            X_train.loc[:, :] = scaler.fit_transform(X_train)
            X_test.loc[:, :] = scaler.transform(X_test)

            # feature selection
            # lasso = LassoLars(alpha=best_params['sel__a'], fit_intercept=False)
            lasso = LassoLars(alpha=alpha, fit_intercept=False)
            selector = SelectFromModel(lasso)
            selector = selector.fit(X_train, y_train)
            to_keep = selector.get_support(True)
            to_keep = X_train.columns[to_keep]

            results['selected'].append(to_keep)
            X_train = X_train[to_keep]
            X_test = X_test[to_keep]

            # upsample
            if upsample:
                smote = SMOTE()
                X_train, y_train = smote.fit_resample(X_train, y_train)

            # fit model
            # clf_params = {k: v for k, v in best_params.items(
            # ) if k != 'scale__scaler' and k != 'sel__a'}
            # clf = clf.set_params(**clf_params)
            clf = clf.fit(X_train, y_train)

            # store results
            try:
                p = clf.decision_function(X_test)
            except:
                p = clf.predict_proba(X_test)[:, 1]
            pred_prob.extend(np.array(p).reshape(-1,))
            pred.extend(clf.predict(X_test).reshape(-1,))
            gt.extend(np.array(y_test).reshape(-1,))

            pbar.set_postfix({'CV': f'{i}/{n}', '# Feats': len(to_keep),
                             't_Acc': np.round(clf.score(X_train, y_train), 2)})

        # compute stats
        fpr, tpr, _ = roc_curve(gt, pred_prob)
        roc_auc = auc(fpr, tpr)
        # acc = accuracy_score(gt, pred)
        acc = balanced_accuracy_score(gt, pred)
        conf_mat = confusion_matrix(gt, pred)
        pbar.set_description(f'{name} [Acc={acc:.2f}, AUC={roc_auc:.2f}]')

        # store results
        results['acc'].append(acc)
        results['auc'].append(roc_auc)
        results['fpr'].append(fpr)
        results['tpr'].append(tpr)
        results['conf_mats'].append(conf_mat)

    return results


# def eval_model(name: str,
#                clf: Callable,
#                params: Dict,
#                X: pd.DataFrame,
#                y: pd.DataFrame,
#                n_repeats: int = 100,
#                alpha: int = 0,
#                selected: list = None,
#                split: bool = False,
#                upsample: bool = False) -> Dict:

#     results = defaultdict(list)

#     pbar = tqdm(range(n_repeats), desc=name)
#     for _ in pbar:
#         # drop duplicate columns
#         idx = np.arange(len(X))
#         np.random.shuffle(idx)
#         XX = X.iloc[idx].drop_duplicates(subset='id').drop('id', axis=1).copy()
#         yy = y.iloc[idx].drop_duplicates(subset='id').drop('id', axis=1).copy()
#         # XX = X.drop('id', axis=1).copy()
#         # yy = y.drop('id', axis=1).copy()
#         if 'id2' in XX:
#             XX = XX.drop('id2', axis=1)

#         # init
#         pred_prob = []
#         pred = []
#         gt = []

#         # LOOCV
#         loo = LeaveOneOut()
#         n = loo.get_n_splits(XX)
#         for i, (train, test) in enumerate(loo.split(XX, yy)):
#             # split
#             X_train, y_train = XX.iloc[train].copy(), yy.iloc[train].copy()
#             X_test, y_test = XX.iloc[test].copy(), yy.iloc[test].copy()

#             # remove dependency
#             idx = X_train.index.isin(
#                 X_test.index.get_level_values('pidn'), level='pidn')

#             X_train = X_train[~idx]
#             y_train = y_train[~idx]

#             # split features
#             if split:
#                 X_train, y_train = splitter(X_train, y_train)
#                 X_test, y_test = splitter(X_test, y_test)

#             # keep select features
#             if selected is not None:
#                 X_train = X_train[selected]
#                 X_test = X_test[selected]

#             y_train = y_train.squeeze()
#             y_test = y_test.squeeze()

#             # param tuning
#             if params:
#                 param_grid = {'clf__'+k: v for k, v in params.items()}
#                 pipe = Pipeline([
#                     ('scale', StandardScaler()),
#                     ('feat_sel', SelectFromModel(
#                         LassoLars(alpha=alpha, fit_intercept=False))),
#                     ('clf', clf),
#                 ])
#                 grid = GridSearchCV(pipe, param_grid, n_jobs=-1)
#                 grid = grid.fit(X_train, y_train)
#                 best_params = grid.best_params_
#                 best_params = {
#                     k.split('__')[-1]: v for k, v in best_params.items()}
#                 clf.set_params(**best_params)

#             # normalize
#             scaler = StandardScaler()
#             X_train.loc[:, :] = scaler.fit_transform(X_train)
#             X_test.loc[:, :] = scaler.transform(X_test)

#             # feature selection
#             if alpha > 0:
#                 selector = SelectFromModel(
#                     LassoLars(alpha=alpha, fit_intercept=False)).fit(X_train, y_train)
#                 to_keep = selector.get_support(True)
#                 to_keep = X_train.columns[to_keep]
#             else:
#                 to_keep = X_train.columns

#             results['selected'].append(to_keep)

#             X_train = X_train[to_keep]
#             X_test = X_test[to_keep]

#             # upsample
#             if upsample:
#                 smote = SMOTE()
#                 X_train, y_train = smote.fit_resample(X_train, y_train)

#             # fit model
#             clf.fit(X_train, y_train)

#             # store results
#             try:
#                 p = clf.decision_function(X_test)
#             except:
#                 p = clf.predict_proba(X_test)[:, 1]
#             pred_prob.extend(np.array(p).reshape(-1,))
#             pred.extend(clf.predict(X_test).reshape(-1,))
#             gt.extend(np.array(y_test).reshape(-1,))

#             pbar.set_postfix({'CV': f'{i}/{n}', '# Feats': len(to_keep),
#                              't_Acc': np.round(clf.score(X_train, y_train), 2)})

#         # compute stats
#         fpr, tpr, _ = roc_curve(gt, pred_prob)
#         roc_auc = auc(fpr, tpr)
#         acc = np.mean(np.array(gt) == np.array(pred))
#         conf_mat = confusion_matrix(gt, pred)
#         pbar.set_description(f'{name} [Acc={acc:.2f}, AUC={roc_auc:.2f}]')

#         # store results
#         results['acc'].append(acc)
#         results['auc'].append(roc_auc)
#         results['fpr'].append(fpr)
#         results['tpr'].append(tpr)
#         results['conf_mats'].append(conf_mat)

#     return results


def model_shap(clf: Callable,
               X: pd.DataFrame,
               y: pd.DataFrame,
               k_fold: int = 5,
               n_repeats: int = 100,
               selected: list = None,
               split: bool = True) -> Dict:

    samples = []
    sl = []
    low = []
    high = []
    for _ in trange(n_repeats, desc=type(clf).__name__):

        # drop duplicates
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        XX = X.iloc[idx].drop_duplicates(subset='id').drop('id', axis=1)
        yy = y.iloc[idx].drop_duplicates(subset='id').drop('id', axis=1)
        feats = XX.columns.copy()

        # CV
        cv = StratifiedKFold(n_splits=k_fold, shuffle=True)
        for train, test in cv.split(XX, yy):
            X_train, y_train = XX.iloc[train].copy(), yy.iloc[train].copy()
            X_test, y_test = XX.iloc[test].copy(), yy.iloc[test].copy()

            if 'id2' in X_train:
                to_keep = []
                targets = X_train['id2']
                for i, r in X_test.iterrows():
                    if r['id2'] not in targets:
                        to_keep.append(i)
                X_train = X_train.drop('id2', axis=1)
                X_test = X_test.drop('id2', axis=1)
                X_test = X_test.loc[to_keep]
                y_test = y_test.loc[to_keep]

            if split:
                X_train, y_train = splitter(X_train, y_train)
                X_test, y_test = splitter(X_test, y_test)
                feats = X_train.columns.copy()

            y_train = y_train.squeeze()
            y_test = y_test.squeeze()

            if selected is not None:
                X_train = X_train[selected]
                X_test = X_test[selected]

            scaler = StandardScaler()
            X_train.loc[:, :] = scaler.fit_transform(X_train)
            X_test.loc[:, :] = scaler.transform(X_test)

            samples.append(X_test.copy())
            sl.append(y_test.copy())
            clf.fit(X_train, y_train)

            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_test)
            low.append(sv[0])
            high.append(sv[1])

    samples = np.vstack(samples)
    sl = np.concatenate(sl)
    low = np.vstack(low)
    high = np.vstack(high)

    return [samples, sl], [low, high]


def count_correct(y_true, y_pred):
    y_true = np.array(y_true).squeeze()
    y_pred = np.array(y_pred).squeeze()
    return np.count_nonzero(y_true == y_pred)
