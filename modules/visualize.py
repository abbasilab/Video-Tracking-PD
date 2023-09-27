import itertools
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from modules.utils import splitter, ci, interp_roc


def VisualizeAll(features: pd.DataFrame,
                 labels: pd.DataFrame,
                 results: dict,
                 split: bool = False,
                 out_dir: Path = None) -> None:
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # preprocess
    features, labels = features.drop('id', axis=1), labels.drop('id', axis=1)
    if 'id2' in features:
        features.drop('id2', axis=1)
    if split:
        features, labels = splitter(features, labels)

    # show stats
    df1 = pd.DataFrame.from_dict(
        results['acc']).describe().loc[['mean', 'std']].T
    df2 = pd.DataFrame.from_dict(
        results['auc']).describe().loc[['mean', 'std']].T
    df1.columns = ('acc_mean', 'acc_std')
    df2.columns = ('auc_mean', 'auc_std')
    df = pd.concat((df1, df2), axis=1).round(3)
    df['acc_ci_99_9'] = pd.DataFrame.from_dict(results['acc']).apply(ci)
    print(df)

    # save stats
    if out_dir:
        df.to_excel(out_dir/'stats.xlsx')

    # compute ROC
    fig = plt.figure(figsize=(10, 10))
    ROCplot(results['fpr'], results['tpr'],
            results['auc'], axes=np.array([plt.gca()]))
    if out_dir:
        plt.savefig(out_dir/'roc.png')
        plt.clf()
    else:
        plt.show()

    # plot confusion matrices
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()
    total = np.sum(pd.DataFrame.from_dict(results['conf_mats']), axis=0)
    for i, (k, v) in enumerate(total.iteritems()):
        ax = axes[i]
        v = v / np.sum(v)
        sns.heatmap(v, annot=True, fmt='.2f', ax=ax, square=True, vmin=0, vmax=0.5,
                    cbar=True, cbar_kws=dict(fraction=0.046, pad=0.04), cmap='Blues')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(k)
    fig.tight_layout()
    if out_dir:
        plt.savefig(out_dir/'conf.png')
        plt.clf()
    else:
        plt.show()

    # get LASSO results
    v = list(results['selected'].values())
    v = list(itertools.chain.from_iterable(v))
    size = len(v)
    v = list(itertools.chain.from_iterable(v))
    unique, counts = np.unique(v, return_counts=True)
    freq = counts / size
    df = pd.DataFrame.from_dict({'feat': unique, "freq": freq})
    df = df[df['freq'] > 0.75]

    # Stable groups
    top = df['feat']
    tmp = features[top].copy()
    scaler = StandardScaler()
    tmp.iloc[:, :] = scaler.fit_transform(tmp)
    tmp['PD state'] = labels
    tmp['PD state'] = tmp['PD state'].apply(
        lambda x: 'mild' if x == 0 else 'severe')

    off = tmp[tmp['PD state'] == 'mild'].iloc[:, :-1]
    on = tmp[tmp['PD state'] != 'mild'].iloc[:, :-1]

    df['p'] = ttest_ind(off, on)[1]

    tmp = pd.melt(tmp, id_vars=['PD state'], ignore_index=False)
    tmp['variable'] = tmp['variable'].astype(str)

    plt.figure(figsize=(10, 20))
    ax = plt.gca()
    sns.boxplot(data=tmp, y='variable', x='value', hue='PD state', ax=ax, medianprops={'color': 'r', 'lw': 3}, meanprops={
                'marker': 'D', 'markeredgecolor': 'black'}, showmeans=True)
    # sns.pointplot(data=tmp, y='variable', x='value', hue='PD state', ax=ax, errorbar=('ci', 99.9))
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.get_legend().remove()
    if out_dir:
        plt.savefig(out_dir/'groups.png')
        plt.clf()
    else:
        plt.show()

    # projection stable
    tmp = features[top].copy()
    scaler = StandardScaler()
    tmp.iloc[:, :] = scaler.fit_transform(tmp)
    fig = project(tmp, labels.squeeze())
    if out_dir:
        plt.savefig(out_dir/'proj_top.png')
        plt.clf()
    else:
        plt.show()

    # projection full
    tmp = features.copy()
    scaler = StandardScaler()
    tmp.iloc[:, :] = scaler.fit_transform(tmp)
    fig = project(tmp, labels.squeeze())
    if out_dir:
        plt.savefig(out_dir/'proj_all.png')
        plt.clf()
    else:
        plt.show()

    df = df.round(3)
    if out_dir:
        df.to_excel(out_dir/'selected.xlsx')
    print(df)

    return df


def ROCplot(all_fprs: dict,
            all_tprs: dict,
            all_aucs: dict,
            orient: str = 'h',
            axes: List = None) -> List:
    # process dict
    all_fprs = pd.DataFrame.from_dict(all_fprs)
    all_tprs = pd.DataFrame.from_dict(all_tprs)
    all_aucs = pd.DataFrame.from_dict(all_aucs)

    # interpolation
    all_tprs = all_fprs.combine(all_tprs, lambda x, y: interp_roc(x, y, 100))
    mean_fpr = np.linspace(0, 1, 100)

    # init figure
    if axes is None:
        if orient == 'h':
            fig, axes = plt.subplots(
                1, all_tprs.shape[1], figsize=(10*all_tprs.shape[1], 10))
        else:
            fig, axes = plt.subplots(
                all_tprs.shape[1], 1, figsize=(10, 10*all_tprs.shape[1]))
    axes = axes.ravel()

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", lw=8,
            color="red", label="Chance", alpha=0.8)
    for i in range(all_tprs.shape[1]):

        tprs = all_tprs.iloc[:, i]
        tprs = [np.asarray(tpr) for tpr in tprs]

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0

        auc = all_aucs.iloc[:, i]
        mean_auc = np.mean(auc)
        std_auc = np.std(auc)

        ax.plot(mean_fpr, mean_tpr,
                label=f"AUC ({all_tprs.columns[i]}) = {mean_auc:.2f} "+r'$\pm$'+f" {std_auc:.2f}", lw=8, alpha=0.6)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')

    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    return axes


def project(data: pd.DataFrame,
            labels: pd.DataFrame = None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    reducers = {
        'PCA': PCA(n_components=2),
        't-SNE': TSNE(init='pca', learning_rate='auto'),
        'MDS': MDS(),
        'UMAP': UMAP(),
    }
    for i, (k, v) in enumerate(reducers.items()):
        ax = axes[i]
        ax.set_title(k)
        ax.set_box_aspect(1)
        tmp = pd.DataFrame(v.fit_transform(data))
        tmp.columns = ['d1', 'd2']
        if labels is None:
            sns.scatterplot(data=tmp, x='d1', y='d2', ax=ax)
        else:
            tmp['PD state'] = np.array(labels.apply(
                lambda x: 'mild' if x == 0 else 'severe'))
            sns.scatterplot(data=tmp, x='d1', y='d2', hue='PD state', ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.get_legend().remove()

    return fig


def raincloudplot(data: pd.DataFrame,
                  orient: str = 'h',
                  palette: str = None,
                  color: str = None,
                  ax: plt.Axes = None) -> plt.Axes:
    # setup
    if not ax:
        if orient == 'h':
            plt.figure(figsize=(6, 2.5*data.shape[1]))
        else:
            plt.figure(figsize=(2.5*data.shape[1], 6))
        ax = plt.gca()
    if palette:
        palette = sns.color_palette(palette)
    elif not color:
        palette = sns.color_palette()

    # draw half violinplots
    sns.violinplot(data=data, width=0.5, cut=0, inner=None, orient=orient, color=color,
                   linewidth=2, palette=palette, ax=ax)
    for collection in ax.collections:
        # offset violinplots
        offset = (0, 0.12) if orient == 'h' else (0.12, 0)
        paths = collection.get_paths()
        for path in paths:
            path._vertices -= offset
        # mask halves and add missing border
        x0, y0, width, height = collection.get_paths()[0].get_extents().bounds
        if orient == 'h':
            collection.set_clip_path(plt.Rectangle((x0 - 0.1, y0 - 0.1),
                                                   width + 0.1,
                                                   height/2 + 0.1,
                                                   transform=ax.transData))
            ax.add_line(plt.Line2D((x0 + 0.001, x0 + width - 0.001),
                                   (y0 + height/2, y0 + height/2),
                                   color=collection.get_edgecolor(),
                                   linewidth=2))
        else:
            collection.set_clip_path(plt.Rectangle((x0 - 0.1, y0 - 0.1),
                                                   width/2 + 0.1,
                                                   height + 0.1,
                                                   transform=ax.transData))
            ax.add_line(plt.Line2D((x0 + width/2, x0 + width/2),
                                   (y0 + 0.001, y0 + height - 0.001),
                                   color=collection.get_edgecolor(),
                                   linewidth=2))

    # draw shifted strip plots
    start = len(ax.collections)
    sns.stripplot(data=data, orient=orient, alpha=0.5,
                  color=color, ax=ax, palette=palette)
    for collection in ax.collections[start:]:
        offset = (0, 0.15) if orient == 'h' else (0.15, 0)
        collection.set_offsets(collection.get_offsets() + offset)

    # draw shifted box plots
    start = len(ax.lines)
    sns.boxplot(data=data, orient=orient, showfliers=False,
                width=0.075, linewidth=2, color=color, palette=palette, ax=ax)
    for line in ax.lines[start:]:
        if orient == 'h':
            data = line.get_ydata()
            data = data - 0.05
            line.set_ydata(data)
        else:
            data = line.get_xdata()
            data = data - 0.05
            line.set_xdata(data)
    for patch in ax.patches:
        path = patch.get_path()
        if orient == 'h':
            path._vertices -= (0, 0.05)
        else:
            path._vertices -= (0.05, 0)

    ax.set_axisbelow(True)
    if orient == 'h':
        ax.grid(axis='x')
    else:
        ax.grid(axis='y')

    return ax
