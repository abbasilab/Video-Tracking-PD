from typing import Tuple, List
from pathlib import Path
import re

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.fft import rfft, rfftfreq
from statsmodels.stats.outliers_influence import variance_inflation_factor


def naive_diff(x: np.ndarray, dt: float) -> np.ndarray:
    stencil = np.array([-1, 1]) / dt
    return np.convolve(x, -stencil, mode='valid')


def to_frequency_domain(x: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert time series to frequency-domain."""
    amp, freq = np.abs(rfft(x)), rfftfreq(x.shape[0], dt)
    return amp, freq


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def find_outliers(arr: np.ndarray,
                  m: float = 3,
                  mode: str = 'median') -> Tuple[np.ndarray, np.ndarray]:
    """Return indices of outliers."""
    arr = np.asarray(arr)
    if mode == 'mean':
        d = np.abs(arr - np.mean(arr))
        dev = np.std(arr)
    elif mode == 'median':
        d = np.abs(arr - np.median(arr))
        dev = np.median(d)
    else:
        raise(f'Unsupported mode: {mode}')
    return np.argwhere(d > m * dev)


def natsort(path: Path, _nsre=re.compile("([0-9]+)")):
    return [int(s) if s.isdigit() else s.lower() for s in _nsre.split(path.name)]


def studentization(x: np.ndarray, mode: str = 'mean') -> np.ndarray:
    """Studendize x into z-scores based on mean or median."""
    if mode == 'mean':
        std = np.std(x)
        mean = np.mean(x)
        x = (x - mean)/(std + 1e-8)
    elif mode == 'median':
        mad = stats.median_abs_deviation(x)
        median = np.median(x)
        x = (x - median)/(mad + 1e-8)
    else:
        raise(f'Unsupported mode {mode}.')
    return x


def interp_roc(fpr: np.ndarray, tpr: np.ndarray, N: int = 100) -> pd.Series:
    int_fpr = np.linspace(0.0, 1.0, N)
    int_tpr = [np.interp(int_fpr, xx, yy) for xx, yy in zip(fpr, tpr)]
    for arr in int_tpr:
        arr[0] = 0.0
        arr[-1] = 1.0
    return pd.Series(int_tpr)


def closestDivisors(n: int) -> Tuple[int, int]:
    a = round(np.sqrt(n))
    while n % a > 0:
        a -= 1
    return a, n//a


def VIF(df: pd.DataFrame) -> np.ndarray:
    vif = [variance_inflation_factor(df.values, ix)
           for ix in range(df.shape[1])]
    return np.array(vif)


def VIF_pruning(df: pd.DataFrame, threshold: float = 10.0) -> List:
    vars = list(df.columns)
    while True:
        vif = VIF(df.loc[:, vars])
        idx = np.argmax(vif)
        if vif[idx] <= threshold:
            break
        del vars[idx]
    return vars


def splitter(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_right = np.array([s.count('left') for s in X.columns]) < 2
    all_left = np.array([s.count('right') for s in X.columns]) < 2
    all_right = X.iloc[:, all_right]
    all_left = X.iloc[:, all_left]
    all_left.columns = all_right.columns
    return pd.concat([all_right, all_left]), pd.concat([y, y])


def ci(data, confidence=0.999):
    a = 1.0 * np.array(data)
    n = len(a)
    _, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return h


def se(data):
    a = 1.0 * np.array(data)
    n = len(a)
    _, se = np.mean(a), stats.sem(a)
    return se
