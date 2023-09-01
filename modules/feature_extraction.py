from typing import Dict, Tuple, List

import numpy as np
import scipy.stats as stats
from scipy.signal import welch
from scipy.spatial import ConvexHull

from modules.utils import naive_diff


def get_all_features(x: np.ndarray, name: str, dt: float, abs: bool = False) -> Tuple[List, List]:
    features, names = [], []

    handle = name + ' - '
    temporal = get_temporal_metrics(x, abs)
    spectral = get_spectral_metrics(x, dt)
    for k, v in temporal.items():
        features.append(v)
        names.append(handle + '|temporal| ' + k)
    for k, v in spectral.items():
        features.append(v)
        names.append(handle + '|spectral| ' + k)

    x = naive_diff(x, dt)
    handle = name + ' {diff} - '
    temporal = get_temporal_metrics(x, abs)
    spectral = get_spectral_metrics(x, dt)
    for k, v in temporal.items():
        features.append(v)
        names.append(handle + '|temporal| ' + k)
    for k, v in spectral.items():
        features.append(v)
        names.append(handle + '|spectral| ' + k)

    return features, names


def get_temporal_metrics(x: np.ndarray, abs: bool = False) -> Dict:
    """Return dict of various temporal metrics of x."""
    if abs:
        x = np.absolute(x)
        return {
            'abs_mean': np.mean(x),
            'abs_median': np.median(x),
            'abs_std': np.std(x),
            'abs_mad': stats.median_abs_deviation(x),
            'abs_skew': stats.skew(x),
            'abs_kurtosis': stats.kurtosis(x),
            # 'abs_root_mean_square': np.sqrt(np.mean(x**2)),
        }
    else: 
        return {
            'mean': np.mean(x),
            'median': np.median(x),
            'std': np.std(x),
            'mad': stats.median_abs_deviation(x),
            'skew': stats.skew(x),
            'kurtosis': stats.kurtosis(x),
            # 'root_mean_square': np.sqrt(np.mean(x**2)),
        }


def get_spectral_metrics(x: np.ndarray, dt: float = 1/30, threshold=1e-8) -> Dict:
    """Return dict of various spectral metrics of x."""
    freq, pwd = welch(x, 1/dt, nperseg=min(256, len(x)))
    total_pwd = np.sum(pwd)
    half_pwr_freq = 0
    partial_sum = 0
    for p, z in zip(pwd, freq):
        partial_sum += p
        if partial_sum * 2 >= total_pwd:
            half_pwr_freq = z
            break
    pdf = pwd / total_pwd
    entropy = stats.entropy(pdf)
    rel_pwr = []
    bands = [(0.5, 1), (1, 2), (2, 4), (4, 6), (6, 128)]
    for (l, h) in bands:
        idx = np.argwhere(np.logical_and(freq > l, freq <= h))
        pwr = np.sum(pdf[idx])
        pwr = 0 if pwr < threshold else pwr
        rel_pwr.append(pwr)
    metrics = {
        'entropy': entropy,
        'half_pwr_freq': half_pwr_freq,
    }
    for i, rp in enumerate(rel_pwr):
        band = bands[i]
        metrics[f'rel_pwr_{band[0]}_to_{band[1]}'] = rp
    return metrics


def get_motion_area(X: List[np.ndarray]) -> float:
    """Return the area of motion of coordinates."""
    X = np.concatenate(X)
    return ConvexHull(X).volume
