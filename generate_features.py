# %% Imports
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

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

# %% Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-a', '--animate_all', action='store_true')
parser.add_argument('--pd_only', action='store_true')
parser.add_argument('--show', action='store_true')
args, _ = parser.parse_known_args()
DEBUG = args.debug
ANIMATE_ALL = args.animate_all
PD_ONLY = args.pd_only
SHOW_FIG = args.show

# %% Load stats and scores
demographic_data = DemographicData(DATA_DIR/'demographics.xlsx')
updrs_data = UPDRSdata(DATA_DIR/'motor_scales_all_2021.03.08.xlsx')
if PD_ONLY:
    pd_patients = demographic_data.get_pidns()
    updrs_data.prune(pd_patients)
    len(pd_patients)

#%% process body videos
pose_files = sorted(list(DATA_DIR.glob('**/*gait*csv')), key=natsort)
poses = []
ids = []
for f in pose_files:
    try:
        pose = PoseSeries(f, min_duration=4, max_duration=8, tolerance=5)
        if pose.seg_cnt == 0:
            continue
        ids.append(pose.id)
        poses.append(pose)
    except Exception as e:
        if DEBUG:
            print(f"Failed to build pose for {f}: {str(e)}.")
ids = np.unique(ids)
print(f"Built {len(poses)} pose series from {len(ids)} patients.")

if ANIMATE_ALL:
    for p in tqdm(poses, desc='Pose Animation'):
        p.animate(visualize=False, show_good_range=True, save_fig=True)

seg_cnts, all_segments = [], []
for p in poses:
    seg_cnts.append(p.seg_cnt)
    all_segments.extend(p.segments)
durations = [(s[1]-s[0])*p.dt for s in all_segments]

print(f'Number of total segments: {np.sum(seg_cnts)}')
print(
    f'Segments per Recording: {np.mean(seg_cnts):.2f} +/- {np.std(seg_cnts):.2f}')
print(
    f'Segment Durations: {np.mean(durations):.2f} +/- {np.std(durations):.2f} s')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(seg_cnts, ax=ax1, discrete=True)
sns.histplot(durations, ax=ax2)
ax1.set_xlabel('# Segments')
ax1.set_title('Good Segments per Gait Recording')
ax2.set_xlabel('Duration [s]')
ax2.set_title('Segment Durations')

plt.savefig(FIGURE_DIR/'body_seg.png')
if SHOW_FIG:
    plt.show()

# %% process hand videos
hand_files = sorted(list(DATA_DIR.glob('**/*fingertap*csv')), key=natsort)
hands = []
ids = []
for f in hand_files:
    try:
        hand = HandSeries(f, min_duration=4, max_duration=8, tolerance=5)
        if hand.seg_cnt == 0:
            continue
        ids.append(hand.id)
        hands.append(hand)
    except Exception as e:
        if DEBUG:
            print(f"Failed to build hand for {f}: {str(e)}.")
ids = np.unique(ids)
print(f"Built {len(hands)} hand series from {len(ids)} patients.")

if ANIMATE_ALL:
    for h in tqdm(hands, desc='Fingertap Animation'):
        h.animate(visualize=False, show_good_range=True, save_fig=True)
        break

seg_cnts, all_segments = [], []
for h in hands:
    seg_cnts.append(h.seg_cnt)
    all_segments.extend(h.segments)
durations = [(s[1]-s[0])*h.dt for s in all_segments]

print(f'Number of total segments: {np.sum(seg_cnts)}')
print(
    f'Segments per Recording: {np.mean(seg_cnts):.2f} +/- {np.std(seg_cnts):.2f}')
print(
    f'Segment Durations: {np.mean(durations):.2f} +/- {np.std(durations):.2f} s')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(seg_cnts, ax=ax1, discrete=True)
sns.histplot(durations, ax=ax2)
ax1.set_xlabel('# Segments')
ax1.set_title('Good Segments per Hand Recording')
ax2.set_xlabel('Duration [s]')
ax2.set_title('Segment Durations')

plt.savefig(FIGURE_DIR/'hand_seg.png')
if SHOW_FIG:
    plt.show()

# %% body features
angle_joints = [
    ('nose', 'middle_shoulder', 'left_shoulder'),
    ('left_elbow', 'left_shoulder', 'left_hip'),
    ('right_elbow', 'right_shoulder', 'right_hip'),
]
distance_joints = [
    ('left_wrist', 'left_shoulder'),
    ('right_wrist', 'right_shoulder'),
    ('left_ankle', 'left_hip'),
    ('right_ankle', 'right_hip'),
]
x_displacement_joints = [
    ('left_ankle', 'right_ankle'),
    ('left_knee', 'right_knee'),
    ('middle_shoulder', 'middle_hip'),
]
y_displacement_joints = [
    ('left_ankle', 'right_ankle'),
    ('left_shoulder', 'right_shoulder'),
    ('left_hip', 'right_hip'),
]

all_feats, all_scores = [], []
feat_cols, score_cols = None, None

for i, p in enumerate(tqdm(poses)):
    info = [p.id]
    scores = updrs_data.get_all_scores(p.id, p.on_med).to_numpy().ravel()
    for j in range(p.seg_cnt):
        feats, col_names = [], []

        for joints in angle_joints:
            base_name = f'{{{joints[0]}, {joints[1]}, {joints[2]}}} (angle)'
            angle = p.get_angle(*joints)[j]
            f, c = get_all_features(angle, base_name, p.dt)
            feats.extend(f)
            col_names.extend(c)

        for joints in distance_joints:
            base_name = f'{{{joints[0]}, {joints[1]}}} (distance)'
            distance = p.get_distance(*joints)[j]
            f, c = get_all_features(distance, base_name, p.dt)
            feats.extend(f)
            col_names.extend(c)

        for joints in x_displacement_joints:
            base_name = f'{{{joints[0]}, {joints[1]}}} (x_displacement)'
            x_displacement = p.get_x_displacement(*joints)[j]
            f, c = get_all_features(x_displacement, base_name, p.dt)
            feats.extend(f)
            col_names.extend(c)

        for joints in y_displacement_joints:
            base_name = f'{{{joints[0]}, {joints[1]}}} (y_displacement)'
            y_displacement = p.get_y_displacement(*joints)[j]
            f, c = get_all_features(y_displacement, base_name, p.dt)
            feats.extend(f)
            col_names.extend(c)

        all_feats.append(np.concatenate([info, feats]))
        all_scores.append(np.concatenate([info, scores]))
        feat_cols = np.concatenate([['pidn'], col_names])
        score_cols = np.concatenate([['pidn'], UPDRSdata.score_names])

feat_df = pd.DataFrame(all_feats, columns=feat_cols).set_index('pidn')
score_df = pd.DataFrame(all_scores, columns=score_cols).set_index('pidn')
feat_df.to_csv(CACHE_DIR/'gait_features.csv')
score_df.to_csv(CACHE_DIR/'gait_scores.csv')

#%% Hand features
distance_joints = [
    ('right_thumb', 'right_index'),
    ('left_thumb', 'left_index'),

    ('right_middle', 'right_wrist'),
    ('left_middle', 'left_wrist'),

    ('right_ring', 'right_wrist'),
    ('left_ring', 'left_wrist'),

    ('right_pinky', 'right_wrist'),
    ('left_pinky', 'left_wrist'),
]

all_feats, all_scores = [], []
feat_cols, score_cols = None, None
is_right = []

for i, h in enumerate(tqdm(hands)):
    info = [h.id]
    scores = updrs_data.get_all_scores(h.id, h.on_med).to_numpy().ravel()
    for j in range(h.seg_cnt):
        feats, col_names = [], []

        wrist_r = h.get_joint('right_wrist', raw=True)[j][:, 1]
        wrist_l = h.get_joint('left_wrist', raw=True)[j][:, 1]
        is_right = wrist_r.mean() < wrist_l.mean()  # note: for y-coordinate, top is 0

        for (joints_r, joints_l) in zip(distance_joints[::2], distance_joints[1::2]):
            distance_r = h.get_distance(*joints_r)[j]
            distance_l = h.get_distance(*joints_l)[j]
            distance = distance_r if is_right else distance_l

            base_name = f"{{{joints_r[0].split('_')[-1]}, {joints_r[1].split('_')[-1]}}} (distance)"
            f, c = get_all_features(distance, base_name, h.dt)
            feats.extend(f)
            col_names.extend(c)

        all_feats.append(np.concatenate([info, feats]))
        all_scores.append(np.concatenate([info, scores]))
        feat_cols = np.concatenate([['pidn'], col_names])
        score_cols = np.concatenate([['pidn'], UPDRSdata.score_names])

feat_df = pd.DataFrame(all_feats, columns=feat_cols).set_index('pidn')
score_df = pd.DataFrame(all_scores, columns=score_cols).set_index('pidn')
feat_df.to_csv(CACHE_DIR/'fingertap_features.csv')
score_df.to_csv(CACHE_DIR/'fingertap_scores.csv')
