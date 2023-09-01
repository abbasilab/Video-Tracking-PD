from typing import List, Union
from collections import namedtuple
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

matplotlib.rc('animation', html='jshtml')
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**30


class UPDRSdata:
    """Class for tracking updrs scores per patient."""

    score_names = ['3.1', '3.2', '3.3.neck',
                   '3.3.rue', '3.3.rle', '3.3.lue',
                   '3.3.lle', '3.4.rue', '3.4.lue',
                   '3.5.rue', '3.5.lue',    '3.6.rue',
                   '3.6.lue', '3.7.rle', '3.7.lle',
                   '3.8.rle',    '3.8.lle', '3.9',
                   '3.10',    '3.11', '3.12',
                   '3.13', '3.14', '3.15.rue',
                   '3.15.lue',    '3.16.rue', '3.16.lue',
                   '3.17.jaw', '3.17.rue', '3.17.lue',
                   '3.17.rle', '3.17.lle', '3.18',
                   '3.total']

    def __init__(self, path_to_excel: str) -> None:
        """Construct dataframe holding only updrs.3 scores."""
        df = pd.read_excel(path_to_excel, sheet_name='Sheet1')
        df.drop('virtual', axis=1, inplace=True)
        df.drop(df.filter(regex='updrs.1|updrs.part.1'), axis=1, inplace=True)
        df.drop(df.filter(regex='updrs.2|updrs.part.2'), axis=1, inplace=True)
        df.drop(df.filter(regex='updrs.4|updrs.part.4'), axis=1, inplace=True)
        df.drop(df.filter(regex='psprs|sara|umsars'), axis=1, inplace=True)

        tmp = df.copy()
        tmp.iloc[:, 3:37] = tmp.iloc[:, 37:71]
        tmp.iloc[:, 37:71] = np.NaN
        idx = df.loc[(df['pd.meds'] == 0) & (df['off.exam'] != 0)].index
        df.loc[idx] = tmp.loc[idx]

        tmp = df.copy()
        tmp.iloc[:, 3:37] = np.NaN
        idx = df.loc[(df['pd.meds'] == 1) & (df['off.exam'] == 0)].index
        df.loc[idx] = tmp.loc[idx]

        df[df < 0] = 0
        df.drop(columns=['pd.meds', 'off.exam'], axis=1, inplace=True)

        self.df = df

    def get_score(self, pidn: int, on_med: bool, score_name: str) -> int:
        name = 'updrs.on.' if on_med else 'updrs.off.'
        name += score_name
        return int(self.df.loc[self.df['pidn'] == pidn, name].values[0])

    def get_all_scores(self, pidn: int, on_med: bool) -> pd.Series:
        if on_med:
            return self.df.loc[self.df['pidn'] == pidn].iloc[:, 35:69]
        else:
            return self.df.loc[self.df['pidn'] == pidn].iloc[:, 1:35]

    def prune(self, keep: list):
        idx = []
        for i, r in self.df.iterrows():
            if r['pidn'] in keep:
                idx.append(i)
        self.df = self.df.iloc[idx]

    def __repr__(self) -> str:
        return repr(self.df)


class DemographicData:
    """Class for tracking metadata per patient."""

    def __init__(self, path_to_excel: str) -> None:
        """Construct dataframe holding only relavent information."""
        self.original = pd.read_excel(path_to_excel,
                                      sheet_name='demographic data',
                                      index_col='pidn')
        self.df = self.original.copy()
        sex = pd.get_dummies(self.df['sex'])
        sex.columns = ['is.female', 'is.male']
        sex.drop('is.male', axis=1, inplace=True)
        handedness = pd.get_dummies(self.df['handedness'])
        handedness[handedness['Ambidextrous'] == 1] += 1
        handedness.drop('Ambidextrous', axis=1, inplace=True)
        handedness.columns = ['left.handed', 'right.handed']
        self.df = self.df[self.df['dx'] == 'pd']
        self.df = self.df[['age', 'dis.dur']]
        self.df = self.df.join(sex).join(handedness)
        self.df[self.df < 0] = np.NaN
        self.df.drop('dis.dur', axis=1, inplace=True)

    def get_pidns(self) -> List:
        return list(self.df.index)

    def get_data(self, pidn: int) -> pd.Series:
        return self.df.loc[pidn]

    def __repr__(self) -> str:
        return repr(self.df)


class LandmarkSeries(ABC):
    """Base class for managing landmark time series."""

    def __init__(self,
                 path_to_csv: str,
                 tolerance: int = 10,
                 min_duration: float = None,
                 max_duration: float = None,
                 dt: float = 1/30) -> None:
        """Construct landmark time series"""
        self.path = Path(path_to_csv)
        self.tolerance = tolerance
        self.min_duration = min_duration if min_duration else 0
        self.max_duration = max_duration
        self.dt = dt
        # initialize variables
        self.base_landmarks = self.landmarks = None
        self.base_joints = self.joints = None
        self.base_scales = self.scales = None
        self.base_biases = self.biases = None
        self.segments = None
        # parse path name
        token = self.path.stem.split('_')
        self.id = int(token[1])
        if len(token) < 4:
            self.on_med = False
        else:
            self.on_med = token[2] == 'on'
        # finish processing
        self._parse_all_landmarks()
        self._update_joints()
        self._find_good_segments()
        self._compute_scale_and_bias()

    def get_joint(self, joint: Union[str, List], use_all: bool = False, raw: bool = False) -> List:
        """Return normalized joint coordinates."""
        joints = self.base_joints if use_all else self.joints
        scales = self.base_scales if use_all else self.scales
        biases = self.base_biases if use_all else self.biases

        if type(joint) is list:
            joints = joint
        else:
            if joint.startswith('middle_'):
                base_joint = joint.split('_')[-1]
                return self.get_midpoint('left_' + base_joint, 'right_' + base_joint, use_all)
            try:
                joints = getattr(joints, joint)
            except:
                raise(f'Cannot find joint with name {joint}.')

        if raw:
            return joints
        else:
            return [(j-b)/s for j, s, b in zip(joints, scales, biases)]

    def get_displacement(self,
                         joint1: Union[str, List],
                         joint2: Union[str, List],
                         use_all: bool = False) -> List[np.ndarray]:
        """Return relative displacement between normalized joints."""
        joint1 = self.get_joint(joint1, use_all)
        joint2 = self.get_joint(joint2, use_all)
        return [j1 - j2 for j1, j2 in zip(joint1, joint2)]

    def get_x_displacement(self,
                           joint1: Union[str, List],
                           joint2: Union[str, List],
                           use_all: bool = False) -> List[np.ndarray]:
        """Return relative x/horizontal displacement between normalized joints."""
        displacement = self.get_displacement(joint1, joint2, use_all)
        return [d[:, 0] for d in displacement]

    def get_y_displacement(self,
                           joint1: Union[str, List],
                           joint2: Union[str, List],
                           use_all: bool = False) -> List[np.ndarray]:
        """Return relative y/vertical displacement between normalized joints."""
        displacement = self.get_displacement(joint1, joint2, use_all)
        return [d[:, 1] for d in displacement]

    def get_distance(self,
                     joint1: Union[str, List],
                     joint2: Union[str, List],
                     use_all: bool = False) -> List[np.ndarray]:
        """Return distance between normalized joints."""
        displacements = self.get_displacement(joint1, joint2, use_all)
        return [np.linalg.norm(d, axis=1) for d in displacements]

    def get_midpoint(self,
                     joint1: Union[str, List],
                     joint2: Union[str, List],
                     use_all: bool = False) -> List[np.ndarray]:
        """Return midpoints between two normalized joints."""
        joint1 = self.get_joint(joint1, use_all)
        joint2 = self.get_joint(joint2, use_all)
        return [(j1 + j2)/2 for j1, j2 in zip(joint1, joint2)]

    def get_angle(self,
                  joint1: Union[str, List],
                  joint2: Union[str, List],
                  default: float = 0,
                  use_all: bool = False) -> List[np.ndarray]:
        """
        Return angle between two normalized joints w.r.t horizontal axis.
        """
        x = self.get_displacement(joint1, joint2, use_all)
        return [np.arctan2(xx[:, 1], xx[:, 0]) * 180 / np.pi - default for xx in x]

    def get_angle(self,
                  joint1: Union[str, List],
                  joint2: Union[str, List],
                  joint3: Union[str, List],
                  default: float = 0,
                  use_all: bool = False) -> List[np.ndarray]:
        """
        Return angle defined by three normalized joints. 
        *joint2* is defined to be the midpoint.
        """
        x1 = self.get_displacement(joint1, joint2, use_all)
        x2 = self.get_displacement(joint3, joint2, use_all)
        angles = []
        for x11, x22 in zip(x1, x2):
            tmp = [np.dot(i, j)/np.linalg.norm(i)/np.linalg.norm(j)
                   for i, j in zip(x11, x22)]
            angles.append(np.arccos(tmp) * 180 / np.pi - default)
        return angles

    @abstractmethod
    def animate(self,
                visualize: bool = True,
                show_good_range: bool = True,
                output_dir: str = None,
                save_fig: bool = False) -> None:
        pass

    def _animate_helper(self,
                        show_good_range: bool,
                        connections: List,
                        color_lr: bool = False) -> FuncAnimation:
        fig = plt.figure(constrained_layout=False, figsize=(10, 15))
        ax = plt.gca()
        plt.title(self.path.stem)
        plt.xlim(0, 1)
        plt.ylim(0, 1.6)
        plt.gca().invert_yaxis()
        skeleton = []
        connections = np.asarray(connections)
        for c in connections:
            if color_lr:
                if c[0] % 2 == 0:  # if left side
                    line, = plt.plot([], [], 'ro-')
                else:  # if right side
                    line, = plt.plot([], [], 'bo-')
                skeleton.append(line)
            else:
                line, = plt.plot([], [], 'o-')
                skeleton.append(line)

        def update(i):
            for j, line in enumerate(skeleton):
                line.set_data(
                    self.base_landmarks[i, 2*connections[j]],
                    self.base_landmarks[i, 2*connections[j]+1]
                )
            if show_good_range:
                if np.any([s[0] <= i <= s[1] for s in self.segments]):
                    ax.set_facecolor('honeydew')
                else:
                    ax.set_facecolor('lightcoral')
            return skeleton

        anim = FuncAnimation(
            fig, update, frames=range(self.base_landmarks.shape[0]))

        return anim

    @abstractmethod
    def _parse_all_landmarks(self) -> None:
        pass

    def _parser_helper(self, df: pd.DataFrame, landmark_labels: List) -> None:
        # read all entries
        column_name = ['timestep', 'inst']
        for lm in landmark_labels:
            column_name.extend([lm + ':x', lm + ':y', lm + ':z'])
        df.columns = column_name
        # get time
        if self.min_duration:
            self.min_duration /= self.dt
        if self.max_duration:
            self.max_duration /= self.dt
        # prune entries
        xy_columns = [i for i in range(len(column_name)-2) if i % 3 != 2]
        df = df.loc[:, :].rolling(
            window=5, win_type='gaussian', center=True).mean(std=0.5)[2:-2]

        landmarks = df.iloc[:, 2:].to_numpy()[:, xy_columns]
        self.base_landmarks = self.landmarks = landmarks

    @abstractmethod
    def _update_joints(self, joint_values: List = None) -> None:
        pass

    @abstractmethod
    def _find_good_segments(self) -> None:
        pass

    def _segment_helper(self, abnormalities: List) -> None:
        # find error clusters
        start, end = 0, self.landmarks.shape[0]
        good_segments = []

        if all([a.size == 0 for a in abnormalities]):
            good_segments.append((start, end))
        else:
            abnormalities = np.unique(np.concatenate(abnormalities))
            error_clusters = []
            cluster = []
            prev = 0
            for a in abnormalities:
                if a - prev < self.tolerance:
                    prev = a
                    cluster.append(a)
                else:
                    error_clusters.append(cluster)
                    cluster = [a]
                    prev = a
            error_clusters.append(cluster)

            # find good segments
            for c in error_clusters:
                if len(c) > 1:
                    if start < c[0] - self.tolerance:
                        good_segments.append((start, c[0] - self.tolerance))
                    start = c[-1] + self.tolerance
            good_segments.append((start, end))
            good_segments = sorted(
                good_segments, key=lambda x: x[1] - x[0], reverse=True)

        # split segments if too large
        self.segments = []
        for s in good_segments:
            duration = s[1] - s[0]
            if duration < self.min_duration:
                continue
            if self.max_duration:
                start = s[0]
                while duration >= self.min_duration:
                    if self.min_duration * 2 / 3 + self.max_duration / 3 <= duration // 2 <= self.max_duration:
                        size = int(duration // 2)
                    else:
                        size = int(min(self.max_duration, duration))
                    self.segments.append((start, start + size))
                    duration -= size
                    start += size
            else:
                self.segments.append((s[0], s[1]))

        self.seg_cnt = len(self.segments)
        joints = [[j[0][slice(*s)] for s in self.segments]
                  for j in self.joints]
        self._update_joints(joints)

    @abstractmethod
    def _compute_scale_and_bias(self) -> None:
        pass


class PoseSeries(LandmarkSeries):
    """Class for managing full-body landmark time series"""

    def animate(self,
                visualize: bool = True,
                show_good_range: bool = True,
                output_dir: str = None,
                save_fig: bool = False) -> None:
        """Animate landmarks."""
        connections = [
            (0, 2), (0, 5), (8, 5),
            (2, 7), (10, 9), (12, 11),
            (12, 14), (12, 24), (14, 16),
            (16, 18), (16, 20), (16, 22),
            (18, 20), (11, 13), (11, 23),
            (13, 15), (15, 17), (15, 19),
            (15, 21), (24, 23), (24, 26),
            (26, 28), (25, 27), (28, 30),
            (28, 32), (30, 32), (23, 25),
            (27, 29), (27, 31), (29, 31)
        ]
        anim = self._animate_helper(
            show_good_range, connections, color_lr=True)
        if save_fig:
            if not output_dir:
                output_dir = self.path.parent
            else:
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            anim.save(str(output_dir/f'{self.path.stem}.mp4'), fps=33)
        if visualize:
            plt.show()
        else:
            plt.close('all')

    def _parse_all_landmarks(self) -> None:
        landmark_labels = [
            '0 - nose', '1 - left eye (inner)', '2 - left eye',
            '3 - left eye (outer)', '4 - right eye (inner)', '5 - right eye',
            '6 - right eye (outer)', '7 - left ear', '8 - right ear',
            '9 - mouth (left)', '10 - mouth (right)', '11 - left shoulder',
            '12 - right shoulder', '13 - left elbow', '14 - right elbow',
            '15 - left wrist', '16 - right wrist', '17 - left pinky',
            '18 - right pinky', '19 - left index', '20 - right index',
            '21 - left thumb', '22 - right thumb', '23 - left hip',
            '24 - right hip', '25 - left knee', '26 - right knee',
            '27 - left ankle', '28 - right ankle', '29 - left heel',
            '30 - right heel', '31 - left foot index', '32 - right foot index'
        ]
        df = pd.read_csv(self.path, header=None)
        self._parser_helper(df, landmark_labels)

    def _update_joints(self, joint_values: List = None) -> None:
        labels = ['nose', 'left_eye', 'right_eye',
                  'left_shoulder', 'right_shoulder', 'left_elbow',
                  'right_elbow', 'left_wrist', 'right_wrist',
                  'left_hip', 'right_hip', 'left_knee',
                  'right_knee', 'left_ankle', 'right_ankle',
                  'left_heel', 'right_heel']
        joint_tuple = namedtuple('Joints', labels)
        if not joint_values:
            nose = [self.base_landmarks[:, :2]]
            left_eye = [self.base_landmarks[:, 2:4]]
            right_eye = [self.base_landmarks[:, 10:12]]
            left_shoulder = [self.base_landmarks[:, 22:24]]
            right_shoulder = [self.base_landmarks[:, 24:26]]
            left_elbow = [self.base_landmarks[:, 26:28]]
            right_elbow = [self.base_landmarks[:, 28:30]]
            left_wrist = [self.base_landmarks[:, 32:34]]
            right_wrist = [self.base_landmarks[:, 34:36]]
            left_hip = [self.base_landmarks[:, 46:48]]
            right_hip = [self.base_landmarks[:, 48:50]]
            left_knee = [self.base_landmarks[:, 52:54]]
            right_knee = [self.base_landmarks[:, 54:56]]
            left_ankle = [self.base_landmarks[:, 56:58]]
            right_ankle = [self.base_landmarks[:, 58:60]]
            left_heel = [self.base_landmarks[:, 60:62]]
            right_heel = [self.base_landmarks[:, 62:64]]
            joint_values = [
                nose, left_eye, right_eye,
                left_shoulder, right_shoulder, left_elbow,
                right_elbow, left_wrist, right_wrist,
                left_hip, right_hip, left_knee,
                right_knee, left_ankle, right_ankle,
                left_heel, right_heel,
            ]
            self.base_joints = self.joints = joint_tuple(*joint_values)
        else:
            self.joints = joint_tuple(*joint_values)

    def _find_good_segments(self) -> None:
        # get all relevant joints
        ls = self.joints.left_shoulder[0]
        rs = self.joints.right_shoulder[0]
        lw = self.joints.left_wrist[0]
        rw = self.joints.right_wrist[0]
        lh = self.joints.left_hip[0]
        rh = self.joints.right_hip[0]
        lk = self.joints.left_knee[0]
        rk = self.joints.right_knee[0]
        la = self.joints.left_ankle[0]
        ra = self.joints.right_ankle[0]
        nose = self.joints.nose[0]

        abnormalities = []
        sos = signal.butter(3, 1/15, btype='low', output='sos')

        # detect tracking error
        x_shoulder_dist = ls[:, 0] - rs[:, 0]
        x_shoulder_dist_flt = signal.sosfiltfilt(sos, x_shoulder_dist)
        abnormalities.append(np.where(
            np.abs(x_shoulder_dist - x_shoulder_dist_flt) > 0.02)[0])

        # detect if patient is sideways, part 1
        x_wrist_hip_dist_l = lw[:, 0] - ls[:, 0]
        x_wrist_hip_dist_r = rw[:, 0] - rs[:, 0]
        abnormalities.append(np.where(
            np.sign(x_wrist_hip_dist_l) == np.sign(x_wrist_hip_dist_r))[0])

        # detect if patient is sideways, part 2
        x_nose_shoulder_dist_l = nose[:, 0] - ls[:, 0]
        x_nose_shoulder_dist_r = nose[:, 0] - rs[:, 0]
        abnormalities.append(np.where(
            np.sign(x_nose_shoulder_dist_l) == np.sign(x_nose_shoulder_dist_r))[0])

        # detect if patient is seated
        hip_knee_len_l = np.abs(lh[:, 1] - lk[:, 1])
        hip_knee_len_r = np.abs(rh[:, 1] - rk[:, 1])
        knee_ankle_len_l = np.abs(lk[:, 1] - la[:, 1])
        knee_ankle_len_r = np.abs(rk[:, 1] - ra[:, 1])
        ratio_l = knee_ankle_len_l / hip_knee_len_l
        ratio_r = knee_ankle_len_r / hip_knee_len_r
        abnormalities.append(np.where(
            np.logical_or(ratio_l > 2, ratio_r > 2))[0])

        self._segment_helper(abnormalities)

    def _compute_scale_and_bias(self) -> None:
        """Compute scale abd bias used in coordinate normalization."""
        self.base_scales = []
        self.base_biases = []
        ls = self.base_joints.left_shoulder[0]
        rs = self.base_joints.right_shoulder[0]
        lh = self.base_joints.left_hip[0]
        rh = self.base_joints.right_hip[0]
        ms = (ls + rs) / 2
        mh = (lh + rh) / 2
        self.base_scales.append(np.linalg.norm(
            ms - mh, axis=1).reshape((-1, 1)))
        self.base_biases.append(mh)
        self.scales = [self.base_scales[0][slice(*s)] for s in self.segments]
        self.biases = [self.base_biases[0][slice(*s)] for s in self.segments]


class HandSeries(LandmarkSeries):

    def get_joint(self, joint: Union[str, List], use_all: bool = False, raw: bool = False) -> List:
        """Return normalized joint coordinates."""
        joints = self.base_joints if use_all else self.joints
        scales = self.base_scales if use_all else self.scales
        biases = self.base_biases if use_all else self.biases

        if type(joint) is list:
            joints = joint
            is_right = True
        else:
            if joint.startswith('middle_'):
                base_joint = joint.split('_')[-1]
                return self.get_midpoint('left_' + base_joint, 'right_' + base_joint, use_all)
            try:
                joints = getattr(joints, joint)
            except:
                raise(f'Cannot find joint with name {joint}.')
            is_right = joint.startswith('right_')

        scales = [s[0] if is_right else s[1] for s in scales]
        biases = [b[0] if is_right else b[1] for b in biases]

        if raw:
            return joints
        else:
            return [(j-b)/s for j, s, b in zip(joints, scales, biases)]

    def animate(self,
                visualize: bool = True,
                show_good_range: bool = True,
                output_dir: str = None,
                save_fig: bool = False) -> None:
        """Animate landmarks."""
        connections = [
            (0, 1), (0, 5), (0, 17),
            (1, 2), (2, 3), (3, 4),
            (5, 6), (5, 9), (6, 7),
            (7, 8), (9, 10), (9, 13),
            (10, 11), (11, 12), (13, 14),
            (13, 17), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (21, 22), (21, 26), (21, 38),
            (22, 23), (23, 24), (24, 25),
            (26, 27), (26, 30), (27, 28),
            (28, 29), (30, 31), (30, 34),
            (31, 32), (32, 33), (34, 35),
            (34, 38), (35, 36), (36, 37),
            (38, 39), (39, 40), (40, 41),
        ]
        anim = self._animate_helper(
            show_good_range, connections, color_lr=True)
        if save_fig:
            if not output_dir:
                output_dir = self.path.parent
            else:
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            anim.save(str(output_dir/f'{self.path.stem}.mp4'), fps=33)
        if visualize:
            plt.show()
        else:
            plt.close('all')

    def _animate_helper(self,
                        show_good_range: bool,
                        connections: List,
                        color_lr: bool = False) -> FuncAnimation:
        fig = plt.figure(constrained_layout=False, figsize=(10, 15))
        ax = plt.gca()
        plt.title(self.path.stem)
        plt.xlim(0, 1)
        plt.ylim(0, 1.6)
        plt.gca().invert_yaxis()
        skeleton = []
        connections = np.asarray(connections)
        for c in connections:
            if color_lr:
                if c[0] >= 21:  # if left side
                    line, = plt.plot([], [], 'ro-')
                else:  # if right side
                    line, = plt.plot([], [], 'bo-')
                skeleton.append(line)
            else:
                line, = plt.plot([], [], 'o-')
                skeleton.append(line)

        def update(i):
            for j, line in enumerate(skeleton):
                line.set_data(
                    self.base_landmarks[i, 2*connections[j]],
                    self.base_landmarks[i, 2*connections[j]+1]
                )
            if show_good_range:
                if np.any([s[0] <= i <= s[1] for s in self.segments]):
                    ax.set_facecolor('honeydew')
                else:
                    ax.set_facecolor('lightcoral')
            return skeleton

        anim = FuncAnimation(
            fig, update, frames=range(self.base_landmarks.shape[0]))

        return anim

    def _parse_all_landmarks(self) -> None:
        landmark_labels = [
            '0 - right wrist', '1 - right thumb_cmc', '2 - right thumb_mcp',
            '3 - right thumb_ip', '4 - right thumb_tip', '5 - right index_mcp',
            '6 - right index_pip', '7 - right index_dip', '8 - right index_tip',
            '9 - right middle_mcp', '10 - right middle_pip', '11 - right middle_dip',
            '12 - right middle_tip', '13 - right ring_mcp', '14 - right ring_pip',
            '15 - right ring_dip', '16 - right ring_tip', '17 - right pinky_mcp',
            '18 - right pinky_pip', '19 - right pinky_dip', '20 - right pinky_tip',
            '21 - left wrist', '22 - left thumb_cmc', '23 - left thumb_mcp',
            '24 - left thumb_ip', '25 - left thumb_tip', '26 - left index_mcp',
            '27 - left index_pip', '28 - left index_dip', '29 - left index_tip',
            '30 - left middle_mcp', '31 - left middle_pip', '32 - left middle_dip',
            '33 - left middle_tip', '34 - left ring_mcp', '35 - left ring_pip',
            '36 - left ring_dip', '37 - left ring_tip', '38 - left pinky_mcp',
            '39 - left pinky_pip', '40 - left pinky_dip', '41 - left pinky_tip',
        ]
        num_columns = len(landmark_labels) * 3 + 2
        df = pd.read_fwf(self.path, header=None)
        df = df[0].str.split(',', expand=True)
        df = df.iloc[:, :num_columns].apply(pd.to_numeric, errors='coerce')
        self._parser_helper(df.astype(float), landmark_labels)

    def _update_joints(self, joint_values: List = None) -> None:
        labels = ['right_wrist', 'right_thumb', 'right_index',
                  'right_middle', 'right_ring', 'right_pinky',
                  'left_wrist', 'left_thumb', 'left_index',
                  'left_middle', 'left_ring', 'left_pinky',
                  'right_index_base', 'right_pinky_base', 'left_index_base',
                  'left_pinky_base']
        joint_tuple = namedtuple('Joints', labels)
        if not joint_values:
            right_wrist = [self.base_landmarks[:, :2]]
            right_thumb = [self.base_landmarks[:, 8:10]]
            right_index = [self.base_landmarks[:, 16:18]]
            right_middle = [self.base_landmarks[:, 24:26]]
            right_ring = [self.base_landmarks[:, 32:34]]
            right_pinky = [self.base_landmarks[:, 40:42]]
            left_wrist = [self.base_landmarks[:, 42:44]]
            left_thumb = [self.base_landmarks[:, 50:52]]
            left_index = [self.base_landmarks[:, 58:60]]
            left_middle = [self.base_landmarks[:, 66:68]]
            left_ring = [self.base_landmarks[:, 74:76]]
            left_pinky = [self.base_landmarks[:, 82:84]]
            right_index_base = [self.base_landmarks[:, 10:12]]
            right_pinky_base = [self.base_landmarks[:, 34:36]]
            left_index_base = [self.base_landmarks[:, 52:54]]
            left_pinky_base = [self.base_landmarks[:, 76:78]]
            joint_values = [right_wrist, right_thumb, right_index,
                            right_middle, right_ring, right_pinky,
                            left_wrist, left_thumb, left_index,
                            left_middle, left_ring, left_pinky,
                            right_index_base, right_pinky_base, left_index_base,
                            left_pinky_base]
            self.base_joints = self.joints = joint_tuple(*joint_values)
        else:
            self.joints = joint_tuple(*joint_values)

    def _find_good_segments(self) -> None:
        # get all relevant joints
        right_wrist = self.joints.right_wrist[0]
        left_wrist = self.joints.left_wrist[0]
        right_index_base = self.joints.right_index_base[0]
        left_index_base = self.joints.left_index_base[0]
        right_index = self.joints.right_index[0]
        left_index = self.joints.left_index[0]
        right_pinky_base = self.joints.right_pinky_base[0]
        left_pinky_base = self.joints.left_pinky_base[0]
        joints = [right_wrist, left_wrist, right_index_base,
                  left_index_base, right_pinky_base, left_pinky_base,
                  right_index, left_index]

        abnormalities = []
        for j in joints:
            abnormalities.append(np.where(np.isnan(j).any(axis=1))[0])

        # detect if patient has both hands down
        y1 = right_index[:, 1] - right_wrist[:, 1]
        y2 = left_index[:, 1] - left_wrist[:, 1]
        abnormalities.append(np.where(
            np.sign(y1) == np.sign(y2))[0])

        self._segment_helper(abnormalities)

    def _compute_scale_and_bias(self) -> None:
        """Compute scale abd bias used in coordinate normalization."""
        self.base_scales = []
        self.base_biases = []

        rw = self.base_joints.right_wrist[0]
        lw = self.base_joints.left_wrist[0]
        rib = self.base_joints.right_index_base[0]
        lib = self.base_joints.left_index_base[0]
        rpb = self.base_joints.right_pinky_base[0]
        lpb = self.base_joints.left_pinky_base[0]

        rr = (rib + rpb) / 2
        ll = (lib + lpb) / 2

        self.base_scales.append(np.array([np.linalg.norm(rr - rw, axis=1),
                                          np.linalg.norm(ll - lw, axis=1)]))
        self.base_biases.append(np.array([rw, lw]))

        self.scales = [self.base_scales[0][:, s[0]:s[1], None]
                       for s in self.segments]
        self.biases = [self.base_biases[0][:, s[0]:s[1]]
                       for s in self.segments]


class FaceSeries(LandmarkSeries):

    def animate(self,
                visualize: bool = True,
                show_good_range: bool = True,
                output_dir: str = None,
                save_fig: bool = False) -> None:
        connections = [
            # silhouette
            (10, 338), (338, 297), (297, 332),
            (332, 284), (284, 251), (251, 389),
            (389, 356), (356, 454), (454, 323),
            (323, 361), (361, 288), (288, 397),
            (397, 365), (365, 379), (379, 378),
            (378, 400), (400, 377), (377, 152),
            (152, 148), (148, 176), (176, 149),
            (149, 150), (150, 136), (136, 172),
            (172, 58), (58, 132), (132, 93),
            (93, 234), (234, 127), (127, 162),
            (162, 21), (21, 54), (54, 103),
            (103, 67), (67, 109), (109, 10),
            # outer lips
            (61, 185), (185, 40), (40, 39),
            (39, 37), (37, 0), (0, 267),
            (267, 269), (269, 270), (270, 409),
            (409, 291), (61, 146), (146, 91),
            (91, 181), (181, 84), (84, 17),
            (17, 314), (314, 405), (405, 321),
            (321, 375), (375, 291),
            # inner lips
            (78, 191), (191, 80), (80, 81),
            (81, 82), (82, 13), (13, 312),
            (312, 311), (311, 310), (310, 415),
            (415, 308), (78, 95), (95, 88),
            (88, 178), (178, 87), (87, 14),
            (14, 317), (317, 402), (402, 318),
            (318, 324), (324, 308),
            # right eye
            (246, 161), (161, 160), (160, 159),
            (158, 157), (157, 173), (246, 33),
            (246, 33), (33, 7), (7, 163),
            (163, 144), (144, 145), (145, 153),
            (153, 154), (154, 155), (155, 133),
            (133, 173),
            # left eye
            (466, 388), (388, 387), (387, 386),
            (386, 385), (385, 384), (384, 398),
            (466, 263), (263, 249), (249, 390),
            (390, 373), (373, 374), (374, 380),
            (380, 381), (381, 382), (382, 362),
            (362, 398),
            # right eyebrow
            (156, 70), (70, 63), (63, 105),
            (105, 66), (66, 107), (107, 55),
            (55, 65), (156, 35), (35, 124),
            (124, 46), (46, 53), (53, 52),
            (52, 65),
            # left eyebrow
            (383, 300), (300, 293), (293, 334),
            (334, 296), (296, 336), (336, 285),
            (285, 295), (383, 265), (265, 353),
            (353, 276), (276, 283), (283, 282),
            (282, 295),
        ]
        anim = self._animate_helper(show_good_range, connections)
        if save_fig:
            if not output_dir:
                output_dir = self.path.parent
            else:
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            anim.save(str(output_dir/f'{self.path.stem}.mp4'), fps=33)
        if visualize:
            plt.show()
        else:
            plt.close('all')

    def _parse_all_landmarks(self) -> None:
        landmark_labels = [str(i) for i in range(468)]
        df = pd.read_csv(self.path, header=None)
        self._parser_helper(df, landmark_labels)

    def _update_joints(self, joint_values: List = None) -> None:
        labels = [
            # silhouette
            'forehead', 'chin',
            # eyes
            'right_upper_eye', 'left_upper_eye',
            'right_lower_eye', 'left_lower_eye',
            # eyebrows
            'right_inner_eyebrow', 'left_inner_eyebrow',
            'right_outer_eyebrow', 'left_outer_eyebrow',
            # lips
            'lips_right', 'lips_left',
            'upper_lip', 'lower_lip',
        ]
        joint_tuple = namedtuple('Joints', labels)
        if not joint_values:
            forehead = [self.base_landmarks[:, 20:22]]
            chin = [self.base_landmarks[:, 304:306]]

            right_upper_eye = [self.base_landmarks[:, 318:320]]
            left_upper_eye = [self.base_landmarks[:, 772:774]]
            right_lower_eye = [self.base_landmarks[:, 290:292]]
            left_lower_eye = [self.base_landmarks[:, 748:750]]

            right_inner_eyebrow = [self.base_landmarks[:, 214:216]]
            left_inner_eyebrow = [self.base_landmarks[:, 672:674]]
            right_outer_eyebrow = [self.base_landmarks[:, 312:314]]
            left_outer_eyebrow = [self.base_landmarks[:, 766:768]]

            lips_right = [self.base_landmarks[:, 122:124]]
            lips_left = [self.base_landmarks[:, 582:584]]
            upper_lip = [self.base_landmarks[:, 26:28]]
            lower_lip = [self.base_landmarks[:, 28:30]]

            joint_values = [forehead, chin,
                            right_upper_eye, left_upper_eye,
                            right_lower_eye, left_lower_eye,
                            right_inner_eyebrow, left_inner_eyebrow,
                            right_outer_eyebrow, left_outer_eyebrow,
                            lips_right, lips_left,
                            upper_lip, lower_lip]
            self.base_joints = self.joints = joint_tuple(*joint_values)
        else:
            self.joints = joint_tuple(*joint_values)

    def _find_good_segments(self) -> None:
        abnormalities = []
        self._segment_helper(abnormalities)

    def _compute_scale_and_bias(self) -> None:
        self.base_scales = []
        self.base_biases = []
        forehead = self.base_joints.forehead[0]
        chin = self.base_joints.chin[0]
        self.base_scales.append(np.linalg.norm(
            forehead-chin, axis=1).reshape(-1, 1))
        self.base_biases.append(chin)
        self.scales = [self.base_scales[0][slice(*s)] for s in self.segments]
        self.biases = [self.base_biases[0][slice(*s)] for s in self.segments]
