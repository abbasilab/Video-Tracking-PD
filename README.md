# Video-Based Tracking of Clinical States in Patients with Parkinson's Disease

Authors: Daniel Deng, Jill L. Ostrem, Vy Nguyen, Daniel D. Cummins, Julia Sun, Anupam Pathak, Simon Little, Reza Abbasi-Asl

Manuscript: [TODO]

## Installation

In the root directory, install Python dependencies by run the command

```bash
pip install -r requirements.txt
```

## Modules

`./modules/tracking.py`: classes for parsing and preprocessing demographic data, UPDRS data, and MediaPipe landmark timeseries data

`./modules/feature_extraction.py:` defines temporal and spectral features to be extracted from landmark timeseries

`./modules/cross_validation.py`: custom leave-one-subject-out cross-validation (CV) for selecting LASSO feature selection and model parameters

`./modules/model_selection.py`: training and validation of models; aggregation of model results

`./modules/visualize.py`: visualization methods for datasets and CV results

`./modules/utils.py`: miscellaneous helper methods

## Video Preprocessing

MediaPipe (https://developers.google.com/mediapipe) should first be used to extract kinematic landmarks as time series. The resultant `.csv` files should be placed under `./dataset`. Currently, only hand landmark timeseries and pose landmark timeseries are supported by the software. Code for parsing additional modalities should extend the `LandmarkSeries` class in `./modules/tracking.py`.

## Usage

To run the analysis, follow the code instructions in `./analysis.ipynb`. By default, extracted features and validation results are stored under `./cache`.

## Version

Last update: Sept 2023, v1.0

## Citation

[TODO]
