import os
import os.path as op

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import RidgeCV, GammaRegressor, BayesianRidge, TweedieRegressor, SGDRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupShuffleSplit, KFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

import mne
from mne import Epochs
from mne.datasets.fieldtrip_cmc import data_path

import config as cfg
from library.spfiltering import (
    ProjIdentitySpace, ProjCommonSpace, ProjSPoCSpace)
from library.featuring import Riemann, LogDiag, NaiveVec
import library.preprocessing_david as pp

from subject_number import subject_number

from joblib import Parallel, delayed


def preprocess (directory        =  directory,
                number_subject   =  number_subject,
                extension        = '.bdf',
                high_pass_filter =  high_pass_filter):
    """
    Reads subject file in directory and preprocess signals
    
    :return: preprocessed raw (EEG) and EDA of one subject 
    """
    
    fname = op.join(directory, 's'+ number_subject + extension)
    raw = mne.io.read_raw_bdf(fname, preload=True) #(?) Is it necessary to preload or load_data? Why?
    
    # crop for memory purposes (use to try things)
    #raw.crop(450., 650.)

    # Separate EDA data
    eda = raw.copy().pick_channels(['GSR1'])
    
    # EDA Band-pass filter
    eda.filter(high_pass_filter, 5, fir_design='firwin') 

    # Common channels
    common_chs = set(raw.info['ch_names'])
    
    # Discard non-relevant channels #(?) Is it OK to descard EOG channels?
    common_chs -= {'EXG1', 'EXG2', 'EXG3', 'EXG4',
                   'EXG5', 'EXG6', 'EXG7', 'EXG8',
                   'GSR2', 'Erg1', 'Erg2', 'Resp',
                   'Plet', 'Temp', 'GSR1', 'Status'}

    # EEG Band-pass filter
    raw.pick_channels(list(common_chs))
    raw.filter(None, 120., fir_design='firwin')
    
    return eda, raw

def compute_cov_matrices (raw=raw, eda=eda):
    """
    Inputs raw (EEG) and calculate covariance matrices ('X' in our model)
    
    :return: cov matrices of one subject's epochs
    """
    preprocess()
    
    X = []
    
    for fb in pp.fbands:
        rf = raw.copy().load_data().filter(fb[0], fb[1])
        events = mne.make_fixed_length_events(rf,
                                              id=3000,
                                              duration=pp.duration,
                                              overlap=2.0)
        
        ec = mne.Epochs(rf, events,
                        event_id=3000, tmin=0, tmax=pp.duration,
                        proj=True, baseline=None, reject=None, preload=True, decim=1,
                        picks=None)
        X.append([mne.compute_covariance(
                                        ec[ii], method='oas')['data'][None]
                                        for ii in range(len(ec))])   

    X = np.array(X)   
    # Delete one axis     
    X = np.squeeze(X)
    # order axis of ndarray (first n_sub, then n_fb)
    X = X.transpose(1,0,2,3)
                
    n_sub, n_fb, n_ch, _ = X.shape
        
    return X, n_sub, n_fb, n_ch

def compute_model_output (eda=eda, model=model):
    """
    Inputs EDA and calculate model output ('y' in our model)
    
    :return: 'y' time serie of one subject
    """
    events_shift = mne.make_fixed_length_events(eda, id=3000, start=1.5,
                                                duration=pp.duration, overlap=2.0)

    eda_epochs_shift = Epochs(eda, events_shift, event_id=3000, tmin=0, tmax=pp.duration,
                              proj=True, baseline=None, preload=True, decim=1)
    
    if model == 'mean':
        y = eda_epochs_shift.get_data().mean(axis=2)[:, 0]  
    elif model == 'delta':
        y = eda_epochs_shift.get_data().max(axis=2)[:, 0] - eda_epochs_shift.get_data().min(axis=2)[:, 0]
    else:
        y = eda_epochs_shift.get_data().var(axis=2)[:, 0]     
        
    return y


def run_low_rank(n_components, X, y, cv, estimators, scoring, groups):
    out = dict(n_components=n_components)
    for name, est in estimators.items():
        print(name)
        this_est = est
        this_est.steps[0][1].n_compo = n_components
        scores = cross_val_score(
            X=X, y=y, cv=cv, estimator=this_est, n_jobs=1,
            groups=groups,
            scoring=scoring)
        if scoring == 'neg_mean_absolute_error':
            scores = -scores
        print(np.mean(scores), f"+/-{np.std(scores)}")
        out[name] = scores
    return out


def low_rank(out=out):

    low_rank_estimators = {k: v for k, v in pipelines.items()
                        if k in ('riemann')} #'spoc', 

    out_list = Parallel(n_jobs=n_jobs)(delayed(run_low_rank)(
                        n_components=cc, X=X, y=y,
                        groups=groups,
                        cv=GroupShuffleSplit(
                            n_splits=10, train_size=.8, test_size=.2),
                        estimators=low_rank_estimators, scoring='r2')
                        for cc in n_components)
    out_frames = list()
    for this_dict in out_list:
        this_df = pd.DataFrame({#'spoc': this_dict['spoc'],
                            'riemann': this_dict['riemann']})
        this_df['n_components'] = this_dict['n_components']
        this_df['fold_idx'] = np.arange(len(this_df))
        out_frames.append(this_df)
    out_df = pd.concat(out_frames)

    out_df.to_csv("./DEAP_component_scores.csv")

    mean_df = out_df.groupby('n_components').mean().reset_index()
    

def select_best_components():
    best_components = {
    #'spoc': mean_df['n_components'][mean_df['spoc'].argmax()],
    'riemann': mean_df['n_components'][mean_df['riemann'].argmax()]
    }
    return best_components

def models(estimator=estimator):
    riemann_model = make_pipeline(
    ProjCommonSpace(scale=scale, n_compo= 10, #best_components['riemann'],
                    reg=1.e-05),
    Riemann(n_fb=n_fb, metric=metric),
    StandardScaler(),
#   RidgeCV(alphas=ridge_shrinkage))
#   GammaRegressor())
#   BayesianRidge())
#   TweedieRegressor())
#   SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
#              coef0=1))
#  SGDRegressor())
    estimator())

    
def pipeline_all_subjects(directory        =  'data',
                          number_subject   =  'all',
                          high_pass_filter =   0.01, 
                          shift_EDA        =   1.5,
                          tune_components  =   True,
                          target           =   'var',
                          model            =  'TweedieRegressor',):

    """

    Reads the all subject files in 'data', preprocess signals, predicts EDA from EEG,
    and outputs a .csv file per subject in 'outputs'

    :directory:          String.   directory in which files are stored. Default = data
    :number_subject:     String.   '01' to '32' (DEAP database). Default = 'all'
    :high_pass_filter:   Boolean.  True (0.01 Hz High-pass filter) or False (No filter --> None)
                         Default = True
    :shift_EDA:          Float.    Length shift between EEG and EDA epochs (in seconds).
                         Default = 1.5
    :target:             String.   Y used in our model 
                         Default = 'mean' --> mean of EDA
    :estimator:          String.   Model used to predict EDA
                         Default = 'TweedieRegressor'
                            
    
    :return: .csv file per subject with predicted EDA (specifying other parameters)
    """ 

    preprocess (directory        =  directory,
                number_subject   =  number_subject,
                extension        = '.bdf',
                high_pass_filter =  high_pass_filter):
