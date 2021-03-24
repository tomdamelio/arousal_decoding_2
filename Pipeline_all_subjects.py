#%%
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import config as cfg

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import RidgeCV, GammaRegressor, BayesianRidge, TweedieRegressor, SGDRegressor
#from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupShuffleSplit, KFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
#from mne import Epochs
#from mne.datasets.fieldtrip_cmc import data_path
from joblib import Parallel, delayed

from library.spfiltering import (
    ProjIdentitySpace, ProjCommonSpace, ProjSPoCSpace)
from library.featuring import Riemann, LogDiag, NaiveVec
import library.preprocessing_david as pp
from subject_number import subject_number

def run_low_rank(n_components, X, y, cv, estimators, scoring, groups):
    out = dict(n_components=n_components)
    for name, est in estimators.items(): #e.g. name --> riemann // est --> make_pipeline(...)
        print(name)
        this_est = est # est --> make_pipeline(...)
        this_est.steps[0][1].n_compo = n_components #151 -> n_components inside Riemann inside pipeline
        scores = cross_val_score(
            X=X, y=y, cv=cv, estimator=this_est, n_jobs=1,
            groups=groups,
            scoring=scoring)
        if scoring == 'neg_mean_absolute_error':
            scores = -scores
        print(np.mean(scores), f"+/-{np.std(scores)}")
        out[name] = scores
    return out

#%%
def pipeline_all_subjects  (directory        =  'data',
                            number_subject   =   subject_number,
                            crop             =   True,
                            extension        =  '.bdf',
                            high_pass_filter =   0.01,
                            shift_EDA        =   1.5,
                            tune_components  =   True,
                            target           =   'var',
                            estimator        =   TweedieRegressor):

    """

    Reads the all subject files in 'data', preprocess signals, predicts EDA from EEG,
    and outputs a .csv file per subject in 'outputs'

    :directory:          String. Directory in which files are stored. Default = 'data'
    :number_subject:     String or list. '01' to '32' (DEAP database).
                         Default = 'subject_number' --> all subjects
    :high_pass_filter:   Float. High-pass filter (in Hz). No filter = None.
                         Default = 0.01               
    :shift_EDA:          Float.    Length shift between EEG and EDA epochs (in seconds).
                         Default = 1.5
    :tune_components:    Boolean. Tune n_components (rank of cov matrices)
                         Default = True                    
    :target:             String.   'Y' used in our model 
                         Default = 'var' --> var of EDA epochs
    :estimator:          String.   Model used to predict EDA
                         Default = TweedieRegressor
                            
    
    :return: .csv file per subject with predicted EDA (specifying other parameters)
    
    """ 
    # container
    exp = []
    
    # Return arguments and values passed to a function
    func_args = locals()
    
    ###########################################################################
    n_compo = 32
    scale = 'auto'
    metric = 'riemann'
    seed = 42
    n_splits = 2
    n_jobs = 20
    ###########################################################################

    # Making the function work with only one subject as input
    if type(number_subject) == str:
        number_subject = [number_subject]
 
    for i in number_subject:   
    
        fname = op.join(directory, 's'+ i + extension)
        raw = mne.io.read_raw_bdf(fname, preload=True)
        #(?) Is it necessary to preload or better to load_data? Why?
        
        # crop for memory purposes (use to test things)
        if crop == True:
            raw = raw.crop(100., 500.)
        
        picks_eda = mne.pick_channels(ch_names = raw.ch_names ,include=['GSR1'])
        
        if int(i) < 23:
            raw.apply_function(fun=lambda x: x/1000, picks=picks_eda)
        else:
            raw.apply_function(fun=lambda x: (10**9/x)/1000, picks=picks_eda)

        # Separate EDA data
        eda = raw.copy().pick_channels(['GSR1'])
        
        # EDA Band-pass filter
        eda.filter(high_pass_filter, 5, fir_design='firwin') 

        # Common channels
        common_chs = set(raw.info['ch_names'])
        
        # Discard non-relevant channels 
        common_chs -= {'EXG1', 'EXG2', 'EXG3', 'EXG4',
                    'EXG5', 'EXG6', 'EXG7', 'EXG8',
                    'GSR2', 'Erg1', 'Erg2', 'Resp',
                    'Plet', 'Temp', 'GSR1', 'Status'}
        #(?) Is it OK to descard EOG channels?

        # EEG Band-pass filter
        raw.pick_channels(list(common_chs))
        raw.filter(None, 120., fir_design='firwin')
        
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
        
        events_shift = mne.make_fixed_length_events(eda, id=3000, start=1.5,
                                                    duration=pp.duration, overlap=2.0)

        eda_epochs_shift = mne.Epochs(eda, events_shift, event_id=3000, tmin=0, tmax=pp.duration,
                                proj=True, baseline=None, preload=True, decim=1)
        
        if target == 'mean':
            y = eda_epochs_shift.get_data().mean(axis=2)[:, 0]  
        elif target == 'delta':
            y = eda_epochs_shift.get_data().max(axis=2)[:, 0] - eda_epochs_shift.get_data().min(axis=2)[:, 0]
        else:
            y = eda_epochs_shift.get_data().var(axis=2)[:, 0]     
            
        n_components = np.arange(1, 32, 1) # max components --> 32 --> 32 EEG channels
        # now let's do group shuffle split
        splits = np.array_split(np.arange(len(y)), n_splits)
        groups = np.zeros(len(y), dtype=np.int)
        for val, inds in enumerate(splits):
            groups[inds] = val

        ##################################################################
        ridge_shrinkage = np.logspace(-3, 5, 100)
        spoc_shrinkage = np.linspace(0, 1, 5)
        common_shrinkage = np.logspace(-7, -3, 5)
        ##################################################################

        pipelines = {
            'dummy':  make_pipeline(
                ProjIdentitySpace(), LogDiag(), StandardScaler(), DummyRegressor()),
            'naive': make_pipeline(ProjIdentitySpace(), NaiveVec(method='upper'),
                                StandardScaler(),
                                RidgeCV(alphas=ridge_shrinkage)),
            'log-diag': make_pipeline(ProjIdentitySpace(), LogDiag(),
                                    StandardScaler(),
                                    RidgeCV(alphas=ridge_shrinkage)),
            'spoc': make_pipeline(
                    ProjSPoCSpace(n_compo=n_compo,
                                scale=scale, reg=0, shrink=0.5),
                    LogDiag(),
                    StandardScaler(),
                    RidgeCV(alphas=ridge_shrinkage)),
            'riemann':
                make_pipeline(
                    ProjCommonSpace(scale=scale, n_compo=n_compo, reg=1.e-05),
                    Riemann(n_fb=n_fb, metric=metric),
                    StandardScaler(),
                    RidgeCV(alphas=ridge_shrinkage)),
            'riemann_gamma': #GammaRegressor
                make_pipeline(
                    ProjCommonSpace(scale=scale, n_compo=n_compo, reg=1.e-05),
                    Riemann(n_fb=n_fb, metric=metric),
                    StandardScaler(),
                    GammaRegressor())
            }    
        
        if tune_components == True:
            
            low_rank_estimators = {k: v for k, v in pipelines.items()
                                if k in ('riemann')} #'spoc', 

            out_list = Parallel(n_jobs=n_jobs)(delayed(run_low_rank)(
                                n_components=cc, X=X, y=y,
                                groups=groups,
                                cv=GroupShuffleSplit(
                                    n_splits=2, train_size=.5, test_size=.5),
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

            best_components = {
            #'spoc': mean_df['n_components'][mean_df['spoc'].argmax()],
            'riemann': mean_df['n_components'][mean_df['riemann'].argmax()]
            }

            riemann_model = make_pipeline(
            ProjCommonSpace(scale=scale, n_compo=best_components['riemann'],
                            reg=1.e-05),
            Riemann(n_fb=n_fb, metric=metric),
            StandardScaler(),
            estimator())
        
        else:
            riemann_model = make_pipeline(
            ProjCommonSpace(scale=scale, n_compo= 32,
                            reg=1.e-05),
            Riemann(n_fb=n_fb, metric=metric),
            StandardScaler(),
            estimator)
        
        cv = KFold(n_splits=2, shuffle=False)

        y_preds = cross_val_predict(riemann_model, X, y, cv=cv)
        
        r2_riemann_model = cross_val_score(riemann_model, X, y, cv=cv,  groups=groups)
        print("mean of R2 cross validation Riemannian Model : ", np.mean(r2_riemann_model))
        
        for scoring in ("r2", "neg_mean_absolute_error"):
            all_scores = dict()
            for key, estimator in pipelines.items():
                cv = GroupShuffleSplit(n_splits=2, train_size=.5, test_size=.5)
                scores = cross_val_score(X=X, y=y, estimator=estimator,
                                        cv=cv, n_jobs=min(2, n_jobs),
                                        groups=groups,
                                        scoring=scoring)
                if scoring == 'neg_mean_absolute_error':
                    scores = -scores
                all_scores[key] = scores
            score_name = scoring if scoring == 'r2' else 'mae'
            np.save(op.join('data',
                            f'all_scores_models_fieldtrip_spoc_{score_name}.npy'),
                    all_scores)
        
        exp.append [i, func_args, raw, eda, ec, y, y_preds]
        
    return exp
        
#%%
experiment_results = pipeline_all_subjects(number_subject=['01', '02'],
                                           crop = True,
                                           tune_components = False)

#%%
fig, ax = plt.subplots(1, 1, figsize=[10, 4])
times = raw.times[ec.events[:, 0] - raw.first_samp]
ax.plot(times, y_preds, color='b', label='Predicted EDA', linewidth=2, alpha=0.3)
ax.plot(times, y, color='r', label='True EDA', linewidth=2, alpha=0.3)
ax.set_xlabel('Time (s)')
ax.set_ylabel('EDA mean')
ax.set_title('Riemann EDA Predictions')
plt.legend()
#plt.xlim(0,300)
mne.viz.tight_layout()
plt.show()

# %%
"""
Integrar todo en la funcion final
-  Dataframe: use itertools
-  Que se pueda correr en Drago

"""