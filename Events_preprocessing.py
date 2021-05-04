#%%
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import Epochs
from mne.decoding import SPoC
from mne.datasets.fieldtrip_cmc import data_path
from mne import pick_types

from autoreject import get_rejection_threshold
from autoreject.autoreject import _GlobalAutoReject
from autoreject.bayesopt import expected_improvement, bayes_opt

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, check_cv

from preprocessing import extract_signal, transform_negative_to_zero, out_of_range, get_rejection_threshold
from channel_names import channels_geneva, channels_twente 
import os.path as op
from subject_number import subject_number

def save_bad_no_stim_annotations (subject_number = subject_number,
                                  save_EDA_EEG_bad_resp = False,
                                  plot_EDA_EEG_bad_resp = False,
                                  plot_events = True,
                                  plot_annotations = True,
                                  save_EDA_EEG_bad_no_stim = False):
    
    """
    First, join EDA, EEG and bad respiraition annotations (and save it).
    Then, mark and  remove all the events that do not correspond to stimulus
    presentation (e.g. pre-experiement, self-reporting, inter-block interval,
    after last stimulus).

    :subject_number:            List. Subjects to analyze. Default = all subject.
    :save_EDA_EEG_bad_resp:     Boolean. Save EDA, EEG and bad respiraition annotations .fif file
    :plot_EDA_EEG_bad_resp:     Boolean. Plot EDA and EEG with bad respiraition annotations.
    :plot_events:               Boolean. Plot all channels with events marks.
    :plot_annotations:          Boolean. Plot all channles with bad_no_stim annotations.
    :save_EDA_EEG_bad_no_stim:  Boolean. Save all channels with bad_no_stim annotations. 

    """ 
    subject_number = ['01']
    for i in subject_number: 
        
        # Read .fif files (with respiration annotations)
        directory = 'outputs/data/EDA+Resp_with_resp_annotations/'
        number_subject = i
        extension = '.fif'
        fname = op.join(directory + 's'+ number_subject + extension)

        raw_fif = mne.io.read_raw_fif(fname, preload=True) 

        # Read bdf files (without annotations) --> all channels
        extension = '.bdf'
        directory = 'data/'
        #Read bdf
        fname_bdf = op.join(directory + 's'+ number_subject + extension)
        raw_bdf = mne.io.read_raw_bdf(fname_bdf, preload=True) 

        mne.rename_channels(info= raw_bdf.info , mapping={'GSR1':'EDA'})

        raw_bdf.set_channel_types({ 'EXG1': 'eog',
                                    'EXG2': 'eog',
                                    'EXG3': 'eog',
                                    'EXG4': 'eog',
                                    'EXG5': 'emg',
                                    'EXG6': 'emg',
                                    'EXG7': 'emg',
                                    'EXG8': 'emg',
                                    'EDA' : 'misc',
                                    'GSR2': 'misc',
                                    'Erg1': 'misc',
                                    'Erg2': 'misc',
                                    'Resp': 'misc',
                                    'Plet': 'misc',
                                    'Temp': 'misc'})

        # Pick STIM, EEG and EOG from .bdf
        raw_bdf2 = raw_bdf.pick_types(stim = True, eeg = True, eog = True,
                                    misc = False, emg = False)

        # Add channels from raw_bdf (stim, EEG and EOG) to raw_fif (resp and EDA)
        raw_fif.add_channels([raw_bdf2])

        # plot EDA and resp
        if plot_EDA_EEG_bad_resp == True:    
            %matplotlib
            raw_fif.plot(order=[1,0], scalings=dict(misc='1e-1', emg='1e-1'))

        # In case in want to resave EDA + EEG + bad_resp annotations
        extension = '.fif'
        directory = 'outputs/data/EDA+EEG+bad_resp/'
        fname_2 = op.join(directory,'s'+ number_subject + extension)
        
        if save_EDA_EEG_bad_resp == True:
            raw_fif.save(fname = fname_2, overwrite=True)
            
        #######################################################################

        if int(i) > 28:
            raw_fif.rename_channels(mapping={'-1': 'Status'} )
            raw_fif.drop_channels('-0')
        
        elif int(i) > 23:
            raw_fif.rename_channels(mapping={'': 'Status'} )
        
        # Create events based on stim channel
        events = mne.find_events(raw_fif, stim_channel='Status')
        
        if int(i) < 24:
            mapping = { 1: 'rating_screen',
                        2: 'video_synch',
                        3: 'fixation_screen ',
                        4: 'music_stim',
                        5: 'fixation_screen_after',
                        7: 'end_exp'}
        else:
            mapping = { 1638145: 'rating_screen',
                        1638149: 'fixation_screen_after ',
                        1638147: 'fixation_screen',
                        1638148: 'music_stim',
                        1638146: 'video_synch',
                        1638151: 'end_exp',
                        }
            
        annot_from_events = mne.annotations_from_events(
            events=events, event_desc=mapping, sfreq=raw_fif.info['sfreq'],
            orig_time=raw_fif.info['meas_date'])
        raw_fif.set_annotations(annot_from_events)
        
        if plot_events == True:
            %matplotlib
            raw_fif.plot()

        ##### Programmatically annotate bad signal ####
        # bad signal annotate as 'bad_no_stim':
        # - Time before the first trial
        # - Time between trials 
        # - Time between blocks (only one bad_no_stim in each subject)
        # - Time after the experimental task
        
        # Select events with stim value == 5 [fix after] or 3 [fixation]
        # Between this two events -->  music stimulus
        
        # Select events of fixation screen after trial [5]
        if int(i) < 24:
            rows_fix_after = np.where(events[:,2] == 5)
        else:
            rows_fix_after = np.where(events[:,2] == 1638149)    
        events_fix_after= events[rows_fix_after]
        onset_fix_after = events_fix_after[:,0]/raw_fif.info['sfreq'] 
        

        # Select events of fixation screen before beginning of trial [3]
        if int(i) < 24:
            rows_fix_before = np.where(events[:,2] == 3)
        else:
            rows_fix_before = np.where(events[:,2] == 1638147)
        events_fix_before = events[rows_fix_before]
        onset_stim_fix_before = events_fix_before[:,0]/raw_fif.info['sfreq']
        onset_stim_fix_before_2 = onset_stim_fix_before[1:]
        
        if int(i) > 22:
            # Sobra un valor de onset_stim_fix_before_2 al inicio del registro
            onset_stim_fix_before_2 = onset_stim_fix_before_2[1:]
            # Sobra un valor de onset_fix_after antes del baseline
            onset_fix_after = onset_fix_after[2:]
            # Delete index 22 from onset_fix_after
            # instead of videosync there is an extra fixation mark
            if int (i) == 28:
                onset_stim_fix_before_2 = np.delete(onset_stim_fix_before_2, 22)

            
        onset_stim_fix_before_2 = np.append(onset_stim_fix_before_2, (len(raw_fif)/raw_fif.info['sfreq']))

        diff_onset_music_stim = onset_stim_fix_before_2 - onset_fix_after

        # Select events pre first music stimulus
        if int(i) > 22:
            diff_onset_music_stim = np.append(onset_stim_fix_before[1], diff_onset_music_stim)
        else:
            diff_onset_music_stim = np.append(onset_stim_fix_before[0], diff_onset_music_stim)

            
        onset_music_stim= np.append(0, onset_fix_after)
        
        if int(i) == 28:
            n_stims = 37+1
        else:
            n_stims = 40+1
        
        later_annot = mne.Annotations(onset=onset_music_stim,
                                    duration=diff_onset_music_stim,
                                    description=['bad_no_stim']*n_stims)

        raw2 = raw_fif.copy().set_annotations(later_annot)
        
        if plot_annotations == True:
            %matplotlib
            raw2.plot(events=events)
        
        if save_EDA_EEG_bad_no_stim ==True:
            extension = '.fif'
            directory = 'outputs/data/EDA+EEG+bad_no_stim/'
            fname_3 = op.join(directory,'s'+ number_subject + extension)
            raw2.save(fname = fname_3, overwrite=True)

#%%
save_bad_no_stim_annotations()

# %%
