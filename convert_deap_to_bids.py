#%%
import os
import pathlib
from pathlib import Path
import mne
from mne_bids import write_raw_bids, BIDSPath, print_dir_tree, read_raw_bids
from mne_bids.stats import count_events

# setup paths
input_path = Path('/storage/store/data/DEAP/original_biosemi_data/')

bids_root = Path('/storage/store2/derivatives/DEAP-bids')

# declare montage
montage = mne.channels.make_standard_montage(kind="biosemi32", head_size=0.095)

if not bids_root.exists():
    os.makedirs(bids_root)

task = 'rest'

for path in input_path.glob("*.bdf"):

    raw = mne.io.read_raw_bdf(path)

    # Rename channel EDA 
    raw.rename_channels(mapping={'GSR1': 'EDA'})
                      
    raw.set_channel_types({'EXG1': 'eog',
                           'EXG2': 'eog',
                           'EXG3': 'eog',
                           'EXG4': 'eog',
                           'EXG5': 'emg',
                           'EXG6': 'emg',
                           'EXG7': 'emg',
                           'EXG8': 'emg',
                           'EDA': 'misc',
                           'GSR2': 'misc',
                           'Erg1': 'misc',
                           'Erg2': 'misc',
                           'Resp': 'misc',
                           'Plet': 'misc',
                           'Temp': 'misc'}) 
    
    subject_id = path.name.split('.bdf')[0] 
    subject_number = int(subject_id.strip('s'))
    subject_id = subject_id.replace('s', '')

    if subject_number > 28:
        raw.rename_channels(mapping={'-1': 'Status'})
        raw.drop_channels('-0')

    elif subject_number > 23:
        raw.rename_channels(mapping={'': 'Status'})
        
    raw.set_channel_types({ 'Status': 'stim'})

    raw.set_montage(montage)
    
    raw.info['line_freq'] = 50  # specify power line frequency as required by BIDS

    # Create events based on stim channel
    events = mne.find_events(raw, stim_channel='Status')

    if subject_number < 24:
        event_id = {'rating_screen': 1,
                    'video_synch': 2,
                    'fixation_screen': 3,
                    'music_stim': 4,
                    'fixation_screen_after': 5,
                    'unknown': 6,
                    'unknown2': 65536,
                    'end_exp': 7}
    else:
        event_id = {'rating_screen': 1638145,
                    'fixation_screen_after': 1638149,
                    'fixation_screen': 1638147,
                    'music_stim': 1638148,
                    'video_synch': 1638146,
                    'end_exp': 1638151,
                    'unknown_1': 5832448,
                    'unknown_2': 5832449,
                    'unknown_3': 5832451,
                    'unknown_4': 5832452,
                    'unknown_5': 5832453}

    bids_path = BIDSPath(subject=subject_id, task=task, root=bids_root)
    write_raw_bids(raw, bids_path, event_id=event_id,
                   events_data=events, overwrite=True)

print_dir_tree(bids_root)

counts = count_events(bids_root)
counts


#%%
raw = read_raw_bids(bids_path=bids_path)

# %%
