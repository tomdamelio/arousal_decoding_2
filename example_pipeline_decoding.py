# Author: Alexandre Barachant <alexandre.barachant@gmail.com>
#         Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)
# Link https://mne.tools/dev/auto_examples/decoding/plot_decoding_spoc_CMC.html#sphx-glr-auto-examples-decoding-plot-decoding-spoc-cmc-py

#%%

import matplotlib.pyplot as plt

import mne
from mne import Epochs
from mne.decoding import SPoC
from mne.datasets.fieldtrip_cmc import data_path

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

from my_functions import extract_signal

# Define parameters
number_subject = '01'

#Extract signal
raw = extract_signal(directory = 'data', number_subject=number_subject,
                     extension = '.bdf')

# Rename channel EDA and set GSR as channel type
mne.rename_channels(info= raw.info , mapping={'GSR1':'EDA'})
raw.set_channel_types({'EDA': 'misc'})
#%%

# Pick EDA
eda = raw.copy().pick_channels(['EDA'])
# load_data would not be necessary. 
# The function load_data() returns a list of paths that the requested data files located.

# Filter EDA:
#  - Low pass  --> 5.00 Hz
#  - High pass --> 0.05 Hz

eda.filter(0.05, 5., fir_design='firwin', picks=['EDA'])

#%%
# Select and filter EEG data (not EOG)
raw.pick_types(meg=False, ref_meg=False, eeg=True, eog=False).load_data()
raw.filter(0.1, 120., fir_design='firwin')

#%%
# Build epochs as sliding windows over the continuous raw file
events = mne.make_fixed_length_events(raw, id=1, duration=10.0, overlap= 2.0)

# Epoch length is 1.5 second
eeg_epochs = Epochs(raw, events, tmin=0., tmax=1.500, baseline=None,
                    detrend=1, decim=8)
eda_epochs = Epochs(eda, events, tmin=0., tmax=1.500, baseline=None)


#%%
# Prepare classification
X = eeg_epochs.get_data()
#y = eda_epochs.get_data().var(axis=2)[:, 0]  # target is EDA power
y = eda_epochs.get_data()

# Classification pipeline with SPoC spatial filtering and Ridge Regression
spoc = SPoC(n_components=2, log=True, reg='oas', rank='full')
clf = make_pipeline(spoc, Ridge())
# Define a two fold cross-validation
cv = KFold(n_splits=2, shuffle=False)

#%%
# Run cross validaton
y_preds = cross_val_predict(clf, X, y, cv=cv)

# Plot the True EMG power and the EMG power predicted from MEG data
fig, ax = plt.subplots(1, 1, figsize=[10, 4])
times = raw.times[eeg_epochs.events[:, 0] - raw.first_samp]
ax.plot(times, y_preds, color='b', label='Predicted EMG')
ax.plot(times, y, color='r', label='True EMG')
ax.set_xlabel('Time (s)')
ax.set_ylabel('EDA Power')
ax.set_title('SPoC EEG Predictions')
plt.legend()
mne.viz.tight_layout()
plt.show()
# %%