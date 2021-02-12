import os
import mne
from EEG_channels_twente import channels 
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
import numpy as np

# Read bdf
number_subject = '25'
path = os.path.join('data', 's'+ number_subject + '.bdf')
s1 = mne.io.read_raw_bdf(path, preload=True)
# Print info of the subject 1 signal
print(s1.info)

# Select EDA data
s1_temp = s1.copy()
print('Number of channels in s1_temp:')
print(len(s1_temp.ch_names), end=' → pick only EDA → ')
s1_temp.pick_channels(['GSR1'])
#s1_temp.crop(tmax = (s1_temp.n_times - 1) / s1_temp.info['sfreq'])
print(len(s1_temp.ch_names))
# Plot  EDA. Plot only first part of the signal
#%matplotlib qt
#s1_temp.plot(title='EDA' , scalings='auto')

# Plot the EDA power spectral density 
#s1_temp.plot_psd()
# Create dataframe of EDA subject 1
df_s1_EDA = s1_temp.to_data_frame()

#Rename column
df_s1_EDA.rename(columns={'GSR1': 'EDA'}, inplace=True)

# Transform EDA (participant 23-32 in Geneva) --> GSR geneva = 10**9 / GSR twente
if int(number_subject) < 23:
    #df_s1_EDA["EDA"] = 10**9/df_s1_EDA["EDA"]
    print('funca')
else:
    df_s1_EDA["EDA"] = (10**9/df_s1_EDA["EDA"])*1000
    print('funca')
    
# Plot EDA: whole data v.2
df_s1_EDA.plot.line(ylim= x='time', y='EDA')




# Plot EDA: whole data
#sns.lineplot(x="time", y="GSR1",
#             data=df_s1_EDA)



# Select only EEG data
#s1_temp2 = s1.copy()
##print('Number of channels in s1_temp:')
##print(len(s1_temp2.ch_names), end=' → pick only EEG → ')
##s1_temp2.pick_channels(channels)
##print(len(s1_temp2.ch_names))
# Plot EEG 
#s1_temp2.plot()
# Plot the EEG power spectral density 
#s1_temp2.plot_psd()
# Create dataframe EEG subject 1
#df_s1_EEG = s1_temp2.to_data_frame()