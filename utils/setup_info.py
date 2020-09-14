from utils.helper_funcs import *
import numpy as np
import matplotlib.pyplot as plt

# info about subjects and filenames
bdf_home = 'D:/Flicker_data/'
response_filename = 'data\EEG_Flicker_Questionnaire_Spread_8.19.20.xlsx'
sub_names = ['Till', 'Timo', 'Adu']  # subject names
sub_IDs = ['01TN', '02TS', '03AM']  # .bdf file IDs
cmap = plt.get_cmap('jet')
sub_colors = dict(zip(sub_IDs, [cmap(i) for i in np.linspace(0, 1, len(sub_IDs))]))
ID_to_name = dict(zip(sub_IDs, sub_names))
name_to_ID = dict(zip(sub_names, sub_IDs))

# pre-processing params
lcutoff = 2  # low frequency cutoff
hcutoff = 45  # high frequency cutoff
ds_hz = 1000  # downsampling hz
fft_step = .05  # fourier transform resolution

# analysis parameters, in Hz and seconds
test_freqs = [8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12.]
runs = range(2)
flick_block_len = 180
flick_on_bounds = (-5, 185)
flick_off_bounds = (0, 60)  # only ~ 1 minute, to be safe
pulse_on_bounds = (-1, 1)
rest_bounds = (-5, 60)  # only 1 minutes, to be safe (Adu's final rest epoch clipped early, ~71s)
session_events = ['rest', 'flick_on', 'flick_off', 'pulse_on']
len_events = dict(zip(session_events, [rest_bounds, flick_on_bounds, flick_off_bounds, pulse_on_bounds]))
questions = ['I felt sleepy',
             'I felt bodiless',
             'I experienced space and time as if I were dreaming',
             'I felt I was in another, wonderful world',
             'I saw regular patterns',
             'I saw colors before me',
             'I saw shapes',
             'The dynamics were constantly changing',
             'I felt annoyed']
SSVEP_band = (5, 15)  # band over which to calculate SSVEP
ref_channels = ['FCz']  # reference channel(s) to apply to EEG before analysis

# raw bdf channels by 1020
chs_not_1020 = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1',
                'Erg2', 'Resp', 'Plet', 'Temp']
chs_not_1020_used = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'Erg1']
chs_not_1020_unused = ['EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg2', 'Resp', 'Plet', 'Temp']
chs_in_1020 = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
               'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
               'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
               'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
op_channels = list(filter(lambda x: x.startswith('P') or x.startswith('O'), chs_in_1020))  # occipitoparietal channels

# event_dict = dict(init_rest=1, flick_test=2, flick_on=3, flick_off=4, pulse=5)  # event dictionary for MNE epochs

# creating expanded dicitionary to include each frequency
event_names = make_event_name_list(test_freqs)
event_dict = dict(zip(event_names, range(len(event_names))))

# # creating dict of event times for each subject
sub_times = dict()
# flick_on times should be first sec after block start
# pulse_on times should be first .2 sec after pulse start

# Till
Till_times = dict(init_rest_on=np.array([6]),
                  flick_on=np.array(
                      [316, 621, 885, 1129, 1387, 1634, 1878, 2127, 2396,
                       2636, 2892, 3160, 3529, 3767, 4013, 4290, 4534, 4771]),
                  pulse_on=np.array(
                      [5058.51, 5063.55, 5069.52, 5074.51, 5078.62, 5082.68, 5086.57,
                       5091.49, 5096.52, 5100.58, 5106.61, 5112.52, 5116.64, 5121.54, 5125.75]),
                  bad_channels=['P2', 'FC1', 'FC3', 'F1', 'F3'])
Till_times.update(dict(end_rest_on=np.array([Till_times['flick_on'][-1] + 300])))
sub_times.update(dict(Till=Till_times))

# Timo
Timo_times = dict(init_rest_on=np.array([9]),
                  flick_on=np.array([289, 596, 908, 1199, 1483, 1772, 2057, 2369, 2653,
                                     3000, 3292, 3597, 3904, 4190, 4493, 4787, 5091, 5364]),
                  pulse_on=np.array([5881.71, 5886.32, 5888.70, 5891.34, 5900.5, 5907.46,
                                     5910.37, 6045.72, 6049.09, 6052.73, 6062.75, 6065.74, 6068.80,
                                     6072.77, 6079.72, 6082.72, 6085.76, 6088.73, 6092.08]),
                  bad_channels=['P2', 'AF3', 'FC4'])
Timo_times.update(dict(end_rest_on=np.array([Timo_times['flick_on'][-1] + 300])))
sub_times.update(dict(Timo=Timo_times))

# Adu
Adu_times = dict(init_rest_on=np.array([6]),
                 flick_on=np.array([260, 549, 807, 1073, 1370, 1661, 1949, 2286, 2579,
                                    2900, 3279, 3358, 3849, 4177, 4450, 4774, 5658, 5975]),
                 pulse_on=np.array([5450.57, 5461.56, 5471.72, 5474.52, 5484.49, 5491.51, 5495.52, 5499.80,
                                    5499.52, 5502.57, 5518.57, 5521.50, 5525.54, 5534.72, 5554.51, 5556.57]),
                 bad_channels=['F1', 'AF7', 'C3', 'C5', 'CP3', 'AF8', 'F8', 'P2'],
                 end_rest_on=np.array([6166]))  #end at 6226, only
sub_times.update(dict(Adu=Adu_times))
