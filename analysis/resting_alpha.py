# Calculates PSD and resting alpha

from utils.mne_funcs import read_eeg
from utils.st_adjudication_funcs import compile_all_session_info, discretize_into_events, identify_resting_alpha
from utils.setup_info import ID_to_name, event_dict, len_events, op_chs, \
    bdf_home, sub_IDs, ds_hz, fft_step
from utils.helper_funcs import save_obj, load_obj
import os

if os.path.isfile('data/subs_dict.pkl'):
    subs_dict = load_obj('data/subs_dict.pkl')
else:
    subs_dict = compile_all_session_info()

for sub_ID in sub_IDs:  # sub_IDs:

    name = ID_to_name[sub_ID]

    # processing data
    processed_1020, events = read_eeg(sub_ID, subs_dict=subs_dict, ds_hz=ds_hz, load_from_file=True,
                                      replace_events=False, save=False)

    # getting epochs from pulses
    pulse_epochs = discretize_into_events(processed_1020, which='pulse', sub_ID=sub_ID, events=events,
                                          len_events=len_events, event_dict=event_dict, bdf_home=bdf_home, save=False)

    # calculating power over epochs
    psds_b_pulse, psds_a_pulse, freqs, r_alpha, calc_method = \
        identify_resting_alpha(pulse_epochs, psd_channels=op_chs, sub_ID=sub_ID, calc_method='tfr', plot=False,
                               fft_step=fft_step)

    # updating the master subject data dictionary
    subs_dict[name].update(dict(power_after_pulse=psds_a_pulse, power_before_pulse=psds_b_pulse,
                                freqs=freqs, r_alpha=r_alpha, calc_method=calc_method))
    del processed_1020

for sub_ID in sub_IDs:  # sub_IDs:
    print(subs_dict[name]["r_alpha"])

# save sub_dicts
save_obj(subs_dict, 'data/subs_dict.pkl')
