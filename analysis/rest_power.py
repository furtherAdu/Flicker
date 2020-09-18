from utils.mne_funcs import read_eeg
from utils.st_adjudication_funcs import discretize_into_events, calc_psd
from utils.setup_info import ID_to_name, event_dict, len_events, hcutoff, lcutoff, op_chs, \
    bdf_home, sub_IDs, ds_hz, fft_step, test_freqs, rest_bounds
from utils.helper_funcs import save_obj, load_obj
import numpy as np
import os

# load the master subject dictionary
if os.path.isfile('data/subs_dict_psds_alpha_SSVEP.pkl'):
    subs_dict = load_obj('data/subs_dict_psds_alpha_SSVEP.pkl')
else:
    print('running SSVEP.py')
    from analysis import SSVEP

    subs_dict = load_obj('data/subs_dict_psds_alpha_SSVEP.pkl')


def calc_rest_psd(subs_dict, rest_type):
    assert rest_type in ['init', 'flicker', 'end']

    if rest_type == 'flicker': #  calculating all post-flicked PSDs
        which = [f'rest/{x}' for x in test_freqs]
    else:
        which = rest_type

    # looping over subjects for analysis
    for sub_ID in sub_IDs:
        name = ID_to_name[sub_ID]
        block_order = subs_dict[name]['response']['Frequency (Hz)']

        # preprocess/load data
        processed_1020, events = read_eeg(sub_ID, subs_dict=subs_dict, ds_hz=ds_hz, load_from_file=True,
                                          replace_events=False, save=False)

        # TODO: fix for last post-flicker rest epoch in Adu's ['TOO SHORT']
        rest_epochs = discretize_into_events(processed_1020, which=which, sub_ID=sub_ID, events=events,
                                             len_events=len_events, event_dict=event_dict, bdf_home=bdf_home,
                                             save=False)

        # flicker_psd.shape == (epoch x channel x frequencies)
        rest_psd, rest_psd_freqs = calc_psd(rest_epochs, psd_channels=op_chs, lcutoff=lcutoff,
                                            hcutoff=hcutoff, fft_step=fft_step, ds_hz=ds_hz,
                                            calc_method='multitaper')

        # determining the frequencies matching to dimension in rest_psd
        rest_freqs = np.array([x for x in block_order if type(x) != str])
        mne_ff = np.array([float(list(event_dict.keys())[list(event_dict.values()).index(x)][len('rest')+1:])
                           for x in rest_epochs.events[:, 2]])

        assert len(rest_freqs) == len(mne_ff), 'Length mismatch between MNE-derived and excel-derived test frequency'
        assert all(rest_freqs == mne_ff), 'Frequency mismatch between MNE-derived and excel-derived test frequency'

        subs_dict[name].update({f'rest_{rest_type}_psd': rest_psd, f'rest_{rest_type}_psd_freqs': rest_psd_freqs})

    return subs_dict


# running SSVEP analysis and plotting
print('calculating rest PSD...')
subs_dict = calc_rest_psd(subs_dict, rest_type='flicker')
print('saving subs_dict with rest PSD data...')
save_obj(subs_dict, 'data/subs_dict_psds_alpha_SSVEP_rest.pkl')
