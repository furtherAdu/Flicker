# # For each flicker frequency block:
#       calculate channel-averaged PSD and take peaks as the SSVEP
#       calculate difference between (log) resting alpha and SSVEP peak (akak as_dif)
# for each question:
#        calculate the correlation, across frequencies, between as_dif and ratings

from utils.mne_funcs import read_eeg
from utils.st_adjudication_funcs import discretize_into_events, calc_psd
from utils.setup_info import ID_to_name, event_dict, len_events, hcutoff, lcutoff, op_channels, \
    bdf_home, sub_IDs, ds_hz, fft_step, test_freqs, SSVEP_band
from utils.helper_funcs import save_obj, load_obj
import numpy as np
import matplotlib.pyplot as plt
import os

# load the master subject dictionary
if os.path.isfile('data/subs_dict_psds_alpha.pkl'):
    subs_dict = load_obj('data/subs_dict_psds_alpha.pkl')
else:
    print('running resting_alpha.py')
    from analysis import resting_alpha  # run resting alpha analysis
    subs_dict = load_obj('data/subs_dict_psds_alpha.pkl')


def calc_SSVEP(subs_dict, SSVEP_band=(5, 15)):
    # looping over subjects for analysis
    for sub_ID in sub_IDs:
        name = ID_to_name[sub_ID]
        block_order = subs_dict[name]['response']['Frequency (Hz)']

        # preprocess/load data
        processed_1020, events = read_eeg(sub_ID, subs_dict=subs_dict, ds_hz=ds_hz, load_from_file=True,
                                          replace_events=False, save=False)

        flicker_epochs = discretize_into_events(processed_1020, which='flick', sub_ID=sub_ID, events=events,
                                                len_events=len_events, event_dict=event_dict, bdf_home=bdf_home,
                                                save=False)

        # flicker_psd.shape == (epoch x channel x frequencies)
        flicker_psd, flicker_psd_freqs = calc_psd(flicker_epochs, psd_channels=op_channels, lcutoff=lcutoff,
                                                  hcutoff=hcutoff, fft_step=fft_step, ds_hz=ds_hz,
                                                  calc_method='multitaper')

         # TODO: interpolation to freqs == np.arange(2,45+.1,.1)

        # determining the frequencies matching to dimension in flicker_psd
        flicker_freqs = np.array([x for x in block_order if type(x) != str])
        mne_ff = np.array([float(list(event_dict.keys())[list(event_dict.values()).index(x)][len('flick')+1:])
                           for x in flicker_epochs.events[:, 2]])
        assert all(flicker_freqs == mne_ff), 'Mismatch between MNE-derived and excel-derived test frequency'

        ca_flicker_psd = flicker_psd.mean(axis=1)  # average over channels
        ca_flicker_psd -= ca_flicker_psd.mean(axis=0)  # subtract avg VEP (hurrah for broadcasting)

        # calculate SSVEPs, as peaks in PSD
        if SSVEP_band:  # mask PSD entries with corresponding frequencies outside of band
            not_in_band = np.array([flicker_psd_freqs < SSVEP_band[0]]) + np.array([flicker_psd_freqs > SSVEP_band[1]])
            temp = ca_flicker_psd.copy()
            temp[:, not_in_band.squeeze()] = -np.inf
            SSVEPs = flicker_psd_freqs[np.nanargmax(temp, axis=1)]

        else:
            SSVEPs = flicker_psd_freqs[np.nanargmax(ca_flicker_psd, axis=1)]

        subs_dict[name].update(dict(SSVEPs=SSVEPs, flicker_psd=flicker_psd, flicker_psd_freqs=flicker_psd_freqs))

    return subs_dict


def plot_PSDs_per_flicker_freq(subs_dict):
    # looping over subjects for plotting
    for sub_ID in sub_IDs:
        # setting up figure
        fig, axs2 = plt.subplots(3, 3, figsize=(7, 7), sharex=True, sharey=True)
        fig.subplots_adjust(hspace=.3, wspace=.05)
        fig.suptitle(f'Log-PSD by flicker frequency, {sub_ID}\n(averaged over trials)')
        axs2 = axs2.ravel()

        name = ID_to_name[sub_ID]
        response = subs_dict[name]['response']
        flicker_psd_freqs = subs_dict[name]['flicker_psd_freqs']
        flicker_psd = subs_dict[name]['flicker_psd']

        flicker_freq_order = np.array([x for x in response['Frequency (Hz)'] if type(x) != str])  # order of flicker frequency exposure
        ca_flicker_psd = flicker_psd.mean(axis=1)  # channel averaged PSD

        # plot, doubling up each frequency's psd in one subplot
        for i, test_freq in enumerate(test_freqs):
            axs2[i].set_title(f'{test_freq} Hz')

            test_freq_args = np.argwhere(flicker_freq_order == test_freq).squeeze()

            # plotting blocks' average
            axs2[i].plot(flicker_psd_freqs, np.log(ca_flicker_psd[test_freq_args, :].mean(axis=0)))

            # # plotting individual blocks
            # for j, flicker_freq in enumerate(test_freq_args):
            #     axs2[i].plot(flicker_psd_freqs, np.log(ca_flicker_psd[flicker_freq]), label=f'run {j}')

            if i >= 6:
                axs2[i].set_xlabel('EEG Frequency (Hz)')

            if i in [0, 3, 6]:
                axs2[i].set_ylabel('Log-Power (V^2 / Hz)')

        # axs2[0].legend()

        plt.show()
        fig.savefig(f'figures/PSD_by_flicker_freq/PSD_by_flicker_freq_{sub_ID}.png')


# running SSVEP analysis and plotting
print('calculating SSVEP...')
subs_dict = calc_SSVEP(subs_dict, SSVEP_band)
print('saving subs_dict with SSVEP data...')
save_obj(subs_dict, 'data/subs_dict_psds_alpha_SSVEP.pkl')

# plot_PSDs_per_flicker_freq(subs_dict)

# TODO: analysis if change in Alpha power will depend on the FLS frequency
