# # spatiotemporal adjudication functions -- reading event times, bad channels, and questionnaire responses

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from utils.helper_funcs import myround
from utils.setup_info import sub_IDs, test_freqs, ID_to_name, runs, lcutoff, hcutoff, fft_step, questions, \
    v1_chs, v4_chs, v5_chs
from mne.time_frequency import tfr_multitaper
import xarray as xr
from pingouin import normality, wilcoxon, ttest
from scipy.stats import pearsonr
from mne.viz.utils import center_cmap
from mne.stats import permutation_cluster_1samp_test as pcluster_test

from utils.setup_info import event_dict, flick_block_len, response_filename, sub_names, sub_times, flick_on_bounds
from utils.helper_funcs import make_event_name_list, Bunch


def read_block_starts(response):
    """ Reads the block start times from a dataframe.

    :param response: dataframe of responses collected during flicker experiment
    :return: (array) for each event, seconds from first event
    """
    # reading in block start times
    starts = response['Start time'].to_numpy()  # start times for each block
    starts_sec = np.array([datetime.timedelta(hours=x.hour, minutes=x.minute).total_seconds() for x in starts])
    starts_sec -= starts_sec[0]  # block starts, in seconds, wrt first block

    return starts_sec


def get_eventarr_badchannels(sub_name, response_filename, sub_times):
    """ Not the most pythonic, but storing subject event times and bad channels, as adjudicated from the raw EEG signal
     in a function for easy access.

    :param sub_timing_dict: nested dictionary {sub_name: {event timings}}
    :param response_filename: spreadsheet with patient responses
    :param sub_name: subject name
    :return: (list) required event times to build event array for MNE
    """
    response = pd.read_excel(response_filename, sheet_name=sub_name, index_col=0, skiprows=1, header=None).T
    flick_order = np.array(list(filter(lambda x: type(x) is not str, response['Frequency (Hz)'])))

    # reading times from dictionary
    bunch = Bunch(sub_times[sub_name])

    # calculating intertwined flicker times
    flick_all = np.repeat(bunch.flick_on, 2)
    flick_all[1::2] += flick_block_len

    # create list of keys using the freq to determing event_ID
    sub_event_names = make_event_name_list(freqs=flick_order, pulses=bunch.pulse_on)

    # creating event array to pass to MNE functions
    events_col = np.concatenate([bunch.init_rest_on, flick_all, bunch.end_rest_on, bunch.pulse_on])  # ordered events
    zero_fill = np.zeros_like(events_col)
    id_fill = np.array([event_dict[x] for x in sub_event_names])

    event_arr = np.vstack([events_col, zero_fill, id_fill]).T  # event array necessary to pass to mne
    event_arr = event_arr[event_arr[:, 0].argsort()]  # sorting for nice plotting

    return event_arr, bunch.bad_channels


def compile_single_session_info(response_filename, sub_name, sub_times):
    # read in the spreadsheet info
    response = pd.read_excel(response_filename, sheet_name=sub_name, index_col=0, skiprows=1, header=None).T

    # extracting block starts as noted
    block_starts = read_block_starts(response)

    # create the event array
    event_arr, bad_channels = get_eventarr_badchannels(sub_name, response_filename, sub_times)

    return [response, block_starts, event_dict, event_arr, bad_channels]


# # creating master subs_dict to hold all event and questionnaire info
def compile_all_session_info():
    subs_session_info = dict()  # allocating
    sub_dict_keys = ['response', 'block_starts', 'event_dict', 'event_arr',
                     'bad_channels']  # keys to store in subject dictionary

    for name in sub_names:
        sub_vals = compile_single_session_info(response_filename, name, sub_times)  # calculate values
        sub_dict = dict(zip(sub_dict_keys, sub_vals))  # create subject's info dict
        sub_dict = {name: sub_dict}

        # update master sub_dict
        subs_session_info = {**subs_session_info, **sub_dict}

    return subs_session_info


def compile_ratings(subs_dict):
    print('\n', 'Compiling subject ratings')

    # allocating xarray for response, (subjects, test frequency, run, ratings)
    xr_ratings = np.ones((len(sub_IDs), len(test_freqs), len(runs), len(questions))) * np.nan
    xr_ratings = xr.DataArray(xr_ratings,
                              coords=dict(subject=sub_IDs, flicker_freq=test_freqs, run=runs, question=questions),
                              dims=['subject', 'flicker_freq', 'run', 'question'])

    # calculating and rating dispersion across all subjects
    for sub_ID in sub_IDs:
        name = ID_to_name[sub_ID]
        response = subs_dict[name]['response']

        epoch_freqs_args = [type(x) != str for x in response['Frequency (Hz)']]  # specifying frequency epochs

        # compiling responses
        for question in questions:
            rating = response[question].loc[epoch_freqs_args].to_numpy()
            for i, run in enumerate(runs):
                run_rating = rating[i * 9: (i + 1) * 9]  # 1st and 2nd half of frequencies
                run_epoch_freqs = response['Frequency (Hz)'].loc[epoch_freqs_args][i * 9: (i + 1) * 9]
                xr_ratings.loc[dict(subject=sub_ID, question=question,
                                    flicker_freq=run_epoch_freqs, run=run)] = run_rating

    return xr_ratings



def discretize_into_events(processed_1020, sub_ID, events, len_events, event_dict, bdf_home, which='pulse',
                           save=True):  # TODO: implement load

    event_groups = ['flick', 'pulse', 'rest']

    assert type(which) == list or type(which) == str, "Param {which} must be a list or str"

    if type(which) == str:
        assert which in event_groups or which in list(event_dict.keys()), f'{which} not a valid epoch type'
        first_letters = which[0][:4]  # identify the starting letters
    elif type(which) == list:
        first_letters = which[0][:4]  # identify the starting letters
        assert all([x.startswith(first_letters) for x in which]), 'All events in {which} must be of same type'
        assert any([x.startswith(first_letters) for x in event_groups]), f'Only events starting with {event_groups} are valid'

    epochs_home = 'processed/epochs'

    # print('plotting the event distribution...')
    # plot_events(events, eeg=processed_1020, event_id=event_dict)  # plot event distribution

    print('discretizing file into epochs...')

    # TODO: add baseline correction to all epochs, if desired
    if 'flick'.startswith(first_letters):
        epochs = mne.Epochs(processed_1020.copy(), events, event_id=event_dict, event_repeated='drop',
                            tmin=len_events['flick_on'][0], tmax=len_events['flick_on'][1])['flick']

    elif 'rest'.startswith(first_letters):  # accommodates both group and specific rest epoch
        epochs = mne.Epochs(processed_1020.copy(), events, event_id=event_dict, event_repeated='drop',
                            tmin=len_events['rest'][0], tmax=len_events['rest'][1],
                            reject_by_annotation=False)[which]

    elif 'pulse'.startswith(first_letters):  # todo: confirm 'drop' helps with multiple event error
        epochs = mne.Epochs(processed_1020.copy(), events, event_id=event_dict['pulse'], event_repeated='drop',
                            tmin=len_events['pulse_on'][0], tmax=len_events['pulse_on'][1], baseline=None)

    if save and which in event_groups:
        epochs.save(os.path.join(bdf_home, epochs_home, f'{which}/{which}_{sub_ID}.fif'), overwrite=True)

    return epochs


# NOTE: method == "multitaper" or "welch" leads to resolution ~1Hz
def identify_resting_alpha(epochs, psd_channels, sub_ID, plot=False, calc_method='tfr', fft_step=.05,
                           lcutoff=2, hcutoff=45, ds_hz=1000):
    """Calculates power and estimates resting alpha as frequency with greatest ERDS decrease

    :param epochs: mne Epochs object with only the light pulse events
    :param psd_channels: channels over which to calcualte power
    :param plot: (bool) whether to plot results
    :param calc_method: calculation method, choose between  'multitaper', 'welch', 'tfr' (time-frequency ratio)
    :return: the power before and after the pulses, frequencies, resting alpha, and calculation method
    """
    assert calc_method in ['multitaper', 'welch', 'tfr'], 'Please choose another method.'

    win = 2 / lcutoff * ds_hz  # window size: two full cycles of the lowest frequency of interest

    # Time-frequency decomposition, clustering, and plotting
    if calc_method == 'tfr':
        print('calculating time-frequency analysis...')
        # Run TF decomposition overall epochs

        freqs = np.arange(4, 16 + fft_step, fft_step)  # only looking at 4 - 16 Hz range

        #  tfr.data.shape == (n_epochs, n_channels, n_freqs, n_times)
        tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, picks=psd_channels,
                             use_fft=True, return_itc=False, average=False,
                             decim=2)
        tfr.crop(-1, 1)  # define epochs around events (in s)

        # baseline = [-1, 0]  # baseline interval (in s)
        # tfr.apply_baseline(baseline, mode="percent")

        arg0 = np.abs(tfr.times).argmin()  # should be 0
        before = tfr.data[:, :, :, :arg0].mean(axis=-1)  # averaging over n_times
        after = tfr.data[:, :, :, arg0:].mean(axis=-1)

        psds_b, psds_a, freqs_a = before, after, tfr.freqs

        log_before = np.log(before.mean(axis=(0, 1)))  # averaging over n_epochs and n_channels, taking log
        log_after = np.log(after.mean(axis=(0, 1)))

        r_alpha = tfr.freqs[(log_before - log_after).argmax()]  # max ERD (decrease)

        # if plot:
        #     vmin, vmax = -2, 2  # set min and max ERDS values in plot
        #     cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
        #     kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
        #                   buffer_size=None)  # for cluster test
        #     event = "20"  # pulse event ID
        #     tfr_ev = tfr[event]  # select desired epochs for visualization
        #
        #     fig, axes = plt.subplots(5, 5, figsize=(20, 20),
        #                              gridspec_kw={"width_ratios": [10, 10, 10, 10, 1]})
        #     for row in range(axes.shape[0]):
        #         for ch, ax in enumerate(axes[row, :-1]):  # for each channel
        #             if ch == 19:
        #                 continue
        #             ch += row * 4
        #             # positive clusters
        #             _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=1, **kwargs)
        #             # negative clusters
        #             _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=-1,
        #                                          **kwargs)
        #
        #             # note that we keep clusters with p <= 0.05 from the combined clusters
        #             # of two independent tests; in this example, we do not correct for
        #             # these two comparisons
        #             c = np.stack(c1 + c2, axis=2)  # combined clusters
        #             p = np.concatenate((p1, p2))  # combined p-values
        #             mask = c[..., p <= 0.05].any(axis=-1)
        #
        #             # plot TFR (ERDS map with masking)
        #             tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
        #                                   axes=ax, colorbar=False, show=False, mask=mask,
        #                                   mask_style="mask")
        #
        #             ax.set_title(epochs.ch_names[ch], fontsize=10)
        #             ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        #             if not ax.is_first_col():
        #                 ax.set_ylabel("")
        #                 ax.set_yticklabels("")
        #
        #         fig.colorbar(axes[row, 0].images[-1], cax=axes[row, -1])
        #
        #     fig.suptitle(f"pulse ERDS, Subject: {sub_ID}")
        #     fig.show()

    elif calc_method == 'multitaper':
        print('calculating multitaper power spectral density...')

        # psds.shape == (n_epochs, n_channels, n_freqs)
        psds_b, freqs_b = mne.time_frequency.psd_multitaper(epochs, fmin=lcutoff, fmax=hcutoff,
                                                            picks=psd_channels, tmax=0)  # TODO: increase resolution
        psds_a, freqs_a = mne.time_frequency.psd_multitaper(epochs, fmin=lcutoff, fmax=hcutoff,
                                                            picks=psd_channels, tmin=0)
        assert all(freqs_a == freqs_b), 'Frequencies unequal, please check the multitaper output.'

        # calculating r_alpha
        r_alpha = freqs_a[(np.log(psds_a) - np.log(psds_b)).mean(axis=(0, 1)).argmin()]  # channel- and epoch-mean of log-difference curves
        # r_alpha = freqs_a[(np.log(psds_a.mean(axis=(0, 1))) - np.log(psds_b.mean(axis=(0, 1)))).argmin()]
        # r_alpha = freqs_a[(np.log(psds_a.mean(axis=(0, 1))) - np.log(psds_b.mean(axis=(0, 1)))).argmin()]

    elif calc_method == 'welch':  # lower resolution, advisable not to use
        print('calculating welch power spectral density...')
        from scipy import signal

        # # Welch method with dataframe
        df = epochs.to_data_frame()
        before = df.loc[df.time < 0, psd_channels]
        after = df.loc[df.time > 0, psd_channels]

        # NOTE: for welch, the only thing that increases frequency resolution is time
        freqs_b, psds_b = signal.welch(before.T, fs=ds_hz, nperseg=win)  # TODO: save freq, psd in dataframe
        freqs_a, psds_a = signal.welch(after.T, fs=ds_hz, nperseg=win)

        r_alpha = freqs_a[(psds_a - psds_b).mean(axis=(0, 1)).argmin()]  # identifying resting alpha

    if plot:
        psds_a_log = np.log(psds_a)
        psds_b_log = np.log(psds_b)
        psds = psds_a_log - psds_b_log
        freqs = freqs_b

        psds_mean = psds.mean(0).mean(0)
        psds_std = psds.mean(0).std(0)

        f, ax = plt.subplots(2, sharex=True)
        ax = ax.ravel()
        ax[0].plot(freqs_a, psds_a_log.mean(axis=(0, 1)), label='before pulse')
        ax[0].plot(freqs_a, psds_b_log.mean(axis=(0, 1)), label='after pulse')

        ax[1].plot(freqs, psds_mean, color='k', label='difference')
        ax[1].fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                           color='k', alpha=.5, label='difference (with error)')
        f.suptitle('Multitaper PSD (gradiometers)\nchannel-averaged')
        ax[1].set(xlabel='Frequency (Hz)', ylabel='Power Spectral Density (log V^2 / Hz)')
        ax[0].set(ylabel='Power Spectral Density')
        ax[0].legend()
        ax[1].legend()

        plt.show()

        # # playing with plotting types
        # epochs.plot_psd_topomap() # topoplots by frequency bands
        # epochs.plot_projs_topomap(ch_type='eeg') # ZeroDivisionError: division by zero
        # epochs.plot_topo_image()  # plots per channel image maps, over all epochs

    return psds_b, psds_a, freqs_a, r_alpha, calc_method


def calc_psd(epochs, psd_channels, lcutoff=2, hcutoff=45, fft_step=.25, ds_hz=1000, calc_method='welch',
             epoch_length=None):

    """Calculates power spectral density

    :param epoch_length: (int) seconds after start of epochs to use to calculate PSD
    :param fft_step: frequency step over which to calculate PSD
    :param lcutoff:  high pass threshold
    :param hcutoff: low pass threshold
    :param epochs: mne Epochs object with the events
    :param psd_channels: channels over which to calculate power
    :return: the PSD over each event, frequencies
    """

    win = 2 / lcutoff * ds_hz  # window size: two full cycles of the lowest frequency of interest

    if calc_method == 'tfr':

        # Time-frequency decomposition
        print('calculating time-frequency analysis...')
        freqs = np.arange(lcutoff, hcutoff + fft_step, fft_step)  # only looking at specific range

        #  tfr.data.shape == (n_epochs, n_channels, n_freqs, n_times)
        tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, picks=psd_channels,
                             use_fft=True, return_itc=False, average=False,
                             decim=2)

        tfr.crop(0, epoch_length)  # define epochs around events (in s) #TODO: fix for rest epochs

        # baseline = [flick_on_bounds[0], 0]  # baseline interval (in s)
        # tfr.apply_baseline(baseline, mode="mean")  # subtract mean of first 5 seconds

        psd = tfr.data.mean(axis=-1)  # averaging over seconds

    elif calc_method == 'multitaper':
        print('calculating multitaper power spectral density...')

        # psd.shape == (n_epochs, n_channels, n_freqs)
        psd, freqs = mne.time_frequency.psd_multitaper(epochs, fmin=lcutoff, fmax=hcutoff,
                                                       picks=psd_channels)

    elif calc_method == 'welch':  # lower resolution, advisable not to use
        print('calculating welch power spectral density...')
        from scipy import signal

        # # Welch method with dataframe
        df = epochs.to_data_frame()
        df = df.loc[psd_channels]  # only channels of interest

        # NOTE: for welch, the only thing that increases frequency resolution is time
        freqs, psd = signal.welch(df.T, fs=ds_hz, nperseg=win)

    return psd, freqs


def calc_secondary_power_features(subs_dict, bin_edge='left', lcutoff=lcutoff, hcutoff=hcutoff, fft_step=fft_step):
    """
    Calculates SSVEP and of alpha power in bins.
    :param bin_edge: {'left', 'center'} bin start w.r.t. lcutoff.
        'center' bins from [lcutoff - fft_step/2, lcutoff + fft_step/2]
        'left' bins from [lcutoff, lcutoff + fft_step]
    :param fft_step: resolution of bins
    :param hcutoff: upper frequency limit to calculate power
    :param lcutoff: lower frequency limit to calculate power
    :param subs_dict: master subject dictionary containing PSDs of post-flicker rest and flicker epochs
    :return:(xarray) of power data, with coords {subjects, runs, feature, and flicker frequency}
    """
    assert bin_edge in ['left', 'center'], f"{bin_edge} is an invalid bin edge type."

    # allocating array to hold test results
    features = ['alpha_power_during', 'alpha_power_after', 'SSVEP']

    # .shape == # (subjects, flicker freqs, EEG freqs, alpha power/SSVEP, runs)
    secondary_power_features = np.ones((len(sub_IDs), len(test_freqs), len(features),
                                     len(runs))) * np.nan
    secondary_power_features = xr.DataArray(secondary_power_features,
                                         coords=dict(subject=sub_IDs, flicker_freq=test_freqs,
                                                     run=runs, feature=features),
                                         dims=['subject', 'flicker_freq', 'feature', 'run'])

    # looping over subjects for plotting
    for sub_ID in sub_IDs:
        name = ID_to_name[sub_ID]
        SSVEPs = subs_dict[name]['SSVEPs']
        psd_bin_res = fft_step  # resolution of binned psd
        psd_freqs = np.arange(lcutoff, hcutoff, psd_bin_res)  # the binned frequencies
        r_alpha = myround(subs_dict[name]['r_alpha'], base=psd_bin_res)  # rounding resting alpha
        flicker_psd_freqs = subs_dict[name]['flicker_psd_freqs']  # TODO: Note. floats represented as 2.0999999999999996
        rest_psd_freqs = subs_dict[name]['rest_flicker_psd_freqs']
        flicker_freqs = np.array([x for x in subs_dict[name]['response']['Frequency (Hz)'] if type(x) != str])

        # channel averaging PSDs
        ca_flicker_psd = subs_dict[name]['flicker_psd'].mean(axis=1)
        ca_rest_psd = subs_dict[name]['rest_flicker_psd'].mean(axis=1)

        if bin_edge == 'left':
            edges = (0, 1)
        elif bin_edge == 'center':
            edges = (-.5, .5)

        # left-edge binning and summing PSD, in bin_res resolution
        freq_bin_args = np.array([np.argwhere((flicker_psd_freqs > x + psd_bin_res * edges[0]) * # binning
                                              (flicker_psd_freqs < x + psd_bin_res * edges[1])).squeeze() for x in psd_freqs])
        rest_freq_bin_args = np.array([np.argwhere((rest_psd_freqs > x + psd_bin_res * edges[0]) *
                                                   (rest_psd_freqs < x + psd_bin_res * edges[1])).squeeze() for x in psd_freqs])
        psd_binned = np.array([ca_flicker_psd[:, x].sum(axis=1) for x in freq_bin_args]).T  # summing over bins
        rest_psd_binned = np.array([ca_rest_psd[:, x].sum(axis=1) for x in rest_freq_bin_args]).T

        # normalize frequencies to alpha TODO: what is normalized in Koch et al. 2006?
        # normed_flicker_psd_freqs = (psd_freqs - r_alpha).round(2)
        # normed_SSVEPs_freq = myround(SSVEPs - r_alpha, base=psd_bin_res)

        # calculating SSVEP power
        SSVEP_psd_freqs_args = np.array([np.argwhere(myround(psd_freqs, psd_bin_res) == x).squeeze()
                                         for x in myround(SSVEPs, base=psd_bin_res)]).squeeze()
        SSVEP_power = [psd_binned[i, x] for i, x in enumerate(SSVEP_psd_freqs_args)]

        # calculating alpha power, during and after flicker
        alpha_psd_freqs_arg = np.argwhere(myround(psd_freqs, psd_bin_res) == r_alpha).squeeze()
        alpha_power_during = psd_binned[:, alpha_psd_freqs_arg]  # alpha power during flicker
        alpha_power_after = rest_psd_binned[:, alpha_psd_freqs_arg]  # alpha power after flicker

        # add power to xarray
        for i, run in enumerate(runs):
            run_alpha_power_during = alpha_power_during[i * 9: (i + 1) * 9]  # 1st and 2nd half of frequencies
            run_alpha_power_after = alpha_power_after[i * 9: (i + 1) * 9]  # 1st and 2nd half of frequencies
            run_SSVEP_power = SSVEP_power[i * 9: (i + 1) * 9]
            run_flicker_freqs = flicker_freqs[i * 9: (i + 1) * 9]
            secondary_power_features.loc[dict(subject=sub_ID, flicker_freq=run_flicker_freqs,
                                           run=run, feature='alpha_power_during')] = run_alpha_power_during
            secondary_power_features.loc[dict(subject=sub_ID, flicker_freq=run_flicker_freqs,
                                           run=run, feature='alpha_power_after')] = run_alpha_power_after
            secondary_power_features.loc[dict(subject=sub_ID, flicker_freq=run_flicker_freqs,
                                           run=run, feature='SSVEP')] = run_SSVEP_power

    return secondary_power_features

# # Connectivity test idea (2)
#   For every (flicker frequency x block), only using occipitotemporal channels,
#   calculate t-tests between PLI during the stimulation vs. the average rest epochs' PLI.
#   Then correlate the t-test statistic with the questionnaire ratings.
#   Results would show how deviations from normal connectivity in V1, V4, V5 correspond to changes in subjective
#   effects. This also takes nice advantage of PLI's theoretic elimination of volume conduction, allowing us to
#   assume independence of channel pair connections.

def calc_PLI_dist(flicker_con_measures, rest_con_measures):
    """
    Calculates PLI distance measures {mean difference, parametric/non-parametric test statistcs}
        between rest and flicker epochs
    :param flicker_con_measures: (xarray) of flicker connectivity measures under dim ['con_method']
     with coords ['chan_1', 'chan_2', 'flicker_freq', 'subject', 'con_method', 'run']
    :param rest_con_measures: (xarray) of rest connectivity measures with same specifications
    :return: (dataframe) with distance measures as columns
    """

    # prepare connectivity xarrays for analysis
    try:  # drop events channel
        flicker_con_measures = flicker_con_measures.drop_sel(dict(chan_1='Status', chan_2='Status'))
    except KeyError:
        pass
    flicker_pli = flicker_con_measures.loc[dict(con_method='pli')]  # connectivity during flicker
    flicker_pli.name = 'flicker_pli'
    flicker_pli = flicker_pli.assign_coords(dict(con_method='flicker_pli'))

    try:  # drop events channel
        rest_con_measures = rest_con_measures.drop_sel(dict(chan_1='Status', chan_2='Status'))
    except KeyError:
        pass
    rest_pli = rest_con_measures.loc[dict(con_method='pli')]  # connectivity at rest
    rest_pli.name = 'rest_pli'
    rest_pli = rest_pli.assign_coords(dict(con_method='rest_pli'))

    pli = xr.concat([rest_pli, flicker_pli], 'con_method')  # concatenate rest_pli and flicker_pli

    ot_chs = np.hstack([v1_chs, v4_chs, v5_chs])  # occipitotemporal channels in lower visual "what" stream
    # not_chs = np.array([x for x in flicker_pli.chan_1.data if x not in ot_chs])  # non-occipitotemporal channels
    ot_tril = np.tril_indices(len(ot_chs), -1)  # lower triangle indices

    # test normality of connectivity in all (flicker frequency x run)
    df_pli = pli.loc[
        dict(chan_1=ot_chs[ot_tril[0]], chan_2=ot_chs[ot_tril[1]])].to_dataframe()  # TODO: fix indexing of lower tri
    df_pli.rename(columns={'rest_pli': 'pli'}, inplace=True)
    df_pli_long = df_pli.copy(deep=True).reset_index()

    # calculate distance between connectivity during and after stimulation
    group = ['flicker_freq', 'run', 'subject']  # group by these dimensions
    pli_dist = df_pli.loc['flicker_pli'].groupby(group).mean() - df_pli.loc['rest_pli'].groupby(group).mean()
    pli_dist.rename(columns={'pli': 'mean_diff'}, inplace=True)

    pli_dist['test_stat'] = np.ones(len(pli_dist)) * np.nan  # adding new column for stats tests

    # for each combination of (flicker_freq x run) determine distance during vs. after with stats test
    pli_normality = df_pli.groupby([*group, 'con_method']).apply(normality)

    for subject in pli.subject.data:
        for flicker_freq in pli.flicker_freq.data:
            for run in pli.run.data:
                temp = df_pli_long.loc[(df_pli_long.flicker_freq == flicker_freq) & (df_pli_long.run == run) &
                                       (df_pli_long.subject == subject)]

                if bool(pli_normality.mean(axis=0)['normal']):  # perform parametric tests
                    t = ttest(temp.loc[temp.con_method == 'flicker_pli']['pli'],
                              temp.loc[temp.con_method == 'rest_pli']['pli'])
                    pli_dist.loc[(flicker_freq, run)]['test_stat'] = float(t['T'])  # test statistic

                else:  # perform non-parametric tests
                    W = wilcoxon(temp.loc[temp.con_method == 'flicker_pli']['pli'],
                                 temp.loc[temp.con_method == 'rest_pli']['pli'])
                    pli_dist.loc[(flicker_freq, run)]['test_stat'] = float(W['W-val'])  # test statistic

    r, p = pearsonr(pli_dist['mean_diff'], pli_dist['test_stat'])  # correlation between distance measures

    print(f"Correlation between calculated test statistic and mean difference: "
          f"{r:.2}, p = {p:.2}")  # note t-test, but not Wilcoxon correlated with mean.

    return pli_dist

