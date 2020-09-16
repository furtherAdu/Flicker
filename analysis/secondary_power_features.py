from utils.helper_funcs import myround
from utils.setup_info import sub_IDs, test_freqs, ID_to_name, runs, lcutoff, hcutoff, fft_step
import xarray as xr
import numpy as np


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
