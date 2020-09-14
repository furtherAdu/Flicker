import matplotlib.pyplot as plt
import scipy.stats
from utils.helper_funcs import load_obj, save_obj, myround
from utils.setup_info import name_to_ID, SSVEP_band, sub_IDs, test_freqs, ID_to_name, runs, lcutoff, hcutoff, fft_step
import xarray as xr
import numpy as np
import os


def calc_power_flicker_freq_dependence(subs_dict, bin_edge='left'):
    """
    Calculate frequency dependence of alpha power and SSVEP power
    :param subs_dict: master subject dictionary containing PSDs of post-flicker rest and flicker epochs
    :return:(xarray) of power data, with coords {subjects, runs, feature, and flicker frequency}
    """
    assert bin_edge in ['left', 'center'], f"{bin_edge} is an invalid bin edge type."

    # allocating array to hold test results
    features = ['alpha_power_during', 'alpha_power_after', 'SSVEP']

    # .shape == # (subjects, flicker freqs, EEG freqs, alpha power/SSVEP, runs)
    power_freq_dependence = np.ones((len(sub_IDs), len(test_freqs), len(features),
                                     len(runs))) * np.nan
    power_freq_dependence = xr.DataArray(power_freq_dependence,
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
        rest_psd_freqs = subs_dict[name]['rest_flicker_psd_freqs']  # TODO: Note. floats represented as 2.0999999999999996
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
        alpha_power_during = psd_binned[:, alpha_psd_freqs_arg]
        alpha_power_after = rest_psd_binned[:, alpha_psd_freqs_arg]

        # add power to xarray
        for i, run in enumerate(runs):
            run_alpha_power_during = alpha_power_during[i * 9: (i + 1) * 9]  # 1st and 2nd half of frequencies
            run_alpha_power_after = alpha_power_after[i * 9: (i + 1) * 9]  # 1st and 2nd half of frequencies
            run_SSVEP_power = SSVEP_power[i * 9: (i + 1) * 9]
            run_flicker_freqs = flicker_freqs[i * 9: (i + 1) * 9]
            power_freq_dependence.loc[dict(subject=sub_ID, flicker_freq=run_flicker_freqs,
                                           run=run, feature='alpha_power_during')] = run_alpha_power_during
            power_freq_dependence.loc[dict(subject=sub_ID, flicker_freq=run_flicker_freqs,
                                           run=run, feature='alpha_power_after')] = run_alpha_power_after
            power_freq_dependence.loc[dict(subject=sub_ID, flicker_freq=run_flicker_freqs,
                                           run=run, feature='SSVEP')] = run_SSVEP_power

    return power_freq_dependence

def plot_power_flicker_freq_dependence(power_freq_dependence):
    """
    Plots frequency dependence of alpha power and SSVEP power
    :param power_freq_dependence: (xarray) of power data, with coords {subjects, runs, feature, and flicker frequency}
    :return: None
    """

    # setting up figure
    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True)
    fig.subplots_adjust(hspace=.15, bottom=.1, top=.9, left=.15)
    # fig.suptitle(r'Frequency dependence of $\alpha$-power and SSVEP')
    axs = axs.ravel()

    all_alpha_during = power_freq_dependence.loc[dict(feature='alpha_power_during')].stack(collapsed=('run', 'subject')).T
    all_alpha_after = power_freq_dependence.loc[dict(feature='alpha_power_after')].stack(collapsed=('run', 'subject')).T
    all_SSVEP = power_freq_dependence.loc[dict(feature='alpha_power_after')].stack(collapsed=('run', 'subject')).T

    # calculate SEM
    during_sem = scipy.stats.sem(all_alpha_during)
    after_sem = scipy.stats.sem(all_alpha_after)
    SSVEP_sem = scipy.stats.sem(all_SSVEP)

    # plot, with error bars
    axs[0].errorbar(x=power_freq_dependence.flicker_freq.data, y=all_SSVEP.mean(axis=0), yerr=SSVEP_sem, label='SSVEP', c='r')
    axs[1].errorbar(x=power_freq_dependence.flicker_freq.data, y=all_alpha_during.mean(axis=0),  yerr=during_sem, label='during flicker')
    axs[1].errorbar(x=power_freq_dependence.flicker_freq.data, y=all_alpha_after.mean(axis=0), yerr=after_sem, label='after flicker', alpha=.6)

    # adding alpha line
    # axs[1].axvline(, c='r', label=r'resting $\alpha$'', linestyle='--')  # TODO: add mean alpha

    # setting axis labels
    axs[0].set_ylabel('SSVEP power (V ^ 2) / Hz')
    axs[1].set_ylabel(r'$\alpha$-power (V ^ 2) / Hz')
    axs[1].set_xlabel('Flicker frequency (Hz)')

    axs[1].legend()

    plt.show()

    fig.savefig(f'figures/power_flicker_freq_dependence.png')


# load the master subject dictionary
if os.path.isfile('data/subs_dict_psds_alpha_SSVEP_rest.pkl'):
    subs_dict = load_obj('data/subs_dict_psds_alpha_SSVEP_rest.pkl')
else:
    from analysis.rest_power import rest_power
    subs_dict = load_obj('data/subs_dict_psds_alpha_SSVEP_rest.pkl')

xarr = calc_power_flicker_freq_dependence(subs_dict, bin_edge='left')
plot_power_flicker_freq_dependence(xarr)