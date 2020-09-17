import matplotlib.pyplot as plt
import scipy.stats
from utils.helper_funcs import load_obj
from utils.setup_info import sub_names
from analysis.secondary_power_features import calc_secondary_power_features
import numpy as np
import os


def plot_power_flicker_freq_dependence(secondary_power_features, avg_alpha):
    """
    Plots frequency dependence of alpha power and SSVEP power
    :param avg_alpha: (float) average alpha frequency for all subjects
    :param secondary_power_features: (xarray) of power data, with coords {subjects, runs, feature, and flicker frequency}
    :return: None
    """

    # setting up figure
    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True)
    fig.subplots_adjust(hspace=.15, bottom=.1, top=.9, left=.15)
    # fig.suptitle(r'Frequency dependence of $\alpha$-power and SSVEP')
    axs = axs.ravel()

    all_alpha_during = secondary_power_features.loc[dict(feature='alpha_power_during')].stack(
        collapsed=('run', 'subject')).T
    all_alpha_after = secondary_power_features.loc[dict(feature='alpha_power_after')].stack(
        collapsed=('run', 'subject')).T
    all_SSVEP = secondary_power_features.loc[dict(feature='alpha_power_after')].stack(collapsed=('run', 'subject')).T

    # calculate SEM
    during_sem = scipy.stats.sem(all_alpha_during)
    after_sem = scipy.stats.sem(all_alpha_after)
    SSVEP_sem = scipy.stats.sem(all_SSVEP)

    # plot, with error bars
    axs[0].errorbar(x=secondary_power_features.flicker_freq.data, y=all_SSVEP.mean(axis=0), yerr=SSVEP_sem,
                    c='r')  # label='SSVEP'
    axs[1].errorbar(x=secondary_power_features.flicker_freq.data, y=all_alpha_during.mean(axis=0), yerr=during_sem,
                    label='during flicker')
    axs[1].errorbar(x=secondary_power_features.flicker_freq.data, y=all_alpha_after.mean(axis=0), yerr=after_sem,
                    label='after flicker', alpha=.6)

    # adding alpha line
    axs[0].axvline(avg_alpha, c='k', label=r'mean resting $\alpha$ = ' + f'{avg_alpha:.2}', linestyle='--')
    axs[1].axvline(avg_alpha, c='k', linestyle='--')

    # setting axis labels
    axs[0].set_ylabel('SSVEP power (V ^ 2) / Hz')
    axs[1].set_ylabel(r'$\alpha$-power (V ^ 2) / Hz')
    axs[1].set_xlabel('Flicker frequency (Hz)')

    axs[0].legend()
    axs[1].legend()

    plt.show()

    fig.savefig(f'figures/power_flicker_freq_dependence.png')


# load the master subject dictionary
if os.path.isfile('data/subs_dict_psds_alpha_SSVEP_rest.pkl'):
    subs_dict = load_obj('data/subs_dict_psds_alpha_SSVEP_rest.pkl')
else:
    from analysis.rest_power import rest_power

    subs_dict = load_obj('data/subs_dict_psds_alpha_SSVEP_rest.pkl')

xarr = calc_secondary_power_features(subs_dict, bin_edge='left')

avg_alpha = np.mean([subs_dict[name]['r_alpha'] for name in sub_names])
plot_power_flicker_freq_dependence(xarr, avg_alpha)
