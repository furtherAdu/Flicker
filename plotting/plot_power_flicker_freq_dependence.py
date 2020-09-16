import matplotlib.pyplot as plt
import scipy.stats
from utils.helper_funcs import load_obj
from analysis.secondary_power_features import calc_secondary_power_features
import os

def plot_power_flicker_freq_dependence(secondary_power_features):
    """
    Plots frequency dependence of alpha power and SSVEP power
    :param secondary_power_features: (xarray) of power data, with coords {subjects, runs, feature, and flicker frequency}
    :return: None
    """

    # setting up figure
    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True)
    fig.subplots_adjust(hspace=.15, bottom=.1, top=.9, left=.15)
    # fig.suptitle(r'Frequency dependence of $\alpha$-power and SSVEP')
    axs = axs.ravel()

    all_alpha_during = secondary_power_features.loc[dict(feature='alpha_power_during')].stack(collapsed=('run', 'subject')).T
    all_alpha_after = secondary_power_features.loc[dict(feature='alpha_power_after')].stack(collapsed=('run', 'subject')).T
    all_SSVEP = secondary_power_features.loc[dict(feature='alpha_power_after')].stack(collapsed=('run', 'subject')).T

    # calculate SEM
    during_sem = scipy.stats.sem(all_alpha_during)
    after_sem = scipy.stats.sem(all_alpha_after)
    SSVEP_sem = scipy.stats.sem(all_SSVEP)

    # plot, with error bars
    axs[0].errorbar(x=secondary_power_features.flicker_freq.data, y=all_SSVEP.mean(axis=0), yerr=SSVEP_sem, label='SSVEP', c='r')
    axs[1].errorbar(x=secondary_power_features.flicker_freq.data, y=all_alpha_during.mean(axis=0), yerr=during_sem, label='during flicker')
    axs[1].errorbar(x=secondary_power_features.flicker_freq.data, y=all_alpha_after.mean(axis=0), yerr=after_sem, label='after flicker', alpha=.6)

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

xarr = calc_secondary_power_features(subs_dict, bin_edge='left')
plot_power_flicker_freq_dependence(xarr)