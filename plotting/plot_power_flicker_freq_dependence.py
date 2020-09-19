import matplotlib.pyplot as plt
import scipy.stats
from utils.helper_funcs import load_obj
from utils.setup_info import sub_names, sub_IDs, runs, sub_colors, ID_to_name
from utils.st_adjudication_funcs import calc_secondary_power_features
from scipy.stats import pearsonr
import numpy as np
import os


def plot_power_flicker_freq_dependence(secondary_power_features, avg_alpha):
    """
    Plots frequency dependence of alpha power and alpha entrainment
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


def plot_alpha_power_by_sub(secondary_power_features):
    """
    Plots frequency dependence of alpha power and SSVEP power
    :param secondary_power_features: (xarray) of power data, with coords {subjects, runs, feature, and flicker frequency}
    :return: None
    """

    ls = ['-', '--']  # linestyles

    # setting up figure
    fig, axs = plt.subplots(2, 3, figsize=(10, 5), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=.15, bottom=.1, top=.95, left=.15)

    # calculating alpha power and entrainment
    all_alpha_during = secondary_power_features.loc[dict(feature='alpha_power_during')].stack(
        collapsed=('run', 'subject')).T
    all_alpha_after = secondary_power_features.loc[dict(feature='alpha_power_after')].stack(
        collapsed=('run', 'subject')).T
    all_entrainment = all_alpha_during - all_alpha_after
    flicker_freq = secondary_power_features.flicker_freq.data

    # plot
    for i, sub_ID in enumerate(sub_IDs):

        r_alpha = subs_dict[ID_to_name[sub_ID]]['r_alpha']
        axs[0, i].axvline(r_alpha, c='k', label=r'resting $\alpha$ = ' + f'{r_alpha:.2}')
        axs[1, i].axvline(r_alpha, c='k')

        axs[0, i].legend()

        for run in runs:
            axs[0, i].plot(flicker_freq, all_alpha_during.loc[dict(subject=sub_ID, run=run)].data,
                           c=sub_colors[sub_ID], linestyle=ls[run], label=f'block {run}')
            axs[1, i].plot(flicker_freq, all_entrainment.loc[dict(subject=sub_ID, run=run)].data,
                           c=sub_colors[sub_ID], linestyle=ls[run], label=f'block {run}')

        axs[1, i].set_xlabel('Flicker frequency (Hz)')

        # setting axis labels
        if i == 0:
            axs[0, i].set_ylabel(r'$\alpha$-power (V ^ 2) / Hz')
            axs[1, i].set_ylabel(r'$\alpha$-entrainment (V ^ 2) / Hz')

        else:            # turn uneccesaryy xlabels off
            axs[0, i].tick_params(axis='x',  # changes apply to the x-axis
                                  which='both',  # both major and minor ticks are affected
                                  bottom=False,  # ticks along the bottom edge are off
                                  labelbottom=False)  # labels along the bottom edge are off

            for j in range(2):  # turn ylabels off
                axs[j, i].tick_params(axis='y',  # changes apply to the y-axis
                                      which='both',  # both major and minor ticks are affected
                                      left=False,  # ticks along the left edge are off
                                      labelleft=False)  # labels along the left edge are off


    axs[1, 2].legend()

    plt.show()

    fig.savefig(f'figures/power_flicker_freq_dependence_by_sub.png')


def calc_alpha_entrainment_corr_across_runs(secondary_power_features):
    """
    Calculates correlation of alpha entrainment, across each run
    :param secondary_power_features: (xarray) of power data, with coords {subjects, runs, feature, and flicker frequency}
    :return: None
    """

    # calculating alpha power and entrainment
    all_alpha_during = secondary_power_features.loc[dict(feature='alpha_power_during')].stack(
        collapsed=('run', 'subject')).T
    all_alpha_after = secondary_power_features.loc[dict(feature='alpha_power_after')].stack(
        collapsed=('run', 'subject')).T
    all_entrainment = all_alpha_during - all_alpha_after
    flicker_freq = secondary_power_features.flicker_freq.data

    # print
    for i, sub_ID in enumerate(sub_IDs):
        print(pearsonr(all_entrainment.loc[dict(subject=sub_ID, run=0)].data,
                       all_entrainment.loc[dict(subject=sub_ID, run=1)].data))

# load the master subject dictionary
if os.path.isfile('data/subs_dict.pkl'):
    subs_dict = load_obj('data/subs_dict.pkl')
else:
    from analysis.rest_power import rest_power

    subs_dict = load_obj('data/subs_dict.pkl')

xarr = calc_secondary_power_features(subs_dict, bin_edge='left')

# avg_alpha = np.mean([subs_dict[name]['r_alpha'] for name in sub_names])
# plot_power_flicker_freq_dependence(xarr, avg_alpha)
plot_alpha_power_by_sub(secondary_power_features=xarr)
# calc_alpha_entrainment_corr_across_runs(secondary_power_features=xarr)
