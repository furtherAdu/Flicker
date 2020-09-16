from utils.setup_info import ID_to_name, sub_IDs, questions, sub_colors
from utils.helper_funcs import save_obj, load_obj 
from utils.setup_info import SSVEP_band
from scipy.stats import pearsonr, zscore
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

# load the master subject dictionary
if os.path.isfile('data\subs_dict_psds_alpha_SSVEP.pkl'):
    subs_dict = load_obj('data\subs_dict_psds_alpha_SSVEP.pkl')
else:
    from analysis.SSVEP import calc_SSVEP

    subs_dict = load_obj('data/subs_dict_psds_alpha.pkl')
    subs_dict = calc_SSVEP(subs_dict, SSVEP_band)
    save_obj(subs_dict, 'data\subs_dict_psds_alpha_SSVEP.pkl')


def plot_SSVEP_ratings_corr(subs_dict):
    # setting up figure
    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True)
    fig.subplots_adjust(hspace=.2, bottom=.4, top=.9, left=.2)
    fig.suptitle('SSVEP and rating intensity correlation')
    axs = axs.ravel()

    axs[0].set_ylabel('Pearson\'s r')
    axs[1].set_ylabel('p - value')

    # allocating array to hold test results
    SSVEP_ratings_corr = np.ones((len(sub_IDs), len(questions), 2)) * np.nan  # (subjects, ratings, r/p)
    SSVEP_ratings_corr = xr.DataArray(SSVEP_ratings_corr,
                                      coords=dict(subject=sub_IDs, question=questions, statistic=['r', 'p']),
                                      dims=['subject', 'question', 'statistic'])

    # looping over subjects for plotting
    for sub_ID in sub_IDs:
        name = ID_to_name[sub_ID]
        response = subs_dict[name]['response']
        SSVEPs = subs_dict[name]['SSVEPs']

        # calculating per-epoch SSVEP frequency
        # correlating with each rating
        for question in questions:
            rating = response[question].loc[[type(x) != str for x in response['Frequency (Hz)']]].to_numpy()
            r, p = pearsonr(rating, SSVEPs)

            # save in xarray
            SSVEP_ratings_corr.loc[dict(subject=sub_ID, question=question, statistic='r')] = r
            SSVEP_ratings_corr.loc[dict(subject=sub_ID, question=question, statistic='p')] = p

        # plot all subjects' r, p values in two plots (x = frequency, y = pearson statistic)
        all_r = SSVEP_ratings_corr.loc[dict(subject=sub_ID, statistic='r')]
        all_p = SSVEP_ratings_corr.loc[dict(subject=sub_ID, statistic='p')]

        axs[0].plot(all_r, label=sub_ID, c=sub_colors[sub_ID])
        axs[1].plot(all_p, c=sub_colors[sub_ID])

    # annotating with questions
    axs[1].axhline(.05, c='r', label='.05', linestyle='--')
    for _, ax in enumerate(axs):
        ax.set_xticks(range(len(questions)))
        ax.set_xticklabels(questions, rotation=60, fontdict={'horizontalalignment': 'right'}, wrap=True)
        ax.legend()

    plt.show()
    fig.savefig(f'figures/rating_corr/SSVEP_rating_corr.png')


def plot_flicker_freq_ratings_corr(subs_dict):
    """
    Plots the correlation between the flicker frequency and the intensity ratings.
    :param subs_dict: nested dictionary, with subjects[{flicker frequencies, intensity ratings}]
    :return:
    """
    # setting up figure
    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True)
    fig.subplots_adjust(hspace=.2, bottom=.4, top=.9, left=.2)
    fig.suptitle('Flicker frequency and rating intensity correlation')
    axs = axs.ravel()

    axs[0].set_ylabel('Pearson\'s r')
    axs[1].set_ylabel('p - value')

    # allocating array to hold test results
    flicker_freq_ratings_corr = np.ones((len(sub_IDs), len(questions), 2)) * np.nan  # (subjects, ratings, r/p)
    flicker_freq_ratings_corr = xr.DataArray(flicker_freq_ratings_corr,
                                             coords=dict(subject=sub_IDs, question=questions, statistic=['r', 'p']),
                                             dims=['subject', 'question', 'statistic'])

    # looping over subjects for plotting
    for sub_ID in sub_IDs:
        name = ID_to_name[sub_ID]
        response = subs_dict[name]['response']
        flicker_freqs = np.array([x for x in response['Frequency (Hz)'] if type(x) != str])

        # correlating with each rating
        for question in questions:
            rating = response[question].loc[[type(x) != str for x in response['Frequency (Hz)']]].to_numpy()
            r, p = pearsonr(rating, flicker_freqs)

            # save in xarray
            flicker_freq_ratings_corr.loc[dict(subject=sub_ID, question=question, statistic='r')] = r
            flicker_freq_ratings_corr.loc[dict(subject=sub_ID, question=question, statistic='p')] = p

        # plot all subjects' r, p values in two plots (x = frequency, y = pearson statistic)
        all_r = flicker_freq_ratings_corr.loc[dict(subject=sub_ID, statistic='r')]
        all_p = flicker_freq_ratings_corr.loc[dict(subject=sub_ID, statistic='p')]

        axs[0].plot(all_r, label=sub_ID, c=sub_colors[sub_ID])
        axs[1].plot(all_p, c=sub_colors[sub_ID])

    # annotating with questions
    axs[1].axhline(.05, c='r', label='.05', linestyle='--')
    for _, ax in enumerate(axs):
        ax.set_xticks(range(len(questions)))
        ax.set_xticklabels(questions, rotation=60, fontdict={'horizontalalignment': 'right'}, wrap=True)
        ax.legend()

    plt.show()
    fig.savefig(f'figures/rating_corr/flicker_freq_rating_corr.png')


def plot_alpha_power_ratings_corr(subs_dict):
    """
    Plots the correlation between resting alpha power and ratings.
    :param subs_dict:
    :return:
    """
    # setting up figure
    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True)
    fig.subplots_adjust(hspace=.2, bottom=.4, top=.9, left=.2)
    # fig.suptitle('Resting alpha power and rating intensity correlation')
    axs = axs.ravel()

    axs[0].set_ylabel('Pearson\'s r')
    axs[1].set_ylabel('p - value')

    # allocating array to hold test results
    alpha_ratings_corr = np.ones((len(sub_IDs), len(questions), 2)) * np.nan  # (subjects, ratings, r/p)
    alpha_ratings_corr = xr.DataArray(alpha_ratings_corr,
                                      coords=dict(subject=sub_IDs, question=questions, statistic=['r', 'p']),
                                      dims=['subject', 'question', 'statistic'])

    # looping over subjects for plotting
    for sub_ID in sub_IDs:
        name = ID_to_name[sub_ID]
        response = subs_dict[name]['response']
        r_alpha = subs_dict[name]['r_alpha']
        flicker_psd_freqs = subs_dict[name]['flicker_psd_freqs']
        flicker_psd = subs_dict[name]['flicker_psd']

        # correlating with each rating
        alpha_arg = np.abs(flicker_psd_freqs - r_alpha).argmin()  # resting alpha arg in flicker_psd
        for question in questions:
            rating = response[question].loc[[type(x) != str for x in response['Frequency (Hz)']]].to_numpy()
            r, p = pearsonr(rating, flicker_psd.mean(axis=1)[:, alpha_arg])

            # save in xarray
            alpha_ratings_corr.loc[dict(subject=sub_ID, question=question, statistic='r')] = r
            alpha_ratings_corr.loc[dict(subject=sub_ID, question=question, statistic='p')] = p

        # plot all subjects' r, p values in two plots (x = frequency, y = pearson statistic)
        all_r = alpha_ratings_corr.loc[dict(subject=sub_ID, statistic='r')]
        all_p = alpha_ratings_corr.loc[dict(subject=sub_ID, statistic='p')]

        axs[0].plot(all_r, label=sub_ID, c=sub_colors[sub_ID])
        axs[1].plot(all_p, c=sub_colors[sub_ID])

    # annotating with questions
    axs[1].axhline(.05, c='r', label='.05', linestyle='--')
    for _, ax in enumerate(axs):
        ax.set_xticks(range(len(questions)))
        ax.set_xticklabels(questions, rotation=60, fontdict={'horizontalalignment': 'right'}, wrap=True)
        ax.legend()

    plt.show()
    fig.savefig(f'figures/rating_corr/alpha_power_rating_corr.png')


def plot_fa_distance_ratings_corr(subs_dict):
    """
    Plots the correlation between the distance (flicker frequency, alpha) and the intensity ratings.
    :param subs_dict: nested dictionary, with subjects[{flicker frequencies, intensity ratings, resting alpha}]
    :return:
    """

    # setting up figure
    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True)
    fig.subplots_adjust(hspace=.2, bottom=.4, top=.9, left=.2)
    fig.suptitle('Flicker-alpha distance and rating intensity correlation')
    axs = axs.ravel()

    axs[0].set_ylabel('Pearson\'s r')
    axs[1].set_ylabel('p - value')

    # allocating array to hold test results
    fa_distance_ratings_corr = np.ones((len(sub_IDs), len(questions), 2)) * np.nan  # (subjects, ratings, r/p)
    fa_distance_ratings_corr = xr.DataArray(fa_distance_ratings_corr,
                                            coords=dict(subject=sub_IDs, question=questions, statistic=['r', 'p']),
                                            dims=['subject', 'question', 'statistic'])

    # looping over subjects for plotting
    for sub_ID in sub_IDs:
        name = ID_to_name[sub_ID]
        response = subs_dict[name]['response']
        flicker_freqs = np.array([x for x in response['Frequency (Hz)'] if type(x) != str])

        # calculating per-epoch difference in resting alpha and flicker frequency
        r_alpha = subs_dict[name]['r_alpha']
        # af_dif = zscore(flicker_freqs - r_alpha)  # z-score
        af_dif = np.abs(flicker_freqs - r_alpha)  # absolute difference

        # correlating with each rating
        for question in questions:
            rating = response[question].loc[[type(x) != str for x in response['Frequency (Hz)']]].to_numpy()
            r, p = pearsonr(rating, af_dif)

            # save in xarray
            fa_distance_ratings_corr.loc[dict(subject=sub_ID, question=question, statistic='r')] = r
            fa_distance_ratings_corr.loc[dict(subject=sub_ID, question=question, statistic='p')] = p

        # plot all subjects' r, p values in two plots (x = frequency, y = pearson statistic)
        all_r = fa_distance_ratings_corr.loc[dict(subject=sub_ID, statistic='r')]
        all_p = fa_distance_ratings_corr.loc[dict(subject=sub_ID, statistic='p')]

        axs[0].plot(all_r, label=sub_ID, c=sub_colors[sub_ID])
        axs[1].plot(all_p, c=sub_colors[sub_ID])

    # annotating with questions
    axs[1].axhline(.05, c='r', label='.05', linestyle='--')
    for _, ax in enumerate(axs):
        ax.set_xticks(range(len(questions)))
        ax.set_xticklabels(questions, rotation=60, fontdict={'horizontalalignment': 'right'}, wrap=True)
        ax.legend()

    plt.show()
    fig.savefig(f'figures/rating_corr/fa_distance_rating_corr.png')


def plot_sa_distance_ratings_corr(subs_dict):
    """
    Plots the correlation between SSVEP-alpha distance and ratings.
    :param subs_dict:
    :return:
    """
    # setting up figure
    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True)
    fig.subplots_adjust(hspace=.2, bottom=.4, top=.9, left=.2)
    fig.suptitle('SSVEP-alpha distance and rating intensity correlation')
    axs = axs.ravel()

    axs[0].set_ylabel('Pearson\'s r')
    axs[1].set_ylabel('p - value')

    # allocating array to hold test results
    SSVEP_ratings_corr = np.ones((len(sub_IDs), len(questions), 2)) * np.nan  # (subjects, ratings, r/p)
    SSVEP_ratings_corr = xr.DataArray(SSVEP_ratings_corr,
                                      coords=dict(subject=sub_IDs, question=questions, statistic=['r', 'p']),
                                      dims=['subject', 'question', 'statistic'])

    # looping over subjects for plotting
    for sub_ID in sub_IDs:
        name = ID_to_name[sub_ID]
        response = subs_dict[name]['response']
        SSVEPs = subs_dict[name]['SSVEPs']

        # calculating per-epoch difference in resting alpha and SSVEP
        r_alpha = subs_dict[name]['r_alpha']
        # as_dif = zscore(SSVEPs - r_alpha)  # z-score
        as_dif = np.abs(SSVEPs - r_alpha)  # absolute difference

        # correlating with each rating
        for question in questions:
            rating = response[question].loc[[type(x) != str for x in response['Frequency (Hz)']]].to_numpy()
            r, p = pearsonr(rating, as_dif)

            # save in xarray
            SSVEP_ratings_corr.loc[dict(subject=sub_ID, question=question, statistic='r')] = r
            SSVEP_ratings_corr.loc[dict(subject=sub_ID, question=question, statistic='p')] = p

        # plot all subjects' r, p values in two plots (x = frequency, y = pearson statistic)
        all_r = SSVEP_ratings_corr.loc[dict(subject=sub_ID, statistic='r')]
        all_p = SSVEP_ratings_corr.loc[dict(subject=sub_ID, statistic='p')]

        axs[0].plot(all_r, label=sub_ID, c=sub_colors[sub_ID])
        axs[1].plot(all_p, c=sub_colors[sub_ID])

    # annotating with questions
    axs[1].axhline(.05, c='r', label='.05', linestyle='--')
    for _, ax in enumerate(axs):
        ax.set_xticks(range(len(questions)))
        ax.set_xticklabels(questions, rotation=60, fontdict={'horizontalalignment': 'right'}, wrap=True)
        ax.legend()

    plt.show()
    fig.savefig(f'figures/rating_corr/sa_distance_rating_corr.png')


def plot_fs_distance_ratings_corr(subs_dict):
    """
    Plots the correlation between SSVEP-alpha distance and ratings.
    :param subs_dict:
    :return:
    """
    # setting up figure
    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True)
    fig.subplots_adjust(hspace=.2, bottom=.4, top=.9, left=.2)
    fig.suptitle('Flicker-SSVEP distance and rating intensity correlation')
    axs = axs.ravel()

    axs[0].set_ylabel('Pearson\'s r')
    axs[1].set_ylabel('p - value')

    # allocating array to hold test results
    fs_distance_ratings_corr = np.ones((len(sub_IDs), len(questions), 2)) * np.nan  # (subjects, ratings, r/p)
    fs_distance_ratings_corr = xr.DataArray(fs_distance_ratings_corr,
                                            coords=dict(subject=sub_IDs, question=questions, statistic=['r', 'p']),
                                            dims=['subject', 'question', 'statistic'])

    # looping over subjects for plotting
    for sub_ID in sub_IDs:
        name = ID_to_name[sub_ID]
        response = subs_dict[name]['response']
        SSVEPs = subs_dict[name]['SSVEPs']
        flicker_freqs = np.array([x for x in response['Frequency (Hz)'] if type(x) != str])

        # calculating per-epoch difference in resting alpha and SSVEP
        # fs_dif = zscore(SSVEPs - flicker_freqs)  # z-score
        fs_dif = np.abs(SSVEPs - flicker_freqs)  # TODO: absolute difference

        # correlating with each rating
        for question in questions:
            rating = response[question].loc[[type(x) != str for x in response['Frequency (Hz)']]].to_numpy()
            r, p = pearsonr(rating, fs_dif)

            # save in xarray
            fs_distance_ratings_corr.loc[dict(subject=sub_ID, question=question, statistic='r')] = r
            fs_distance_ratings_corr.loc[dict(subject=sub_ID, question=question, statistic='p')] = p

        # plot all subjects' r, p values in two plots (x = frequency, y = pearson statistic)
        all_r = fs_distance_ratings_corr.loc[dict(subject=sub_ID, statistic='r')]
        all_p = fs_distance_ratings_corr.loc[dict(subject=sub_ID, statistic='p')]

        axs[0].plot(all_r, label=sub_ID, c=sub_colors[sub_ID])
        axs[1].plot(all_p, c=sub_colors[sub_ID])

    # annotating with questions
    axs[1].axhline(.05, c='r', label='.05', linestyle='--')
    for _, ax in enumerate(axs):
        ax.set_xticks(range(len(questions)))
        ax.set_xticklabels(questions, rotation=60, fontdict={'horizontalalignment': 'right'}, wrap=True)
        ax.legend()

    plt.show()
    fig.savefig(f'figures/rating_corr/fs_distance_rating_corr.png')


plot_SSVEP_ratings_corr(subs_dict)
plot_flicker_freq_ratings_corr(subs_dict)
plot_alpha_power_ratings_corr(subs_dict)

plot_sa_distance_ratings_corr(subs_dict)
plot_fa_distance_ratings_corr(subs_dict)
plot_fs_distance_ratings_corr(subs_dict)  # entrainment

# TODO: Note, z-scoring a distance assuming gaussian receptive field of freq-dependent hallucination process
