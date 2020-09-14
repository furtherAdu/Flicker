from utils.st_adjudication_funcs import compile_all_session_info
from utils.setup_info import ID_to_name, sub_IDs, questions, test_freqs, sub_colors
from utils.helper_funcs import load_obj
from scipy.stats import pearsonr
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

if os.path.isfile('data/subs_dict_psds_alpha.pkl'):
    subs_dict = load_obj('data/subs_dict_psds_alpha.pkl')
else:
    subs_dict = compile_all_session_info()


def plot_retest_reliability(subs_dict):
    # allocating array to hold test results
    retest_reliability = np.ones((len(sub_IDs), len(test_freqs), 2)) * np.nan  # (subjects, frequencies, r/p)
    retest_reliability = xr.DataArray(retest_reliability,
                                      coords=dict(subject=sub_IDs, frequency=test_freqs, statistic=['r', 'p']),
                                      dims=['subject', 'frequency', 'statistic'])
    # setting up figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    fig.subplots_adjust(hspace=.05)
    # fig.suptitle('Flicker frequency retest reliability')
    axs = axs.ravel()

    axs[0].set_title('Pearson\'s r')
    axs[1].set_title('p - value')

    # calculating and plotting retest reliability in loop
    for sub_ID in sub_IDs:
        name = ID_to_name[sub_ID]
        response = subs_dict[name]['response']

        for freq in test_freqs:
            freq_ratings = response.loc[response['Frequency (Hz)'] == freq][questions].to_numpy()
            r, p = pearsonr(freq_ratings[0], freq_ratings[1])  # test-retest reliability analysis, ideally r > .7

            # save in xarray
            retest_reliability.loc[dict(subject=sub_ID, frequency=freq, statistic='r')] = r
            retest_reliability.loc[dict(subject=sub_ID, frequency=freq, statistic='p')] = p

        # plot all subjects' r, p values in two plots (x = frequency, y = pearson statistic)
        all_r = retest_reliability.loc[dict(subject=sub_ID, statistic='r')]
        all_p = retest_reliability.loc[dict(subject=sub_ID, statistic='p')]
        all_freq = retest_reliability.frequency.data

        axs[0].plot(all_freq, all_r, label=sub_ID, c=sub_colors[sub_ID])
        axs[1].plot(all_freq, all_p, c=sub_colors[sub_ID])

    axs[1].axhline(.05, c='r', label='.05', linestyle='--')
    for _, ax in enumerate(axs):
        ax.set_xticks(all_freq)
        ax.legend()
    plt.show()
    fig.savefig('figures/freq_retest_reliability.png')


plot_retest_reliability(subs_dict)