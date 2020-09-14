from utils.setup_info import ID_to_name, questions, test_freqs, runs, sub_IDs
from utils.helper_funcs import load_obj
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# load the master subject dictionary
if os.path.isfile('data/subs_dict_psds_alpha.pkl'):
    subs_dict = load_obj('data/subs_dict_psds_alpha.pkl')
else:
    from analysis import resting_alpha  # run resting alpha analysis

    subs_dict = load_obj('data/subs_dict_psds_alpha.pkl')


def plot_rating_dispersion(subs_dict):
    # allocating xarray for response, (subjects, test frequency, run, ratings)
    rating_dispersion = np.ones((len(sub_IDs), len(test_freqs), len(runs), len(questions))) * np.nan
    rating_dispersion = xr.DataArray(rating_dispersion,
                                     coords=dict(subject=sub_IDs, test_freq=test_freqs, run=runs, question=questions),
                                     dims=['subject', 'test_freq', 'run', 'question'])

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
                rating_dispersion.loc[dict(subject=sub_ID, question=question,
                                           test_freq=run_epoch_freqs, run=run)] = run_rating

    # setting up figure
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=.35, top=.92)
    # fig.suptitle('Post-flicker block ratings, all subjects')

    ax.boxplot(rating_dispersion.stack(collapsed=('subject', 'test_freq', 'run')))

    ax.set_xticks(np.arange(len(questions)) + 1)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xticklabels(questions, rotation=60, fontdict={'horizontalalignment': 'right'}, wrap=True)

    plt.show()
    fig.savefig(f'figures/rating_dispersion.png')

plot_rating_dispersion(subs_dict)