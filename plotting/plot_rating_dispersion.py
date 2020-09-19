from utils.setup_info import ID_to_name, questions, test_freqs, runs, sub_IDs, sub_colors
from utils.helper_funcs import load_obj
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# load the master subject dictionary
if os.path.isfile('data/subs_dict.pkl'):
    subs_dict = load_obj('data/subs_dict.pkl')
else:
    from analysis import resting_alpha  # run resting alpha analysis

    subs_dict = load_obj('data/subs_dict.pkl')

figsize = (17, 8)
rating_lim = (0,100)

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
    ax.set_ylabel('Post-Epoch Questionnaire ratings')

    ax.boxplot(rating_dispersion.stack(collapsed=('subject', 'test_freq', 'run')))

    ax.set_xticks(np.arange(len(questions)) + 1)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xticklabels(questions, rotation=60, fontdict={'horizontalalignment': 'right'}, wrap=True)

    plt.show()
    fig.savefig(f'figures/rating_dispersion.png')

    return rating_dispersion


def plot_rating_per_block_freq(subs_dict):
    # setting up figure
    fig, axs = plt.subplots(3, 3, figsize=figsize, sharey=True, sharex=True)
    fig.subplots_adjust(hspace=.2, wspace=.05, bottom=.1, top=.95)
    axs = axs.ravel()

    ls = ['-', '--']  # linestyles

    # calculating and plotting ratingsin loop
    for j, question in enumerate(questions):

        if j % 3 == 0:
            axs[j].set_ylabel('rating')
        else:
            axs[j].tick_params(axis='y',  # changes apply to the y-axis
                               which='both',  # both major and minor ticks are affected
                               left=False,  # ticks along the left edge are off
                               labelleft=False)  # labels along the left edge are off
        if j >= 6:
            axs[j].set_xlabel('Flicker frequency (Hz)')
        else:
            axs[j].tick_params(axis='x',  # changes apply to the x-axis
                               which='both',  # both major and minor ticks are affected
                               bottom=False,  # ticks along the bottom edge are off
                               labelbottom=False)  # labels along the bottom edge are off

        axs[j].set_xticks(range(len(test_freqs)))
        axs[j].set_xticklabels(test_freqs)
        axs[j].set_yticks(range(rating_lim[0], rating_lim[1] + 1, 20))
        axs[j].set_ylim(rating_lim)
        axs[j].set_title(question, wrap=True)

        for i, sub_ID in enumerate(sub_IDs):
            name = ID_to_name[sub_ID]
            response = subs_dict[name]['response']
            flicker_freqs = np.array([x for x in response['Frequency (Hz)'] if type(x) != str])
            rating = response[question].loc[[type(x) != str for x in response['Frequency (Hz)']]].to_numpy()

            for run in runs:
                rating_ord_by_freq = flicker_freqs[int(run * 9):int(run * 9) + 9].argsort()
                rating_per_run = rating[int(run * 9):int(run * 9) + 9][rating_ord_by_freq]
                axs[j].plot(range(len(test_freqs)), rating_per_run,
                            c=sub_colors[sub_ID], linestyle=ls[run], label=f'block {run + 1}')

        # if j == 8:
        #     axs[j].legend()

    plt.show()
    fig.savefig('figures/rating_per_block_freq.png')


def plot_rating_per_sub_block_freq(subs_dict):

    ls = ['-', '--']  # linestyles

    for i, sub_ID in enumerate(sub_IDs):

        # setting up figure
        fig, axs = plt.subplots(3, 3, figsize=figsize, sharey=True, sharex=True)
        fig.subplots_adjust(hspace=.2, wspace=.05, bottom=.1, top=.95)
        axs = axs.ravel()

        # reading in subject data
        name = ID_to_name[sub_ID]
        response = subs_dict[name]['response']
        flicker_freqs = np.array([x for x in response['Frequency (Hz)'] if type(x) != str])

        # calculating and plotting ratings in loop
        for j, question in enumerate(questions):

            rating = response[question].loc[[type(x) != str for x in response['Frequency (Hz)']]].to_numpy()

            if j % 3 == 0:
                axs[j].set_ylabel('rating')
            else:
                axs[j].tick_params(axis='y',  # changes apply to the y-axis
                                   which='both',  # both major and minor ticks are affected
                                   left=False,  # ticks along the left edge are off
                                   labelleft=False)  # labels along the left edge are off
            if j >= 6:
                axs[j].set_xlabel('Flicker frequency (Hz)')
            else:
                axs[j].tick_params(axis='x',  # changes apply to the x-axis
                                   which='both',  # both major and minor ticks are affected
                                   bottom=False,  # ticks along the bottom edge are off
                                   labelbottom=False)  # labels along the bottom edge are off

            axs[j].set_xticks(range(len(test_freqs)))
            axs[j].set_yticks(range(rating_lim[0], rating_lim[1] + 1, 20))
            axs[j].set_ylim(rating_lim)
            axs[j].set_xticklabels(test_freqs)
            axs[j].set_title(question, wrap=True)

            for run in runs:
                rating_ord_by_freq = flicker_freqs[int(run * 9):int(run * 9) + 9].argsort()
                rating_per_run = rating[int(run * 9):int(run * 9) + 9][rating_ord_by_freq]
                axs[j].plot(range(len(test_freqs)), rating_per_run,
                            c=sub_colors[sub_ID], linestyle=ls[run], label=f'block {run + 1}')

            if j == 8:
                axs[j].legend()

        plt.show()
        fig.savefig(f'figures/rating_per_block_freq_{sub_ID}.png')


def plot_rating_per_sub_block_freq2(subs_dict):

    ls = ['-', '--']  # linestyles

    # setting up figure
    fig, axs = plt.subplots(9, 3, sharey=True, sharex=True)
    fig.set_size_inches(4, 9)
    fig.subplots_adjust(hspace=.9, wspace=.03, bottom=.1, top=.95, right=.98)

    scale_factor = 5
    xloc = np.arange(len(test_freqs)) / scale_factor

    for i, sub_ID in enumerate(sub_IDs):

        # reading in subject data
        name = ID_to_name[sub_ID]
        response = subs_dict[name]['response']
        flicker_freqs = np.array([x for x in response['Frequency (Hz)'] if type(x) != str])

        # calculating and plotting ratings in loop
        for j, question in enumerate(questions):

            rating = response[question].loc[[type(x) != str for x in response['Frequency (Hz)']]].to_numpy()

            if i == 0:
                axs[j, i].set_ylabel('rating')
            else:
                axs[j, i].tick_params(axis='y',  # changes apply to the y-axis
                                   which='both',  # both major and minor ticks are affected
                                   left=False,  # ticks along the left edge are off
                                   labelleft=False)  # labels along the left edge are off
            if j == 8:
                axs[j, i].set_xlabel('Flicker frequency (Hz)')
            else:
                axs[j, i].tick_params(axis='x',  # changes apply to the x-axis
                                   which='both',  # both major and minor ticks are affected
                                   bottom=False,  # ticks along the bottom edge are off
                                   labelbottom=False)  # labels along the bottom edge are off

            axs[j, i].set_xticks(xloc)
            axs[j, i].set_yticks(range(rating_lim[0], rating_lim[1] + 1, 50))
            axs[j, i].set_ylim(rating_lim)
            axs[j, i].set_xticklabels(test_freqs)

            if i == 1:
                axs[j, i].set_title(question, wrap=True)

            for run in runs:
                rating_ord_by_freq = flicker_freqs[int(run * 9):int(run * 9) + 9].argsort()
                rating_per_run = rating[int(run * 9):int(run * 9) + 9][rating_ord_by_freq]
                axs[j, i].plot(xloc, rating_per_run,
                            c=sub_colors[sub_ID], linestyle=ls[run], label=f'block {run + 1}')

            if j == 0 and i == 2:
                axs[j, i].legend()

    plt.show()
    fig.savefig(f'figures/rating_per_block_freq_2.png')


# rating_dispersion = plot_rating_dispersion(subs_dict)
# med = rating_dispersion.groupby('question').median(['subject', 'run', 'test_freq']) # get median and IQR
# iqr = rating_dispersion.stack(collapsed=('subject', 'test_freq', 'run')).quantile([.25, .75], dim='collapsed')

# plot_rating_per_block_freq(subs_dict)
# plot_rating_per_sub_block_freq(subs_dict)
plot_rating_per_sub_block_freq2(subs_dict)
