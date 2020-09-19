# calculate resting alpha for pulses per Till’s paper
# subtract 1s before - 1s after for each pulse (ERD)
# average over all channels
# apply log
# make figure

import matplotlib.pyplot as plt
from utils.helper_funcs import load_obj
from utils.setup_info import name_to_ID, sub_colors
import numpy as np
import os


def plot_average_occipital_power(subs_dict, save=True):
    fig, axs = plt.subplots(1, len(subs_dict), figsize=(10, 4), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=.05, wspace=.05, top=.85)
    # fig.suptitle('Average parieto-occipital power')
    axs = axs.ravel()

    for i, sub in enumerate(list(subs_dict.keys())):

        # for each subject
        sd = subs_dict[sub]
        ID = name_to_ID[sub]

        # plot the power during the pulse
        before = np.log(sd['power_before_pulse'].mean(axis=(0, 1)))
        after = np.log(sd['power_after_pulse'].mean(axis=(0, 1)))
        freqs = sd['freqs']

        axs[i].plot(freqs, after, label='after pulse', c=sub_colors[ID], linestyle='--')
        axs[i].plot(freqs, before, label='before pulse', c=sub_colors[ID])
        axs[i].set_title(name_to_ID[sub])

        if i == 0:
            axs[i].set_ylabel('Log-Power (V^2 / Hz)')

        if i == 2:
            axs[i].legend()

    if save:
        fig.savefig(f'figures/pulse_occ_power.png')


def plot_ERD(subs_dict, save=True):
    fig2, axs2 = plt.subplots(1, len(subs_dict), figsize=(10, 4), sharex=True, sharey=True)
    fig2.subplots_adjust(hspace=.05, wspace=.05, top=.85)
    # fig2.suptitle('Event-Related Desynchronization (ERD)')
    axs2 = axs2.ravel()

    for i, sub in enumerate(list(subs_dict.keys())):

        # for each subject
        sd = subs_dict[sub]
        ID = name_to_ID[sub]

        log_before = np.log(sd['power_before_pulse'].mean(axis=(0, 1)))
        log_after = np.log(sd['power_after_pulse'].mean(axis=(0, 1)))

        freqs = sd['freqs']

        # plot the event-related desynchronization (ERD)
        diff = log_after - log_before
        alpha = freqs[np.argmin(diff)]

        axs2[i].plot(freqs, diff, c=sub_colors[ID], label=ID)
        axs2[i].axvline(alpha, c='k', label=r'$\alpha$ = ' + str(alpha.round(2)))
        axs2[i].set_xlabel('Frequency (Hz)')
        # axs2[i].set_title(name_to_ID[sub] + r', $\alpha$ = ' + str(alpha.round(2)))
        axs2[i].legend()

        if i == 0:
            axs2[i].set_ylabel('Δ Log-Power (V^2 / Hz)')

        if i == 2:
            axs2[i].legend()

    if save:
        fig2.savefig(f'figures/pulse_ERD.png')


# load the master subject dictionary
if os.path.isfile('data\subs_dict_psds_alpha_SSVEP.pkl'):
    subs_dict = load_obj('data\subs_dict_psds_alpha_SSVEP.pkl')
elif os.path.isfile('data/subs_dict.pkl'):
    subs_dict = load_obj('data/subs_dict.pkl')

plot_average_occipital_power(subs_dict)
plot_ERD(subs_dict)
