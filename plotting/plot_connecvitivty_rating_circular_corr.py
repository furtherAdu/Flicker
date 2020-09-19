# # Connectivity test idea (1)
#   For each question on the questionnaire, plot connectivity of channel pairs whose PLI highly correlates
#  (p < 1e-5)  with the ratings (collapsing both channel pair PLI and ratings across 1st/2nd flicker block,
#  flicker frequency, and subject).
#  This would visualize nicely, show how connectivity mediates subjective experiences on each dimension of the
#   questionnaire.

from mne.viz import plot_connectivity_circle, circular_layout
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import xarray as xr
import numpy as np
import mne
import gc
import os

from utils.setup_info import questions, ds_hz, v1_chs, v4_chs, v5_chs
from utils.st_adjudication_funcs import compile_ratings
from utils.helper_funcs import load_obj

gc.collect()


def calc_channel_connectivity_rating_corr(flick_con_measures, xr_ratings, p_thresh=1e-5):
    try:
        flick_con_measures = flick_con_measures.drop_sel(dict(chan_1='Status', chan_2='Status'))  # drop events channel
    except KeyError:
        pass
    channels = flick_con_measures.chan_1.data
    tril = np.tril_indices(len(channels), -1)  # lower triangle indices
    flick_pli = flick_con_measures.loc[dict(con_method='pli')]

    # allocating array to hold correlation results
    xr_corr = np.ones((len(channels), len(channels), len(questions), 2)) * np.nan  # (subjects, ratings, r/p)
    xr_corr = xr.DataArray(xr_corr,
                           coords=dict(chan_1=channels, chan_2=channels, question=questions, statistic=['r', 'p']),
                           dims=['chan_1', 'chan_2', 'question', 'statistic'])

    # calculate pearson r for all (question x lower triangle entries)
    print('calculating pearson r for all (questions x lower triangle entries in connectivity matrix)...')
    for question in xr_ratings.question.data:
        for idxs in np.vstack(tril).T:
            r, p = pearsonr(flick_pli.loc[dict(chan_1=channels[idxs[0]], chan_2=channels[idxs[1]])].data.ravel(),
                            xr_ratings.loc[dict(question=question)].data.ravel())
            # save in xarray
            xr_corr.loc[dict(chan_1=channels[idxs[0]], chan_2=channels[idxs[1]], question=question, statistic='r')] = r
            xr_corr.loc[dict(chan_1=channels[idxs[0]], chan_2=channels[idxs[1]], question=question, statistic='p')] = p

            if p < p_thresh:
                print(question, idxs, r, p)
    print('...correlation calculation complete.')

    return xr_corr


def plot_corr_connectivity_by_rating(flick_con_measures, xr_corr, chan_groups=None, n_lines=None, p_thresh=1e-5):
    """
    Plots circular plot with connectivity that's correlated (p < p_thresh) with rating, for every question
    :param flick_con_measures: (xarray) with connectivity measure during flicker
    :param xr_corr: (xarray) with correlation results between PLI and ratings
    :param chan_groups: (list) of lists containing channel names to group in plot
    :param p_thresh: (float) threshold to plot PLI with significant rating correlation
    :return: None
    """
    try:
        flick_con_measures = flick_con_measures.drop_sel(dict(chan_1='Status', chan_2='Status'))  # drop events channel
    except KeyError:
        pass

    # parsing channels
    channels = flick_con_measures.chan_1.data.tolist()
    if chan_groups == None:
        chan_order = channels
        boundaries = np.arange(0, len(channels), 8)
    else:
        grouped_ch = np.concatenate(chan_groups)
        ungrouped_ch = [x for x in channels if x not in grouped_ch]
        chan_order = np.concatenate([grouped_ch, ungrouped_ch]).tolist()
        boundaries = np.cumsum([len(x) for x in chan_groups]).tolist()
        boundaries.insert(0, 0)

    flick_pli = flick_con_measures.loc[dict(con_method='pli')]

    big_mask = xr_corr.loc[dict(statistic='p')] < p_thresh  # boolean mask
    big_masked_r = xr_corr.loc[dict(statistic='r')] * big_mask  # masking all pearson r
    vmin, vmax = big_masked_r.min().data, big_masked_r.max().data
    vabsmax = max([abs(x) for x in [vmin, vmax]])

    for i, question in enumerate(questions):

        # collapsing over dimensions
        mask = big_mask.loc[dict(question=question)]  # boolean mask
        if mask.sum().data == 0:  # if no significant correlations, plot nothing
            continue
        # con = flick_pli.stack(collapsed=('subject', 'run', 'flicker_freq')).mean('collapsed')
        # con_masked = con * mask
        r_masked = big_masked_r.loc[dict(question=question)]

        info = mne.create_info(ch_names=channels, sfreq=ds_hz, ch_types='eeg')  # create info for plotting
        info.set_montage(mne.channels.make_standard_montage('biosemi64'))
        node_angles = circular_layout(node_names=channels, node_order=chan_order, start_pos=90,
                                      group_boundaries=boundaries)

        print(f'plotting highly correlated connections. Question: "{question}"')
        fig, axs = plot_connectivity_circle(r_masked.data, channels, n_lines=n_lines, node_angles=node_angles,
                                            node_colors=list(plt.get_cmap('tab20b').colors),
                                            vmin=-vabsmax, vmax=vabsmax,
                                            facecolor='k',
                                            colormap='Spectral_r',
                                            # colormap='bwr',
                                            title=f'All-to-All correlation, PLI & "{question}"')

        fig.savefig(f'figures/PLI_rating_circular_corr/PLI_rating_{question}',
                    facecolor=fig.get_facecolor(), edgecolor='none')


# load the master subject dictionary
if os.path.isfile('data/subs_dict.pkl'):
    subs_dict = load_obj('data/subs_dict.pkl')
else:
    # TODO: may need to import from connectivity
    from analysis.rest_power import rest_power

    subs_dict = load_obj('data/subs_dict.pkl')

# load the flicker connectivity xarray
try:
    flick_con_measures = xr.load_dataarray('data\subs_con_measures_flick.nc')
except FileNotFoundError:
    raise Exception('Please run first run analysis/connectivity.py for flicker epochs')

xr_ratings = compile_ratings(subs_dict)  # get ratings

try:
    xr_corr = xr.load_dataarray('data/pli_rating_corr_results.nc')
except FileNotFoundError:
    xr_corr = calc_channel_connectivity_rating_corr(flick_con_measures, xr_ratings, p_thresh=1e-5)  # calc correlation
    xr_corr.to_netcdf('data/pli_rating_corr_results.nc')

# NOTE: opting to encode line colors with correlation, not PLI
plot_corr_connectivity_by_rating(flick_con_measures, xr_corr,
                                 chan_groups=[v1_chs, v4_chs, v5_chs], n_lines=None, p_thresh=1e-5)  # plot
