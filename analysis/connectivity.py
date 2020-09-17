from utils.helper_funcs import myround, save_obj, load_obj
from utils.mne_funcs import read_eeg
from utils.st_adjudication_funcs import discretize_into_events
from utils.setup_info import sub_IDs, ID_to_name, len_events, event_dict, bdf_home, ds_hz, test_freqs, \
    runs, flick_block_len, postprocessed_channels
from mne.connectivity import spectral_connectivity
from mne.viz import plot_sensors_connectivity
import xarray as xr
import numpy as np
import mne
import os
import gc


def calc_connectivity(subs_dict, con_methods):
    """
    Calculates multiple connectivity methods for all subject and returns labeled xarray
    :param subs_dict: (dict) of master subject data
    :param con_methods: (list) connectivity methods to calculate over flicker epochs
    :return: (dict) updated subs_dict, (xarray) containing calculated connectivity for all subjects
    """
    # connectivity calculatiion params
    fmin, fmax = (8.), (12.)  # specifying band over which to calculated connectivity
    tmin = 0.0  # exclude the baseline period
    n_sub = 6  # number of sub_epochs, over which to calculate connectivity

    # allocating empty xarray for results
    con_measures = np.ones(
        (len(sub_IDs), len(test_freqs), len(postprocessed_channels), len(postprocessed_channels),
         len(runs), len(con_methods))) * np.nan

    con_measures = xr.DataArray(con_measures, coords=dict(subject=sub_IDs,
                                                          flicker_freq=test_freqs,
                                                          chan_1=postprocessed_channels,
                                                          chan_2=postprocessed_channels,
                                                          run=runs,
                                                          con_method=con_methods),
                                dims=['subject', 'flicker_freq', 'chan_1', 'chan_2', 'run', 'con_method'])

    # looping over subjects for analysis
    for sub_ID in sub_IDs:

        name = ID_to_name[sub_ID]

        if 'connectivity' in list(subs_dict[name].keys()):  # if connectivity already calculated, go to next subject
            con_measures.loc[dict(subject=sub_ID)] = subs_dict[name]['connectivity']
            continue

        block_order = subs_dict[name]['response']['Frequency (Hz)']

        # preprocess/load data
        processed_1020, events = read_eeg(sub_ID, subs_dict=subs_dict, ds_hz=ds_hz, load_from_file=True,
                                          replace_events=False, save=False)

        flicker_epochs = discretize_into_events(processed_1020, which='flick', sub_ID=sub_ID, events=events,
                                                len_events=len_events, event_dict=event_dict, bdf_home=bdf_home,
                                                save=False)

        # sanity check for dimensional aligment
        mne_events = [list(flicker_epochs.event_id.values()).index(x) for x in flicker_epochs.events[:, -1]]
        mne_event_names = np.array(list(flicker_epochs.event_id.keys()))[mne_events]
        mne_block_order = [float(x[6:]) for x in mne_event_names]  # assumes all(mne_even_names) .starts_with("flick/")
        assert [x for x in block_order.to_list() if type(x) != str] == mne_block_order, 'Block frequency order is not aligned in flicker_epochs'

        for i, _ in enumerate(flicker_epochs):
            epoch = flicker_epochs[i:i + 1]
            run = i // len(test_freqs)  # floor divide, assumes flicker_epochs in order[0,1]

            print(f'\nCalcualting connectivity for subject {sub_ID} in a {mne_block_order[i]} Hz epoch (run {run}),'
                  f' subdivided into  {n_sub} sub epochs')

            # discretizing epoch into (N sub_epochs of  x channels x times)
            bounds = epoch.times[(epoch.times >= 0)][
                     ::int(flick_block_len / 5) * ds_hz] * ds_hz  # assumes epoch.times exceed flick_block_len
            df = epoch.to_data_frame()  # pushing to dataframe for easy manipulation
            sub_epochs_data = np.array(
                [df[epoch.info.ch_names][(df.time >= bounds[i]) & (df.time < bounds[i + 1])].to_numpy().T for i in
                 range(n_sub)])
            sub_epochs = mne.EpochsArray(sub_epochs_data, epoch.info)  # creating mne epochs object from subepochs

            # calculating connectivity measure(s)
            con, freqs, times, n_epochs, n_tapers = spectral_connectivity(sub_epochs, method=con_methods,
                                                                          mode='multitaper',
                                                                          sfreq=ds_hz, fmin=fmin, fmax=fmax,
                                                                          faverage=True,
                                                                          tmin=0, tmax=30,
                                                                          n_jobs=2)  # speeding up with 2 jobs

            for j, con_method in enumerate(con_methods):  # saving to xarray
                con_measures.loc[
                    dict(subject=sub_ID, flicker_freq=mne_block_order[i], run=run, con_method=con_method)] = \
                    con[j].squeeze()

        print(f'Finished calcualting connectivity for subject {sub_ID}\n')
        subs_dict[name].update(dict(connectivity=con_measures.loc[dict(subject=sub_ID)]))

    # # Now, visualize connectivity in 3D for an example
    # plot_sensors_connectivity(flicker_epochs.info, con[0][:, :, 0])

    return subs_dict, con_measures


gc.collect()  # clearing unused memory

# load the master subject dictionary
if os.path.isfile('data/subs_dict_psds_alpha_SSVEP_rest_con.pkl'):
    subs_dict = load_obj('data/subs_dict_psds_alpha_SSVEP_rest_con.pkl')
else:
    from analysis.rest_power import rest_power

    subs_dict = load_obj('data/subs_dict_psds_alpha_SSVEP_rest.pkl')

con_methods = ['coh', 'pli']  # calculating coherence and phase lag index
subs_dict, con_measures = calc_connectivity(subs_dict, con_methods)

con_measures.to_netcdf('data\subs_con_measures.nc')  # saving as xarray
save_obj(subs_dict, 'data/subs_dict_psds_alpha_SSVEP_rest_con.pkl')  # saving in master subject dict
