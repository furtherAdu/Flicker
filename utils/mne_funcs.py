# functions built from MNE, preprocessing EEG signal, adding/plotting events

import mne
import os
import gc
from utils.setup_info import lcutoff, hcutoff, ds_hz, chs_not_1020, bdf_home, ID_to_name, ref_channels

def preprocess_file(fname, bad_channels, load_from_file=True):
    """ Filtering, downsampling and interpolating of raw MNE file

    :param fname: saved name for eeg file
    :param bad_channels: bad channels to interpolate
    :return: processed file
    """

    filenames = list(filter(lambda x: x.startswith(fname), os.listdir(os.path.join(bdf_home, 'raw'))))
    filenames.sort()

    if len(filenames) > 1:
        raws = []
        for filename in filenames:
            raws.append(mne.io.read_raw_bdf(os.path.join(bdf_home, 'raw', filename)))
        raw = mne.concatenate_raws(raws)  #  Boundaries of the concatenated raw files are annotated bad
        raw.annotations.delete([0, 1])  # deleting BAD and EDGE boundary
    else:
        raw = mne.io.read_raw_bdf(os.path.join(bdf_home, 'raw', f'{fname}.bdf'))  # read in .bdf file

    # specifying filenames for saving
    fname_raw_1020 = os.path.join(bdf_home, 'isolated_1020', f'{fname}_1020only.fif')
    fname_processed_1020 = os.path.join(bdf_home, 'processed', f'{fname}_1020_processed.fif')

    if os.path.isfile(fname_processed_1020) and load_from_file:
        raw_1020 = mne.io.read_raw_fif(fname_processed_1020) # TODO: re run preprocessing ofr adu's file
        try:
            raw_1020.annotations.delete([0, 1])  # deleting BAD and EDGE boundary, if saved to file
        except IndexError:
            pass

    else:
        if os.path.isfile(fname_raw_1020):
            raw_1020 = mne.io.read_raw_fif(fname_raw_1020)  # loading raw files with montage
            raw_1020.load_data()  # loading data for preprocessing
            raw_1020.plot_psd()
            # try:
            #     raw_not_1020 = mne.io.read_raw_fif('data/01TN_not1020.fif')  # TIMO: EOG not necessary for ICA artifact removal
            # except FileNotFoundError:
            #     pass

        else:
            # separating raw file into 1020 and non-1020  channels
            raw_1020 = raw.copy().load_data()
            raw_1020.drop_channels(chs_not_1020)  # remove unused channels from loaded BDF files

            # applying montage to raw file
            biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
            raw_1020.set_montage(biosemi64_montage)

            # saving raw 1020
            raw_1020.save(fname_raw_1020)

            # raw_not_1020 = raw.copy().load_data() # loading non-1020 channels
            # raw_not_1020.drop_channels(chs_in_1020+chs_not_1020_unused)

        raw_1020.filter(l_freq=lcutoff, h_freq=hcutoff)  # high and low pass filtering
        raw_1020.resample(sfreq=ds_hz)  # downsampling to 1000 hz
        raw_1020.info['bads'] += bad_channels  # identifying shit channels
        raw_1020.interpolate_bads(reset_bads=False)  # interpolating
        raw_1020.save(fname_processed_1020, overwrite=True)  # saving processed file

    return raw_1020


def plot_events(events, eeg, event_id):
    # ensuring correctly noted events by plotting
    events[:, 0] = events[:, 0] * eeg.info['sfreq']  # multiplying second by sampling frequency
    fig = mne.viz.plot_events(events, event_id=event_id, sfreq=eeg.info['sfreq'],
                              first_samp=eeg.first_samp)


def read_eeg(sub_ID, subs_dict, ds_hz=1000, save=False, replace_events=False, load_from_file=True):
    gc.collect()  # cleaning up RAM

    sub_name = ID_to_name[sub_ID]  # corresponding subject name
    sub_dict = subs_dict[sub_name]  # dict with all subject info
    events = sub_dict['event_arr']  # event array

    if events[0, 0] / ds_hz < 1:  # if events have not yet been scaled to the sampling frequency
        events[:, 0] *= ds_hz  # TODO: note this to be source of problem
    # elif events[0, 0] / ds_hz**2 > 1: # assumed first event to be >= 1000s into recording

    events = events.astype(int)
    raw_home = 'processed/raw_w_events'

    # # read in existing or add event data to fif
    event_raw = os.path.join(bdf_home, raw_home, f"processed_w_events_{sub_ID}.fif")
    if os.path.isfile(event_raw) and load_from_file:
        processed_1020 = mne.io.read_raw_fif(event_raw)

    else:
        print('preprocessing and adding event data ...')
        processed_1020 = preprocess_file(sub_ID, sub_dict['bad_channels'], load_from_file=True)  # pre-processed file
        processed_1020.set_annotations(None)  # deleting annotations that may cause rejected epochs
        mne.add_reference_channels(processed_1020, ref_channels=ref_channels)  # TODO: add reference channels

    if replace_events:
        processed_1020 = processed_1020.load_data()
        processed_1020.add_events(events, 'Status', replace=True)  # adding events

    if save and replace_events:
        processed_1020.save(os.path.join(bdf_home, raw_home, f'processed_w_events_{sub_ID}.fif'),
                            overwrite=True)

    return processed_1020, events
