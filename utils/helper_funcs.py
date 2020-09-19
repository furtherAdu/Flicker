import pickle
import numpy as np

def make_event_name_list(freqs, pulses=None):
    """
    Creates list of events from frequencies and number of pulses
    :param freqs: (array) unique frequencies to create list from
    :param pulses: (array) times of pulses
    :return:
    """
    event_names = ['rest/init']
    for x in freqs:
        event_names.append(f'flick/{x}')
        event_names.append(f'rest/{x}')
    event_names.append('rest/end')
    event_names.append('pulse')

    try:
        for i in range(len(pulses) - 1):
            event_names.append('pulse')
    except TypeError:
        pass

    return event_names


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def myround(x, base=5):
    return base * np.round(x/base)


