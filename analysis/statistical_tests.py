from utils.stats_funcs import calc_rm_anova, get_sig_anovas, paired_tests
from utils.helper_funcs import load_obj
from utils.st_adjudication_funcs import compile_ratings, calc_secondary_power_features
import pandas as pd
import xarray as xr
import os
import gc

gc.collect()

pd.options.mode.chained_assignment = None  # Note: solves "SettingWithCopyWarning: A value is trying to be  set on a copy of a slice from a DataFrame"
pd.set_option('display.width', 1000)  # setting printing options for statistics
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

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

# load in rest connectivity xarray
try:
    rest_con_measures = xr.load_dataarray('data\subs_con_measures_rest.nc')
except FileNotFoundError:
    raise Exception('Please run first run analysis/connectivity.py for rest epochs')

# # (A) statistical tests using IVs {ranked alpha power, run, flicker frequency}
# creating xarrays to hold alpha power and ratings
xr_alpha_power = calc_secondary_power_features(subs_dict).drop_sel(
    dict(feature=['SSVEP', 'alpha_power_after']))  # get secondary power features
xr_alpha_power.name = 'alpha_power'
xr_ratings = compile_ratings(subs_dict)  # get ratings
xr_ratings.name = 'ratings'

# merging into xarray of ratings and alpha power
ds = xr.merge([xr_ratings, xr_alpha_power])

# # # running tests
stats_dict = calc_rm_anova(ds, p_thresh=.05)  # calculating and printing ANOVAs
sig_anovas = get_sig_anovas(stats_dict, p_thresh=.05)  # getting significant ANOVAs
paired_tests_dict = paired_tests(stats_dict, sig_anovas, p_thresh=.05, p_adjust='bonf')  # calcualting pairwise t-tests


# # TODO: (B) statistical tests using IVs {coherence, PLI}
# # Connectivity test idea (3)
#   RM ANOVA to determine effect of ranked, occipitotemporal channel-averaged PLI on
#   the questionnaire ratings.
#   Results would show how ordinal connectivity (lower vs. higher)  affects subjective experience.
