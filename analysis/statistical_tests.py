from utils.setup_info import ID_to_name, questions, sub_IDs, test_freqs, runs
from utils.helper_funcs import load_obj
from analysis.secondary_power_features import calc_secondary_power_features
from pingouin import rm_anova, normality, sphericity, friedman
import pandas as pd
import xarray as xr
import numpy as np
import os

pd.options.mode.chained_assignment = None  # TODO: solves "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame"


def compile_ratings(subs_dict):
    # allocating xarray for response, (subjects, test frequency, run, ratings)
    xr_ratings = np.ones((len(sub_IDs), len(test_freqs), len(runs), len(questions))) * np.nan
    xr_ratings = xr.DataArray(xr_ratings,
                              coords=dict(subject=sub_IDs, flicker_freq=test_freqs, run=runs, question=questions),
                              dims=['subject', 'flicker_freq', 'run', 'question'])

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
                xr_ratings.loc[dict(subject=sub_ID, question=question,
                                    flicker_freq=run_epoch_freqs, run=run)] = run_rating

    return xr_ratings


def rm_aligned_rank_transform(df, DV=None, IV=None):
    """
    Calculates aligned rank transformed RM ANOVA per https://sci-hub.se/10.1145/1978942.1978963
    :param df: (dataframe) with index names=['subject', *IV] and columns=[*DV]
    :param DV: (list) dependent variable
    :param IV: (list) independent variable
    :return: rm_anova results for factor A, B, and their interaction
    """

    # (1) compute residuals
    residual = df.subtract(df.groupby(IV).mean())  # cell means from matching levels of IVs

    # (2) Compute estimated effects for all main and interaction effects
    # df.reset_index(inplace=True)  # removing nested/hierarchical structure
    mu = df.mean()[0]  # grand mean
    est_oneway_A = df.groupby(IV[0]).mean() - mu  # estimated main effect for factor A
    est_oneway_B = df.groupby(IV[1]).mean() - mu  # estimated main effect for factor B
    est_twoway = df - df.groupby(IV[0]).mean() - df.groupby(
        IV[1]).mean() + mu  # estimated effects for all main and interaction effects

    # (3) Compute aligned response Y' (y prime)
    Y_pA = residual + est_oneway_A
    Y_pB = residual + est_oneway_B
    Y_p_twoway = residual + est_twoway

    # (4) Assign averaged ranks Y″ (y double prime)
    Y_ppA = Y_pA.rank()
    Y_ppB = Y_pB.rank()
    Y_pp2 = Y_p_twoway.rank()

    # # sanity check:
    # First, every column of aligned responses Y′ should sum to zero
    # Second, a full-factorial ANOVA performed on the aligned (not ranked) responses Y′ should show all effects
    #   stripped out(F=0.00, p=1.00) except for the effect for which the data were aligned.
    ensure_A = rm_anova(Y_pA.reset_index(), dv=DV, within=IV, subject='subject', detailed=True)
    assert ensure_A['p-unc'][1:].mean() == 1, 'Something went wrong calculating effect of factor A'
    ensure_B = rm_anova(Y_pB.reset_index(), dv=DV, within=IV, subject='subject', detailed=True)
    assert ensure_B['p-unc'][[0, 2]].mean() == 1, 'Something went wrong calculating effect of factor B'
    ensure_AB = rm_anova(Y_p_twoway.reset_index(), dv=DV, within=IV, subject='subject', detailed=True)
    assert ensure_AB['p-unc'][:2].mean() == 1, 'Something went wrong calculating interaction of factors A and B'

    # calculating ANOVAs
    A_anova = rm_anova(Y_ppA.reset_index(), dv=DV, within=IV[0], subject='subject', detailed=True)
    B_anova = rm_anova(Y_ppB.reset_index(), dv=DV, within=IV[1], subject='subject', detailed=True)
    AB_anova = rm_anova(Y_pp2.reset_index(), dv=DV, within=IV, subject='subject', detailed=True)

    return A_anova, B_anova, AB_anova


def calc_rm_anova(ds):
    """
    Calculates normality and sphericity of data. Performs RM anova when appropriate
    :param ds: (xarray dataset) must contain variables 'ratings' and 'secondary_power_features',
        with coordinates {run, test_freq, subject, flicker_freq, feature:'alpha_power_during, question}
    :return:
    """

    # # calculating stats for alpha power
    alpha_power = ds['alpha_power'].to_dataframe()
    alpha_power.reset_index(inplace=True)

    # is power normally distributed by flicker frequency (for each freq, N=6)
    ap_normality = normality(alpha_power, dv='alpha_power', group='flicker_freq', method='shapiro', alpha=0.05)

    # is power spherical by flicker frequency
    ap_sphericity = sphericity(alpha_power, dv='alpha_power', within='flicker_freq', subject='subject',
                               method='mauchly', alpha=0.05)

    print('\nAlpha power')
    if bool(ap_sphericity[0]):
        print(f'data are spherical. Use uncorrected p-values.')
    else:
        print(f'data are not spherical. Use GG-corrected p-values.')

    if bool(ap_normality.product(axis=0)['normal']):
        print(f'data are normal.')
        ap_anova = rm_anova(data=alpha_power, dv='alpha_power', within=['run', 'flicker_freq'], subject='subject',
                            detailed=True)
        print(ap_anova)

    else:  # calculate rank over all observations
        print(f'data do not meet RM ANOVA assumptions. Performing RM ANOVA with ranked observations.')
        run_anova, flicker_freq_anova, rf_anova = \
            rm_aligned_rank_transform(df=ds['alpha_power'].to_dataframe(), DV=['alpha_power'],
                                      IV=['run', 'flicker_freq'])
        print(pd.concat([run_anova[0:1], flicker_freq_anova[0:1], rf_anova[2:3]]))  # omitting error rows

    # # calculating stats for question ratings
    for question in questions:
        question_rating = ds['ratings'].loc[dict(question=question)].to_dataframe()
        question_rating.reset_index(inplace=True)

        # is the rating normally distributed by flicker frequency (for each freq, N=6)
        question_normality = normality(question_rating, dv='ratings', group='flicker_freq', method='shapiro',
                                       alpha=0.05)

        # is rating spherical by flicker frequency
        question_sphericity = sphericity(question_rating, dv='ratings', within='flicker_freq', subject='subject',
                                         method='mauchly', alpha=0.05)

        # creating datafram for IV = ranked alpha rm_anova
        df_qa = ds['alpha_power'].to_dataframe().droplevel('feature')
        df_qa.reset_index(inplace=True)
        df_qa['ratings'] = question_rating['ratings']  # question ratings to dataframe

        # ranking 'alpha_power'(average across runs) for each subject range(8)
        ranked_alpha = df_qa.groupby(['flicker_freq', 'subject']).mean()  # taking mean alpha power over runs
        ranked_alpha.reset_index(inplace=True)  # removing nested structure
        ranked_alpha.alpha_power = ranked_alpha.groupby('subject')['alpha_power'].rank()  # ranking alpha

        # updating df with ranked alpha (could probably do with a multiindex)
        for sub_ID in sub_IDs:
            for freq in df_qa['flicker_freq'].unique():
                df_idx = (df_qa['flicker_freq'] == freq) & (df_qa['subject'] == sub_ID)
                ra_idx = (ranked_alpha['flicker_freq'] == freq) & (
                        ranked_alpha['subject'] == sub_ID)  # ranked alpha index
                df_qa['alpha_power'].loc[df_idx] = np.repeat(ranked_alpha.loc[ra_idx]['alpha_power'].values, 2)

        print(f'\n"{question}"')
        if bool(question_sphericity[0]):
            print(f'data are spherical. Use uncorrected p-values.')
        else:
            print(f'data are not spherical. Use GG-corrected p-values.')

        if bool(question_normality.product(axis=0)['normal']):
            print(f'data are normal.')
            question_anova = rm_anova(data=question_rating, dv='ratings', within=['run', 'flicker_freq'],
                                      subject='subject', detailed=True)
            print(question_anova)

            # resting alpha - rating anova
            rar_anova = rm_anova(data=df_qa, dv='ratings', within=['run', 'alpha_power'],
                                 subject='subject', detailed=True)
            print('\n', rar_anova)

        else:  # calculate rank over all observations
            print(f'data do not meet RM ANOVA assumptions. Performing RM ANOVA with ranked observations.')
            run_anova, flicker_freq_anova, rf_anova = rm_aligned_rank_transform(df=ds['ratings'].loc[dict(question=question)].to_dataframe(),
                                                                                DV='ratings',
                                                                                IV=['run', 'flicker_freq'])
            print(pd.concat([run_anova[0:1], flicker_freq_anova[0:1], rf_anova[2:3]]))  # omitting error rows

            a_run_anova, alpha_anova, rar_anova = rm_aligned_rank_transform(df=df_qa.set_index(['subject', 'run', 'alpha_power']),
                                                                            DV='ratings',
                                                                            IV=['run', 'alpha_power'])
            print('\n', pd.concat([a_run_anova[0:1], alpha_anova[0:1], rar_anova[2:3]]))  # omitting error rows

        # TODO: (extra) calculate two-way, mixed effects RM ANOVA with within=['flicker_freq','fs_distance']


# load the master subject dictionary
if os.path.isfile('data/subs_dict_psds_alpha_SSVEP_rest.pkl'):
    subs_dict = load_obj('data/subs_dict_psds_alpha_SSVEP_rest.pkl')
else:
    from analysis.rest_power import rest_power

    subs_dict = load_obj('data/subs_dict_psds_alpha_SSVEP_rest.pkl')

# creating xarrays to hold alpha power and ratings
xr_alpha_power = calc_secondary_power_features(subs_dict).drop_sel(dict(feature=['SSVEP', 'alpha_power_after']))
xr_alpha_power.name = 'alpha_power'
xr_ratings = compile_ratings(subs_dict)
xr_ratings.name = 'ratings'
ds = xr.merge([xr_ratings, xr_alpha_power])  # merging xarray of ratings and alpha power

# setting printing options for statistics
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)

calc_rm_anova(ds)  # calculating and printing stats
