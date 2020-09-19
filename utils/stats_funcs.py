from utils.setup_info import ID_to_name, questions, sub_IDs, test_freqs, runs
from pingouin import rm_anova, normality, sphericity, pairwise_ttests
import pandas as pd
import xarray as xr
import numpy as np


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


def calc_rm_anova(ds, p_thresh=.05):
    """
    Calculates normality and sphericity of data. Performs RM anova when appropriate
    :param ds: (xarray dataset) must contain variables 'ratings' and 'secondary_power_features',
        with coordinates {run, test_freq, subject, flicker_freq, feature:'alpha_power_during, question}
    :return:
    """

    print('\n', 'Calculating repeated measures ANOVAs')
    stats_dict = dict()  # master dict to contain all stats results

    # # calculating stats for alpha power
    alpha_dict = dict()  # dictionary for dv=alpha_power tests
    alpha_power = ds['alpha_power'].to_dataframe()
    alpha_power.reset_index(inplace=True)

    # is power normally distributed by flicker frequency (for each freq, N=6)
    ap_normality = normality(alpha_power, dv='alpha_power', group='flicker_freq', method='shapiro', alpha=p_thresh)

    # is power spherical by flicker frequency
    ap_sphericity = sphericity(alpha_power, dv='alpha_power', within='flicker_freq', subject='subject',
                               method='mauchly', alpha=p_thresh)

    print('\nAlpha power')
    if bool(ap_sphericity[0]):
        print(f'data are spherical. Use uncorrected p-values.')
        alpha_dict.update(dict(spherical=True))
    else:
        print(f'data are not spherical. Use GG-corrected p-values.')
        alpha_dict.update(dict(spherical=False))

    if bool(ap_normality.product(axis=0)['normal']):
        print(f'data are normal.')
        ap_anova = rm_anova(data=alpha_power, dv='alpha_power', within=['run', 'flicker_freq'], subject='subject',
                            detailed=True)
        print(ap_anova)

        alpha_dict.update(dict(normal=True))
        alpha_dict.update(dict(anova=ap_anova))

    else:  # calculate rank over all observations
        print(f'data do not meet RM ANOVA assumptions. Performing RM ANOVA with ranked observations.')
        run_anova, flicker_freq_anova, rf_anova = \
            rm_aligned_rank_transform(df=ds['alpha_power'].to_dataframe(), DV=['alpha_power'],
                                      IV=['run', 'flicker_freq'])
        ranked_anova = pd.concat([run_anova[0:1], flicker_freq_anova[0:1], rf_anova[2:3]])
        print(ranked_anova)  # omitting error rows

        alpha_dict.update(dict(normal=False))
        alpha_dict.update(dict(anova=ranked_anova))

    alpha_dict.update(dict(df=alpha_power))  # adding data to master dict
    stats_dict.update(dict(alpha_power=alpha_dict))  # updating master stats dict with ANOVA

    # # calculating stats for question ratings
    for question in questions:

        question_freq_dict = dict()  # dictionary for dv=ratings, IV=flicker_freq tests
        question_rap_dict = dict()  # dictionary for dv=ratings, IV=ranked alpha_power

        question_rating = ds['ratings'].loc[dict(question=question)].to_dataframe()
        question_rating.reset_index(inplace=True)

        # is the rating normally distributed by flicker frequency (for each freq, N=6)
        question_normality = normality(question_rating, dv='ratings', group='flicker_freq', method='shapiro',
                                       alpha=p_thresh)

        # is rating spherical by flicker frequency
        question_sphericity = sphericity(question_rating, dv='ratings', within='flicker_freq', subject='subject',
                                         method='mauchly', alpha=p_thresh)

        # creating datafram for IV = ranked alpha rm_anova
        df_qa = ds['alpha_power'].to_dataframe().droplevel('feature')
        df_qa.reset_index(inplace=True)
        df_qa['ratings'] = question_rating['ratings']  # question ratings to dataframe

        # ranking 'alpha_power'(average across runs) for each subject range(8)
        ranked_alpha = df_qa.groupby(['flicker_freq', 'subject']).mean()  # taking mean alpha power over runs
        ranked_alpha.reset_index(inplace=True)  # removing nested structure
        ranked_alpha.alpha_power = ranked_alpha.groupby('subject')['alpha_power'].rank()  # ranking alpha, 1 is lowest

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
            question_freq_dict.update(dict(spherical=True))
            question_rap_dict.update(dict(spherical=True))
        else:
            print(f'data are not spherical. Use GG-corrected p-values.')
            question_freq_dict.update(dict(spherical=False))
            question_rap_dict.update(dict(spherical=False))

        if bool(question_normality.product(axis=0)['normal']):
            print(f'data are normal.')
            question_anova = rm_anova(data=question_rating, dv='ratings', within=['run', 'flicker_freq'],
                                      subject='subject', detailed=True)
            print(question_anova)

            question_freq_dict.update(dict(normal=True))  # updating results dict for question
            question_freq_dict.update(dict(anova=question_anova))

            # iv=resting alpha, dv=rating anova
            rar_anova = rm_anova(data=df_qa, dv='ratings', within=['run', 'alpha_power'],
                                 subject='subject', detailed=True)
            print('\n', rar_anova)

            question_rap_dict.update(dict(normal=True))  # updating results dict for question
            question_rap_dict.update(dict(anova=rar_anova))

        else:  # calculate rank over all observations
            print(f'data do not meet RM ANOVA assumptions. Performing RM ANOVA with ranked observations.')
            run_anova, flicker_freq_anova, rf_anova = rm_aligned_rank_transform(
                df=ds['ratings'].loc[dict(question=question)].to_dataframe(),
                DV='ratings',
                IV=['run', 'flicker_freq'])
            ranked_freq_anova = pd.concat([run_anova[0:1], flicker_freq_anova[0:1], rf_anova[2:3]])
            print(ranked_freq_anova)  # omitting error rows

            question_freq_dict.update(dict(normal=False))  # updating results dict for question
            question_freq_dict.update(dict(anova=ranked_freq_anova))

            a_run_anova, alpha_anova, rar_anova = rm_aligned_rank_transform(
                df=df_qa.set_index(['subject', 'run', 'alpha_power']),
                DV='ratings',
                IV=['run', 'alpha_power'])

            ranked_rap_anova = pd.concat([a_run_anova[0:1], alpha_anova[0:1], rar_anova[2:3]])
            print('\n', ranked_rap_anova)  # omitting error rows

            question_freq_dict.update(dict(normal=False))  # updating results dict for question
            question_rap_dict.update(dict(anova=ranked_rap_anova))

        question_freq_dict.update(dict(df=question_rating))  # adding data to master dict
        question_rap_dict.update(dict(df=df_qa))
        stats_dict.update({f'freq_{question}': question_freq_dict})  # updating master stats dict with ANOVA
        stats_dict.update({f'rap_{question}': question_rap_dict})

        # TODO: (extra) calculate two-way, mixed effects RM ANOVA with within=['flicker_freq','fs_distance']

    return stats_dict


def get_sig_anovas(stats_dict, p_thresh=.05):
    """
    Prints and returns signficant ANOVA results
    :param stats_dict: (nested dict) with {name of test {ANOVA results and data}}
    :param p_thresh: (float) threshold for p-value
    :return: (list) significant ANOVA results
    """

    print('\n', 'Calculating significant ANOVAs')
    sig_anovas = []  # list of all significant tests

    # for all statistical tests
    for test_key in list(stats_dict.keys()):
        test = stats_dict[test_key]
        spherical = test['spherical']  # determine sphericality
        anova = test['anova']

        # identify ANOVAs with significant p-value
        if spherical:
            pval = anova['p-unc']
        else:
            pval = anova['p-GG-corr']

        if not anova.loc[pval < p_thresh].empty:  # if there's a significant result
            print('\n', test_key, '\n', anova.loc[pval < p_thresh])  # print the result
            sig_anovas.append({f'{test_key}': anova.loc[pval < p_thresh]})  # add to significant results list

    return sig_anovas


def paired_tests(stats_dict, sig_anovas, p_thresh=.05, p_adjust='bonf'):
    """
    Calculates, prints, and returns pairwise t-tests for all significant anovas.
    Non parametric tests are used if vast majority (>80%) of DVs across within-subject conditions aren't normal
    Note: t-tests 'T' value

    :param stats_dict: nested dict) with {name of test {ANOVA results and data}}
    :param sig_anovas: (list) of dataframs with significant ANOVAs
    :param p_thresh: (float) threshold for determining significance, used for multiple comparison corrections
    :param p_adjust: (str) p-value adjustment method to use, per pingouin.pairwise_ttests
    :return: (dict) results of pairwise test, key formatted as (anova test)_(question)_(paired_effect)
    """

    print('\n', 'Calculating pairwise t-tests for significant ANOVAs')
    paired_tests_dict = dict()

    for sig_anova in sig_anovas:  # for all signfiicant ANOVAs

        test_key = list(sig_anova.keys())[0]  # sig test name
        df = stats_dict[test_key]['df']  # corresponding sig data
        variables = sig_anova[test_key].Source.to_list()  # DV/interactions with a sig effect

        for i, variable in enumerate(variables):  # string parsing components of interactions
            variables[i] = variable.split(' * ')

        for variable in variables:  # for all DV/interactions

            # checking normality across condition
            uniq_vars = np.unique(df[variable])
            all_normality = [normality(df['ratings'][(df[variable] == uniq_var).to_numpy().squeeze()]) for uniq_var in
                             uniq_vars]
            all_normality = pd.concat(all_normality)

            # run paired t-tests/Wilcoxon
            if all_normality['normal'].mean() > .8:  # if large majority of between group scores are normal
                pwt = pairwise_ttests(parametric=True, dv='ratings', subject='subject', data=df, alpha=p_thresh,
                                      padjust=p_adjust, within=variable)  # note within = the repeated measure
            else:
                pwt = pairwise_ttests(parametric=False, dv='ratings', subject='subject', data=df, alpha=p_thresh,
                                      padjust=p_adjust, within=variable)

            # key for results is (anova test)_(question)_(paired_effect)
            paired_tests_dict.update({f'{test_key}_{"_".join(variable)}': pwt})

    return paired_tests_dict
