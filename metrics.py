import numpy as np
import pycatch22
from numpy import array as a

def fill_lc_gaps(mags,times,cadence):
    # lc.data.sort("times")
    missing = np.where(np.diff(times) > 10*cadence)[0]
    
    missing_stretches = []
    for m in missing:
        missing_stretches.extend(np.arange(times[m],times[m+1],cadence))
    
    missing_stretches = a(missing_stretches)
    new_times = np.concatenate((a(times),missing_stretches))
    new_mags = np.concatenate((a(mags),np.mean(mags)*np.ones_like(missing_stretches)))
    new_uncerts = np.concatenate((a(mags),np.full(len(missing_stretches),np.inf)))
    
    time_idxs = np.argsort(new_times)
    
    new_times = new_times[time_idxs]
    new_mags = new_mags[time_idxs]
    new_uncerts = new_uncerts[time_idxs]

    return new_times, new_mags, new_uncerts

def calc_metrics(mags, times, cadence):
    _, filled, _ = fill_lc_gaps(mags, times, cadence)
    return a(pycatch22.catch22_all(filled,short_names=False)["values"] + [np.std(mags),np.mean(mags)])

short_metric_names = [
    'mode_5',
    'mode_10',
    'acf_timescale',
    'acf_first_min',
    'ami2',
    'trev',
    'high_fluctuation',
    'stretch_high',
    'transition_matrix',
    'periodicity',
    'embedding_dist',
    'ami_timescale',
    'whiten_timescale',
    'outlier_timing_pos',
    'outlier_timing_neg',
    'centroid_freq',
    'stretch_decreasing',
    'entropy_pairs',
    'rs_range',
    'dfa',
    'low_freq_power',
    'forecast_error',
    "std",
    "mean"
]

long_metric_names = [
    'DN_HistogramMode_5',
    'DN_HistogramMode_10',
    'CO_f1ecac',
    'CO_FirstMin_ac',
    'CO_HistogramAMI_even_2_5',
    'CO_trev_1_num',
    'MD_hrv_classic_pnn40',
    'SB_BinaryStats_mean_longstretch1',
    'SB_TransitionMatrix_3ac_sumdiagcov',
    'PD_PeriodicityWang_th0_01',
    'CO_Embed2_Dist_tau_d_expfit_meandiff',
    'IN_AutoMutualInfoStats_40_gaussian_fmmi',
    'FC_LocalSimple_mean1_tauresrat',
    'DN_OutlierInclude_p_001_mdrmd',
    'DN_OutlierInclude_n_001_mdrmd',
    'SP_Summaries_welch_rect_area_5_1',
    'SB_BinaryStats_diff_longstretch0',
    'SB_MotifThree_quantile_hh',
    'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
    'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
    'SP_Summaries_welch_rect_centroid',
    'FC_LocalSimple_mean3_stderr',
    "std",
    "mean"
]
