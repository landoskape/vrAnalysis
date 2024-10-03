# add path that contains the vrAnalysis package
import sys
import os

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from argparse import ArgumentParser
from functools import partial
from tqdm import tqdm
import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from vrAnalysis.helpers import cutoff_type, positive_float, AttributeDict, fit_exponentials
from vrAnalysis import database
from vrAnalysis import tracking
from vrAnalysis import analysis
from vrAnalysis.analysis.variance_structure import (
    load_spectra_data,
    plot_spectral_data,
    plot_spectral_averages,
    plot_spectral_energy,
    plot_fourier_data,
    compare_spectral_averages,
    plot_spectral_averages_comparison,
    plot_all_exponential_fits,
    predict_exp_fits_across_mice,
    plot_value_comparison,
    plot_value_to_value_comparison,
    plot_total_variance_comparison,
    predict_total_variance_across_mice,
    plot_svca_vs_cvpca,
)

CUTOFFS = (0.4, 0.7)
MAXCUTOFFS = None


def handle_inputs():
    """method for creating and parsing input arguments"""
    parser = ArgumentParser(description="do summary plots for a mouse")
    parser.add_argument(
        "--mouse-names", type=str, nargs="*", default="all", help="which mice to compare (list of mouse names, or like default), (default='all')"
    )
    parser.add_argument("--cutoffs", nargs="*", type=cutoff_type, default=CUTOFFS, help=f"cutoffs for reliability (default={CUTOFFS})")
    parser.add_argument("--maxcutoffs", nargs="*", type=cutoff_type, default=MAXCUTOFFS, help="maxcutoffs for reliability cells (default=None)")
    parser.add_argument("--do-spectra", default=False, action="store_true", help="create spectrum plots for mouse (default=False)")
    parser.add_argument("--dist-step", default=1, type=float, help="dist-step for creating spkmaps (default=1cm)")
    parser.add_argument("--smooth", default=0.1, type=positive_float, help="smoothing width for spkmaps (default=0.1cm)")
    parser.add_argument("--reload-spectra-data", default=False, action="store_true", help="reload spectra data (default=False)")
    args = parser.parse_args()

    # if mouse_names is "all", get all mouse names from the database
    if args.mouse_names == "all":
        # mousedb = database.vrDatabase("vrSessions")
        mousedb = database.vrDatabase("vrMice")
        df = mousedb.getTable(trackerExists=True)
        mouse_names = df["mouseName"].unique()
        args.mouse_names = mouse_names

    # return the parsed arguments
    return args


def get_spectra(mouse_name, args):
    """method for analyzing and plotting spectra with cvPCA and cvFOURIER analyses"""
    # load spectra data (use temp if it matches)
    track = tracking.tracker(mouse_name)  # get tracker object for mouse
    pcm = analysis.placeCellMultiSession(track, autoload=False)  # open up place cell multi session analysis object (don't autoload!!!)

    single_args = AttributeDict(vars(args))
    single_args["mouse_name"] = mouse_name

    spectra_dictionary = load_spectra_data(pcm, single_args, save_as_temp=False, reload=False)

    # return the dictionary
    return spectra_dictionary


def make_comparison_plots(pcms, spectra_data):
    pass
    # ylog_min = None
    # single_env, across_env = compare_spectral_averages(spectra_data)
    # for do_xlog in [False]:
    #     for do_ylog in [True]:
    #         plot_spectral_averages_comparison(
    #             pcms, single_env, across_env, do_xlog=do_xlog, do_ylog=do_ylog, ylog_min=ylog_min, with_show=False, with_save=True
    #         )
    for include_cvpca in [False]:
        plot_svca_vs_cvpca(pcms, spectra_data, include_cvpca=include_cvpca, do_ylog=True, with_show=False, with_save=True)
    # for relative_session in [False]:  # [True, False]:
    #     # plot_all_exponential_fits(pcms, spectra_data, relative_session=relative_session, with_show=False, with_save=True)
    #     plot_total_variance_comparison(pcms, spectra_data, relative_session=relative_session, with_show=False, with_save=True)
    # summary_val = ["map_var", "map_corr", "map_frob_norm"]
    # summary_val = ["pf_norm", "pf_tcorr_mean"]
    # fancy_names = ["PF Amplitude", "PF Consistency"]
    # for relative_value in [True, False]:
    #     for sval_name, fancy_name in zip(summary_val, fancy_names):
    #         plot_value_comparison(
    #             pcms,
    #             spectra_data,
    #             sval_name,
    #             reduction=("", np.mean),
    #             relative_value=relative_value,
    #             relative_session=True,
    #             with_show=False,
    #             with_save=True,  # True,
    #             poster2024=True,
    #             fancy_name=fancy_name,
    #         )
    #         plot_value_to_value_comparison(
    #             pcms,
    #             spectra_data,
    #             sval_name,
    #             "cv_by_env_all",
    #             first_reduction=("", np.mean),
    #             second_reduction="sum",
    #             relative_value=relative_value,
    #             relative_session=True,
    #             first_offset=0,
    #             with_show=False,
    #             with_save=True,
    #         )
    # rel_cor_cutoffs = [0.7]

    # def frac_gt(x, rcc):
    #     return np.sum(x > rcc) / len(x)

    # reductions = [(f"frac_gt_{rcc}", partial(frac_gt, rcc=rcc)) for rcc in rel_cor_cutoffs]
    # for relative_session in [True, False]:
    #     for reduction in reductions:
    #         plot_value_comparison(
    #             pcms,
    #             spectra_data,
    #             "rel_cor",
    #             reduction=reduction,
    #             relative_value=False,
    #             relative_session=relative_session,
    #             with_show=False,
    #             with_save=True,
    #         )
    # predict_exp_fits_across_mice(pcms, spectra_data, with_show=False, with_save=True)
    # predict_total_variance_across_mice(pcms, spectra_data, with_show=False, with_save=True)


if __name__ == "__main__":
    args = handle_inputs()

    if args.do_spectra:
        # analyze spectra and make plots
        pcms = []
        spectra_data = []
        for mouse in args.mouse_names:
            print(f"Getting spectra data for {mouse}")
            spectra_data.append(get_spectra(mouse, args))  # Each is a dictionary of all the spectral output data
            c_track = tracking.tracker(mouse)
            c_pcm = analysis.placeCellMultiSession(c_track, autoload=False)
            pcms.append(c_pcm)
        make_comparison_plots(pcms, spectra_data)
