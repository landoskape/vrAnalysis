# add path that contains the vrAnalysis package
import sys
import os

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from argparse import ArgumentParser
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt

from vrAnalysis.helpers import (
    cutoff_type,
    positive_float,
)
from vrAnalysis import tracking
from vrAnalysis import analysis
from vrAnalysis.analysis.variance_structure import (
    load_spectra_data,
    plot_spectral_data,
    plot_spectral_averages,
    plot_spectral_energy,
    plot_fourier_data,
    plot_reliability_data,
    plot_pf_var_data,
    plot_svca_data,
    compare_exp_fits,
)

CUTOFFS = (0.4, 0.7)
MAXCUTOFFS = None


def handle_inputs():
    """method for creating and parsing input arguments"""
    parser = ArgumentParser(description="do summary plots for a mouse")
    parser.add_argument("--mouse-name", required=True, type=str, default=None, help="which mouse to run experiments on (required)")
    parser.add_argument("--cutoffs", nargs="*", type=cutoff_type, default=CUTOFFS, help=f"cutoffs for reliability (default={CUTOFFS})")
    parser.add_argument("--maxcutoffs", nargs="*", type=cutoff_type, default=MAXCUTOFFS, help="maxcutoffs for reliability cells (default=None)")
    parser.add_argument("--do-spectra", default=False, action="store_true", help="create spectrum plots for mouse (default=False)")
    parser.add_argument("--dist-step", default=1, type=float, help="dist-step for creating spkmaps (default=1cm)")
    parser.add_argument("--smooth", default=0.1, type=positive_float, help="smoothing width for spkmaps (default=0.1cm)")
    parser.add_argument("--reload-spectra-data", default=False, action="store_true", help="reload spectra data (default=False)")
    return parser.parse_args()


def analyze_spectra(pcm, args):
    """method for analyzing and plotting spectra with cvPCA and cvFOURIER analyses"""
    # load spectra data (use temp if it matches)
    spectra_data = load_spectra_data(pcm, args, save_as_temp=True)

    # make plots
    plt.close("all")
    # for color_by_session in [True, False]:
    #     for normalize in [True, False]:
    #         plot_spectral_data(
    #             pcm,
    #             spectra_data,
    #             color_by_session=color_by_session,
    #             normalize=normalize,
    #             with_show=False,
    #             with_save=True,
    #         )
    #     for covariance in [False, True]:
    #         plot_fourier_data(pcm, spectra_data, color_by_session=color_by_session, covariance=covariance, with_show=False, with_save=True)
    #     plot_reliability_data(pcm, spectra_data, color_by_session=color_by_session, with_show=False, with_save=True)
    #     plot_pf_var_data(pcm, spectra_data, color_by_session=color_by_session, with_show=False, with_save=True)
    # for do_xlog in [True, False]:
    #     for do_ylog in [True, False]:
    #         plot_spectral_averages(pcm, spectra_data, do_xlog=do_xlog, do_ylog=do_ylog, with_show=False, with_save=True)
    # plot_spectral_energy(pcm, spectra_data, with_show=False, with_save=True)
    # for normalize in [True, False]:
    #     plot_svca_data(pcm, spectra_data, normalize=normalize, with_show=False, with_save=True)
    for color_by_session in [True, False]:
        compare_exp_fits(pcm, spectra_data, amplitude=True, color_by_session=color_by_session, with_show=False, with_save=True)


if __name__ == "__main__":
    args = handle_inputs()
    mouse = args.mouse_name
    cutoffs = args.cutoffs
    maxcutoffs = args.maxcutoffs

    track = tracking.tracker(mouse)  # get tracker object for mouse
    pcm = analysis.placeCellMultiSession(track, autoload=False)  # open up place cell multi session analysis object (don't autoload!!!)

    if args.do_spectra:
        # analyze spectra and make plots
        analyze_spectra(pcm, args)
