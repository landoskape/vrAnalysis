import os
import sys
import argparse
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import scipy as sp
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

from scripts.dimilibi.helpers import make_position_basis, filter_timepoints, figure_folder

from vrAnalysis import analysis
from vrAnalysis import helpers
from vrAnalysis import database
from vrAnalysis import faststats as fs
from dimilibi import Population
from dimilibi import SVCA
from dimilibi import PCA
from dimilibi import RidgeRegression

sessiondb = database.vrDatabase("vrSessions")
FIGURE_FOLDER = "svca_placefields"


def get_sessions():
    ises = sessiondb.iterSessions(imaging=True, vrRegistration=True, experimentID=1)
    return list(ises)


def svca_placefield_tempfile(ses):
    return f"svca_placefield_{ses.mouseName}_{ses.dateString}_{ses.sessionid}"


def process_session(ses):
    print(f"Processing session {ses.sessionPrint()}...")
    keep_planes = [1, 2, 3, 4]
    onefile = "mpci.roiActivityDeconvolvedOasis"
    ospks = ses.loadone(onefile)
    keep_idx = ses.idxToPlanes(keep_planes=keep_planes)
    ospks = ospks[:, keep_idx]
    time_split_prms = dict(
        num_groups=2,
        chunks_per_group=-3,  # each chunk has 3 samples in it
        num_buffer=3,  # each chunk is buffered by 3 samples
    )
    npop = Population(ospks.T, generate_splits=True, time_split_prms=time_split_prms)
    pcss = analysis.placeCellSingleSession(ses, keep_planes=keep_planes, onefile=onefile, autoload=True)

    assert len(pcss.environments) == 1, "Only one environment is supported for this analysis"

    # Get source/target data for SVCA (we're not cross-validating the SVCA here so can use the full dataset across all timepoints)
    source, target = npop.get_split_data(None, center=False, scale=True, pre_split=False, scale_type="preserve")
    train_source, train_target = npop.get_split_data(0, center=False, scale=True, pre_split=False, scale_type="preserve")
    test_source, test_target = npop.get_split_data(1, center=False, scale=True, pre_split=False, scale_type="preserve")

    print("Running SVCA")

    # Fit an SVCA model to source / target (across all timepoints)
    svca = SVCA(centered=True).fit(source, target)

    print("Measuring place fields")

    # Get place fields from ROIs
    envnum = pcss.environments[0]
    train_spkmap = pcss.get_spkmap(envnum=envnum, average=True, trials="train")[0]  # only one environment here, but it's output as a list
    source_spkmap = train_spkmap[npop.cell_split_indices[0]]
    target_spkmap = train_spkmap[npop.cell_split_indices[1]]

    # Get test place fields from ROIs for a comparison across timepoints...
    test_spkmap = pcss.get_spkmap(envnum=envnum, average=True, trials="test")[0]  # only one environment here, but it's output as a list
    source_spkmap_test = test_spkmap[npop.cell_split_indices[0]]
    target_spkmap_test = test_spkmap[npop.cell_split_indices[1]]

    # Get full place fields
    full_spkmap = pcss.get_spkmap(envnum=envnum, average=True, trials="full")[0]
    source_spkmap_full = full_spkmap[npop.cell_split_indices[0]]
    target_spkmap_full = full_spkmap[npop.cell_split_indices[1]]

    # Check where the nans are and ignore those locations
    idx_nan = (
        np.any(np.isnan(source_spkmap), axis=0)
        | np.any(np.isnan(target_spkmap), axis=0)
        | np.any(np.isnan(source_spkmap_test), axis=0)
        | np.any(np.isnan(target_spkmap_test), axis=0)
        | np.any(np.isnan(source_spkmap_full), axis=0)
        | np.any(np.isnan(target_spkmap_full), axis=0)
    )

    # Run PCA on the place fields to get the principal components
    source_pca = PCA().fit(source_spkmap[:, ~idx_nan])
    target_pca = PCA().fit(target_spkmap[:, ~idx_nan])
    train_source_components = source_pca.get_components()
    train_target_components = target_pca.get_components()

    # Also on the test place fields
    source_pca_test = PCA().fit(source_spkmap_test[:, ~idx_nan])
    target_pca_test = PCA().fit(target_spkmap_test[:, ~idx_nan])
    test_source_components = source_pca_test.get_components()
    test_target_components = target_pca_test.get_components()

    # Measuring shared variance across SVCs or PF-PCs
    svca_standard = SVCA(centered=True).fit(train_source, train_target)
    svpf_source = torch.tensor(source_spkmap_full)[:, ~idx_nan].float()
    svpf_target = torch.tensor(target_spkmap_full)[:, ~idx_nan].float()
    num_components = min(svpf_source.shape[0], svpf_target.shape[0], (~idx_nan).sum())
    svca_placefields = SVCA(centered=True, num_components=num_components).fit(svpf_source, svpf_target)

    sv_scores_svcs = svca_standard.score(test_source, test_target)[0]
    sv_scores_pcpfs = svca_placefields.score(test_source, test_target)[0]

    print("Comparing place fields to SVCs")

    # Compare the PCA map of place fields on train / test trials -- this is a control comparison
    source_traintest_map = np.dot(train_source_components.T, test_source_components)
    target_traintest_map = np.dot(train_target_components.T, test_target_components)

    # For U, V, and components, each column is a component (so each row is a neuron)
    # Measure the correlation coefficient between the SVCs and the Place field PCs
    source_map = helpers.crossCorrelation(np.array(train_source_components), np.array(svca.U))
    target_map = helpers.crossCorrelation(np.array(train_target_components), np.array(svca.V))

    print("Making place fields out of SVCs")

    # Make place fields out of the activity projected onto the SVCs
    u_activity = fs.zscore(np.array(svca.U.T @ npop.data[npop.cell_split_indices[0]]), axis=1)
    v_activity = fs.zscore(np.array(svca.V.T @ npop.data[npop.cell_split_indices[1]]), axis=1)
    u_rawspkmap = helpers.getBehaviorAndSpikeMaps(ses, onefile=u_activity.T)[3]
    v_rawspkmap = helpers.getBehaviorAndSpikeMaps(ses, onefile=v_activity.T)[3]
    uspkmap_train = pcss.get_spkmap(envnum=envnum, average=True, trials="train", rawspkmap=u_rawspkmap)[0]
    vspkmap_train = pcss.get_spkmap(envnum=envnum, average=True, trials="train", rawspkmap=v_rawspkmap)[0]
    uspkmap_test = pcss.get_spkmap(envnum=envnum, average=True, trials="test", rawspkmap=u_rawspkmap)[0]
    vspkmap_test = pcss.get_spkmap(envnum=envnum, average=True, trials="test", rawspkmap=v_rawspkmap)[0]

    # Handle nuance of reliability values
    def select_env(tup, idx):
        return list(map(lambda x: x[idx], tup))

    print("measuring reliability and sorting place fields")

    # Measure the reliability of the place fields
    urelmse, urelcor = select_env(pcss.get_reliability_values(envnum=envnum, rawspkmap=u_rawspkmap), 0)
    vrelmse, vrelcor = select_env(pcss.get_reliability_values(envnum=envnum, rawspkmap=v_rawspkmap), 0)
    relmse, relcor = select_env(pcss.get_reliability_values(envnum=envnum), 0)

    # Percentile to show:
    prctile_cutoff = 95
    urelcor_prctile = np.percentile(urelcor[~np.isnan(urelcor)], prctile_cutoff)
    vrelcor_prctile = np.percentile(vrelcor[~np.isnan(vrelcor)], prctile_cutoff)
    relcor_prctile = np.percentile(relcor[~np.isnan(relcor)], prctile_cutoff)

    # cor_cutoff = 0.2
    # u_rel_idx = urelcor > cor_cutoff
    # v_rel_idx = vrelcor > cor_cutoff
    # rel_idx = relcor > cor_cutoff

    u_rel_idx = urelcor > urelcor_prctile
    v_rel_idx = vrelcor > vrelcor_prctile
    rel_idx = relcor > relcor_prctile

    # Sort by place field location on train trials
    uidx = pcss.get_place_field(uspkmap_train[u_rel_idx], method="max")[1]
    vidx = pcss.get_place_field(vspkmap_train[v_rel_idx], method="max")[1]
    tspkmapidx = pcss.get_place_field(train_spkmap[rel_idx], method="max")[1]

    # Predict position from SVCs (and compare to prediction from place fields)
    frame_position, frame_environment, _ = ses.get_frame_behavior(speedThreshold=1)
    valid_u_activity, valid_position, valid_environment = filter_timepoints(u_activity.T, frame_position, frame_environment)
    valid_roi_activity = filter_timepoints(ospks, frame_position, frame_environment)[0]
    position_basis = make_position_basis(valid_position, valid_environment, num_basis=10)

    upospop = Population(
        valid_u_activity.T,
        time_split_prms={"num_groups": 3, "relative_size": [5, 5, 1], "chunks_per_group": -3, "num_buffer": 3},
        dtype=torch.float32,
    )
    scale_params = dict(center=True, scale=True)
    train_valid_u = upospop.apply_split(valid_u_activity.T, 0, **scale_params)
    val_valid_u = upospop.apply_split(valid_u_activity.T, 1, **scale_params)
    test_valid_u = upospop.apply_split(valid_u_activity.T, 2, **scale_params)
    train_valid_ospks = upospop.apply_split(valid_roi_activity.T, 0, **scale_params)
    val_valid_ospks = upospop.apply_split(valid_roi_activity.T, 1, **scale_params)
    test_valid_ospks = upospop.apply_split(valid_roi_activity.T, 2, **scale_params)
    train_pos_basis = upospop.apply_split(position_basis.T, 0, **scale_params)
    val_pos_basis = upospop.apply_split(position_basis.T, 1, **scale_params)
    test_pos_basis = upospop.apply_split(position_basis.T, 2, **scale_params)

    # generate random data with same covariance structure as the u activity
    u_mean = np.mean(valid_u_activity, axis=0)
    u_covariance = np.cov(valid_u_activity.T)
    u_random = np.random.multivariate_normal(u_mean, u_covariance, valid_u_activity.shape[0])
    train_valid_u_random = upospop.apply_split(u_random.T, 0, **scale_params)
    val_valid_u_random = upospop.apply_split(u_random.T, 1, **scale_params)
    test_valid_u_random = upospop.apply_split(u_random.T, 2, **scale_params)

    num_u_features = train_valid_u.shape[0]
    keep_roi_features = torch.randperm(train_valid_ospks.shape[0])[:num_u_features]
    train_valid_ospks = train_valid_ospks[keep_roi_features]
    val_valid_ospks = val_valid_ospks[keep_roi_features]
    test_valid_ospks = test_valid_ospks[keep_roi_features]

    def test_alphas(train_source, train_target, val_source, val_target, name, alphas=torch.logspace(1, 9, 9)):
        models = []
        scores = []
        for alpha in tqdm(alphas, desc=f"Testing alphas for {name}", leave=True):
            model = RidgeRegression(alpha=alpha, fit_intercept=True).fit(train_source.T, train_target.T)
            models.append(model)
            scores.append(model.score(val_source.T, val_target.T))
        best_model = models[np.argmax(scores)]
        best_score = np.max(scores)
        best_alpha = alphas[np.argmax(scores)]
        return best_model, best_score, best_alpha

    best_model_svc = test_alphas(train_valid_u, train_pos_basis, val_valid_u, val_pos_basis, "SVC")[0]
    best_model_roi = test_alphas(train_valid_ospks, train_pos_basis, val_valid_ospks, val_pos_basis, "ROI")[0]
    best_model_random = test_alphas(train_valid_u_random, train_pos_basis, val_valid_u_random, val_pos_basis, "RANDOM")[0]
    svc_model_score = best_model_svc.score(test_valid_u.T, test_pos_basis.T)
    roi_model_score = best_model_roi.score(test_valid_ospks.T, test_pos_basis.T)
    random_model_score = best_model_random.score(test_valid_u_random.T, test_pos_basis.T)
    print(f"Ridge regression score for SVCs: {svc_model_score}")
    print(f"Ridge regression score for ROIs: {roi_model_score}")
    print(f"Ridge regression score for random data: {random_model_score}")
    print(f"Ridge regression SVC model best alpha: {best_model_svc.alpha}")
    print(f"Ridge regression ROI model best alpha: {best_model_roi.alpha}")
    print(f"Ridge regression random model best alpha: {best_model_random.alpha}")

    results = dict(
        mouse_name=ses.mouseName,
        dateString=ses.dateString,
        sessionid=ses.sessionid,
        svc_model_score=svc_model_score,
        roi_model_score=roi_model_score,
        random_model_score=random_model_score,
        urelmse=urelmse,
        urelcor=urelcor,
        vrelmse=vrelmse,
        vrelcor=vrelcor,
        relmse=relmse,
        relcor=relcor,
        sv_scores_svcs=sv_scores_svcs,
        sv_scores_pcpfs=sv_scores_pcpfs,
    )
    # pcss.save_temp_file(results, svca_placefield_tempfile(ses))

    vmin = -1
    vmax = 1
    fontsize = 12

    # Plot of Place Fields
    plt.rcParams.update({"font.size": fontsize})

    u_rand_idx = np.random.choice(np.sum(u_rel_idx), 5, replace=False)
    r_rand_idx = np.random.choice(np.sum(rel_idx), 5, replace=False)

    num_plot = 5
    cmap = mpl.cm.get_cmap("viridis").resampled(num_plot)

    fig, ax = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    for ii in range(num_plot):
        ax[0, 0].plot(uspkmap_train[u_rel_idx][uidx][u_rand_idx[ii]], color=cmap(ii))
        ax[0, 1].plot(train_spkmap[rel_idx][tspkmapidx][r_rand_idx[ii]], color=cmap(ii))
        ax[1, 0].plot(uspkmap_test[u_rel_idx][uidx][u_rand_idx[ii]], color=cmap(ii))
        ax[1, 1].plot(test_spkmap[rel_idx][tspkmapidx][r_rand_idx[ii]], color=cmap(ii))
    ax[0, 0].set_title("SVCs")
    ax[0, 1].set_title("ROIs")
    ax[0, 0].set_ylabel("Train Trials")
    ax[1, 0].set_ylabel("Test Trials")
    ax[1, 0].set_xlabel("Position (cm)")
    ax[1, 1].set_xlabel("Position (cm)")

    # ax[0, 0].imshow(uspkmap_train[u_rel_idx][uidx], aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)  # , interpolation="none")
    # ax[0, 1].imshow(vspkmap_train[v_rel_idx][vidx], aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)  # , interpolation="none")
    # ax[0, 2].imshow(train_spkmap[rel_idx][tspkmapidx], aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)  # , interpolation="none")
    # ax[1, 0].imshow(uspkmap_test[u_rel_idx][uidx], aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)  # , interpolation="none")
    # ax[1, 1].imshow(vspkmap_test[v_rel_idx][vidx], aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)  # , interpolation="none")
    # ax[1, 2].imshow(test_spkmap[rel_idx][tspkmapidx], aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)  # , interpolation="none")
    # ax[1, 0].set_xlabel("Position (cm)")  # , fontsize=fontsize)
    # ax[1, 1].set_xlabel("Position (cm)")  # , fontsize=fontsize)
    # ax[1, 2].set_xlabel("Position (cm)")  # , fontsize=fontsize)
    # ax[0, 0].set_ylabel("Train Trials")  # , fontsize=fontsize)
    # ax[1, 0].set_ylabel("Test Trials")  # , fontsize=fontsize)
    # ax[0, 0].set_title("U (SVCs - Source)")  # , fontsize=fontsize)
    # ax[0, 1].set_title("V (SVCs - Target)")  # , fontsize=fontsize)
    # ax[0, 2].set_title("ROIs")  # , fontsize=fontsize)
    # pcss.saveFigure(fig.number, FIGURE_FOLDER + "PF_TrainTest")
    # plt.show()

    fig, ax = plt.subplots(2, 3, figsize=(9, 6), layout="constrained")
    ax[0, 0].imshow(uspkmap_train[u_rel_idx][uidx], aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)  # , interpolation="none")
    ax[0, 1].imshow(vspkmap_train[v_rel_idx][vidx], aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)  # , interpolation="none")
    ax[0, 2].imshow(train_spkmap[rel_idx][tspkmapidx], aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)  # , interpolation="none")
    ax[1, 0].imshow(uspkmap_test[u_rel_idx][uidx], aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)  # , interpolation="none")
    ax[1, 1].imshow(vspkmap_test[v_rel_idx][vidx], aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)  # , interpolation="none")
    ax[1, 2].imshow(test_spkmap[rel_idx][tspkmapidx], aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)  # , interpolation="none")
    ax[1, 0].set_xlabel("Position (cm)")  # , fontsize=fontsize)
    ax[1, 1].set_xlabel("Position (cm)")  # , fontsize=fontsize)
    ax[1, 2].set_xlabel("Position (cm)")  # , fontsize=fontsize)
    ax[0, 0].set_ylabel("Train Trials")  # , fontsize=fontsize)
    ax[1, 0].set_ylabel("Test Trials")  # , fontsize=fontsize)
    ax[0, 0].set_title("U (SVCs - Source)")  # , fontsize=fontsize)
    ax[0, 1].set_title("V (SVCs - Target)")  # , fontsize=fontsize)
    ax[0, 2].set_title("ROIs")  # , fontsize=fontsize)
    # pcss.saveFigure(fig.number, FIGURE_FOLDER + "PF_TrainTest")
    # plt.show()

    # Plot of Reliability Metrics
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")
    ax[0].ecdf(urelmse[~np.isnan(urelmse)], label="U")
    ax[0].ecdf(vrelmse[~np.isnan(vrelmse)], label="V")
    ax[0].ecdf(relmse[~np.isnan(relmse)], label="PCA")
    ax[1].ecdf(urelcor[~np.isnan(urelcor)], label="U")
    ax[1].ecdf(vrelcor[~np.isnan(vrelcor)], label="V")
    ax[1].ecdf(relcor[~np.isnan(relcor)], label="PCA")
    ax[0].set_xlabel("Reliability (MSE)")
    ax[1].set_xlabel("Reliability (COR)")
    ax[0].set_ylabel("Cumulative probability")
    ax[1].set_ylabel("Cumulative probability")
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlim(-3, 1)
    pcss.saveFigure(fig.number, FIGURE_FOLDER + "ReliabilityValues")
    # plt.show()

    # Plot of PCA Map
    max_components = train_source_components.shape[1] - 1

    # Create figure and GridSpec
    fig = plt.figure(figsize=(13, 4))
    gs = mpl.gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.1])

    # Create subplots
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0, sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharey=ax0, sharex=ax0)
    cax = fig.add_subplot(gs[3])  # subplot for colorbar

    # Create the three subplots
    im0 = ax0.imshow(np.abs(source_map[:, :max_components]), aspect="auto", interpolation="none", cmap="plasma", vmin=0, vmax=1)
    ax1.imshow(np.abs(target_map[:, :max_components]), aspect="auto", interpolation="none", cmap="plasma", vmin=0, vmax=1)
    ax2.imshow(np.abs(source_traintest_map[:, :max_components]), aspect="auto", interpolation="none", cmap="plasma", vmin=0, vmax=1)

    ax0.set_title("PC / SVC - Cell Set 1")
    ax1.set_title("PC / SVC - Cell Set 2")
    ax2.set_title("Train / Test ROIs")
    ax0.set_xlabel("SVC")
    ax1.set_xlabel("SVC")
    ax2.set_xlabel("Test PC")
    ax0.set_ylabel("PC")
    ax1.set_ylabel("PC")
    ax2.set_ylabel("Train PC")

    ax1.set_xlim(-0.5, 25.5)
    ax1.set_ylim(25.5, -0.5)

    # Add colorbar
    plt.colorbar(im0, cax=cax, label="Intensity")

    # Adjust spacing between subplots
    plt.tight_layout()

    pcss.saveFigure(fig.number, FIGURE_FOLDER + "PC_SVC_Correlation")

    # plt.show()

    plt.close("all")


def add_results_per_session(ses):
    print(f"Processing session {ses.sessionPrint()}...")
    keep_planes = [1, 2, 3, 4]
    onefile = "mpci.roiActivityDeconvolvedOasis"
    ospks = ses.loadone(onefile)
    keep_idx = ses.idxToPlanes(keep_planes=keep_planes)
    ospks = ospks[:, keep_idx]
    time_split_prms = dict(
        num_groups=2,
        chunks_per_group=-3,  # each chunk has 3 samples in it
        num_buffer=3,  # each chunk is buffered by 3 samples
    )
    npop = Population(ospks.T, generate_splits=True, time_split_prms=time_split_prms)
    pcss = analysis.placeCellSingleSession(ses, keep_planes=keep_planes, onefile=onefile, autoload=True)

    assert len(pcss.environments) == 1, "Only one environment is supported for this analysis"

    # Get source/target data for SVCA (we're not cross-validating the SVCA here so can use the full dataset across all timepoints)
    # source, target = npop.get_split_data(None, center=False, scale=True, pre_split=False, scale_type="preserve")
    train_source, train_target = npop.get_split_data(0, center=False, scale=True, pre_split=False, scale_type="preserve")
    test_source, test_target = npop.get_split_data(1, center=False, scale=True, pre_split=False, scale_type="preserve")

    # Get place fields from ROIs
    envnum = pcss.environments[0]
    train_spkmap = pcss.get_spkmap(envnum=envnum, average=True, trials="train")[0]  # only one environment here, but it's output as a list
    source_spkmap = train_spkmap[npop.cell_split_indices[0]]
    target_spkmap = train_spkmap[npop.cell_split_indices[1]]

    source_relcor = pcss.get_reliability_values(envnum=envnum)[1][0][npop.cell_split_indices[0]]
    target_relcor = pcss.get_reliability_values(envnum=envnum)[1][0][npop.cell_split_indices[1]]

    # Get test place fields from ROIs for a comparison across timepoints...
    test_spkmap = pcss.get_spkmap(envnum=envnum, average=True, trials="test")[0]  # only one environment here, but it's output as a list
    source_spkmap_test = test_spkmap[npop.cell_split_indices[0]]
    target_spkmap_test = test_spkmap[npop.cell_split_indices[1]]

    # Get full place fields
    full_spkmap = pcss.get_spkmap(envnum=envnum, average=True, trials="full")[0]
    source_spkmap_full = full_spkmap[npop.cell_split_indices[0]]
    target_spkmap_full = full_spkmap[npop.cell_split_indices[1]]

    # Check where the nans are and ignore those locations
    idx_nan = (
        np.any(np.isnan(source_spkmap), axis=0)
        | np.any(np.isnan(target_spkmap), axis=0)
        | np.any(np.isnan(source_spkmap_test), axis=0)
        | np.any(np.isnan(target_spkmap_test), axis=0)
        | np.any(np.isnan(source_spkmap_full), axis=0)
        | np.any(np.isnan(target_spkmap_full), axis=0)
    )

    # Measuring shared variance across SVCs or PF-PCs
    svca_standard = SVCA(centered=True).fit(train_source, train_target)
    svpf_source = torch.tensor(source_spkmap_full)[:, ~idx_nan].float()
    svpf_target = torch.tensor(target_spkmap_full)[:, ~idx_nan].float()
    num_components = min(svpf_source.shape[0], svpf_target.shape[0], (~idx_nan).sum())
    svca_placefields = SVCA(centered=True, num_components=num_components).fit(svpf_source, svpf_target)

    # Project out the place field components
    u_pf_proj = svca_placefields.U @ svca_placefields.U.T @ test_source
    v_pf_proj = svca_placefields.V @ svca_placefields.V.T @ test_target
    test_source_nopf = test_source - u_pf_proj
    test_target_nopf = test_target - v_pf_proj

    sv_scores_svcs = svca_standard.score(test_source, test_target)[0]
    sv_scores_pcpfs = svca_placefields.score(test_source, test_target)[0]
    sv_scores_svcs_nopf = svca_standard.score(test_source_nopf, test_target_nopf)[0]

    usvc_pf_proj = svca_standard.U.T @ svca_placefields.U
    vsvc_pf_proj = svca_standard.V.T @ svca_placefields.V
    mindim0 = min(usvc_pf_proj.shape[0], vsvc_pf_proj.shape[0])
    mindim1 = min(usvc_pf_proj.shape[1], vsvc_pf_proj.shape[1])
    avg_svc_pfpc_map = torch.mean(
        torch.stack((torch.abs(usvc_pf_proj[:mindim0][:, :mindim1]), torch.abs(vsvc_pf_proj[:mindim0][:, :mindim1])), dim=0), dim=0
    )

    u_explained = torch.sum(usvc_pf_proj**2, dim=1)
    v_explained = torch.sum(vsvc_pf_proj**2, dim=1)
    svc_fraction_explained = (u_explained + v_explained) / 2

    results = pcss.load_temp_file(svca_placefield_tempfile(ses))
    results["sv_scores_svcs"] = sv_scores_svcs
    results["sv_scores_svcs_nopf"] = sv_scores_svcs_nopf
    results["sv_scores_pcpfs"] = sv_scores_pcpfs
    results["svc_fraction_explained"] = svc_fraction_explained
    results["svc_pfpc_map"] = avg_svc_pfpc_map
    pcss.save_temp_file(results, svca_placefield_tempfile(ses))


def load_data():
    session_data = []
    for ses in get_sessions():
        pcss = analysis.placeCellSingleSession(ses, autoload=False)
        session_data.append(pcss.load_temp_file(svca_placefield_tempfile(ses)))

    mouse_names = np.array([res["mouse_name"] for res in session_data])
    dates = np.array([res["dateString"] for res in session_data])
    sessionids = np.array([res["sessionid"] for res in session_data])
    mice = sorted(list(set(mouse_names)))

    # For each mouse, get the index corresponding to which session that was for each mouse (to analyze familiarity)
    idx_session = np.zeros(len(session_data), dtype=int)
    for mouse in mice:
        idx_to_mouse = np.where(mouse_names == mouse)[0]
        c_datestrings = dates[idx_to_mouse]
        idx_date_order = np.argsort(c_datestrings)
        idx_session[idx_to_mouse] = idx_date_order

    # Compile each result into a single dictionary
    results = dict(
        mouse_names=mouse_names,
        dates=dates,
        sessionids=sessionids,
        idx_session=idx_session,
        svc_model_scores=np.array([res["svc_model_score"] for res in session_data]),
        roi_model_scores=np.array([res["roi_model_score"] for res in session_data]),
        random_model_score=np.array([res["random_model_score"] for res in session_data]),
        urelmse=[res["urelmse"] for res in session_data],
        urelcor=[res["urelcor"] for res in session_data],
        vrelmse=[res["vrelmse"] for res in session_data],
        vrelcor=[res["vrelcor"] for res in session_data],
        relmse=[res["relmse"] for res in session_data],
        relcor=[res["relcor"] for res in session_data],
        sv_scores_svcs=[res["sv_scores_svcs"] for res in session_data],
        sv_scores_pcpfs=[res["sv_scores_pcpfs"] for res in session_data],
        sv_scores_svcs_nopf=[res["sv_scores_svcs_nopf"] for res in session_data],
        sv_fraction_explained=[res["svc_fraction_explained"] for res in session_data],
        sv_pfpc_map=[res["svc_pfpc_map"] for res in session_data],
    )
    return results


def compare_figures(results):
    # Plot the SVC and ROI model scores
    plt.rcParams.update({"font.size": 24})

    mice = sorted(list(set(results["mouse_names"])))
    num_mice = len(mice)
    num_sessions = len(results["mouse_names"])
    cmap = mpl.colormaps["turbo"].resampled(num_mice)

    # Get color for each mouse
    icolors = [mice.index(mouse) for mouse in results["mouse_names"]]
    colors = [cmap(icolor) for icolor in icolors]
    all_scores = np.stack((results["roi_model_scores"], results["svc_model_scores"], results["random_model_score"])).T

    idx = [0, 1, 2]
    labels = ["ROIs", "SVCs", "Random"]
    fig, ax = plt.subplots(figsize=(5, 3))
    for ii, ascore in enumerate(all_scores):
        plt.plot(idx, ascore, c=colors[ii], marker="o", linestyle="-")
    plt.xticks(idx, labels)
    plt.ylabel("Model Score")
    # cb = plt.colorbar(scatter, ax=ax, label="Mouse", norm=norm, boundaries=bounds, ticks=np.arange(num_mice))
    # cb.set_ticklabels(mice)
    plt.tight_layout()
    plt.show()

    mse_edges = np.linspace(-3, 1, 31)
    cor_edges = np.linspace(-1, 1, 31)
    mse_centers = helpers.edge2center(mse_edges)
    cor_centers = helpers.edge2center(cor_edges)

    # Measure fractional counts for each session of all the reliability values
    urelmse_counts = np.zeros((num_sessions, len(mse_centers)))
    urelcor_counts = np.zeros((num_sessions, len(cor_centers)))
    vrelmse_counts = np.zeros((num_sessions, len(mse_centers)))
    vrelcor_counts = np.zeros((num_sessions, len(cor_centers)))
    relmse_counts = np.zeros((num_sessions, len(mse_centers)))
    relcor_counts = np.zeros((num_sessions, len(cor_centers)))
    for i in range(num_sessions):
        urelmse_counts[i], _ = helpers.fractional_histogram(results["urelmse"][i], bins=mse_edges)
        urelcor_counts[i], _ = helpers.fractional_histogram(results["urelcor"][i], bins=cor_edges)
        vrelmse_counts[i], _ = helpers.fractional_histogram(results["vrelmse"][i], bins=mse_edges)
        vrelcor_counts[i], _ = helpers.fractional_histogram(results["vrelcor"][i], bins=cor_edges)
        relmse_counts[i], _ = helpers.fractional_histogram(results["relmse"][i], bins=mse_edges)
        relcor_counts[i], _ = helpers.fractional_histogram(results["relcor"][i], bins=cor_edges)

    cols = "krb"
    labels = ["SVC-U", "SVC-V", "ROIs"]
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")
    for i, (counts, label) in enumerate(zip([urelmse_counts, vrelmse_counts, relmse_counts], labels)):
        ax[0].plot(mse_centers, np.nanmean(counts, axis=0), c=cols[i], label=label)
        ax[0].fill_between(
            mse_centers,
            np.nanmean(counts, axis=0) + np.nanstd(counts, axis=0) / np.sqrt(np.sum(~np.isnan(counts), axis=0)),
            np.nanmean(counts, axis=0) - np.nanstd(counts, axis=0) / np.sqrt(np.sum(~np.isnan(counts), axis=0)),
            color=cols[i],
            alpha=0.3,
        )
    for i, (counts, label) in enumerate(zip([urelcor_counts, vrelcor_counts, relcor_counts], labels)):
        ax[1].plot(cor_centers, np.nanmean(counts, axis=0), c=cols[i], label=label)
        ax[1].fill_between(
            cor_centers,
            np.nanmean(counts, axis=0) + np.nanstd(counts, axis=0) / np.sqrt(np.sum(~np.isnan(counts), axis=0)),
            np.nanmean(counts, axis=0) - np.nanstd(counts, axis=0) / np.sqrt(np.sum(~np.isnan(counts), axis=0)),
            color=cols[i],
            alpha=0.3,
        )
    ax[0].set_xlabel("Reliability (MSE)")
    ax[1].set_xlabel("Reliability (COR)")
    ax[0].set_ylabel("Fractional count")
    ax[1].set_ylabel("Fractional count")
    ax[0].legend()
    ax[1].legend()
    plt.show()

    # Analyze SV from SVCs vs PC-PFs comparison
    total_sv_svcs = np.zeros(num_sessions)
    total_sv_pcpfs = np.zeros(num_sessions)
    total_sv_svcs_nopf = np.zeros(num_sessions)
    total10_svcs = np.zeros(num_sessions)
    total10_pcpfs = np.zeros(num_sessions)
    total10_svcs_nopf = np.zeros(num_sessions)
    for ires, (sv_svcs, sv_pcpfs, sv_svcs_nopf) in enumerate(
        zip(results["sv_scores_svcs"], results["sv_scores_pcpfs"], results["sv_scores_svcs_nopf"])
    ):
        total_sv_svcs[ires] = sv_svcs[: len(sv_pcpfs)].sum()
        total_sv_pcpfs[ires] = sv_pcpfs.sum()
        total_sv_svcs_nopf[ires] = sv_svcs_nopf[: len(sv_pcpfs)].sum()
        total10_svcs[ires] = sv_svcs[:10].sum()
        total10_pcpfs[ires] = sv_pcpfs[:10].sum()
        total10_svcs_nopf[ires] = sv_svcs_nopf[:10].sum()

    min_dim = min(min([len(sv_svcs) for sv_svcs in results["sv_scores_svcs"]], [len(sv_pcpfs) for sv_pcpfs in results["sv_scores_pcpfs"]]))
    all_sv_svcs = np.stack([sv_svcs[:min_dim] for sv_svcs in results["sv_scores_svcs"]])
    all_sv_svcs_nopf = np.stack([sv_svcs[:min_dim] for sv_svcs in results["sv_scores_svcs_nopf"]])
    all_sv_pcpfs = np.stack([sv_pcpfs[:min_dim] for sv_pcpfs in results["sv_scores_pcpfs"]])

    rel_sv_svcs = all_sv_svcs / total_sv_svcs[:, None]
    rel_sv_svcs_nopf = all_sv_svcs_nopf / total_sv_svcs[:, None]
    rel_sv_pcpfs = all_sv_pcpfs / total_sv_svcs[:, None]

    plt.rcParams.update({"font.size": 20})

    fig, ax = plt.subplots(1, 2, figsize=(8, 7), width_ratios=[1, 0.5], layout="constrained")
    helpers.errorPlot(range(1, min_dim + 1), rel_sv_svcs, axis=0, se=True, ax=ax[0], color="k", label="SVCs", alpha=0.3)
    helpers.errorPlot(range(1, min_dim + 1), rel_sv_svcs_nopf, axis=0, se=True, ax=ax[0], color="r", label="SVCs (-PFs)", alpha=0.3)
    helpers.errorPlot(range(1, min_dim + 1), rel_sv_pcpfs, axis=0, se=True, ax=ax[0], color="b", label="PC-PFs", alpha=0.3)
    ax[0].set_xlabel("Dimension")
    ax[0].set_ylabel("Normalized Variance")
    ax[0].set_xscale("log")
    ax[0].set_ylim(0)
    ax[0].legend()

    total_variance = np.stack((total_sv_svcs, total_sv_svcs_nopf, total_sv_pcpfs))
    relative_variance = total_variance / total_variance[0]
    ax[1].plot([0, 1, 2], relative_variance, linewidth=1, c="k", marker=".", markersize=12)
    ax[1].scatter(0 * np.ones(relative_variance.shape[1]), relative_variance[0], c="k", s=10, label="SVCs", zorder=1000)
    ax[1].scatter(1 * np.ones(relative_variance.shape[1]), relative_variance[1], c="r", s=10, label="SVCs (-PFs)", zorder=1000)
    ax[1].scatter(2 * np.ones(relative_variance.shape[1]), relative_variance[2], c="b", s=10, label="PC-PFs", zorder=1000)
    ax[1].set_xticks([0, 1, 2], labels=["SVCs", "SVCs (-PFs)", "PC-PFs"], rotation=45, ha="right")
    ax[1].set_ylabel("Relative Total Variance")
    ax[1].set_xlim(-0.3, 2.3)
    ax[1].set_ylim(0)
    plt.show()

    with_poster2024_save = False
    if with_poster2024_save:
        save_directory = figure_folder()
        save_name = "SVC_StandardAndPFPCs"
        save_path = save_directory / save_name
        helpers.save_figure(fig, save_path)

    # Also show fractional variance of each component
    mice = sorted(np.unique(results["mouse_names"]))
    mice_names = helpers.short_mouse_names(mice)
    num_mice = len(mice)
    cmap = mpl.cm.get_cmap("turbo").resampled(num_mice)
    icolors = [mice.index(mouse) for mouse in results["mouse_names"]]
    colors = [cmap(icolor) for icolor in icolors]

    min_svc_dim = min([len(sv_svcs) for sv_svcs in results["sv_fraction_explained"]])
    all_sv_fraction_explained = np.stack([sv_fe[:min_svc_dim] for sv_fe in results["sv_fraction_explained"]])

    all_by_mouse = np.zeros((num_mice, min_svc_dim))
    dev_by_mouse = np.zeros((num_mice, min_svc_dim))
    for i, mouse in enumerate(mice):
        idx_mouse = np.where(results["mouse_names"] == mouse)[0]
        all_by_mouse[i] = np.nanmean(all_sv_fraction_explained[idx_mouse], axis=0)
        dev_by_mouse[i] = np.nanstd(all_sv_fraction_explained[idx_mouse], axis=0) / np.sqrt(
            np.sum(~np.isnan(all_sv_fraction_explained[idx_mouse]), axis=0)
        )

    plt.rcParams.update({"font.size": 24})

    figdim = 5.3
    fig, ax = plt.subplots(1, 1, figsize=(figdim, figdim), layout="constrained")
    # helpers.errorPlot(range(1, min_svc_dim + 1), all_sv_fraction_explained, axis=0, se=True, ax=ax, color="k", alpha=0.3)
    for i, mouse in enumerate(mice_names):
        label = "Each Mouse" if i == 0 else None
        ax.plot(range(1, min_svc_dim + 1), all_by_mouse[i], color="k", label=label)
    ax.set_xlabel("SVC-Time Dimension")
    ax.set_ylabel(f"fraction variance shared\nwith SVC-POS dimensions")
    ax.set_xscale("log")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1)
    # plt.show()

    with_poster2024_save = True
    if with_poster2024_save:
        save_directory = figure_folder()
        save_name = "SVC_FractionExplained_PFPCs"
        save_path = save_directory / save_name
        helpers.save_figure(fig, save_path)

    # Show how the SVCs and PFPCs relate to each other
    min_svc_dim = min([sv_svcs.shape[0] for sv_svcs in results["sv_pfpc_map"]])
    min_pfpc_dim = min([sv_svcs.shape[1] for sv_svcs in results["sv_pfpc_map"]])
    all_sv_pfpc_map = np.stack([sv_pm[:min_svc_dim][:, :min_pfpc_dim] for sv_pm in results["sv_pfpc_map"]])
    show_dims = 50
    show_map = np.mean(all_sv_pfpc_map, axis=0)[:show_dims][:, :show_dims]
    iexample = 9  # most correlated with average
    example_map = all_sv_pfpc_map[iexample][:show_dims][:, :show_dims]
    vmax = np.ceil(10 * np.max(np.abs(example_map))) / 10
    # vmax = 1.0

    plt.rcParams.update({"font.size": 24})

    extent = [0, show_dims, show_dims, 0]
    # vmax = np.round(20 * np.max(np.abs(show_map)) * 1.5) / 20
    figdim = 5.3
    fig, ax = plt.subplots(1, 1, figsize=(figdim, figdim), layout="constrained")
    im = ax.imshow(show_map.T, extent=extent, interpolation="none", cmap="gray_r", vmin=0, vmax=vmax, aspect="equal")
    ax.set_xlim(0, show_dims)
    ax.set_ylim(show_dims, 0)
    ax.set_xlabel("SVC-Time  Dimension", labelpad=-15)
    ax.set_ylabel("SVC-Pos  Dimension", labelpad=-15)
    ax.set_xticks([0, show_dims])
    ax.set_yticks([0, show_dims])
    ticks = np.round(1000 * np.linspace(0, vmax, 3)) / 1000
    inset_position = [0.88, 0.1, 0.075, 0.65]
    inset_colorbar = ax.inset_axes(inset_position)
    plt.colorbar(im, cax=inset_colorbar, ticks=[])
    ax.text(inset_position[0] + inset_position[2] / 2, inset_position[1] - 0.01, ticks[0], ha="center", va="top", transform=ax.transAxes)
    ax.text(
        inset_position[0] + inset_position[2] / 2,
        inset_position[1] + inset_position[3] + 0.01,
        ticks[-1],
        ha="center",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.text(
        inset_position[0] - 0.01,
        inset_position[1] + inset_position[3] / 2,
        "|dot(SVC-T, SVC-P)|",
        rotation=90,
        ha="right",
        va="center",
        color="k",
        transform=ax.transAxes,
        zorder=1000,
    )
    # plt.tight_layout()

    inset = ax.inset_axes([0.05, 0.05, 0.3, 0.3])
    inset.imshow(example_map.T, extent=extent, interpolation="none", cmap="gray_r", vmin=0, vmax=vmax, aspect="equal")
    inset.set_xlim(-0.5, show_dims)
    inset.set_ylim(show_dims, -0.5)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.text(show_dims / 2, -2, "Example\nSession", ha="center", va="bottom", color="k", fontsize=24)

    # plt.show()

    with_poster2024_save = False
    if with_poster2024_save:
        save_directory = figure_folder()
        save_name = "SVC_PFPC_Map"
        save_path = save_directory / save_name
        helpers.save_figure(fig, save_path)

    plt.rcParams.update({"font.size": 24})

    cosyne = True
    if cosyne:
        fig_height = 6.5
        fig_width = 4.4
        show_10 = False
    else:
        fig_height = 5.3
        fig_width = 4.4
        show_10 = True

    # Make focused plot on relative total variance
    if show_10:
        total_variance = np.stack((total_sv_svcs, total_sv_pcpfs, total10_svcs, total10_pcpfs))
    else:
        total_variance = np.stack((total_sv_svcs, total_sv_pcpfs))
    relative_variance = total_variance / total_variance[0]
    xd = [0, 1]
    labels = ["Time", "Pos"]
    cols = ["k", "#9F9FFF"]
    patch = [False, False]
    if show_10:
        xd = xd + [2, 3]
        labels = labels + labels
        cols = cols + cols
        patch = patch + [True, True]

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), layout="constrained")
    ax.plot(xd, relative_variance, linewidth=1, c="k", marker="o", markersize=10)
    for ii, (xx, rv) in enumerate(zip(xd, relative_variance)):
        ax.plot(
            xx * np.ones(len(rv)),
            rv,
            color=(cols[ii], 0.6),
            linewidth=2,
            marker="o",
            linestyle="none",
            markersize=8,
            zorder=1000,
        )
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax)
    # Make a gray patch around the x value spanning the full ylim where "True" is set
    for x, p in zip(xd, patch):
        if p:
            ax.fill_between([x - 0.5, x + 0.5], 0, ymax, color="gray", edgecolor="none", alpha=0.2)
    if show_10:
        ax.text(2.5, ymax * 0.95, "1st 10\nDims\nOnly", ha="center", va="top")
    ax.set_xticks(xd, labels=labels, rotation=0, ha="center")
    ax.set_xlabel("SVCs")
    ax.set_ylabel("Relative Total Var.")
    ax.set_xlim(-0.3, 1.3 + 2 * show_10)
    ax.set_ylim(0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    plt.show()

    with_poster2024_save = False
    if with_poster2024_save:
        save_directory = figure_folder()
        save_name = "RelativeTotalVariance"
        save_name = save_name + ("_cosyne" if cosyne else "")
        save_path = save_directory / save_name
        helpers.save_figure(fig, save_path)

    # Make focused plot on relative total variance
    total_variance = np.stack((total_sv_svcs, total_sv_pcpfs))
    total_variance10 = np.stack((total10_svcs, total10_pcpfs))
    relative_variance = total_variance / total_variance[0]
    relative_variance10 = total_variance10 / total_variance10[0]
    xd = [[0, 1], [2, 3]]
    labels = ["SVC-Time", "SVC-Pos"]
    cols = ["k", "#9F9FFF"]

    fig, ax = plt.subplots(1, 1, figsize=(4.4, 5.3), layout="constrained")
    ax.plot(xd[0], relative_variance, linewidth=1, c="k", marker="o", markersize=10)
    ax.plot(xd[1], relative_variance10, linewidth=1, c="k", marker="o", markersize=10)
    for ii, (xx, rv) in enumerate(zip(xd[0], relative_variance)):
        ax.plot(
            xx * np.ones(len(rv)),
            rv,
            color=(cols[ii], 0.6),
            linewidth=2,
            marker="o",
            linestyle="none",
            markersize=8,
            zorder=1000,
        )
    for ii, (xx, rv) in enumerate(zip(xd[1], relative_variance10)):
        ax.plot(
            xx * np.ones(len(rv)),
            rv,
            color=(cols[ii], 0.6),
            linewidth=2,
            marker="o",
            linestyle="none",
            markersize=8,
            zorder=1000,
        )
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax)
    # Make a gray patch around the x value spanning the full ylim where "True" is set
    for x in xd[1]:
        ax.fill_between([x - 0.5, x + 0.5], 0, ymax, color="gray", edgecolor="none", alpha=0.2)
    ax.text(1.75, ymax * 0.05, "Only\n1st 10\nDims", ha="left", va="bottom")
    ax.set_xticks(xd[0] + xd[1], labels=labels + labels, rotation=45, ha="right")
    ax.set_ylabel("Relative Total Variance")
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(0)
    plt.show()

    with_poster2024_save = True
    if with_poster2024_save:
        save_directory = figure_folder()
        save_name = "RelativeTotalVarianceSplit10"
        save_path = save_directory / save_name
        helpers.save_figure(fig, save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_sessions", action="store_true", default=False, help="Process the sessions and save temp data / figures")
    parser.add_argument("--add_results_per_session", action="store_true", default=False, help="Add results per session")
    parser.add_argument("--compare_mice", action="store_true", default=False, help="Compare the results across mice")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.process_sessions:
        for ses in tqdm(get_sessions()):
            process_session(ses)

    if args.add_results_per_session:
        for ses in tqdm(get_sessions()):
            add_results_per_session(ses)

    if args.compare_mice:
        results = load_data()
        compare_figures(results)
