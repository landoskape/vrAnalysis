import os
from copy import copy

from vrAnalysis import analysis
from vrAnalysis import tracking
from vrAnalysis import helpers
from vrAnalysis import database
from tqdm import tqdm
import numpy as np
import faststats as fs
import pickle

from dimilibi import Population, SVCA

mousedb = database.vrDatabase("vrMice")
df = mousedb.getTable(trackerExists=True)
mouse_names = df["mouseName"].unique()

cvMethod = helpers.cvPCA_paper_neurons


def add_to_spectra_data(pcm, args):
    """skeleton for adding something without reloading everything"""
    with open(pcm.saveDirectory("temp") / f"{args.mouse_name}_spectra_data.pkl", "rb") as f:
        temp_files = pickle.load(f)

    vss = []
    for p in pcm.pcss:
        vss.append(analysis.VarianceStructure(p.vrexp, distStep=args.dist_step, autoload=False))

    # first load session data (this can take a while)
    for v in tqdm(vss, leave=True, desc="loading session data"):
        v.load_data()

    cv_by_env_trial = []
    cv_by_env_trial_rdm = []
    cv_by_env_trial_cvrdm = []
    svc_shared_position = []
    for v in tqdm(vss, leave=False, desc="doing special cvPCA and SVCA analyses on trial position data"):
        c_spkmaps_full = v.get_spkmap(average=False, smooth=args.smooth, trials="full")
        c_spkmaps_full_avg = [np.nanmean(c, axis=1) for c in c_spkmaps_full]

        # Do cvPCA comparison between full spkmap with poisson noise across fake trials and on the actual trials
        c_spkmaps_full = [c[:, np.random.permutation(c.shape[1])] for c in c_spkmaps_full]  # randomize trials for easy splitting
        c_spkmaps_full_train = [c[:, : int(c.shape[1] / 2)] for c in c_spkmaps_full]
        c_spkmaps_full_test = [c[:, int(c.shape[1] / 2) : int(c.shape[1] / 2) * 2] for c in c_spkmaps_full]

        # Find positions with nans
        idx_nans = [np.isnan(ctr).any(axis=(0, 1)) | np.isnan(cte).any(axis=(0, 1)) for ctr, cte in zip(c_spkmaps_full_train, c_spkmaps_full_test)]

        # Filter nans
        c_spkmaps_full_train = [ctr[:, :, ~idx_nan] for ctr, idx_nan in zip(c_spkmaps_full_train, idx_nans)]
        c_spkmaps_full_test = [cte[:, :, ~idx_nan] for cte, idx_nan in zip(c_spkmaps_full_test, idx_nans)]
        c_spkmaps_full_avg = [c[:, ~idx_nan] for c, idx_nan in zip(c_spkmaps_full_avg, idx_nans)]
        c_spkmaps_full = [c[:, :, ~idx_nan] for c, idx_nan in zip(c_spkmaps_full, idx_nans)]

        # Generate random samples from poisson distribution with means set by average and number of trials equivalent to measured
        c_spkmaps_full_train_rdm = [
            np.moveaxis(np.random.poisson(np.max(c, 0), [ctr.shape[1], *c.shape]), 0, 1) for c, ctr in zip(c_spkmaps_full_avg, c_spkmaps_full_train)
        ]
        c_spkmaps_full_test_rdm = [
            np.moveaxis(np.random.poisson(np.max(c, 0), [cte.shape[1], *c.shape]), 0, 1) for c, cte in zip(c_spkmaps_full_avg, c_spkmaps_full_test)
        ]

        # Get average of these particular train/test split
        c_spkmaps_full_train_avg = [np.nanmean(c, axis=1) for c in c_spkmaps_full_train]
        c_spkmaps_full_test_avg = [np.nanmean(c, axis=1) for c in c_spkmaps_full_test]

        # Generate cross-validated samples from averages on train/test
        c_spkmaps_full_train_avg_rdm = [
            np.moveaxis(np.random.poisson(np.max(c, 0), [ctr.shape[1], *c.shape]), 0, 1)
            for c, ctr in zip(c_spkmaps_full_train_avg, c_spkmaps_full_train)
        ]
        c_spkmaps_full_test_avg_rdm = [
            np.moveaxis(np.random.poisson(np.max(c, 0), [cte.shape[1], *c.shape]), 0, 1)
            for c, cte in zip(c_spkmaps_full_test_avg, c_spkmaps_full_test)
        ]

        # Flatten along positions and trials
        c_spkmaps_full_train_rdm = [c.reshape(c.shape[0], -1) for c in c_spkmaps_full_train_rdm]
        c_spkmaps_full_test_rdm = [c.reshape(c.shape[0], -1) for c in c_spkmaps_full_test_rdm]
        c_spkmaps_full_train = [c.reshape(c.shape[0], -1) for c in c_spkmaps_full_train]
        c_spkmaps_full_test = [c.reshape(c.shape[0], -1) for c in c_spkmaps_full_test]
        c_spkmaps_full_train_avg_rdm = [c.reshape(c.shape[0], -1) for c in c_spkmaps_full_train_avg_rdm]
        c_spkmaps_full_test_avg_rdm = [c.reshape(c.shape[0], -1) for c in c_spkmaps_full_test_avg_rdm]

        # perform cvpca on full spkmap with poisson noise
        s_rdm = [
            np.nanmean(helpers.shuff_cvPCA(csftr.T, csfte.T, nshuff=5, cvmethod=helpers.cvPCA_paper_neurons), axis=0)
            for csftr, csfte in zip(c_spkmaps_full_train_rdm, c_spkmaps_full_test_rdm)
        ]
        s_trial = [
            np.nanmean(helpers.shuff_cvPCA(csftr.T, csfte.T, nshuff=5, cvmethod=helpers.cvPCA_paper_neurons), axis=0)
            for csftr, csfte in zip(c_spkmaps_full_train, c_spkmaps_full_test)
        ]
        s_cvrdm = [
            np.nanmean(helpers.shuff_cvPCA(csftr.T, csfte.T, nshuff=5, cvmethod=helpers.cvPCA_paper_neurons), axis=0)
            for csftr, csfte in zip(c_spkmaps_full_train_avg_rdm, c_spkmaps_full_test_avg_rdm)
        ]

        # Also do SVCA on the full spkmap across trials
        c_spkmaps_full_rs = [c.reshape(c.shape[0], -1) for c in c_spkmaps_full]

        # Create population
        time_split_prms = dict(
            num_groups=2,
            chunks_per_group=-5,
            num_buffer=1,
        )
        npops = [Population(c, time_split_prms=time_split_prms) for c in c_spkmaps_full_rs]

        # Split population
        train_source, train_target = helpers.named_transpose(
            [npop.get_split_data(0, center=False, scale=True, scale_type="preserve") for npop in npops]
        )
        test_source, test_target = helpers.named_transpose(
            [npop.get_split_data(1, center=False, scale=True, scale_type="preserve") for npop in npops]
        )

        # Fit SVCA on position averaged data across trials...
        svca_position = [SVCA().fit(ts, tt) for ts, tt in zip(train_source, train_target)]
        svc_shared_position = [sv.score(ts, tt)[0].numpy() for sv, ts, tt in zip(svca_position, test_source, test_target)]

        cv_by_env_trial.append(s_trial)
        cv_by_env_trial_rdm.append(s_rdm)
        cv_by_env_trial_cvrdm.append(s_cvrdm)
        svc_shared_position.append(svc_shared_position)

    # # get spkmaps of all cells / just reliable cells
    # for v in tqdm(vss, leave=False, desc="preparing spkmaps"):
    #     make_kwargs = lambda min_speed, max_speed: dict(
    #         distStep=v.distStep,
    #         onefile=v.onefile,
    #         speedThreshold=min_speed,
    #         speedMaxThreshold=max_speed,
    #         speedSmoothing=v.smoothWidth,
    #         standardizeSpks=v.standardizeSpks,
    #         idxROIs=v.idxUseROI,
    #         get_spkmap=True,
    #     )
    #     occmap_slow, _, _, spkmap_slow, _, distedges = helpers.getBehaviorAndSpikeMaps(v.vrexp, **make_kwargs(1, 5))
    #     bool_full_slow, idxFullTrials_slow, idxFullTrialEachEnv_slow = v._return_trial_indices(occmap_slow, distedges)
    #     occmap_fast, _, _, spkmap_fast, _, distedges = helpers.getBehaviorAndSpikeMaps(v.vrexp, **make_kwargs(5, 10))
    #     bool_full_fast, idxFullTrials_fast, idxFullTrialEachEnv_fast = v._return_trial_indices(occmap_slow, distedges)
    #     occmap_speedy, _, _, spkmap_speedy, _, distedges = helpers.getBehaviorAndSpikeMaps(v.vrexp, **make_kwargs(10, np.inf))
    #     bool_full_speedy, idxFullTrials_speedy, idxFullTrialEachEnv_speedy = v._return_trial_indices(occmap_slow, distedges)

    #     occmap_all, _, _, spkmap_all, _, distedges = helpers.getBehaviorAndSpikeMaps(v.vrexp, **make_kwargs(1, np.inf))

    #     c_spkmaps_slow = v.get_spkmap(average=False, smooth=args.smooth, trials="full", rawspkmap=spkmap_slow, occmap=occmap_slow)
    #     c_spkmaps_fast = v.get_spkmap(average=False, smooth=args.smooth, trials="full", rawspkmap=spkmap_fast, occmap=occmap_fast)
    #     c_spkmaps_speedy = v.get_spkmap(average=False, smooth=args.smooth, trials="full", rawspkmap=spkmap_speedy, occmap=occmap_speedy)
    #     c_spkmaps = v.get_spkmap(average=False, smooth=args.smooth, trials="full", rawspkmap=spkmap_all, occmap=occmap_all)

    #     idx_nan = np.any(np.isnan(c_spkmaps[0]), axis=(0, 1)) | np.any(np.isnan(c_spkmaps[1]), axis=(0, 1))
    #     idx = [np.argsort(np.argmax(np.mean(c_spkmap[:, :, ~idx_nan], axis=1), axis=1)) for c_spkmap in c_spkmaps]

    #     # fig, ax = plt.subplots(2, 4, figsize=(15, 10), layout="constrained", sharex=True, sharey=True)
    #     # for i, (spkmaps, name) in enumerate(zip([c_spkmaps_slow, c_spkmaps_fast, c_spkmaps_speedy, c_spkmaps], ["slow", "fast", "speedy", "all"])):
    #     #     mean0 = np.mean(spkmaps[0], axis=1)
    #     #     mean1 = np.mean(spkmaps[1], axis=1)
    #     #     ax[0, i].imshow(mean0[idx[0]], aspect="auto", cmap="plasma", origin="lower", vmin=0, vmax=1)
    #     #     ax[0, i].set_title(f"Env 0 spkmap")
    #     #     ax[1, i].imshow(mean1[idx[1]], aspect="auto", cmap="plasma", origin="lower", vmin=0, vmax=1)
    #     #     ax[1, i].set_title(f"Env 1 spkmap")
    #     # plt.show()
    #     # keep playing with this and maybe be smart about how to combine different speeds for an eventual cvPCA analysis

    # # make analyses consistent by using same (randomly subsampled) numbers of trials & neurons for each analysis
    # all_max_trials = min([int(v._get_min_trials(allmap) // 2) for v, allmap in zip(vss, allcell_maps)])
    # all_max_neurons = min([int(v._get_min_neurons(allmap)) for v, allmap in zip(vss, allcell_maps)])

    # # get cvPCA and cvFOURIER analyses for all cells / just reliable cells
    # cv_by_env_all = []
    # cv_across_all = []
    # for allmap, v in tqdm(zip(allcell_maps, vss), leave=False, desc="running cvPCA and cvFOURIER", total=len(vss)):
    #     # get cvPCA for all cell spike maps (always do by_trial=False until we have a theory for all trial=True)
    #     c_env, c_acc = v.do_cvpca(allmap, by_trial=False, max_trials=all_max_trials, max_neurons=all_max_neurons)
    #     cv_by_env_all.append(c_env)
    #     cv_across_all.append(c_acc)

    update_dict = dict()

    temp_files.update(update_dict)

    pcm.save_temp_file(temp_files, f"{args.mouse_name}_spectra_data.pkl")


if __name__ == "__main__":
    for mouse_name in mouse_names:
        print(f"Analyzing {mouse_name}")

        # load spectra data for target mouse
        track = tracking.tracker(mouse_name)  # get tracker object for mouse
        pcm = analysis.placeCellMultiSession(track, autoload=False)  # open up place cell multi session analysis object (don't autoload!!!)

        # load spectra data (use temp if it matches)
        args = helpers.AttributeDict(
            dict(
                mouse_name=mouse_name,
                dist_step=1,
                smooth=0.1,
                cutoffs=(0.4, 0.7),
                maxcutoffs=None,
                reload_spectra_data=False,
            )
        )

        add_to_spectra_data(pcm, args)

        # os.system(f"python scripts/mouse_summary.py --mouse-name {mouse_name} --do-spectra")
