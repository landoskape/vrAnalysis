from _old_vrAnalysis import session
from dimilibi import Population
from dimilibi import ReducedRankRegression

if __name__ == "__main__":
    # A little script for comparing time split methods for the same session (on RRR only)

    # Start by picking a session (use the same one with the second line or a random one with the first)

    # vrexp = random.choice(sessiondb.iterSessions(imaging=True, vrRegistration=True))
    vrexp = session.vrExperiment("CR_Hippocannula6", "2022-08-26", "702")
    print(vrexp.sessionPrint())  # show which session you chose

    # Load the deconvolved activity data
    ospks = vrexp.loadone("mpci.roiActivityDeconvolvedOasis")
    keep_idx = vrexp.idxToPlanes(keep_planes=[1, 2, 3])
    ospks = ospks[:, keep_idx]

    # We need a few population objects that are split in time in different ways
    # 1. Chunks of time (which will probably lead to train/test set having slightly different underlying structure)
    # 2. Random splits of time (without buffering) - which we should probably think of as "cheating" due to autocorrelations
    # 3. Random splits of time with buffering - which should be more realistic, but still have some autocorrelation
    time_split_chunks = dict(num_groups=2, relative_size=[5, 1], chunks_per_group=25, num_buffer=10)
    time_split_random = dict(num_groups=2, relative_size=[5, 1], chunks_per_group=-1, num_buffer=0)
    time_split_randbuff = dict(num_groups=2, relative_size=[5, 1], chunks_per_group=-1, num_buffer=2)

    # make populations with desired splits
    npop_chunks = Population(ospks.T, generate_splits=True, time_split_prms=time_split_chunks)
    npop_random = Population(ospks.T, generate_splits=True, time_split_prms=time_split_random)
    npop_randbuff = Population(ospks.T, generate_splits=True, time_split_prms=time_split_randbuff)

    # choose parameters for retrieving the train / test data
    center = False
    scale = True
    pre_split = False
    scale_type = "preserve"

    def get_data(npop, center, scale, pre_split, scale_type):
        train_source, train_target = npop.get_split_data(0, center=center, scale=scale, pre_split=pre_split, scale_type=scale_type)
        test_source, test_target = npop.get_split_data(1, center=center, scale=scale, pre_split=pre_split, scale_type=scale_type)
        return dict(train_source=train_source, train_target=train_target, test_source=test_source, test_target=test_target)

    # get dictionaries with train/test data for source/target neurons
    data_chunks = get_data(npop_chunks, center=center, scale=scale, pre_split=pre_split, scale_type=scale_type)
    data_random = get_data(npop_random, center=center, scale=scale, pre_split=pre_split, scale_type=scale_type)
    data_randbuff = get_data(npop_randbuff, center=center, scale=scale, pre_split=pre_split, scale_type=scale_type)

    # fit RRR models
    def fit_rrr(data):
        return ReducedRankRegression(alpha=1e5, fit_intercept=True).fit(data["train_source"].T, data["train_target"].T)

    def score_rrr(rrr, data):
        return rrr.score(data["test_source"].T, data["test_target"].T)

    rrr_chunks = fit_rrr(data_chunks)
    rrr_random = fit_rrr(data_random)
    rrr_randbuff = fit_rrr(data_randbuff)

    score_chunks = score_rrr(rrr_chunks, data_chunks)
    score_random = score_rrr(rrr_random, data_random)
    score_randbuff = score_rrr(rrr_randbuff, data_randbuff)

    print(f"Score Chunks: {score_chunks}")
    print(f"Score Random: {score_random}")
    print(f"Score RandBuff: {score_randbuff}")
