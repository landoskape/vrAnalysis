import time
import torch

# from cachecache import cache
from joblib import Memory
from tqdm import tqdm

# Need to explicitly create a Memory object
memory = Memory("./cachedir", verbose=0)

from vrAnalysis import database
from vrAnalysis import tracking
from vrAnalysis import analysis
from vrAnalysis import session

from dimilibi import Population, ReducedRankRegression, scaled_mse
from dimilibi import BetaVAE, SVCANet, LocalSimilarity, EmptyRegularizer, BetaVAE_KLDiv, FlexibleFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_sessions():
    # first get list of mice that I tend to use for analyses
    mousedb = database.vrDatabase("vrMice")
    df = mousedb.getTable(trackerExists=True)
    mouse_names = df["mouseName"].unique()

    # for each mouse, get the session identifiers and add them to a list
    all_sessions = {mouse_name: [] for mouse_name in mouse_names}
    for mouse_name in mouse_names:
        track = tracking.tracker(mouse_name)  # get tracker object for mouse
        pcm = analysis.placeCellMultiSession(track, autoload=False)  # open up place cell multi session analysis object (don't autoload!!!)
        for pcss in pcm.pcss:
            datestr, sessionid = pcss.vrexp.dateString, pcss.vrexp.sessionid
            all_sessions[mouse_name].append((datestr, sessionid))

    return all_sessions


def get_rank(subset=False):
    """
    A subset of ranks to use for optimization and testing.
    """
    return [1, 2, 3, 5, 8, 15, 50, 100, 200]


def get_keep_planes():
    return [1, 2, 3, 4]


@memory.cache
def optimize_rrr(mouse_name, datestr, sessionid):
    def get_init_alphas():
        return torch.logspace(6, 10, 5)

    def get_next_alphas(init_alpha):
        return torch.logspace(torch.log10(init_alpha) - 0.2, torch.log10(init_alpha) + 0.2, 5)

    ses = session.vrExperiment(mouse_name, datestr, sessionid)
    ospks = ses.loadone("mpci.roiActivityDeconvolvedOasis")
    keep_idx = ses.idxToPlanes(keep_planes=get_keep_planes())
    ospks = ospks[:, keep_idx]
    time_split_prms = dict(
        num_groups=3,
        relative_size=[5, 1, 1],
        chunks_per_group=25,
    )
    npop = Population(ospks.T, generate_splits=True, time_split_prms=time_split_prms)
    indices_dict = npop.get_indices_dict()

    # split the data into training and validation sets
    train_source, train_target = npop.get_split_data(0, center=True, scale=False)
    val_source, val_target = npop.get_split_data(1, center=True, scale=False)

    init_alphas = get_init_alphas()
    init_rrr = []
    for a in tqdm(init_alphas):
        init_rrr.append(ReducedRankRegression(alpha=a, fit_intercept=False).fit(train_source.T, train_target.T))
    init_scores = [rrr.score(val_source.T, val_target.T) for rrr in init_rrr]
    best_init_alpha = init_alphas[init_scores.index(max(init_scores))]

    next_alpha = get_next_alphas(best_init_alpha)
    next_rrr = []
    for a in tqdm(next_alpha):
        next_rrr.append(ReducedRankRegression(alpha=a, fit_intercept=False).fit(train_source.T, train_target.T))
    next_scores = [rrr.score(val_source.T, val_target.T) for rrr in next_rrr]
    best_index = next_scores.index(max(next_scores))
    best_alpha = next_alpha[best_index]

    results = dict(
        mouse_name=mouse_name,
        datestr=datestr,
        sessionid=sessionid,
        best_alpha=best_alpha,
        val_score=max(next_scores),
        indices_dict=indices_dict,
    )
    return results


@memory.cache
def test_rrr(mouse_name, datestr, sessionid, indices_dict, alpha):
    ses = session.vrExperiment(mouse_name, datestr, sessionid)
    ospks = ses.loadone("mpci.roiActivityDeconvolvedOasis")
    keep_idx = ses.idxToPlanes(keep_planes=get_keep_planes())
    ospks = ospks[:, keep_idx]

    npop = Population.make_from_indices(indices_dict, ospks.T)

    train_source, train_target = npop.get_split_data(0, center=True, scale=False)
    test_source, test_target = npop.get_split_data(2, center=True, scale=False)

    rrr = ReducedRankRegression(alpha=alpha, fit_intercept=False).fit(train_source.T, train_target.T)

    test_score = rrr.score(test_source.T, test_target.T)
    test_scaled_mse = scaled_mse(rrr.predict(test_source.T), test_target.T, reduce="mean")

    ranks = get_rank(subset=False)
    test_score_by_rank = [rrr.score(test_source.T, test_target.T, rank=r) for r in tqdm(ranks)]
    test_scaled_mse_by_rank = [scaled_mse(rrr.predict(test_source.T, rank=r), test_target.T, reduce="mean") for r in tqdm(ranks)]

    results = dict(
        test_score=test_score,
        test_scaled_mse=test_scaled_mse,
        test_score_by_rank=test_score_by_rank,
        test_scaled_mse_by_rank=test_scaled_mse_by_rank,
    )

    return results


def optimize_networks(mouse_name, datestr, sessionid, indices_dict, betavae=False, regularize=False):
    ses = session.vrExperiment(mouse_name, datestr, sessionid)
    ospks = ses.loadone("mpci.roiActivityDeconvolvedOasis")
    keep_idx = ses.idxToPlanes(keep_planes=get_keep_planes())
    ospks = ospks[:, keep_idx]

    npop = Population.make_from_indices(indices_dict, ospks.T)

    train_source, train_target = npop.get_split_data(0, center=True, scale=False)
    val_source, val_target = npop.get_split_data(1, center=True, scale=False)
    test_source, test_target = npop.get_split_data(2, center=True, scale=False)

    num_neurons = train_source.size(0)
    num_hidden = [400]
    num_latent = get_rank(subset=True)
    num_target_neurons = train_target.size(0)
    num_timepoints = train_source.size(1)

    batch_size = min(100, num_timepoints)

    net_constructor = BetaVAE if betavae else SVCANet

    nets = [
        net_constructor(
            num_neurons,
            num_hidden,
            dim_latent,
            num_target_neurons,
            activation=torch.nn.ReLU(),
            nonnegative=False,
        ).to(device)
        for dim_latent in num_latent
    ]

    loss_function = torch.nn.MSELoss()
    if betavae:
        standard_regularizer = [EmptyRegularizer() for _ in range(len(nets))]
        beta_reg = BetaVAE_KLDiv(beta=1)
    else:
        if regularize:
            standard_reg_weight = 1e4
            flex_filter = FlexibleFilter(baseline=0.1, negative_scale=0.0)
            standard_regularizer = [LocalSimilarity(num_neurons, num_timepoints, filter=flex_filter).to(device) for _ in range(len(nets))]
        else:
            standard_regularizer = [EmptyRegularizer() for _ in range(len(nets))]
        beta_reg = EmptyRegularizer()

    weight_decay = 1e1
    opts = [torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=weight_decay) for net in nets]


def do_rrr_optimization(all_sessions):
    for mouse_name, sessions in all_sessions.items():
        for datestr, sessionid in sessions:
            pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
            print(f"Optimizing ridge regression for: {mouse_name}, {datestr}, {sessionid}:")
            t = time.time()
            optimize_results = optimize_rrr(mouse_name, datestr, sessionid)
            print(f"Time: {time.time() - t : .2f}, Best alpha: {optimize_results['best_alpha']}")
            print(f"Testing ridge regression for: {mouse_name}, {datestr}, {sessionid}:")
            test_results = test_rrr(mouse_name, datestr, sessionid, optimize_results["indices_dict"], optimize_results["best_alpha"])
            rrr_results = {**optimize_results, **test_results}
            pcss.save_temp_file(rrr_results, f"rrr_optimization_results_{str(pcss.vrexp)}")


def do_network_optimization(all_sessions):
    for mouse_name, sessions in all_sessions.items():
        for datestr, sessionid in sessions:
            pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
            rrr_results = pcss.load_temp_file("rrr_optimization_results")
            indices_dict = rrr_results["indices_dict"]

            # print(f"Optimizing networks for: {mouse_name}, {datestr}, {sessionid}:")
            # t = time.time()
            # optimize_results = optimize_networks(mouse_name, datestr, sessionid)
            # print(f"Time: {time.time() - t : .2f}")
            # pcss.save_temp_file(optimize_results, "network_optimization_results")


if __name__ == "__main__":
    all_sessions = get_sessions()
    do_rrr_optimization(all_sessions)
    do_network_optimization(all_sessions)
