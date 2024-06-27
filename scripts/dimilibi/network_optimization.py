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
