import os
import sys
import time
from tqdm import tqdm
import torch

from helpers import load_population, get_ranks, memory


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

from vrAnalysis import analysis
from vrAnalysis import session

from dimilibi import train
from dimilibi import BetaVAE, SVCANet, LocalSimilarity, EmptyRegularizer, BetaVAE_KLDiv, FlexibleFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def optimize_networks(mouse_name, datestr, sessionid, ranks, betavae=False):

    npop = load_population(mouse_name, datestr, sessionid)

    train_source, train_target = npop.get_split_data(0, center=True, scale=False)
    val_source, val_target = npop.get_split_data(1, center=True, scale=False)
    test_source, test_target = npop.get_split_data(2, center=True, scale=False)

    num_neurons = train_source.size(0)
    num_hidden = [400]
    num_latent = ranks
    num_target_neurons = train_target.size(0)
    num_timepoints = train_source.size(1)
    num_replicates = 2

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
        for dim_latent in num_latent for _ in range(num_replicates)
    ]

    loss_function = torch.nn.MSELoss(reduction='sum')
    if betavae:
        beta_regularizer = BetaVAE_KLDiv(beta=1, reduction='sum')
    else:
        beta_regularizer = EmptyRegularizer()

    standard_reg_weight = 0
    beta_reg_weight = 10
    standard_regularizers = [EmptyRegularizer() for _ in nets]
    
    weight_decay = 1e1
    opts = [torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=weight_decay) for net in nets]

    num_epochs = 2000

    _results = train(
        nets,
        opts,
        loss_function,
        standard_regularizers,
        beta_regularizer,
        standard_reg_weight,
        beta_reg_weight,
        train_source,
        train_target,
        val_source,
        val_target,
        test_source,
        test_target,
        batch_size,
        betavae=betavae,
        device=device,
        num_epochs=num_epochs,
    )

    # validation just picks the best epoch (e.g. early stopping)
    best_val_epoch = torch.argmax(_results["val_score"], dim=1)
    
    # then we save the results for the best epoch from the test set over training
    test_loss = torch.gather(_results["test_loss"], 1, best_val_epoch.unsqueeze(1)).squeeze(1)
    test_score = torch.gather(_results["test_score"], 1, best_val_epoch.unsqueeze(1)).squeeze(1)
    test_scaled_mse = torch.gather(_results["test_scaled_mse"], 1, best_val_epoch.unsqueeze(1)).squeeze(1)

    # we train multiple networks per rank, so we need to pick the best network for each rank
    val_score = torch.gather(_results["val_score"], 1, best_val_epoch.unsqueeze(1)).squeeze(1)
    best_network_by_rank = torch.argmax(val_score.view(-1, num_replicates), dim=1)

    # pick the test results for the best network for each rank
    test_loss = torch.gather(test_loss.view(-1, num_replicates), 1, best_network_by_rank.unsqueeze(1)).squeeze(1)
    test_score = torch.gather(test_score.view(-1, num_replicates), 1, best_network_by_rank.unsqueeze(1)).squeeze(1)
    test_scaled_mse = torch.gather(test_scaled_mse.view(-1, num_replicates), 1, best_network_by_rank.unsqueeze(1)).squeeze(1)

    # only return val trajectories from the networks with the best validation score
    def _get_trajectory(trajectory, network_index):
        _view_traj = trajectory.view(-1, num_replicates, num_epochs)
        _index = network_index.unsqueeze(1).unsqueeze(2).expand(-1, -1, num_epochs)
        return torch.gather(_view_traj, 1, _index).squeeze(1)
    
    _val_score = _get_trajectory(_results["val_score"], best_network_by_rank)
    _val_loss = _get_trajectory(_results["val_loss"], best_network_by_rank)
    _val_scaled_mse = _get_trajectory(_results["val_scaled_mse"], best_network_by_rank)

    results = dict(
        mouse_name=mouse_name,
        datestr=datestr,
        sessionid=sessionid,
        ranks=ranks,
        best_epoch=best_val_epoch,
        test_loss=test_loss,
        test_score=test_score,
        test_scaled_mse=test_scaled_mse,
        val_loss_trajectory=_val_loss,
        val_score_trajectory=_val_score,
        val_scaled_mse_trajectory=_val_scaled_mse,
    )

    return results


def network_tempfile_name(vrexp, net_name):
    """generate temporary file name for network results"""
    return f"network_optimization_results_{net_name}_{str(vrexp)}"


def do_network_optimization(all_sessions, skip_completed=True):
    """
    Perform optimization and testing of network models on peer prediction.

    Will optimize a beta-vae, a standard network, and a standard network trained with a similarity 
    regularizer for each session in all_sessions and test the best model on the test set. Trains 
    independent models for each rank in ranks. The results are saved as a dictionary and stored in a
    temporary file using the standard analysis temporary file storage system.

    Parameters
    ----------
    all_sessions : dict
        Dictionary containing session identifiers for each mouse.
    """
    ranks = get_ranks()
    network_parameters = dict(
        betavae=dict(betavae=True),
        standard=dict(betavae=False),
    )
    for mouse_name, sessions in all_sessions.items():
        for datestr, sessionid in sessions:
            pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
            for net_name, net_prms in network_parameters.items():
                if skip_completed and pcss.check_temp_file(network_tempfile_name(pcss.vrexp, net_name)):
                    print(f"Found completed optimization for: {mouse_name}, {datestr}, {sessionid}, {net_name}")
                    continue
                print(f"Optimizing network models for: {mouse_name}, {datestr}, {sessionid}, {net_name}:")
                optimize_results = optimize_networks(mouse_name, datestr, sessionid, ranks, **net_prms)
                pcss.save_temp_file(optimize_results, network_tempfile_name(pcss.vrexp, net_name))


@torch.no_grad()
def load_network_results(all_sessions, results='all'):
    network_parameters = dict(
        betavae=dict(betavae=True),
        standard=dict(betavae=False),
    )
    network_results = [[] for _ in network_parameters]
    
    for mouse_name, sessions in all_sessions.items():
        for datestr, sessionid in sessions:
            pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
            for inet, net_name in enumerate(network_parameters):
                net_filename = network_tempfile_name(pcss.vrexp, net_name)
                if not pcss.check_temp_file(net_filename):
                    print(f"Skipping network_results from {mouse_name}, {datestr}, {sessionid}, {net_name} (not found)")
                    continue
                print(f"Loading network_results from {mouse_name}, {datestr}, {sessionid}, {net_name}")
                network_results[inet].append(pcss.load_temp_file(net_filename))
    
    if results=='all':
        return network_results, network_parameters
    
    if results=='test_by_mouse':
        test_results = []
        for network_result in network_results:
            ranks = network_result[0]["ranks"]
            for net_res in network_result:
                assert net_res["ranks"] == ranks, "ranks are not all equal"
            num_ranks = len(ranks)
            mouse_names = list(set([res["mouse_name"] for res in network_result]))
            num_mice = len(mouse_names)
            test_scores = torch.zeros((num_mice, num_ranks))
            test_scaled_mses = torch.zeros((num_mice, num_ranks))
            num_samples = torch.zeros(num_mice)
            for net_res in network_result:
                mouse_idx = mouse_names.index(net_res["mouse_name"])
                test_scores[mouse_idx] += net_res["test_score"]
                test_scaled_mses[mouse_idx] += net_res["test_scaled_mse"]
                num_samples[mouse_idx] += 1
            # get average for each mouse
            test_scores /= num_samples.unsqueeze(1)
            test_scaled_mses /= num_samples.unsqueeze(1)
            test_results.append(dict(
                mouse_names=mouse_names,
                ranks=ranks,
                test_scores=test_scores,
                test_scaled_mses=test_scaled_mses,
            ))
        return test_results, network_parameters
    
    raise ValueError(f"results must be 'all' or 'test_by_mouse', got {results}")