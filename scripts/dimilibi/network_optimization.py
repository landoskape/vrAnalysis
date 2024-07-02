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
        num_epochs=2000,
    )

    # validation just picks the best epoch (e.g. early stopping)
    best_val_epoch = torch.argmax(_results["val_score"], dim=1)

    # then we save the results for the best epoch from the test set over training
    test_loss = torch.gather(_results["test_loss"], 1, best_val_epoch.unsqueeze(0)).squeeze(0)
    test_score = torch.gather(_results["test_score"], 1, best_val_epoch.unsqueeze(0)).squeeze(0)
    test_scaled_mse = torch.gather(_results["test_scaled_mse"], 1, best_val_epoch.unsqueeze(0)).squeeze(0)

    results = dict(
        mouse_name=mouse_name,
        datestr=datestr,
        sessionid=sessionid,
        ranks=ranks,
        best_epoch=best_val_epoch,
        test_loss=test_loss,
        test_score=test_score,
        test_scaled_mse=test_scaled_mse,
        val_loss_trajectory=_results["val_loss"],
        val_score_trajectory=_results["val_score"],
        val_scaled_mse_trajectory=_results["val_scaled_mse"],
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

