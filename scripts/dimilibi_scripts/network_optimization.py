import os
import sys
from tqdm import tqdm
import torch
import optuna

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers import load_population, get_ranks

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

from _old_vrAnalysis import analysis
from _old_vrAnalysis import session

from dimilibi import scaled_mse
from dimilibi import SVCANet, EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# methods for optimizing network hyperparameters
def build_networks(
    train_source,
    train_target,
    num_hidden,
    num_latent,
    lr,
    weight_decay,
    transparent_relu,
    dropout_rate,
    num_networks,
):
    """
    Build networks to be used in an optuna optimization.

    In a previous iteration, I included control over regularization parameters and network type (BetaVAE or SVCANet)
    however I found that my fancy regularizers don't help much and that BetaVAE is not as good as SVCANet (it's just
    really finnicky and probably requires a lot more tuning). So I've simplified this function to just build SVCANets
    with a transparent ReLU activation function.

    Parameters
    ----------
    train_source : torch.Tensor
        Source data for training.
    train_target : torch.Tensor
        Target data for training.
    num_hidden : list
        List of hidden layer sizes.
    num_latent : int
        Number of latent dimensions.
    lr : float, optional
        Learning rate for the optimizer.
    weight_decay : float, optional
        Weight decay for the optimizer.
    transparent_relu : bool, optional
        Whether to use a transparent ReLU activation function, by default True.
    dropout_rate : float, optional
        Dropout rate for the network.
    num_networks : int, optional
        Number of networks to train.
    """
    num_neurons = train_source.size(0)
    num_target_neurons = train_target.size(0)

    nets = [
        SVCANet(
            num_neurons,
            num_hidden,
            num_latent,
            num_target_neurons,
            activation=torch.nn.ReLU(),
            nonnegative=True,
            transparent_relu=transparent_relu,
            dropout_rate=dropout_rate,
        ).to(device)
        for _ in range(num_networks)
    ]

    optimizers = [torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay) for net in nets]
    loss_fn = torch.nn.MSELoss(reduction="sum")
    return nets, optimizers, loss_fn


def train_networks(
    nets, optimizers, loss_fn, train_source, train_target, test_source, test_target, noise_level, num_epochs, reduce=True, verbose=False
):
    """
    Train a networks for a set number of epochs (to be used in an optuna optimization).

    """
    num_nets = len(nets)
    num_timepoints = train_source.size(1)
    batch_size = num_timepoints // 10

    train_loss = torch.zeros((num_nets, num_epochs))
    train_score = torch.zeros((num_nets, num_epochs))
    traintest_loss = torch.zeros((num_nets, num_epochs))
    traintest_score = torch.zeros((num_nets, num_epochs))

    train_source = train_source.to(device)
    train_target = train_target.to(device)
    test_source = test_source.to(device)
    test_target = test_target.to(device)

    # Setup early stopping
    early_stopping = EarlyStopping(num_nets, patience=10, min_delta=0.01, direction="maximize")

    for net in nets:
        net.train()

    if verbose:
        progress = tqdm(range(num_epochs), desc="Training Networks")
    else:
        progress = range(num_epochs)

    for epoch in progress:

        itime = torch.randperm(num_timepoints)[:batch_size]

        source_batch = train_source[:, itime].T
        target_batch = train_target[:, itime].T

        for opt in optimizers:
            opt.zero_grad()

        predictions = [net(source_batch + noise_level * torch.randn_like(source_batch)) for net in nets]
        loss = [loss_fn(pred, target_batch) for pred in predictions]
        for l in loss:
            l.backward()
        for opt in optimizers:
            opt.step()

        scores = [net.score(source_batch, target_batch) for net in nets]

        for inet in range(num_nets):
            train_loss[inet, epoch] = loss[inet].item()
            train_score[inet, epoch] = scores[inet].item()

        with torch.no_grad():
            for net in nets:
                net.eval()
            preds = [net(test_source.T) for net in nets]
            for inet in range(num_nets):
                traintest_loss[inet, epoch] = loss_fn(preds[inet], test_target.T).item()
                traintest_score[inet, epoch] = nets[inet].score(test_source.T, test_target.T).item()
            for net in nets:
                net.train()

        if torch.all(early_stopping(traintest_score[:, epoch])):
            print(f"Early stopping at epoch {epoch}")
            break

        if torch.any(torch.isnan(traintest_score[:, epoch])):
            raise optuna.TrialPruned()

    best_scores = torch.max(traintest_score[:, : epoch + 1], dim=1).values
    best_epochs = torch.argmax(traintest_score[:, : epoch + 1], dim=1)

    if reduce:
        return torch.mean(best_scores), torch.mean(best_epochs.float())

    return best_scores, best_epochs


def objective(trial, train_source, train_target, test_source, test_target, num_latent, num_epochs, num_networks=10):
    # Define the hyperparameter to optimize
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e4, log=True)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    num_hidden = trial.suggest_int("num_hidden", 50, 2000, log=True)
    noise_level = trial.suggest_float("noise_level", 1e-3, 1e2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    transparent_relu = True  # trial.suggest_categorical("transparent_relu", [True, False])

    # Build the network
    nets, optimizers, loss_fn = build_networks(
        train_source,
        train_target,
        [num_hidden],
        num_latent,
        lr,
        weight_decay,
        transparent_relu,
        dropout_rate,
        num_networks,
    )

    # Train the network
    best_score = train_networks(nets, optimizers, loss_fn, train_source, train_target, test_source, test_target, noise_level, num_epochs=num_epochs)[
        0
    ]

    # Return the evaluation metric (the best testing score during training: early stopping is going to be used)
    return best_score


def test_networks(train_source, train_target, val_source, val_target, test_source, test_target, num_latent, params, num_epochs=1000, num_networks=10):
    # Build networks with optimized hyperparameters
    transparent_relu = True
    nets, optimizers, loss_fn = build_networks(
        train_source,
        train_target,
        [params["num_hidden"]],
        num_latent,
        params["lr"],
        params["weight_decay"],
        transparent_relu,
        params["dropout_rate"],
        num_networks,
    )

    # Train the networks
    train_networks(nets, optimizers, loss_fn, train_source, train_target, val_source, val_target, params["noise_level"], num_epochs, reduce=False)

    test_loss = torch.zeros(len(nets))
    test_score = torch.zeros(len(nets))
    test_scaled_mse = torch.zeros(len(nets))

    test_source = test_source.to(device)
    test_target = test_target.to(device)

    for inet, net in enumerate(nets):
        net.eval()
        pred = net(test_source.T)
        test_loss[inet] = torch.nn.MSELoss(reduction="sum")(pred, test_target.T).item()
        test_score[inet] = net.score(test_source.T, test_target.T).item()
        test_scaled_mse[inet] = scaled_mse(pred, test_target.T).item()

    return test_loss, test_score, test_scaled_mse


def optimize_hyperparameters(
    mouse_name,
    datestr,
    sessionid,
    num_latent,
    num_epochs=1000,
    num_networks=3,
    n_trials=50,
    retest_only=False,
    show_progress_bar=False,
    population_name=None,
):
    """
    Optimize hyperparameters for networks trained on peer prediction.

    Loads the population data for the session and splits it into training, validation, and test sets.
    Uses optuna to optimize hyperparameters for networks trained on the training set and evaluated on the validation set.
    The best hyperparameters are then used to train a new set of networks and evaluate them on the test set.

    If retest_only is True, the function will only retest networks that have already been optimized.
    """
    npop = load_population(mouse_name, datestr, sessionid, population_name=population_name)

    train_source, train_target = npop.get_split_data(0, center=False, scale=True, scale_type="preserve")
    val_source, val_target = npop.get_split_data(1, center=False, scale=True, scale_type="preserve")
    test_source, test_target = npop.get_split_data(2, center=False, scale=True, scale_type="preserve")

    if retest_only:
        pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
        hyp_filename = network_tempfile_name(pcss.vrexp, num_latent, population_name=population_name)
        original_results = pcss.load_temp_file(hyp_filename)
        study = original_results["study"]
        best_params = original_results["best_params"]

    else:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, train_source, train_target, val_source, val_target, num_latent, num_epochs, num_networks),
            n_trials=n_trials,
            n_jobs=2,
            show_progress_bar=show_progress_bar,
        )
        best_params = study.best_params

    # Use the best hyperparameters to train a new round of networks and evaluate them on the test data
    test_loss, test_score, test_scaled_mse = test_networks(
        train_source,
        train_target,
        val_source,
        val_target,
        test_source,
        test_target,
        num_latent,
        best_params,
        num_epochs=num_epochs,
        num_networks=num_networks,
    )

    return dict(
        mouse_name=mouse_name,
        datestr=datestr,
        sessionid=sessionid,
        study=study,
        rank=num_latent,
        best_params=best_params,
        test_loss=test_loss,
        test_score=test_score,
        test_scaled_mse=test_scaled_mse,
    )


def retest_networks(mouse_name, datestr, sessionid, num_latent, num_epochs, num_networks, population_name=None):
    npop = load_population(mouse_name, datestr, sessionid, population_name=population_name)

    train_source, train_target = npop.get_split_data(0, center=False, scale=True, scale_type="preserve")
    val_source, val_target = npop.get_split_data(1, center=False, scale=True, scale_type="preserve")
    test_source, test_target = npop.get_split_data(2, center=False, scale=True, scale_type="preserve")

    pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
    hyp_filename = network_tempfile_name(pcss.vrexp, num_latent, population_name=population_name)
    original_results = pcss.load_temp_file(hyp_filename)
    best_params = original_results["best_params"]
    test_loss, test_score, test_scaled_mse = test_networks(
        train_source,
        train_target,
        val_source,
        val_target,
        test_source,
        test_target,
        num_latent,
        best_params,
        num_epochs=num_epochs,
        num_networks=num_networks,
    )

    original_results["test_loss"] = test_loss
    original_results["test_score"] = test_score
    original_results["test_scaled_mse"] = test_scaled_mse

    return original_results


def network_tempfile_name(vrexp, rank, population_name=None):
    """generate temporary file name for network tuning results"""
    name = f"network_optimization_results_rank{rank}_{str(vrexp)}"
    if population_name is not None:
        name += f"_{population_name}"
    return name


def do_network_optimization(all_sessions, retest_only=False, population_name=None, skip_completed=True, save=True):
    """
    Perform network optimization and testing of network models on peer prediction.

    Will use optuna to optimize hyperparameters for networks trained on each session in
    all_sessions and test the best model on the test set. The results are saved as a
    dictionary and stored in a temporary file using the standard analysis temporary file
    storage system.

    Parameters
    ----------
    all_sessions : dict
        Dictionary containing session identifiers for each mouse.
    retest_only : bool, optional
        Whether to only retest networks that have already been optimized, by default False.
    skip_completed : bool, optional
        Whether to skip sessions that have already been optimized, by default True.
        Automatically set to False if retest_only is True.
    save : bool, optional
        Whether to save the results to a temporary file, by default True.
    """
    ranks = get_ranks()
    for mouse_name, sessions in all_sessions.items():
        for datestr, sessionid in sessions:
            pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
            for num_latent in ranks:
                if (
                    not retest_only
                    and skip_completed
                    and pcss.check_temp_file(network_tempfile_name(pcss.vrexp, num_latent, population_name=population_name))
                ):
                    print(f"Found completed hyperparameter optimization for: {mouse_name}, {datestr}, {sessionid}, rank:{num_latent}")
                    continue
                if retest_only and not pcss.check_temp_file(network_tempfile_name(pcss.vrexp, num_latent, population_name=population_name)):
                    print(f"Skipping retest for: {mouse_name}, {datestr}, {sessionid}, rank:{num_latent} (no hyperparameter optimization found)")
                    continue
                print(f"Optimizing network hyperparameters for: {mouse_name}, {datestr}, {sessionid}, rank: {num_latent}:")
                hyperparameter_results = optimize_hyperparameters(
                    mouse_name, datestr, sessionid, num_latent, retest_only=retest_only, show_progress_bar=False, population_name=population_name
                )
                if save:
                    pcss.save_temp_file(hyperparameter_results, network_tempfile_name(pcss.vrexp, num_latent, population_name=population_name))


@torch.no_grad()
def load_network_results(all_sessions, results="all", population_name=None):
    ranks = get_ranks()
    network_results = []
    session_ids = []
    tested_ranks = []  # this is just for confirmation that everything was processed correctly and completely
    for mouse_name, sessions in all_sessions.items():
        for datestr, sessionid in sessions:
            pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
            network_results.append([])
            session_ids.append((mouse_name, datestr, sessionid))
            tested_ranks.append(torch.zeros(len(ranks), dtype=bool))
            for irank, num_latent in enumerate(ranks):
                hyp_filename = network_tempfile_name(pcss.vrexp, num_latent, population_name=population_name)
                if not pcss.check_temp_file(hyp_filename):
                    print(f"Skipping hyperparameter_results from {mouse_name}, {datestr}, {sessionid}, rank:{num_latent} (not found)")
                    tested_ranks[-1][irank] = False
                    continue
                else:
                    tested_ranks[-1][irank] = True
                print(f"Loading hyperparameter_results from {mouse_name}, {datestr}, {sessionid}, rank:{num_latent}")
                network_results[-1].append(pcss.load_temp_file(hyp_filename))

    if results == "all":
        if not torch.stack(tested_ranks).all():
            print(f"Warning!!! Not all ranks were processed for all sessions!")
        return network_results, session_ids, tested_ranks

    if results == "test_by_mouse":
        if not torch.stack(tested_ranks).all():
            raise ValueError("Not all ranks were processed for all sessions, can't consolidate across sessions within mouse!")

        mouse_names = sorted(list(set(map(lambda x: x[0], session_ids))))  # get master list of mouse names
        num_mice = len(mouse_names)
        num_ranks = len(ranks)
        val_scores = torch.zeros((num_mice, num_ranks))
        test_scores = torch.zeros((num_mice, num_ranks))
        test_scaled_mses = torch.zeros((num_mice, num_ranks))
        params = dict(
            zip(network_results[0][0]["best_params"].keys(), [[[] for _ in range(num_ranks)] for _ in network_results[0][0]["best_params"]])
        )
        num_samples = torch.zeros(num_mice)
        for network_result in network_results:
            mouse_idx = mouse_names.index(network_result[0]["mouse_name"])
            for irank, net_res in enumerate(network_result):
                assert (
                    net_res["rank"] == ranks[irank]
                ), f"ranks don't match for {ranks[irank]} in session: {net_res['mouse_name']}/{net_res['datestr']}/{net_res['sessionid']}"
                val_scores[mouse_idx, irank] += net_res["study"].best_value
                test_scores[mouse_idx, irank] += net_res["test_score"].mean().item()
                test_scaled_mses[mouse_idx, irank] += net_res["test_scaled_mse"].mean().item()
                if irank == 0:
                    num_samples[mouse_idx] += 1
                for key, val in net_res["best_params"].items():
                    params[key][irank].append(val)
        # get average for each mouse
        val_scores /= num_samples.unsqueeze(1)
        test_scores /= num_samples.unsqueeze(1)
        test_scaled_mses /= num_samples.unsqueeze(1)
        test_results = dict(
            mouse_names=mouse_names,
            ranks=ranks,
            test_scores=test_scores,
            test_scaled_mses=test_scaled_mses,
            num_samples=num_samples,
            params=params,
        )
        return test_results, session_ids

    raise ValueError(f"results must be 'all' or 'test_by_mouse', got {results}")
