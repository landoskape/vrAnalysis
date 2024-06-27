from tqdm import tqdm
import torch

import sys
import os

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from vrAnalysis.helpers import named_transpose

from .metrics import scaled_mse


def train(
    nets,
    opts,
    loss_function,
    regularizers,
    beta_regularizer,
    alpha_reg,
    train_source,
    train_target,
    val_source,
    val_target,
    num_timepoints,
    batch_size,
    betavae=False,
    device="cuda",
    num_epochs=2000,
):
    # train the network
    num_nets = len(nets)

    train_loss = torch.zeros((num_nets, num_epochs))
    train_score = torch.zeros((num_nets, num_epochs))
    train_reg = torch.zeros((num_nets, num_epochs))
    train_betareg = torch.zeros((num_nets, num_epochs))
    train_scaled_mse = torch.zeros((num_nets, num_epochs))
    trainval_loss = torch.zeros((num_nets, num_epochs))
    trainval_score = torch.zeros((num_nets, num_epochs))
    trainval_scaled_mse = torch.zeros((num_nets, num_epochs))

    train_source = train_source.to(device)
    train_target = train_target.to(device)
    val_source = val_source.to(device)
    val_target = val_target.to(device)

    for net in nets:
        net.train()

    progress = tqdm(range(num_epochs), desc="Training SVCA Net")
    for epoch in progress:

        itime = torch.randperm(num_timepoints)[:batch_size]

        source_batch = train_source[:, itime].T
        target_batch = train_target[:, itime].T

        for opt in opts:
            opt.zero_grad()

        predictions = [net(source_batch) for net in nets]
        if betavae:
            mulogvar = [pred[1:] for pred in predictions]
            predictions = [pred[0] for pred in predictions]
        else:
            mulogvar = [torch.tensor(0.0, device=device) for _ in range(num_nets)]

        losses = [loss_function(pred, target_batch) for pred in predictions]
        standard_regs = [reg(source_batch, pred) for reg, pred in zip(regularizers, predictions)]
        beta_regs = [beta_regularizer(*mlv) for mlv in mulogvar]

        full_loss = [loss + alpha_reg * sreg + breg for loss, sreg, breg in zip(losses, standard_regs, beta_regs)]
        for loss in full_loss:
            loss.backward()

        for opt in opts:
            opt.step()

        scores = [net.score(source_batch, target_batch) for net in nets]

        for inet in range(len(nets)):
            train_loss[inet, epoch] = losses[inet].item()
            train_reg[inet, epoch] = standard_regs[inet].item()
            train_betareg[inet, epoch] = beta_regs[inet].item()
            train_score[inet, epoch] = scores[inet].item()
            train_scaled_mse[inet, epoch] = scaled_mse(predictions[inet], target_batch).item()

            with torch.no_grad():
                for net in nets:
                    net.eval()
                pred = nets[inet](val_source.T)
                if betavae:
                    pred = pred[0]
                trainval_loss[inet, epoch] = loss_function(pred, val_target.T).item()
                trainval_score[inet, epoch] = nets[inet].score(val_source.T, val_target.T).item()
                trainval_scaled_mse[inet, epoch] = scaled_mse(pred, val_target.T).item()
                for net in nets:
                    net.train()

    results = dict(
        train_loss=train_loss,
        train_score=train_score,
        train_reg=train_reg,
        train_betareg=train_betareg,
        train_scaled_mse=train_scaled_mse,
        trainval_loss=trainval_loss,
        trainval_score=trainval_score,
        trainval_scaled_mse=trainval_scaled_mse,
    )
