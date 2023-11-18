import itertools
import warnings
from copy import deepcopy
from typing import Callable, Dict, List, Literal, Optional, Type, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch
from torch.multiprocessing import cpu_count, get_context
from torch.utils.data import DataLoader
from tqdm import tqdm

from devinterp.optim.sgld import SGLD


def sample_single_chain(
    ref_model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    num_draws=100,
    num_burnin_steps=0,
    num_steps_bw_draws=1,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict] = None,
    chain: int = 0,
    seed: Optional[int] = None,
    pbar: bool = False,
    verbose=True,
    device: torch.device = torch.device("cpu"),
    return_weights=False,
    restrict_to_orth_grad: bool = False,
):
    # Initialize new model and optimizer for this chain
    model = deepcopy(ref_model).to(device)

    optimizer_kwargs = optimizer_kwargs or {}
    if sampling_method == SGLD:
        optimizer_kwargs["restrict_to_orth_grad"] = True
    optimizer = sampling_method(model.parameters(), **optimizer_kwargs)

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps

    local_draws = pd.DataFrame(
        index=range(num_draws),
        columns=["chain", "step", "loss"] + (["model_weights"] if return_weights else []),
    )

    iterator = zip(range(num_steps), itertools.cycle(loader))

    if pbar:
        iterator = tqdm(
            iterator, desc=f"Chain {chain}", total=num_steps, disable=not verbose  # TODO: Redundant
        )
    if restrict_to_orth_grad:
        w_0 = nn.utils.parameters_to_vector(model.parameters()).detach().clone().to(device)
        model.train()
        entire_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            entire_loss += loss
        entire_loss /= len(loader)
        gradients = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
        loss_grad_w0 = (
            nn.utils.parameters_to_vector(gradients).detach().clone().to(device)
        )
        # # ce_loss_grad_w0 *= submodule_param_mask
        loss_grad_w0 /= loss_grad_w0.norm(p=2)
        optimizer.zero_grad()
    else:
        model.train()
    for i, (xs, ys) in iterator:
        optimizer.zero_grad()
        xs, ys = xs.to(device), ys.to(device)
        y_preds = model(xs)
        loss = criterion(y_preds, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if restrict_to_orth_grad:
            w_1 = nn.utils.parameters_to_vector(model.parameters())
            # diff = w_1 - w_0
            # proj_diff = diff - torch.dot(diff, loss_grad_w0) * loss_grad_w0
            # w_1 = w_0 + proj_diff
            # make sure the gradient is orthogonal to the previous gradient
            w_1 -= torch.dot(w_1 - w_0, loss_grad_w0) * loss_grad_w0
            nn.utils.vector_to_parameters(w_1, model.parameters())

        if i >= num_burnin_steps and (i - num_burnin_steps) % num_steps_bw_draws == 0:
            draw_idx = (i - num_burnin_steps) // num_steps_bw_draws
            local_draws.loc[draw_idx, "step"] = i
            local_draws.loc[draw_idx, "chain"] = chain
            local_draws.loc[draw_idx, "loss"] = loss.detach().item()
            if return_weights:
                local_draws.loc[draw_idx, "model_weights"] = (
                    model.state_dict()["weights"].clone().detach()
                )

    return local_draws


def _sample_single_chain(kwargs):
    return sample_single_chain(**kwargs)


def sample(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Callable,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    cores: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    pbar: bool = True,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
    return_weights: bool = False,
    restrict_to_orth_grad: bool = False,
):
    """
    Sample model weights using a given optimizer, supporting multiple chains.

    Parameters:
        model (torch.nn.Module): The neural network model.
        step (Literal['sgld']): The name of the optimizer to use to step.
        loader (DataLoader): DataLoader for input data.
        criterion (torch.nn.Module): Loss function.
        num_draws (int): Number of samples to draw.
        num_chains (int): Number of chains to run.
        num_burnin_steps (int): Number of burn-in steps before sampling.
        num_steps_bw_draws (int): Number of steps between each draw.
        cores (Optional[int]): Number of cores for parallel execution.
        seed (Optional[Union[int, List[int]]]): Random seed(s) for sampling.
        progressbar (bool): Whether to display a progress bar.
        optimizer_kwargs (Optional[Dict[str, Union[float, Literal['adaptive']]]]): Keyword arguments for the optimizer.
    """
    if cores is None:
        cores = min(4, cpu_count())

    if seed is not None:
        if isinstance(seed, int):
            seeds = [seed + i for i in range(num_chains)]
        elif len(seed) != num_chains:
            raise ValueError("Length of seed list must match number of chains")
        else:
            seeds = seed
    else:
        seeds = [None] * num_chains

    def get_args(i):
        return dict(
            chain=i,
            seed=seeds[i],
            ref_model=model,
            loader=loader,
            criterion=criterion,
            num_draws=num_draws,
            num_burnin_steps=num_burnin_steps,
            num_steps_bw_draws=num_steps_bw_draws,
            sampling_method=sampling_method,
            optimizer_kwargs=optimizer_kwargs,
            pbar=pbar,
            device=device,
            verbose=verbose,
            return_weights=return_weights,
            restrict_to_orth_grad=restrict_to_orth_grad,
        )

    results = []

    if cores > 1:
        ctx = get_context("spawn")
        with ctx.Pool(cores) as pool:
            results = pool.map(_sample_single_chain, [get_args(i) for i in range(num_chains)])
    else:
        for i in range(num_chains):
            results.append(_sample_single_chain(get_args(i)))

    results_df = pd.concat(results, ignore_index=True)
    return results_df


def estimate_rlct(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Callable,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    cores: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    pbar: bool = True,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> float:
    warnings.warn(
        "estimate_rlct is deprecated. Use `devinterp.slt.estimate_learning_coeff` instead."
    )
    trace = sample(
        model=model,
        loader=loader,
        criterion=criterion,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        num_draws=num_draws,
        num_chains=num_chains,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        cores=cores,
        seed=seed,
        pbar=pbar,
        device=device,
        verbose=verbose,
    )

    baseline_loss = trace.loc[trace["chain"] == 0, "loss"].iloc[0]
    avg_loss = trace.groupby("chain")["loss"].mean().mean()
    num_samples = len(loader.dataset)

    return (avg_loss - baseline_loss) * num_samples / np.log(num_samples)
