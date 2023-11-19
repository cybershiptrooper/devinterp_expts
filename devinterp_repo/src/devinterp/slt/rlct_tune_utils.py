from devinterp.slt import estimate_learning_coeff, estimate_learning_coeff_with_summary
from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
import torch.nn as nn
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import Union, Literal

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# define dataclass for hyperparameters
@dataclass
class ChainConfig:
    batch_size: int = 512
    num_chains: int = 1
    num_draws: int = 400
    num_burnin_steps: int = 0
    num_steps_bw_draws: int = 1
    device: str = "cpu"
    verbose: bool = False
    restrict_to_orth_grad: bool = False
    criterion: nn.Module = nn.CrossEntropyLoss()
    
    def get_parameters(self):
        # return parameters for SLT
        return { 
            "num_chains": self.num_chains,
            "num_draws": self.num_draws,
            "num_burnin_steps": self.num_burnin_steps,
            "num_steps_bw_draws": self.num_steps_bw_draws,
            "device": self.device,
            "verbose": self.verbose,
            "restrict_to_orth_grad": self.restrict_to_orth_grad,
            "criterion": self.criterion,
        }

@dataclass
class OptimConfig():
    lr: float = 1e-5
    num_samples: int = 100
    bounding_box_size: float = None
    
    def get_parameters(self):
        return {
            "lr": self.lr,
            "num_samples": self.num_samples,
            "bounding_box_size": self.bounding_box_size,
        }
    def get_optimizer(self):
        raise NotImplementedError

@dataclass
class SGLDConfig(OptimConfig):
    elasticity: float = 100
    noise_level: float = 1.0
    temperature: Union[Literal["adaptive"], float] = "adaptive",
    def get_parameters(self):
        # return super parameters and parameters for SGLD
        return {
            ** super().get_parameters(),
            "elasticity": self.elasticity,
            "noise_level": self.noise_level,
            "temperature": self.temperature,
        }
    def get_optimizer(self):
        return SGLD

@dataclass
class SGNHTConfig(OptimConfig):
    diffusion_factor: float = 0.01

    def get_parameters(self):
        # return super parameters and parameters for SGLD
        return {
            ** super().get_parameters(),
            "diffusion_factor": self.diffusion_factor,
        }
    
    def get_optimizer(self):
        return SGNHT
    
def rlct_estimate(model, chain_config, optim_config):
    train_data = datasets.MNIST("../data", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_data, batch_size=chain_config.batch_size, shuffle=True)
    return estimate_learning_coeff(
        model, 
        train_loader,
        sampling_method=optim_config.get_optimizer(),
        ** chain_config.get_parameters(),
        optimizer_kwargs=optim_config.get_parameters(),
    )

def rlct_estimate_with_summary(model, chain_config, optim_config, loader=None):
    if loader is None:
        train_data = datasets.MNIST("../data", train=True, transform=transforms.ToTensor(), download=True)
        train_loader = DataLoader(train_data, batch_size=chain_config.batch_size, shuffle=True)
    else:
        train_loader = loader
    return estimate_learning_coeff_with_summary(
        model, 
        train_loader,
        sampling_method=optim_config.get_optimizer(),
        ** chain_config.get_parameters(),
        optimizer_kwargs=optim_config.get_parameters(),
    )


def get_summary_for_hyperparams(model, args_list, device="cpu", loader=None, criterion=None):
    print(args_list)
    if loader is not None:
        print("Using custom loader, len: ", len(loader.dataset))
    chain_config = ChainConfig(
        num_chains=args_list['num_chains'],
        num_draws=args_list['num_draws'],
        num_burnin_steps=0,
        num_steps_bw_draws=1,
        device=device,
        verbose=False,
        restrict_to_orth_grad=args_list['restrict_to_orth_grad'],
        criterion=criterion if criterion is not None else nn.CrossEntropyLoss(),
    )
    if args_list['optimizer'] == SGLD:
        optim_config = SGLDConfig(
            lr=args_list['lr'],
            noise_level=args_list['noise_level'],
            elasticity=args_list['elasticity'],
            num_samples=args_list['num_samples'],
            temperature=args_list['temperature'],
        )
    elif args_list['optimizer'] == SGNHT:
        optim_config = SGNHTConfig(
            lr=args_list['lr'],
            diffusion_factor=args_list['diffusion_factor'],
            num_samples=args_list['num_samples'],
        )

    rlct_estimate_summary = rlct_estimate_with_summary(model, chain_config, optim_config, loader=loader)
    return rlct_estimate_summary

def get_args_for_sgnht(lr_list, diffusion_factor_list, 
                       restrict_to_orth_grad_list, num_samples_list, 
                      num_draws_list, num_chains, **kwargs):
    return [
        {
            "lr": lr,
            "optimizer": SGNHT,
            "diffusion_factor": diffusion_factor,
            "num_draws": num_draw,
            "restrict_to_orth_grad": restrict_to_orth_grad,
            "num_samples": num_sample,
            "num_chains": num_chain,
        }
        for lr in lr_list
        for diffusion_factor in diffusion_factor_list
        for restrict_to_orth_grad in restrict_to_orth_grad_list
        for num_draw in (num_draws_list)
        for num_sample in num_samples_list
        for num_chain in (num_chains)
    ]

def get_args_for_sgld(lr_list, noise_level_list, elasticity_list, temperature_list, 
                      restrict_to_orth_grad_list, num_samples_list, 
                      num_draws_list, num_chains, **kwargs):
    return [
        {
            "lr": lr,
            "optimizer": SGLD,
            "noise_level": noise_level,
            "elasticity": elasticity,
            "temperature": temperature,
            "num_draws": num_draw,
            "num_samples": num_sample,
            "restrict_to_orth_grad": restrict_to_orth_grad,
            "num_chains": num_chain,
        }
        for lr in lr_list
        for noise_level in noise_level_list
        for elasticity in elasticity_list
        for temperature in temperature_list
        for restrict_to_orth_grad in restrict_to_orth_grad_list
        for num_draw in num_draws_list
        for num_chain in (num_chains)
        for num_sample in num_samples_list
    ]

def make_args_list(
        lr_list=[1e-5], optimizer_list=[SGNHT], 
        diffusion_factor_list=[0.1], 
        noise_level_list=[1.0], 
        elasticity_list=[100.0], 
        temperature_list=["adaptive"], 
        num_samples=[60000],
        restrict_to_orth_grad_list=[False], num_draws=[400], num_chains=[1]):
    optimizer_to_args_func = {
        SGLD: get_args_for_sgld,
        SGNHT: get_args_for_sgnht
    }
    args_list = []
    for optimizer in optimizer_list:
        args_list.extend(optimizer_to_args_func[optimizer](
            lr_list=lr_list,
            noise_level_list=noise_level_list,
            elasticity_list=elasticity_list,
            temperature_list=temperature_list,
            restrict_to_orth_grad_list=restrict_to_orth_grad_list,
            num_samples_list=num_samples,
            num_draws_list=num_draws,
            num_chains=num_chains,
            diffusion_factor_list=diffusion_factor_list,
        ))
    return args_list
    
