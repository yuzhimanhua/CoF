"""Implementation of customized optimizer."""

import collections
import math
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version


def get_layer_lrs(
    layer_decay: float,
    n_layers: int,
    offset: Optional[int]=2,
    )-> Dict:
    """Gets a dict mapping from variable name to layerwise learning rate."""
    # TODO: This only supports BERT like models.
    key_to_depths = collections.OrderedDict({
        "embeddings": 0,
        "cls": n_layers + offset,
        "encoder.pooler": n_layers + offset,
        "encode_proj": n_layers + offset,
        "query_instruct_proj": n_layers + offset,
        "discriminator_predictions": n_layers + offset,
        "encoder.expert_gate.fwd": n_layers // 2 + offset,
    })
    for layer in range(n_layers):
        key_to_depths[f"encoder.layer.{layer}"] = layer + 1

        # TODO: Makes this configurable.
        key_to_depths[f"encoder.expert_gate.{layer}"] = layer + 1

    return {
        key: layer_decay ** (n_layers + offset - depth)
        for key, depth in key_to_depths.items()
    }


def get_layer_lrs_for_t5(
    layer_decay: float,
    n_layers: int,
    offset: Optional[int]=2,
    )-> Dict:
    """Gets a dict mapping from variable name to layerwise learning rate."""
    # TODO: This only supports BERT like models.
    key_to_depths = collections.OrderedDict({
        "shared.weight": 0,
        "encode_proj": n_layers + offset,
        "encoder.final_layer_norm.weight": n_layers + offset,
        "encoder.pooler": n_layers + offset,
    })
    for layer in range(n_layers):
        key_to_depths[f"encoder.block.{layer}"] = layer + 1

    return {
        key: layer_decay ** (n_layers + offset - depth)
        for key, depth in key_to_depths.items()
    }


class AdamWLayer(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`__.

    Parameters:
        params (:obj:`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0[")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0[")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"] * group["lr_adapt_weight"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["lr_adapt_weight"] * group["weight_decay"]))

        return loss