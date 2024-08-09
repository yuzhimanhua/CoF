#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Command line arguments utils
"""


import logging
import numpy as np
import os
import random
import socket
import torch

from omegaconf import DictConfig

logger = logging.getLogger()

# TODO: to be merged with conf_utils.py


def set_cfg_params_from_state(state: dict, cfg: DictConfig, resume_train=True):
    """
    Overrides some of the encoder config parameters from a give state object
    """
    if not state:
        return

    if not resume_train:
        logger.info("Not loading encoder params from the model_file!")
        return
        
    cfg.do_lower_case = state["do_lower_case"]
    cfg.encoder.pretrained_model_cfg = state["pretrained_model_cfg"]
    cfg.encoder.encoder_model_type = state["encoder_model_type"]
    cfg.encoder.pretrained_file = state["pretrained_file"]
    cfg.encoder.projection_dim = state["projection_dim"]
    # cfg.encoder.sequence_length = state["sequence_length"]

    # Add more encoder parameters.
    optional_keys = [
        "shared_encoder",
        "use_moe",
        "moe_type",
        "num_expert",
        "use_infer_expert",
        "per_layer_gating",
        "q_rep_method",
        "mean_pool",
        "use_norm_rep",
        "factor_rep",
        "sequence_length",
        "use_attn_gate",
        "sep_instruct",
        "instruct_type",
        "proj_adaptor",
        "fix_instruct_encoder",
        "deep_instruct_fusion",
    ]

    # If resumes training, loads all encoder params to keep it consistent.
    logger.info("Resumes training, loading extra encoder params")
    loads_params = []
    for okey in optional_keys:
        if okey in state:
            loads_params.append(okey)
            if state[okey] != cfg.encoder[okey]:
                logger.warning(f"{okey} is different for the saved checkpoint and current configured run!")
            cfg.encoder[okey] = state[okey]
        else:
            logger.warning(f"{okey} is not found in the saved checkpoint!")
    
    logger.info(loads_params)


def get_encoder_params_state_from_cfg(cfg: DictConfig):
    """
    Selects the param values to be saved in a checkpoint, so that a trained model can be used for downstream
    tasks without the need to specify these parameter again
    :return: Dict of params to memorize in a checkpoint
    """
    return {
        "do_lower_case": cfg.do_lower_case,
        "pretrained_model_cfg": cfg.encoder.pretrained_model_cfg,
        "encoder_model_type": cfg.encoder.encoder_model_type,
        "pretrained_file": cfg.encoder.pretrained_file,
        "projection_dim": cfg.encoder.projection_dim,
        "sequence_length": cfg.encoder.sequence_length,

        # Adds more parameters for better checkpoint loading.
        "shared_encoder": cfg.encoder.shared_encoder,

        "use_moe": cfg.encoder.use_moe,
        "moe_type": cfg.encoder.moe_type,
        "num_expert": cfg.encoder.num_expert,
        "use_infer_expert": cfg.encoder.use_infer_expert,
        "per_layer_gating": cfg.encoder.per_layer_gating,

        "q_rep_method": cfg.encoder.q_rep_method,
        "mean_pool": cfg.encoder.mean_pool,
        "use_norm_rep": cfg.encoder.use_norm_rep,
        "factor_rep": cfg.encoder.factor_rep,
        "use_attn_gate": cfg.encoder.use_attn_gate,
        "sep_instruct": cfg.encoder.sep_instruct,
        "instruct_type": cfg.encoder.instruct_type,
        "proj_adaptor": cfg.encoder.proj_adaptor,
        "fix_instruct_encoder": cfg.encoder.fix_instruct_encoder,
        "deep_instruct_fusion": cfg.encoder.deep_instruct_fusion,
    }


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_cfg_gpu(cfg):
    """
    Setup params for CUDA, GPU & distributed training
    """
    logger.info("args.local_rank %s", cfg.local_rank)
    ws = os.environ.get("WORLD_SIZE")
    cfg.distributed_world_size = int(ws) if ws else 1
    logger.info("WORLD_SIZE %s", ws)
    if cfg.local_rank == -1 or cfg.no_cuda:  # single-node multi-gpu (or cpu) mode
        device = str(
            torch.device(
                "cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"
            )
        )
        cfg.n_gpu = torch.cuda.device_count()
    else:  # distributed mode
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        torch.distributed.init_process_group(backend="nccl", world_size=int(ws))
        cfg.n_gpu = 1

    cfg.device = device

    logger.info(
        "Initialized host %s as d.rank %d on device=%s, n_gpu=%d, world size=%d",
        socket.gethostname(),
        cfg.local_rank,
        cfg.device,
        cfg.n_gpu,
        cfg.distributed_world_size,
    )
    logger.info("16-bits training: %s ", cfg.fp16)
    return cfg


def setup_logger(logger, log_level=logging.INFO):
    logger.setLevel(log_level)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter(
        "[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)
