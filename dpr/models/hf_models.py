#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

from dataclasses import dataclass
import logging
from typing import OrderedDict, Tuple

import math
import torch
from torch import Tensor as T
from torch import nn
from transformers import BertConfig, BertTokenizer, BertModel, BertTokenizerFast
from transformers import AutoModel, AutoConfig
from transformers import ElectraConfig, ElectraModel, ElectraTokenizerFast
from transformers import T5Config, T5EncoderModel, T5TokenizerFast
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.file_utils import ModelOutput
from transformers.optimization import AdamW, Adafactor

from dpr.models.biencoder import BiEncoder, MoEBiEncoder
from dpr.models.biencoder_uni import BiEncoderUni, MoEBiEncoderUni, InstructBiEncoderUni
from dpr.models.adv_biencoder import AdvBiEncoder
from dpr.utils.data_utils import Tensorizer
from .reader import Reader
from dpr.utils.model_utils import load_states_from_checkpoint, get_model_obj
from dpr.models.robust_bert import AdvBertModel
from dpr.models.robust_electra import AdvElectraModel
from dpr.models.robust_t5enc import AdvT5EncoderModel
from dpr.models.robust_roberta import AdvRobertaModel
from dpr.models.moe_models import MoEBertModel
from dpr.models.instruct_bert import InstructBertModel
from dpr.models.optimization import get_layer_lrs, get_layer_lrs_for_t5, AdamWLayer

logger = logging.getLogger(__name__)

model_mapping = {
    'bert': (BertConfig, BertTokenizerFast, BertModel),
    'microsoft/BiomedNLP': (BertConfig, BertTokenizerFast, BertModel),
    'microsoft/mpnet': (AutoConfig, BertTokenizerFast, AutoModel),
    'sentence-transformers/all-mpnet-base': (AutoConfig, BertTokenizerFast, AutoModel),
    'allenai/scibert_scivocab_uncased': (BertConfig, BertTokenizerFast, BertModel),
    'Luyu/co': (BertConfig, BertTokenizerFast, BertModel),
    'michiyasunaga/LinkBERT': (BertConfig, BertTokenizerFast, BertModel),
    'facebook/contriever': (BertConfig, BertTokenizerFast, BertModel),
    'roberta': (RobertaConfig, RobertaTokenizer, RobertaModel),
    'google/electra': (ElectraConfig, ElectraTokenizerFast, ElectraModel),
    't5': (T5Config, T5TokenizerFast, T5EncoderModel),
    'google/flan': (T5Config, T5TokenizerFast, T5EncoderModel),
    'google/t5': (T5Config, T5TokenizerFast, T5EncoderModel),
    'sentence-transformers/gtr-t5': (T5Config, T5TokenizerFast, T5EncoderModel),
}

moe_model_mapping = {
    'facebook/contriever': (BertConfig, BertTokenizerFast, (MoEBertModel, BertModel)),
    'allenai/scibert_scivocab_uncased': (BertConfig, BertTokenizerFast, (MoEBertModel, BertModel)),
    'bert': (BertConfig, BertTokenizerFast, (MoEBertModel, BertModel)),
    'microsoft/BiomedNLP': (BertConfig, BertTokenizerFast, (MoEBertModel, BertModel)),
    'Luyu/co': (BertConfig, BertTokenizerFast, (MoEBertModel, BertModel)),
    'michiyasunaga/LinkBERT': (BertConfig, BertTokenizerFast, BertModel),
}

instruct_model_mapping = {
    'bert': (BertConfig, BertTokenizerFast, (InstructBertModel, BertModel)),
    'microsoft/BiomedNLP': (BertConfig, BertTokenizerFast, (InstructBertModel, BertModel)),
}


adv_model_mapping = {
    'bert': (BertConfig, BertTokenizerFast, AdvBertModel),
    'Luyu/co': (BertConfig, BertTokenizerFast, AdvBertModel),
    'michiyasunaga/LinkBERT': (BertConfig, BertTokenizerFast, BertModel),
    'roberta': (RobertaConfig, RobertaTokenizer, AdvRobertaModel),
    'google/electra': (ElectraConfig, ElectraTokenizerFast, AdvElectraModel),
    't5': (T5Config, T5TokenizerFast, AdvT5EncoderModel),
}


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    question_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )
    ctx_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    fix_ctx_encoder = cfg.fix_ctx_encoder if hasattr(cfg, "fix_ctx_encoder") else False

    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    )

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, biencoder, optimizer


def get_bert_reader_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = (
        get_optimizer(
            reader,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, reader, optimizer


def get_any_biencoder_components(cfg, use_instruct=False, inference_only: bool = False, use_uni_biencoder: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    use_vat = False
    # TODO: fix this.
    use_moe = cfg.encoder.use_moe
    num_expert = 0
    num_q_expert = None
    num_ctx_expert = None

    # TODO: Makes this part of model parameters.
    freeze_backbone = cfg.freeze_backbone if hasattr(cfg, "freeze_backbone") else False
    if use_moe:
        if not hasattr(cfg.encoder, "num_expert") or cfg.encoder.num_expert == -1:
            raise ValueError("When use_moe=True, num_expert is required")
        num_expert = cfg.encoder.num_expert
        use_infer_expert = cfg.encoder.use_infer_expert
        per_layer_gating = cfg.encoder.per_layer_gating
        moe_type = cfg.encoder.moe_type
        if hasattr(cfg.encoder, "num_q_expert"):
            num_q_expert = cfg.encoder.num_q_expert
            num_ctx_expert = num_expert - num_q_expert
        else:
            num_q_expert = cfg.encoder.num_expert
            num_ctx_expert = cfg.encoder.num_expert
    else:
        num_expert = 1
        use_infer_expert = False
        per_layer_gating = False
        moe_type = "mod3"

    if hasattr(cfg, "train") and hasattr(cfg.train, "use_vat"):
        use_vat = cfg.train.use_vat

    mean_pool_q_encoder = False
    if cfg.encoder.mean_pool:
        mean_pool_q_encoder = True
        if cfg.encoder.mean_pool_ctx_only:
            mean_pool_q_encoder = False

    factor_rep = cfg.encoder.factor_rep

    question_encoder = HFEncoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        use_vat=use_vat,
        use_moe=use_moe,
        moe_type=moe_type,
        use_infer_expert=use_infer_expert,
        per_layer_gating=per_layer_gating,
        num_expert=num_expert if cfg.encoder.shared_encoder else num_q_expert,
        use_instruct=use_instruct,
        mean_pool=mean_pool_q_encoder,
        factor_rep=factor_rep,
        pretrained=cfg.encoder.pretrained,
        use_norm_rep=cfg.encoder.use_norm_rep,
        task_header_type=cfg.encoder.task_header_type,
        use_attn_gate=cfg.encoder.use_attn_gate,
        instruct_type=cfg.encoder.instruct_type,
        proj_adaptor=cfg.encoder.proj_adaptor,
        **kwargs
    )


    fix_instruct_enocder = cfg.encoder.fix_instruct_encoder if hasattr(cfg.encoder, "fix_instruct_encoder") else False

    if cfg.encoder.shared_encoder:
        ctx_encoder = question_encoder
    else:
        ctx_encoder = HFEncoder(
            cfg.encoder.pretrained_model_cfg,
            projection_dim=cfg.encoder.projection_dim,
            use_vat=use_vat,
            use_moe=use_moe,
            moe_type=moe_type,
            use_infer_expert=use_infer_expert,
            per_layer_gating=per_layer_gating,
            num_expert=num_ctx_expert if num_ctx_expert else num_expert,
            use_instruct=use_instruct,
            dropout=dropout,
            mean_pool=cfg.encoder.mean_pool,
            factor_rep=factor_rep,
            pretrained=cfg.encoder.pretrained,
            use_norm_rep=cfg.encoder.use_norm_rep,
            task_header_type=cfg.encoder.task_header_type,
            use_attn_gate=cfg.encoder.use_attn_gate,
            instruct_type=cfg.encoder.instruct_type,
            proj_adaptor=cfg.encoder.proj_adaptor,
            **kwargs
        )

    if use_instruct:
        if cfg.encoder.shared_encoder and not fix_instruct_enocder:
            instruct_encoder = question_encoder
        else:
            instruct_encoder = HFEncoder(
                cfg.encoder.pretrained_model_cfg,
                projection_dim=cfg.encoder.projection_dim,
                use_vat=use_vat,
                use_moe=use_moe,
                moe_type=moe_type,
                use_instruct=False,
                use_infer_expert=use_infer_expert,
                per_layer_gating=per_layer_gating,
                num_expert=num_ctx_expert if num_ctx_expert else num_expert,
                dropout=dropout,
                mean_pool=cfg.encoder.mean_pool,
                factor_rep=factor_rep,
                pretrained=cfg.encoder.pretrained,
                use_norm_rep=cfg.encoder.use_norm_rep,
                task_header_type=cfg.encoder.task_header_type,
                **kwargs
            )

    fix_ctx_encoder = cfg.fix_ctx_encoder if hasattr(cfg, "fix_ctx_encoder") else False

    if use_uni_biencoder:
        if not hasattr(cfg.encoder, "q_rep_method"):
            raise ValueError("q_rep_method is not configured!")
        if use_moe or use_vat:
            biencoder = MoEBiEncoderUni(
                question_encoder, ctx_encoder,
                fix_ctx_encoder=fix_ctx_encoder,
                num_expert=num_expert if use_moe else 0,
                num_q_expert=num_q_expert if use_moe else None,
                q_rep_method=cfg.encoder.q_rep_method,
                do_span=(cfg.encoder.span_proposal_prob > 0.0),
            )
        elif use_instruct:
            biencoder = InstructBiEncoderUni(
                question_model=question_encoder,
                ctx_model=ctx_encoder,
                instruct_model=instruct_encoder,
                fix_ctx_encoder=fix_ctx_encoder,
                fix_instruct_encoder=fix_instruct_enocder,
                q_rep_method=cfg.encoder.q_rep_method,
                deep_instruct_fusion=cfg.encoder.deep_instruct_fusion,
            )
        else:
            biencoder = BiEncoderUni(
                question_encoder, ctx_encoder,
                fix_ctx_encoder=fix_ctx_encoder,
                q_rep_method=cfg.encoder.q_rep_method,
            )

    elif use_moe:
        if num_q_expert is not None:
            offet_expert_id = True
        else:
            offset_expert_id = False
        biencoder = MoEBiEncoder(
            question_encoder,
            ctx_encoder,
            fix_ctx_encoder=fix_ctx_encoder,
            num_expert=num_expert,
            num_q_expert=num_q_expert,
            offset_expert_id=offset_expert_id,
        )

    elif use_vat:
        biencoder = AdvBiEncoder(
            question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
        )
    else:
        biencoder = BiEncoder(
            question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
        )

    if cfg.encoder.pretrained_file:
        checkpoint = load_states_from_checkpoint(cfg.encoder.pretrained_file)
        model_to_load = get_model_obj(biencoder)
        model_to_load.load_state(checkpoint)

    if "base" in cfg.encoder.pretrained_model_cfg:
        n_layers = 12
    elif "large" in cfg.encoder.pretrained_model_cfg:
        n_layers = 24
    elif "condenser" in cfg.encoder.pretrained_model_cfg:
        n_layers = 12
    elif "contriever" in cfg.encoder.pretrained_model_cfg:
        n_layers = 12
    elif "scibert" in cfg.encoder.pretrained_model_cfg:
        n_layers = 12
    else:
        raise ValueError("Unknown nlayers for %s" % cfg.encoder.pretrained_model_cfg)

    if not inference_only and hasattr(cfg.train, "moe_factor"):
        moe_factor = cfg.train.moe_factor
    else:
        moe_factor = 1.0
    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            adam_betas=cfg.train.adam_betas,
            weight_decay=cfg.train.weight_decay,
            use_layer_lr=cfg.train.use_layer_lr,
            n_layers=n_layers,
            layer_decay=cfg.train.layer_decay,
            opt_name=cfg.train.opt_name if hasattr(cfg.train, "opt_name") else "adam",
            use_t5=("t5" in cfg.encoder.pretrained_model_cfg),
            moe_factor=moe_factor,
        )
        if not inference_only
        else None
    )

    tensorizer = get_any_tensorizer(cfg)
    return tensorizer, biencoder, optimizer

def get_any_reader_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    encoder = HFEncoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    hidden_size = encoder.encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = (
        get_optimizer(
            reader,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_any_tensorizer(cfg)
    return tensorizer, reader, optimizer

def get_any_tensorizer(cfg, tokenizer=None):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg

    if not tokenizer:
        tokenizer = get_any_tokenizer(
            pretrained_model_cfg, do_lower_case=cfg.do_lower_case
        )
        if cfg.special_tokens:
            _add_special_tokens(tokenizer, cfg.special_tokens)

    # return BertTensorizer(tokenizer, sequence_length)   # this should be fine
    return HFTensorizer(tokenizer, sequence_length)   # this should be fine


def get_any_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    model_name = pretrained_cfg_name.split('-')[0]
    if "sentence-transformer" in pretrained_cfg_name:
        model_name = "-".join(pretrained_cfg_name.split('-')[:-1])
    else:
        model_name = pretrained_cfg_name.split('-')[0]

    tokenizer_class = model_mapping[model_name][1]
    return tokenizer_class.from_pretrained(pretrained_cfg_name)


class SimHeader(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, eps: float = 1e-6):
        super().__init__()
        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_(mean=0.0, std=0.02)
        self.project = nn.Sequential(linear, nn.LayerNorm(out_dim))
        self.out_features = out_dim

    def forward(self, inputs):
        return (self.project(inputs),)

        
class AttentionHeader(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int = 1,
                 dropout_prob: float = 0.0,
                 position_embedding_type: str = "relative_key",
                 max_position_embeddings: int = 128,
                 ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.position_embedding_type = position_embedding_type
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * max_position_embeddings - 1, self.attention_head_size)

        self.out_features = hidden_size

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            distance[0, :] = 0
            distance[:, 0] = 0
            distance = torch.min(distance, torch.full_like(distance, self.max_position_embeddings - 1)) + self.max_position_embeddings - 1
            distance = torch.max(distance, torch.zeros_like(distance))
            positional_embedding = self.distance_embedding(distance)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


import regex
def init_moe_from_pretrained_mapping(pretrained_sd, moe_sd,
                                     moe_layer_name="moe_layer"):
    state_dict = {}
    missing_vars = []
    pattern_list = [
       (f"{moe_layer_name}.", ""),
       (r"interm_layers.\d+", "intermediate"),
       (r"output_layers.\d+", "output"),
       (r"moe_query.\d+", "query"),
       (r"moe_key.\d+", "key"),
       (r"moe_value.\d+", "value"),
       (r"moe_dense.\d+", "dense"),
    ]

    def normalize_var_name(var_name):
        for ptn in pattern_list:
            var_name = regex.sub(ptn[0], ptn[1], var_name)
        return var_name

    moe_print = True
    prev_layer = -1
    for var_name in moe_sd:
        if moe_layer_name in var_name or "moe" in var_name:
            pretrained_var_name = normalize_var_name(var_name)
            if moe_print:
                cur_layer = int(var_name.split("encoder.layer.")[1].split(".")[0])
                if prev_layer >= 0 and prev_layer != cur_layer:
                    moe_print = False
                else:
                    prev_layer = cur_layer
        else:
            pretrained_var_name = var_name

        if pretrained_var_name in pretrained_sd:
            state_dict[var_name] = pretrained_sd[pretrained_var_name]
        else:
            missing_vars.append((var_name, pretrained_var_name))

    again_missing_vars = []
    for var_name, _ in missing_vars:
        if "expert_gate" in var_name:
            state_dict[var_name] = moe_sd[var_name]
        else:
            again_missing_vars.append(var_name)

    return state_dict

    
def init_instruct_from_pretrained_mapping(pretrained_sd, target_sd,
                                          target_layer_names=["self.gate", "self.adaptor"]):
    state_dict = {}
    missing_vars = []

    def is_target_var(var_name):
        for tgt_var_name in target_layer_names:
            if tgt_var_name in var_name:
                return True
        return False

    for var_name in target_sd:
        pretrained_var_name = var_name
        if pretrained_var_name in pretrained_sd:
            state_dict[var_name] = pretrained_sd[pretrained_var_name]
        elif is_target_var(var_name):
            state_dict[var_name] = target_sd[var_name]
        else:
            missing_vars.append(var_name)

    return state_dict


@dataclass
class HFEncoderOutput(ModelOutput):
   sequence_output: torch.FloatTensor = None
   pooled_output: torch.FloatTensor = None
   hidden_states: torch.FloatTensor = None
   total_entropy_loss: torch.FloatTensor = None


class HFEncoder(nn.Module):
    def __init__(self,
                 cfg_name: str,
                 use_vat: bool = False,
                 use_moe: bool = False,
                 moe_type: str = "mod3",
                 num_expert: int = 0,
                 use_infer_expert: bool = False,
                 per_layer_gating: bool = False,
                 use_instruct: bool = False,
                 task_header_type: str = "attn:1",
                 projection_dim: int = 0,
                 dropout: float = 0.1,
                 pretrained: bool = True,
                 seq_rep_method: str = "cls",
                 mean_pool: bool = False,
                 factor_rep: bool = False,
                 use_norm_rep: bool = False,
                 use_attn_gate: bool = False,
                 instruct_type: str = "all",
                 proj_adaptor: bool = False,
                 **kwargs):
        super().__init__()
        if "sentence-transformer" in cfg_name:
            model_name = "-".join(cfg_name.split('-')[:-1])
        else:
            model_name = cfg_name.split('-')[0]
        if use_moe:
            config_class, _, model_class = moe_model_mapping[model_name]
            assert num_expert > 0, "num_expert can't be zero when using MoE."
        elif use_vat:
            config_class, _, model_class = adv_model_mapping[model_name]
        elif use_instruct:
            config_class, _, model_class = instruct_model_mapping[model_name]
        else:
            config_class, _, model_class = model_mapping[model_name]
        cfg = config_class.from_pretrained(cfg_name)
        self.num_expert = cfg.num_expert = num_expert
        self.use_infer_expert = cfg.use_infer_expert = use_infer_expert
        self.per_layer_gating = cfg.per_layer_gating = per_layer_gating
        self.moe_type = cfg.moe_type = moe_type
        self.use_attn_gate = cfg.use_attn_gate = use_attn_gate
        self.instruct_type = cfg.instruct_type = instruct_type
        self.proj_adaptor = cfg.proj_adaptor = proj_adaptor

        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        assert cfg.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.mean_pool = mean_pool
        self.factor_rep = factor_rep
        self.use_moe = use_moe
        self.use_norm_rep = use_norm_rep

        self.seq_rep_method = seq_rep_method

        self.encode_proj = None
        if projection_dim != 0:
            if projection_dim == -1:
                output_dim = cfg.hidden_size
                # self.encode_proj = SimHeader(cfg.hidden_size, cfg.hidden_size, dropout=dropout)
            else:
                output_dim = projection_dim
            
            header_type, nheader = task_header_type.split(":")
            nheader = int(nheader)

            if header_type == "linear":
                self.encode_proj = nn.ModuleList(
                    [SimHeader(cfg.hidden_size, output_dim, dropout=dropout) for _ in range(nheader)])
            elif header_type == "attn":
                self.encode_proj = nn.ModuleList(
                    [AttentionHeader(cfg.hidden_size, dropout_prob=dropout) for _ in range(nheader)])
            else:
                raise ValueError(f"Unknown header {task_header_type}")

        self.use_instruct = False
        if use_moe:
            model_class, orig_model = model_class
            orig_encoder = orig_model.from_pretrained(cfg_name, config=cfg, **kwargs)
            self.encoder =  model_class(config=cfg)
            self.encoder.load_state_dict(
                init_moe_from_pretrained_mapping(
                    orig_encoder.state_dict(),
                    self.encoder.state_dict()))
        elif use_instruct:
            model_class, orig_model = model_class
            orig_encoder = orig_model.from_pretrained(cfg_name, config=cfg, **kwargs)
            self.encoder = model_class(config=cfg)
            self.encoder.load_state_dict(
                init_instruct_from_pretrained_mapping(
                    orig_encoder.state_dict(),
                    self.encoder.state_dict()))
            self.use_instruct = use_instruct
        else:
            self.encoder =  model_class.from_pretrained(cfg_name, config=cfg, **kwargs)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        input_embeds: T=None,
        representation_token_pos=0,
        expert_id=None,
        expert_offset: int=0,
        task_header_id: int=0,
        instruct_hidden_states: T=None,
        instruct_attention_mask: T=None,
        output_hidden_states: bool=False,
        output_attentions: bool=False,
    ) -> Tuple[T, ...]:
        if input_embeds is None:
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        else:
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "input_embeds": input_embeds,
            }
        if expert_id is not None:
            inputs["expert_id"] = expert_id

        if instruct_hidden_states is not None:
            batch_size = input_ids.size()[0]
            instruct_size = instruct_attention_mask.size()[0]
            if batch_size != instruct_size:
                if instruct_size == 1:
                    instruct_attention_mask = instruct_attention_mask.repeat((batch_size, 1))
            inputs["instruct_hidden_states"] = instruct_hidden_states
            inputs["instruct_attention_mask"] = instruct_attention_mask

        if output_hidden_states:
            inputs["output_hidden_states"] = output_hidden_states

        if output_attentions:
            inputs["output_attentions"] = output_attentions

        outputs = self.encoder(**inputs)
        sequence_output = outputs[0]
        if self.encoder.config.output_hidden_states or output_hidden_states:
            if hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states
            else:
                hidden_states = outputs[3]
        else:
            hidden_states = None

        if self.use_moe and self.use_infer_expert:
            total_entropy_loss = outputs[-1]
        else:
            total_entropy_loss = None

        if self.encode_proj:
            sequence_output = self.encode_proj[task_header_id](sequence_output)[0]

        if self.use_norm_rep:
            sequence_output = torch.nn.functional.normalize(sequence_output, p=2, dim=-1)

        if self.mean_pool:
            mask = attention_mask.unsqueeze(-1).float()
            factor = torch.sum(attention_mask, dim=1, keepdim=True).float()
            pooled_output = torch.sum(sequence_output * mask, dim=1) / factor
        elif isinstance(representation_token_pos, int) or representation_token_pos is None:
            pooled_output = sequence_output[:, 0, :]
        else: 
            raise ValueError("Unknown case for pooled_output!")

        # This is required for slicing [0,0] position for representation.
        sequence_output[:, 0, :] = pooled_output

        return sequence_output, pooled_output, hidden_states, total_entropy_loss
        # return HFEncoderOutput(
        #     sequence_output=sequence_output,
        #     pooled_output=pooled_output,
        #     hidden_states=hidden_states,
        #     total_entropy_loss=total_entropy_loss,
        # )

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.encoder.config.hidden_size


def get_bert_tensorizer(cfg, tokenizer=None):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg

    if not tokenizer:
        tokenizer = get_bert_tokenizer(
            pretrained_model_cfg, do_lower_case=cfg.do_lower_case
        )
        if cfg.special_tokens:
            _add_special_tokens(tokenizer, cfg.special_tokens)

    return BertTensorizer(tokenizer, sequence_length)


def _add_special_tokens(tokenizer, special_tokens):
    special_tokens_num = len(special_tokens)
    # TODO: this is a hack-y logic that uses some private tokenizer structure which can be changed in HF code
    assert special_tokens_num < 50
    unused_ids = [
        tokenizer.vocab["[unused{}]".format(i)] for i in range(special_tokens_num)
    ]

    for idx, id in enumerate(unused_ids):
        del tokenizer.vocab["[unused{}]".format(idx)]
        tokenizer.vocab[special_tokens[idx]] = id
        tokenizer.ids_to_tokens[id] = special_tokens[idx]

    tokenizer._additional_special_tokens = list(special_tokens)


def get_roberta_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_roberta_tokenizer(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case
        )
    return RobertaTensorizer(tokenizer, args.sequence_length)


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    use_layer_lr: bool = True,
    n_layers: int = 12,
    layer_decay: float = 0.8,
    use_t5: bool = False,
    opt_name: str = "adam",
    moe_factor: float = 1.0,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]

    def _unpack_group_params(group_params):
        params, wdecays, layer_lrs = zip(**group_params)
        return {
            "params": params,
            "weight_decay": wdecays[0],
            "layer_lrs": layer_lrs,
        }

    if opt_name == "adam" and use_layer_lr:
        if use_t5:
            name_to_adapt_lr = get_layer_lrs_for_t5(
                layer_decay=layer_decay,
                n_layers=n_layers,
            )
        else:
            name_to_adapt_lr = get_layer_lrs(
                layer_decay=layer_decay,
                n_layers=n_layers,
            )
        optimizer_grouped_parameters = []
        for name, param in model.named_parameters():
            update_for_var = False
            for key in name_to_adapt_lr:
                if key in name:
                    update_for_var = True
                    lr_adapt_weight = name_to_adapt_lr[key]

            if not update_for_var:
                # TODO: This has to be refactored.
                if "linear" in name or "span" in name:
                    lr_adapt_weight = 1.0
                else:
                    raise ValueError("No adaptive LR for %s" % name)

            wdecay = weight_decay
            if any(nd in name for nd in no_decay):
                # Parameters with no decay.
                wdecay = 0.0
            if "moe" in name:
                lr_adapt_weight *= moe_factor
            optimizer_grouped_parameters.append({
                "params": param,
                "weight_decay": wdecay,
                "lr_adapt_weight": lr_adapt_weight,
            })
        optimizer = AdamWLayer(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps, betas=adam_betas)
    elif opt_name == "adam":
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps, betas=adam_betas)
    elif opt_name == "adafactor":
        optimizer = Adafactor(
            model.parameters(),
            lr=learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )

    return optimizer


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )


def get_roberta_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return RobertaTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls,
        cfg_name: str,
        projection_dim: int = 0,
        dropout: float = 0.1,
        pretrained: bool = True,
        **kwargs
    ) -> BertModel:
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(
                cfg_name, config=cfg, project_dim=projection_dim, **kwargs
            )
        else:
            return HFBertEncoder(cfg, project_dim=projection_dim)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
    ) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert (
                representation_token_pos.size(0) == bsz
            ), "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack(
                [
                    sequence_output[i, representation_token_pos[i, 1], :]
                    for i in range(bsz)
                ]
            )

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class HFTensorizer(Tensorizer):
    def __init__(
        self, tokenizer, max_length: int, pad_to_max: bool = True
    ):
        self.tokenizer = tokenizer

        self.is_fast_tokenizer = True
        if not issubclass(tokenizer.__class__, PreTrainedTokenizerFast):
            self.is_fast_tokenizer = False

        self.max_length = max_length
        self.pad_to_max = pad_to_max
        if tokenizer.mask_token_id is None:
            self.mask_token_id = tokenizer.unk_token_id
        else:
            self.mask_token_id = tokenizer.mask_token_id

        if tokenizer.pad_token_id is None:
            raise ValueError("The given tokenizer has no pad_token_id!")
        self.pad_token_id = tokenizer.pad_token_id

        #TODO: For encoders wo/ sep token, this is problematic.
        if tokenizer.sep_token is None:
            self.sep_token = ''
        else:
            self.sep_token = tokenizer.sep_token

        if tokenizer.sep_token_id is not None:
            self.eos_token_id = tokenizer.sep_token_id
        elif tokenizer.eos_token_id is not None:
            self.eos_token_id = tokenizer.eos_token_id
        else:
            raise ValueError("The tokenizer has no special token for eos")

    def encode_text(self, text, text_pair=None,
        add_special_tokens: bool = True,
        apply_max_len=True,
        padding='max_length',
        return_overflowing_tokens=False,
        return_special_tokens_mask=True,
        return_tensors='pt'):
        if not self.is_fast_tokenizer:
            raise ValueError("Not supported for non fast tokenizer %s" % self.tokenizer)
        return self.tokenizer(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length if apply_max_len else 10000,
            padding=padding,
            truncation=True,
            return_offsets_mapping=True,
            stride=self.max_length // 2,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_tensors=return_tensors,
        )

    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        text = text.strip()

        if title:
            text_input1 = title
            text_input2 = text
        else:
            text_input1 = text
            text_input2 = None
        token_ids = self.tokenizer.encode(
            text_input1,
            text_pair=text_input2,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length if apply_max_len else 10000,
            padding='max_length',
            truncation=True,
        )
        #     token_ids = self.tokenizer.encode(
        #         title,
        #         text_pair=text,
        #         add_special_tokens=add_special_tokens,
        #         max_length=self.max_length if apply_max_len else 10000,
        #         padding='max_length',
        #         # pad_to_max_length=True,
        #         truncation=True,
        #     )
        # else:
        #     token_ids = self.tokenizer.encode(
        #         text,
        #         add_special_tokens=add_special_tokens,
        #         max_length=self.max_length if apply_max_len else 10000,
        #         padding='max_length',
        #         # pad_to_max_length=True,
        #         truncation=True,
        #     )
        # seq_len = self.max_length
        # if self.pad_to_max and len(token_ids) < seq_len:
        #     token_ids = token_ids + [self.pad_token_id] * (
        #         seq_len - len(token_ids)
        #     )
        # elif len(token_ids) >= seq_len:
        #     token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
        #     token_ids[-1] = self.eos_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return (tokens_tensor != self.get_pad_id()).int()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]


class BertTensorizer(Tensorizer):
    def __init__(
        self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max
        if tokenizer.pad_token_id is None:
            raise ValueError("The given tokenizer has no pad_token_id!")
        self.pad_token_id = tokenizer.pad_token_id
        if tokenizer.sep_token_id is not None:
            self.eos_token_id = tokenizer.sep_token_id
        elif tokenizer.eos_token_id is not None:
            self.eos_token_id = tokenizer.eos_token_id
        else:
            raise ValueError("The tokenizer has no special token for eos")

    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # TODO: move max len to methods params?

        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                padding='max_length',
                # pad_to_max_length=True,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                padding='max_length',
                # pad_to_max_length=True,
                truncation=True,
            )
        # seq_len = self.max_length
        # if self.pad_to_max and len(token_ids) < seq_len:
        #     token_ids = token_ids + [self.pad_token_id] * (
        #         seq_len - len(token_ids)
        #     )
        # elif len(token_ids) >= seq_len:
        #     token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
        #     token_ids[-1] = self.eos_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return (tokens_tensor != self.get_pad_id()).int()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]


class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(
            tokenizer, max_length, pad_to_max=pad_to_max
        )
