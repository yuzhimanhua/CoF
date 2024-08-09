#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

from base64 import encode
import collections
from hashlib import pbkdf2_hmac
import logging
import math
from optparse import Option
import random
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dpr.data.biencoder_data import BiEncoderSample
from dpr.models.biencoder import BiEncoderNllLoss
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import CheckpointState

logger = logging.getLogger(__name__)

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "question_ids",
        "question_segments",
        "context_ids",
        "ctx_segments",
        "is_positive",
        "hard_negatives",
        "encoder_type",
    ],
)
# TODO: it is only used by _select_span_with_token. Move them to utils
rnd = random.Random(0)


class AdvBiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        super(AdvBiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        noise_input_embeds: T=None,
        fix_encoder: bool = False,
        representation_token_pos=0,
    ) -> Tuple[T, T, T, T]:
        sequence_output = None
        pooled_output = None
        hidden_states = None
        input_embeds = None
        if ids is not None or noise_input_embeds is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                        input_embeds=noise_input_embeds,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                    input_embeds=noise_input_embeds,
                )

        return sequence_output, pooled_output, hidden_states

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        q_noise_input_embeds: T = None,
        ctx_noise_input_embeds: T = None,
        encoder_type: str = None,
        representation_token_pos=0,
    ) -> Tuple[T, T]:
        q_encoder = (
            self.question_model
            if encoder_type is None or encoder_type == "question"
            else self.ctx_model
        )
        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            noise_input_embeds=q_noise_input_embeds,
            fix_encoder=self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
        )

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.question_model
        )
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
            ctx_encoder,
            context_ids,
            ctx_segments,
            ctx_attn_mask,
            noise_input_embeds=ctx_noise_input_embeds,
            fix_encoder=self.fix_ctx_encoder,
        )

        return q_pooled_out, ctx_pooled_out

    # TODO delete once moved to the new method
    @classmethod
    def create_biencoder_input(
        cls,
        samples: List,
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of data items (from json) to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only
            if shuffle and shuffle_positives:
                positive_ctxs = sample["positive_ctxs"]
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample["positive_ctxs"][0]

            neg_ctxs = sample["negative_ctxs"]
            hard_neg_ctxs = sample["hard_negative_ctxs"]

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(
                    ctx["text"],
                    title=ctx["title"] if (insert_title and "title" in ctx) else None,
                )
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

            question_tensors.append(tensorizer.text_to_tensor(question))

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
        )

    @classmethod
    def create_biencoder_input2(
        cls,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question = sample.query
            # question = normalize_question(sample.query)

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(
                    ctx.text, title=ctx.title if (insert_title and ctx.title) else None
                )
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(
                        question, tensorizer, token_str=query_token
                    )
                    question_tensors.append(query_span)
                else:
                    question_tensors.append(
                        tensorizer.text_to_tensor(" ".join([query_token, question]))
                    )
            else:
                question_tensors.append(tensorizer.text_to_tensor(question))

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
        )

    def load_state(self, saved_state: CheckpointState):
        # TODO: make a long term HF compatibility fix
        if "question_model.embeddings.position_ids" in saved_state.model_dict:
            del saved_state.model_dict["question_model.embeddings.position_ids"]
            del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        self.load_state_dict(saved_state.model_dict)

    def get_state_dict(self):
        return self.state_dict()


def l2_norm_vector(d, epsilon=1e-6):
    """Normalizes vector into L2 norm ball."""
    d /= torch.sqrt(epsilon + torch.sum(torch.square(d), dim=-1, keepdim=True))
    return d

def l1_norm_vector(d, epsilon=1e-6):
    """Normalizes vector into L1 norm ball."""
    d /= (epsilon + torch.sum(torch.abs(d), dim=-1, keepdim=True))
    return d

def linf_norm_vector(d, epsilon=1e-6):
    """Normalizes vector into L2 norm ball."""
    norm = torch.max(torch.abs(d), dim=-1, keepdim=True)[0]
    d /= (norm + epsilon)
    return d


def _normalize_vector(noise, normalizer="L2"):
    """Normalizes the input tensor."""
    if normalizer == "L2":
        norm_func = l2_norm_vector
    elif normalizer == "L1":
        norm_func = l1_norm_vector
    elif normalizer == "Linf":
        norm_func = linf_norm_vector
    else:
        raise ValueError("Not implemented yet!")
    noise = norm_func(noise)
    return noise


def _generate_noise(input, noise_normalizer="L2"):
    """Generates a randomly sampled noise with Lp normalization."""
    noise = torch.zeros_like(input).normal_(0, 1)
    noise = _normalize_vector(noise, normalizer=noise_normalizer).detach()
    noise.requires_grad_()
    return noise


def js_divergence_w_log_prob(log_p: T, log_q: T) -> T:
    """Computes the JS divergence between two log probs."""
    concat_logprob = torch.cat(
        [log_p.unsqueeze(dim=-1), log_q.unsqueeze(dim=-1)], dim=-1)
    mean_log_prob = math.log(0.5) + torch.logsumexp(concat_logprob, dim=-1)
    loss = 0.5 * (
        F.kl_div(log_p, mean_log_prob, log_target=True, reduce="batchmean") +
        F.kl_div(log_q, mean_log_prob, log_target=True, reduce="batchmean"))
    return loss


def hellinger_distance(log_p: T, log_q: T) -> T:
    """Computes the hellinger distance."""
    return 0.5 * torch.mean(
        torch.sum((torch.exp(0.5 * log_p) - torch.exp(0.5 * log_q)).square(), dim=-1))


class PerturbationEstimator(object):
    def __init__(self,
                 epsilon: Optional[float]=1e-3,
                 loss_type: Optional[str]="kl",
                 step_epsilon: Optional[float]=1e-3,
                 noise_norm: Optional[str]="L2",
                 sample_noise_norm: Optional[str]="L2",
                 ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.loss_type = loss_type
        self.step_epsilon = step_epsilon
        self.noise_norm = noise_norm
        self.sample_noise_norm = sample_noise_norm
        # logger.info("Using %s as VAT loss", self.loss_type)
        # logger.info("Using %s as for norm sample noise", self.sample_noise_norm)
        # logger.info("Using %s as for norm perturb noise", self.noise_norm)

    def estimate_noise_for_ctx(self,
                               model,
                               inputs,
                               q_attn_mask,
                               clean_q_vector,
                               ctx_attn_mask,
                               clean_ctx_vector,
                               encoder_type,
                               representation_token_pos,
                               q_rep_token_pos: T = None,
                               ctx_rep_token_pos: T = None,
                               use_t5: bool = False,
                               loss_scale: float = None,
                               tau: float = 0.01,
                               sim_method: str = "cos",
                               ) -> Tuple[T, T, T]:

        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            if use_t5:
                q_embed_func = model.module.question_model.encoder.shared
                ctx_embed_func = model.module.ctx_model.encoder.shared
            else:
                q_embed_func = model.module.question_model.encoder.embeddings
                ctx_embed_func = model.module.ctx_model.encoder.embeddings
        else:
            if use_t5:
                q_embed_func = model.question_model.encoder.shared
                ctx_embed_func = model.ctx_model.encoder.shared
            else:
                q_embed_func = model.question_model.encoder.embeddings
                ctx_embed_func = model.ctx_model.encoder.embeddings

        local_ctx_input_embeds = ctx_embed_func(
            inputs.context_ids,
        )
        local_q_input_embeds = q_embed_func(
            inputs.question_ids,
        )

        ctx_noise = _generate_noise(
            local_ctx_input_embeds,
            noise_normalizer=self.sample_noise_norm,
        )
        noise_ctx_input_embeds = (
            local_ctx_input_embeds +
            self.epsilon * ctx_noise
        )
        noise_q_input_embeds = local_q_input_embeds

        perturb_inputs = {
            "question_ids": None,
            "question_segments": None,
            "question_attn_mask": q_attn_mask,
            "context_ids": None,
            "ctx_segments": None,
            "ctx_attn_mask": ctx_attn_mask,
            "encoder_type": encoder_type,
            "q_noise_input_embeds": noise_q_input_embeds,
            "ctx_noise_input_embeds": noise_ctx_input_embeds,
        }

        if q_rep_token_pos is not None:
            perturb_inputs["q_rep_token_pos"] =  q_rep_token_pos
            perturb_inputs["ctx_rep_token_pos"] = ctx_rep_token_pos

        # perturb_outputs = model(
        #     None,
        #     None,
        #     q_attn_mask,
        #     None,
        #     None,
        #     ctx_attn_mask,
        #     q_noise_input_embeds=noise_q_input_embeds,
        #     ctx_noise_input_embeds=noise_ctx_input_embeds,
        #     encoder_type=encoder_type,
        #     # representation_token_pos=representation_token_pos,
        # )
        perturb_outputs = model(**perturb_inputs)

        perturb_q_vectors, perturb_ctx_vectors = perturb_outputs

        perturb_scores = BiEncoderNllLoss.get_scores(
            perturb_q_vectors, perturb_ctx_vectors, sim_method=sim_method, tau=tau)
        clean_scores = BiEncoderNllLoss.get_scores(
            clean_q_vector, clean_ctx_vector, sim_method=sim_method, tau=tau)

        if len(perturb_q_vectors.size()) > 1:
            q_num = perturb_q_vectors.size(0)
            perturb_scores = perturb_scores.view(q_num, -1)
            clean_scores = clean_scores.view(q_num, -1)

        perturb_log_scores = F.log_softmax(perturb_scores, dim=1)
        clean_log_scores = F.log_softmax(clean_scores, dim=1).detach()

        if self.loss_type == "kl":
            perturb_loss = F.kl_div(
                perturb_log_scores,
                clean_log_scores,
                reduction="batchmean",
                log_target=True,
            )
        elif self.loss_type == "js":
            perturb_loss = js_divergence_w_log_prob(perturb_log_scores, clean_log_scores)
        elif self.loss_type == "hellinger":
            perturb_loss = hellinger_distance(perturb_log_scores, clean_log_scores)
        elif self.loss_type == "sym_kl":
            perturb_loss = F.kl_div(
                perturb_log_scores,
                clean_log_scores,
                reduction="batchmean",
                log_target=True,
            ) + F.kl_div(
                clean_log_scores,
                perturb_log_scores,
                reduction="batchmean",
                log_target=True,
            )
        else:
            raise ValueError("Unknown loss_type: %s" % self.loss_type)

        adv_ctx_noise = torch.autograd.grad(
            perturb_loss,
            [ctx_noise],
            only_inputs=True,
            retain_graph=False,
        )[0]
        adv_ctx_perturb = _normalize_vector(
            adv_ctx_noise, normalizer=self.noise_norm).detach()
        # adv_q_perturb = _normalize_vector(
        #     adv_q_noise, normalizer=self.noise_norm).detach()

        adv_ctx_input_embeds = (
            local_ctx_input_embeds.clone() +
            self.step_epsilon * adv_ctx_perturb
        )
        adv_q_input_embeds = local_q_input_embeds.clone()

        perturb_inputs["q_noise_input_embeds"] = adv_q_input_embeds
        perturb_inputs["ctx_noise_input_embeds"] = adv_ctx_input_embeds

        # adv_inputs = {
        #     "question_ids": None,
        #     "question_segments": None,
        #     "question_attn_mask": q_attn_mask,
        #     "context_ids": None,
        #     "ctx_segments": None,
        #     "ctx_attn_mask": ctx_attn_mask,
        #     "encoder_type": encoder_type,
        #     "q_noise_input_embeds": adv_q_input_embeds,
        #     "ctx_noise_input_embeds": adv_ctx_input_embeds,
        # }

        # if q_rep_token_pos is not None:
        #     adv_inputs["q_rep_token_pos"] =  q_rep_token_pos
        #     adv_inputs["ctx_rep_token_pos"] = ctx_rep_token_pos

        # adv_outputs = model(
        #     None,
        #     None,
        #     q_attn_mask,
        #     None,
        #     None,
        #     ctx_attn_mask,
        #     q_noise_input_embeds=adv_q_input_embeds,
        #     ctx_noise_input_embeds=adv_ctx_input_embeds,
        #     encoder_type=encoder_type,
        #     representation_token_pos=representation_token_pos,
        # )
        adv_outputs = model(**perturb_inputs)

        adv_q_vectors, adv_ctx_vectors = adv_outputs
        return adv_q_vectors, adv_ctx_vectors


    def estimate_noise(self,
                       model,
                       inputs,
                       q_attn_mask,
                       clean_q_vector,
                       ctx_attn_mask,
                       clean_ctx_vector,
                       encoder_type,
                       representation_token_pos,
                       use_t5: bool = False,
                       q_rep_token_pos: T = None,
                       ctx_rep_token_pos: T = None,
                       loss_scale: float = None,
                       tau: float = 0.01,
                       sim_method: str = "cos",
                       ctx_only: bool = False,
                       use_moe: bool = False,
                       bp_adv: bool = True,
                       ) -> Tuple[T, T, T]:

        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            if use_t5:
                q_embed_func = model.module.question_model.encoder.shared
                ctx_embed_func = model.module.ctx_model.encoder.shared
            else:
                q_embed_func = model.module.question_model.encoder.embeddings
                ctx_embed_func = model.module.ctx_model.encoder.embeddings
        else:
            if use_t5:
                q_embed_func = model.question_model.encoder.shared
                ctx_embed_func = model.ctx_model.encoder.shared
            else:
                q_embed_func = model.question_model.encoder.embeddings
                ctx_embed_func = model.ctx_model.encoder.embeddings

        local_ctx_input_embeds = ctx_embed_func(
            inputs.context_ids,
        )
        local_q_input_embeds = q_embed_func(
            inputs.question_ids,
        )

        ctx_noise = _generate_noise(
            local_ctx_input_embeds,
            noise_normalizer=self.sample_noise_norm,
        )
        noise_ctx_input_embeds = (
            local_ctx_input_embeds +
            self.epsilon * ctx_noise
        )
        q_noise = _generate_noise(
            local_q_input_embeds,
            noise_normalizer=self.sample_noise_norm,
        )
        if ctx_only:
            noise_q_input_embeds = local_q_input_embeds + 0.0 * q_noise
        else:
            noise_q_input_embeds = (
                local_q_input_embeds +
                self.epsilon * q_noise
            )

        perturb_inputs = {
            "question_ids": None,
            "question_segments": None,
            "question_attn_mask": q_attn_mask,
            "context_ids": None,
            "ctx_segments": None,
            "ctx_attn_mask": ctx_attn_mask,
            "q_noise_input_embeds": noise_q_input_embeds,
            "ctx_noise_input_embeds": noise_ctx_input_embeds,
            "encoder_type": encoder_type,
        }
        if use_moe:
            perturb_inputs["q_expert_ids"] =  inputs.query_expert_ids
            perturb_inputs["ctx_expert_ids"] = inputs.ctx_expert_ids

        # TODO: Makes this configurable for context too.
        if hasattr(inputs, "question_rep_pos"):
            perturb_inputs["q_rep_token_pos"] = inputs.question_rep_pos
        else:
            logger.warning("No question rep positions")

        # perturb_outputs = model(
        #     None,
        #     None,
        #     q_attn_mask,
        #     None,
        #     None,
        #     ctx_attn_mask,
        #     q_noise_input_embeds=noise_q_input_embeds,
        #     ctx_noise_input_embeds=noise_ctx_input_embeds,
        #     encoder_type=encoder_type,
        #     representation_token_pos=representation_token_pos,
        # )
        perturb_outputs = model(**perturb_inputs)

        # perturb_q_vectors, perturb_ctx_vectors = perturb_outputs
        perturb_q_vectors = perturb_outputs[0]
        perturb_ctx_vectors = perturb_outputs[1]

        perturb_scores = BiEncoderNllLoss.get_scores(
            perturb_q_vectors, perturb_ctx_vectors, sim_method=sim_method, tau=tau)
        clean_scores = BiEncoderNllLoss.get_scores(
            clean_q_vector, clean_ctx_vector, sim_method=sim_method, tau=tau)

        if len(perturb_q_vectors.size()) > 1:
            q_num = perturb_q_vectors.size(0)
            perturb_scores = perturb_scores.view(q_num, -1)
            clean_scores = clean_scores.view(q_num, -1)

        perturb_log_scores = F.log_softmax(perturb_scores, dim=1)
        clean_log_scores = F.log_softmax(clean_scores, dim=1).detach()

        if self.loss_type == "kl":
            perturb_loss = F.kl_div(
                perturb_log_scores,
                clean_log_scores,
                reduction="batchmean",
                log_target=True,
            )
        elif self.loss_type == "js":
            perturb_loss = js_divergence_w_log_prob(perturb_log_scores, clean_log_scores)
        elif self.loss_type == "hellinger":
            perturb_loss = hellinger_distance(perturb_log_scores, clean_log_scores)
        elif self.loss_type == "sym_kl":
            perturb_loss = F.kl_div(
                perturb_log_scores,
                clean_log_scores,
                reduction="batchmean",
                log_target=True,
            ) + F.kl_div(
                clean_log_scores,
                perturb_log_scores,
                reduction="batchmean",
                log_target=True,
            )
        else:
            raise ValueError("Unknown loss_type: %s" % self.loss_type)

        adv_ctx_noise, adv_q_noise = torch.autograd.grad(
            perturb_loss,
            [ctx_noise, q_noise],
            only_inputs=True,
            retain_graph=False,
        )
        adv_ctx_perturb = _normalize_vector(
            adv_ctx_noise, normalizer=self.noise_norm).detach()
        adv_q_perturb = _normalize_vector(
            adv_q_noise, normalizer=self.noise_norm).detach()

        adv_ctx_input_embeds = (
            local_ctx_input_embeds.clone() +
            self.step_epsilon * adv_ctx_perturb
        )
        if ctx_only:
            adv_q_input_embeds = local_q_input_embeds.clone()
        else:
            adv_q_input_embeds = (
                local_q_input_embeds.clone() +
                self.step_epsilon * adv_q_perturb
            )

        perturb_inputs["q_noise_input_embeds"] = adv_q_input_embeds
        perturb_inputs["ctx_noise_input_embeds"] = adv_ctx_input_embeds
        # adv_outputs = model(
        #     None,
        #     None,
        #     q_attn_mask,
        #     None,
        #     None,
        #     ctx_attn_mask,
        #     q_noise_input_embeds=adv_q_input_embeds,
        #     ctx_noise_input_embeds=adv_ctx_input_embeds,
        #     encoder_type=encoder_type,
        #     representation_token_pos=representation_token_pos,
        # )
        if bp_adv:
            adv_outputs = model(**perturb_inputs)
        else:
            with torch.no_grad():
                adv_outputs = model(**perturb_inputs)

        # adv_q_vectors, adv_ctx_vectors = adv_outputs
        adv_q_vectors = adv_outputs[0]
        adv_ctx_vectors = adv_outputs[1]
        return adv_q_vectors, adv_ctx_vectors


class VNBiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        adv_q_vectors: T,
        adv_ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
        vat_loss_type: float = "kl",
        vat_alpha: float = 0.0,
        tau: float = 0.01,
        sim_method: str = "cos",
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        # scores = BiEncoderNllLoss.get_scores(q_vectors, ctx_vectors, sim_method=sim_method, tau=tau)
        # adv_scores = BiEncoderNllLoss.get_scores(adv_q_vectors, adv_ctx_vectors, sim_method=sim_method, tau=tau)

        # adv_q_cross_scores = BiEncoderNllLoss.get_scores(adv_q_vectors, ctx_vectors, sim_method=sim_method, tau=tau)
        # clean_q_over_adv_q_scores = torch.concat([scores, adv_q_cross_scores], dim=1)

        concat_ctx_vectors = torch.concat([ctx_vectors, adv_ctx_vectors], dim=0)
        clean_q_over_all_ctx_scores = BiEncoderNllLoss.get_scores(q_vectors, concat_ctx_vectors, sim_method=sim_method, tau=tau)

        q_self_scores = BiEncoderNllLoss.get_scores(q_vectors, adv_q_vectors, sim_method=sim_method, tau=tau)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            # scores = scores.view(q_num, -1)
            # adv_q_cross_scores = adv_q_cross_scores.view(q_num, -1)
            # clean_q_over_adv_q_scores = clean_q_over_adv_q_scores.view(q_num, -1)
            clean_q_over_all_ctx_scores = clean_q_over_all_ctx_scores.view(q_num, -1)

        # softmax_scores = F.log_softmax(scores, dim=1)

        # clean_loss = F.nll_loss(
        #     softmax_scores,
        #     torch.tensor(positive_idx_per_question).to(softmax_scores.device),
        #     reduction="mean",
        # )

        # Computes auxiliary losses.
        # cq_cross_softmax_scores = F.log_softmax(clean_q_over_adv_q_scores, dim=1)
        # cq_overall_softmax_scores = F.log_softmax(clean_q_over_all_ctx_scores, dim=1)
        softmax_scores = F.log_softmax(clean_q_over_all_ctx_scores, dim=1)

        # cross_aux_loss = F.nll_loss(
        #    cq_cross_softmax_scores,
        #    torch.tensor(positive_idx_per_question).to(softmax_scores.device),
        #    reduction="mean",
        # )
        clean_loss = F.nll_loss(
           softmax_scores,
           torch.tensor(positive_idx_per_question).to(softmax_scores.device),
           reduction="mean",
        )

        q_self_loss = -torch.diag(F.log_softmax(q_self_scores, dim=1)).mean()

        # all_aux_loss = (cross_aux_loss + overall_aux_loss) / 2.0
        all_aux_loss = q_self_loss

        logger.debug("clean_loss=%.3f", clean_loss)
        logger.debug("all_aux_loss=%.3f", all_aux_loss)

        loss = clean_loss + vat_alpha * all_aux_loss

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
            max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)
        ).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count


class AdvBiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        adv_q_vectors: T,
        adv_ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
        vat_loss_type: float = "kl",
        vat_alpha: float = 0.0,
        tau: float = 0.01,
        sim_method: str = "cos",
        use_sec: bool = False,
        gamma: float = 0.1,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = BiEncoderNllLoss.get_scores(q_vectors, ctx_vectors, sim_method=sim_method, tau=tau)
        adv_scores = BiEncoderNllLoss.get_scores(adv_q_vectors, adv_ctx_vectors, sim_method=sim_method, tau=tau)

        aux_clean_q_cross_scores = BiEncoderNllLoss.get_scores(q_vectors, adv_ctx_vectors, sim_method=sim_method, tau=tau)
        aux_adv_q_cross_scores = BiEncoderNllLoss.get_scores(adv_q_vectors, ctx_vectors, sim_method=sim_method, tau=tau)

        concat_ctx_vectors = torch.concat([ctx_vectors, adv_ctx_vectors], dim=0)

        aux_clean_q_overall_scores = BiEncoderNllLoss.get_scores(q_vectors, concat_ctx_vectors, sim_method=sim_method, tau=tau)
        aux_adv_q_overall_scores = BiEncoderNllLoss.get_scores(adv_q_vectors, concat_ctx_vectors, sim_method=sim_method, tau=tau)

        q_self_scores = BiEncoderNllLoss.get_scores(q_vectors, adv_q_vectors, sim_method=sim_method, tau=tau)
        ctx_self_scores = BiEncoderNllLoss.get_scores(ctx_vectors, adv_ctx_vectors, sim_method=sim_method, tau=tau)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)
            adv_scores = adv_scores.view(q_num, -1)
            aux_clean_q_cross_scores = aux_clean_q_cross_scores.view(q_num, -1)
            aux_adv_q_cross_scores = aux_adv_q_cross_scores.view(q_num, -1)
            aux_clean_q_overall_scores = aux_clean_q_overall_scores.view(q_num, -1)
            aux_adv_q_overall_scores = aux_adv_q_overall_scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        clean_loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        # Computes auxiliary losses.
        cq_cross_softmax_scores = F.log_softmax(aux_clean_q_cross_scores, dim=1)
        aq_cross_softmax_scores = F.log_softmax(aux_adv_q_cross_scores, dim=1)
        cq_overall_softmax_scores = F.log_softmax(aux_clean_q_overall_scores, dim=1)
        aq_overall_softmax_scores = F.log_softmax(aux_adv_q_overall_scores, dim=1)

        cross_aux_loss = F.nll_loss(
           cq_cross_softmax_scores,
           torch.tensor(positive_idx_per_question).to(softmax_scores.device),
           reduction="mean",
        ) + F.nll_loss(
           aq_cross_softmax_scores,
           torch.tensor(positive_idx_per_question).to(softmax_scores.device),
           reduction="mean",
        )

        overall_aux_loss = F.nll_loss(
           cq_overall_softmax_scores,
           torch.tensor(positive_idx_per_question).to(softmax_scores.device),
           reduction="mean",
        ) + F.nll_loss(
           aq_overall_softmax_scores,
           torch.tensor(positive_idx_per_question).to(softmax_scores.device),
           reduction="mean",
        )

        # all_aux_loss = (cross_aux_loss + overall_aux_loss) / 4.0
        all_aux_loss = (cross_aux_loss + overall_aux_loss)

        q_self_loss = -torch.diag(F.log_softmax(q_self_scores, dim=1)).mean()
        ctx_self_loss = -torch.diag(F.log_softmax(ctx_self_scores, dim=1)).mean()

        # self_enf_loss = (q_self_loss + ctx_self_loss) / 2.0
        self_enf_loss = (q_self_loss + ctx_self_loss)

        logger.debug("clean_loss=%.3f", clean_loss)
        if vat_alpha > 0.0:
            adv_log_scores = F.log_softmax(adv_scores, dim=1)
            if vat_loss_type == "kl":
                adv_loss = F.kl_div(
                    adv_log_scores,
                    softmax_scores.detach(),
                    reduction="batchmean",
                    log_target=True,
                )
            elif vat_loss_type == "js":
                adv_loss = js_divergence_w_log_prob(adv_log_scores, softmax_scores.detach())
            elif vat_loss_type == "hellinger":
                adv_loss = hellinger_distance(adv_log_scores, softmax_scores.detach())
            elif vat_loss_type == "sym_kl":
                adv_loss = 0.5 * F.kl_div(
                    softmax_scores.detach(),
                    adv_log_scores,
                    reduction="batchmean",
                    log_target=True,
                ) + 0.5 * F.kl_div(
                    adv_log_scores,
                    softmax_scores.detach(),
                    reduction="batchmean",
                    log_target=True,
                )
            else:
                raise ValueError("Unknown loss_type: %s" % vat_loss_type)

            logger.debug("vat_loss=%.3f", adv_loss)
            loss = clean_loss + vat_alpha * adv_loss
        else:
            loss = clean_loss

        logger.debug("self_enf_loss=%.3f", self_enf_loss)
        logger.debug("all_aux_loss=%.3f", all_aux_loss)

        loss += vat_alpha * self_enf_loss
        # loss += vat_alpha * all_aux_loss * 0.1
        loss += vat_alpha * all_aux_loss

        if use_sec:
            q_norm = q_vectors.norm(dim=1, p=2)
            ctx_norm = ctx_vectors.norm(dim=1, p=2)
            mu_q = torch.mean(q_norm).detach()
            mu_ctx = torch.mean(ctx_norm).detach()
            sec_q_loss = torch.mean((q_norm - mu_q).norm(dim=0, p=2))
            sec_ctx_loss = torch.mean((ctx_norm - mu_ctx).norm(dim=0, p=2))
            sec_loss = (sec_q_loss + sec_ctx_loss) / 2.0
            logger.debug("sec_loss: ", sec_loss.item())
            loss += gamma * sec_loss

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
            max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)
        ).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count


def _select_span_with_token(
    text: str, tensorizer: Tensorizer, token_str: str = "[START_ENT]"
) -> T:
    id = tensorizer.get_token_id(token_str)
    query_tensor = tensorizer.text_to_tensor(text)

    if id not in query_tensor:
        query_tensor_full = tensorizer.text_to_tensor(text, apply_max_len=False)
        token_indexes = (query_tensor_full == id).nonzero()
        if token_indexes.size(0) > 0:
            start_pos = token_indexes[0, 0].item()
            # add some randomization to avoid overfitting to a specific token position

            left_shit = int(tensorizer.max_length / 2)
            rnd_shift = int((rnd.random() - 0.5) * left_shit / 2)
            left_shit += rnd_shift

            query_tensor = query_tensor_full[start_pos - left_shit :]
            cls_id = tensorizer.tokenizer.cls_token_id
            if query_tensor[0] != cls_id:
                query_tensor = torch.cat([torch.tensor([cls_id]), query_tensor], dim=0)

            from dpr.models.reader import _pad_to_len

            query_tensor = _pad_to_len(
                query_tensor, tensorizer.get_pad_id(), tensorizer.max_length
            )
            query_tensor[-1] = tensorizer.tokenizer.sep_token_id
            # logger.info('aligned query_tensor %s', query_tensor)

            assert id in query_tensor, "query_tensor={}".format(query_tensor)
            return query_tensor
        else:
            raise RuntimeError(
                "[START_ENT] toke not found for Entity Linking sample query={}".format(
                    text
                )
            )
    else:
        return query_tensor
