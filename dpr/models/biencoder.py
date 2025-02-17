#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import collections
import logging
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dpr.data.biencoder_data import BiEncoderSample
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


def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    q_vectors = q_vectors.to(torch.float32)
    ctx_vectors = ctx_vectors.to(torch.float32)
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def _normalize(input, p, dim, eps = 1e-6):
    denom = input.norm(p, dim, keepdim=True).clamp_min(eps)
    return input / denom.detach()


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    # q_vector = _normalize(q_vector, 2, dim=1)
    # ctx_vectors = _normalize(ctx_vectors, 2, dim=1)
    q_vector = torch.nn.functional.normalize(q_vector, p=2, dim=-1)
    ctx_vectors = torch.nn.functional.normalize(ctx_vectors, p=2, dim=-1)

    return dot_product_scores(q_vector, ctx_vectors)


def gaussian_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    # q_vector = _normalize(q_vector, 2, dim=1)
    # ctx_vectors = _normalize(ctx_vectors, 2, dim=1)
    q_vector = torch.nn.functional.normalize(q_vector, p=2, dim=-1)
    ctx_vectors = torch.nn.functional.normalize(ctx_vectors, p=2, dim=-1)

    return 1.0 - dot_product_scores(q_vector, ctx_vectors)


def onehot_max(logits):
    _, max_ind = torch.max(logits, dim=-1)
    y = torch.nn.functional.one_hot(max_ind, num_classes=logits.size(-1))
    return y

    
class GroupModel(nn.Module):
    def __init__(self, num_group, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_group = num_group
        self.fwd_func = nn.Linear(self.hidden_dim, self.num_group)
        self.fwd_func.weight.data.normal_(mean=0.0, std=0.02)
        self.fwd_func_2 = nn.Linear(self.num_group, self.hidden_dim, bias=False)
        self.act_func = torch.nn.functional.relu
        self.fwd_func_3 = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x, tau=1.0):
        logits = self.fwd_func(x)

        if self.training:
            one_hot = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
        else:
            one_hot = onehot_max(logits)

        p_y = nn.functional.softmax(logits, dim=-1)
        log_p_y = nn.functional.log_softmax(logits, dim=-1)
        entropic_loss = torch.sum(p_y * log_p_y, dim=-1)

        output = self.act_func(self.fwd_func_3(self.fwd_func_2(one_hot)))

        return one_hot, output, entropic_loss


class GroupBiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
        num_expert: int = 1,
        num_q_expert: int = None,
        offset_expert_id: bool = False,
        num_q_group: int = 1,
        num_ctx_group: int = 100,
    ):
        super(MoEBiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder
        self.num_expert = num_expert
        if num_q_expert:
            self.num_q_expert = num_q_expert
            self.num_ctx_expert = num_expert - num_q_expert
        else:
            self.num_q_expert = num_expert // 2
            self.num_ctx_expert = num_expert // 2
        self.q_gp_model = GroupModel(num_q_group, question_model.get_out_size())
        self.ctx_gp_model = GroupModel(num_ctx_group, ctx_model.get_out_size())
        self.offset_expert_id = offset_expert_id

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        noise_input_embeds: T = None,
        fix_encoder: bool = False,
        representation_token_pos=0,
        expert_id=None,
    ) -> Tuple[T, T, T]:
        sequence_output = None
        pooled_output = None
        hidden_states = None
        outputs = None
        if ids is not None or noise_input_embeds is not None:
            if fix_encoder:
                with torch.no_grad():
                    # sequence_output, pooled_output, hidden_states = sub_model(
                    outputs = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                        input_embeds=noise_input_embeds,
                        expert_id=expert_id,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                # sequence_output, pooled_output, hidden_states = sub_model(
                outputs = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                    input_embeds=noise_input_embeds,
                    expert_id=expert_id,
                )

        # return sequence_output, pooled_output, hidden_states
        if outputs is not None:
            return outputs
        return sequence_output, pooled_output, hidden_states, None
        

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
        task_expert_id: int=None,
    ) -> Tuple[T, T]:
        q_encoder = (
            self.question_model
            if encoder_type is None or encoder_type == "question"
            else self.ctx_model
        )

        if question_ids is not None:
            bsz = question_ids.shape[0]
        elif q_noise_input_embeds is not None:
            bsz = q_noise_input_embeds.shape[0]
        else:
            bsz = 1

        q_expert_ids = None
        if not self.question_model.use_infer_expert:
            q_expert_ids = torch.randint(low=0, high=self.num_q_expert, size=(bsz,)).type(torch.int64)
            assert q_expert_ids.dtype == torch.int64

        # _q_seq, q_pooled_out, _q_hidden = self.get_representation(
        q_outputs = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            noise_input_embeds=q_noise_input_embeds,
            fix_encoder=self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
            expert_id=q_expert_ids,
        )

        q_pooled_out = q_outputs[1]

        _, gp_q_out, q_entropy_loss = self.q_gp_model(q_pooled_out)

        if context_ids is not None:
            bsz = context_ids.shape[0]
        elif ctx_noise_input_embeds is not None:
            bsz = ctx_noise_input_embeds.shape[0]
        else:
            bsz = 1

        ctx_expert_ids = None
        if not self.ctx_model.use_infer_expert:
            ctx_expert_ids = torch.randint(low=self.num_q_expert, high=(self.num_q_expert + self.num_ctx_expert), size=(bsz,)).type(torch.int64)
            assert ctx_expert_ids.dtype == torch.int64

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.question_model
        )
        # _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
        ctx_outputs = self.get_representation(
            ctx_encoder,
            context_ids,
            ctx_segments,
            ctx_attn_mask,
            noise_input_embeds=ctx_noise_input_embeds,
            fix_encoder=self.fix_ctx_encoder,
            representation_token_pos=representation_token_pos,
            expert_id=ctx_expert_ids,
        )

        # if hasattr(ctx_outputs, "pooled_output"):
        #     ctx_pooled_out = ctx_outputs.pooled_output
        #     ctx_entropy_loss = ctx_outputs.total_entropy_loss
        # else:
        #     ctx_pooled_out = ctx_outputs[1]
        #     ctx_entropy_loss = None
        ctx_pooled_out = ctx_outputs[1]
        # ctx_entropy_loss = ctx_outputs[-1]

        _, gp_ctx_out, ctx_entropy_loss = self.ctx_gp_model(ctx_pooled_out)
        entropy_loss = None
        if q_entropy_loss is not None and ctx_entropy_loss is not None:
            entropy_loss = torch.concat([q_entropy_loss, ctx_entropy_loss])

        return q_pooled_out, ctx_pooled_out, entropy_loss, gp_q_out, gp_ctx_out

    def load_state(self, saved_state: CheckpointState):
        # TODO: make a long term HF compatibility fix
        if "question_model.embeddings.position_ids" in saved_state.model_dict:
            del saved_state.model_dict["question_model.embeddings.position_ids"]
            del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        self.load_state_dict(saved_state.model_dict)

    def get_state_dict(self):
        return self.state_dict()



class MoEBiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
        num_expert: int = 1,
        num_q_expert: int = None,
        offset_expert_id: bool = False,
    ):
        super(MoEBiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder
        self.num_expert = num_expert
        if num_q_expert:
            self.num_q_expert = num_q_expert
            self.num_ctx_expert = num_expert - num_q_expert
        else:
            self.num_q_expert = num_expert // 2
            self.num_ctx_expert = num_expert // 2
        self.offset_expert_id = offset_expert_id

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        noise_input_embeds: T = None,
        fix_encoder: bool = False,
        representation_token_pos=0,
        expert_id=None,
    ) -> Tuple[T, T, T]:
        sequence_output = None
        pooled_output = None
        hidden_states = None
        outputs = None
        if ids is not None or noise_input_embeds is not None:
            if fix_encoder:
                with torch.no_grad():
                    # sequence_output, pooled_output, hidden_states = sub_model(
                    outputs = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                        input_embeds=noise_input_embeds,
                        expert_id=expert_id,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                # sequence_output, pooled_output, hidden_states = sub_model(
                outputs = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                    input_embeds=noise_input_embeds,
                    expert_id=expert_id,
                )

        # return sequence_output, pooled_output, hidden_states
        if outputs is not None:
            return outputs
        return sequence_output, pooled_output, hidden_states, None
        

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
        task_expert_id: int=None,
    ) -> Tuple[T, T]:
        q_encoder = (
            self.question_model
            if encoder_type is None or encoder_type == "question"
            else self.ctx_model
        )

        if question_ids is not None:
            bsz = question_ids.shape[0]
        elif q_noise_input_embeds is not None:
            bsz = q_noise_input_embeds.shape[0]
        else:
            bsz = 1

        q_expert_ids = None
        if not self.question_model.use_infer_expert:
            q_expert_ids = torch.randint(low=0, high=self.num_q_expert, size=(bsz,)).type(torch.int64)
            assert q_expert_ids.dtype == torch.int64

        # _q_seq, q_pooled_out, _q_hidden = self.get_representation(
        q_outputs = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            noise_input_embeds=q_noise_input_embeds,
            fix_encoder=self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
            expert_id=q_expert_ids,
        )

        # if hasattr(q_outputs, "pooled_output"):
        #     q_pooled_out = q_outputs.pooled_output
        #     q_entropy_loss = q_outputs.total_entropy_loss
        # else:
        q_pooled_out = q_outputs[1]
        q_entropy_loss = q_outputs[-1]

        if context_ids is not None:
            bsz = context_ids.shape[0]
        elif ctx_noise_input_embeds is not None:
            bsz = ctx_noise_input_embeds.shape[0]
        else:
            bsz = 1

        ctx_expert_ids = None
        if not self.ctx_model.use_infer_expert:
            ctx_expert_ids = torch.randint(low=self.num_q_expert, high=(self.num_q_expert + self.num_ctx_expert), size=(bsz,)).type(torch.int64)
            assert ctx_expert_ids.dtype == torch.int64

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.question_model
        )
        # _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
        ctx_outputs = self.get_representation(
            ctx_encoder,
            context_ids,
            ctx_segments,
            ctx_attn_mask,
            noise_input_embeds=ctx_noise_input_embeds,
            fix_encoder=self.fix_ctx_encoder,
            representation_token_pos=representation_token_pos,
            expert_id=ctx_expert_ids,
        )

        # if hasattr(ctx_outputs, "pooled_output"):
        #     ctx_pooled_out = ctx_outputs.pooled_output
        #     ctx_entropy_loss = ctx_outputs.total_entropy_loss
        # else:
        #     ctx_pooled_out = ctx_outputs[1]
        #     ctx_entropy_loss = None
        ctx_pooled_out = ctx_outputs[1]
        ctx_entropy_loss = ctx_outputs[-1]

        entropy_loss = None
        if q_entropy_loss is not None and ctx_entropy_loss is not None:
            entropy_loss = torch.concat([q_entropy_loss, ctx_entropy_loss])

        return q_pooled_out, ctx_pooled_out, entropy_loss

    def load_state(self, saved_state: CheckpointState):
        # TODO: make a long term HF compatibility fix
        if "question_model.embeddings.position_ids" in saved_state.model_dict:
            del saved_state.model_dict["question_model.embeddings.position_ids"]
            del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        self.load_state_dict(saved_state.model_dict)

    def get_state_dict(self):
        return self.state_dict()


class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
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
        fix_encoder: bool = False,
        representation_token_pos=0,
    ) -> Tuple[T, T, T]:
        sequence_output = None
        pooled_output = None
        hidden_states = None
        outputs = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    # sequence_output, pooled_output, hidden_states = sub_model(
                    outputs = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                # sequence_output, pooled_output, hidden_states = sub_model(
                outputs = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                )

        if outputs is not None:
            return outputs
        return sequence_output, pooled_output, hidden_states

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos=0,
    ) -> Tuple[T, T]:
        q_encoder = (
            self.question_model
            if encoder_type is None or encoder_type == "question"
            else self.ctx_model
        )
        # _q_seq, q_pooled_out, _q_hidden = self.get_representation(
        q_outputs = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
        )

        q_pooled_out = q_outputs[1]

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.question_model
        )
        # _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
        ctx_outputs= self.get_representation(
            ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder
        )

        ctx_pooled_out = ctx_outputs[1]

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


class BiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
        tau: float = 0.01,
        sim_method: str = "cos",
        alpha: float = 100.0,
        use_sec: bool = False,
        gamma: float = 1.0,
        donot_use_as_negative: T = None,
        pos_prior: float = 0.01,
        debiase: bool = False,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors, sim_method=sim_method, tau=tau)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)
        if sim_method == "gaussian":
            scores = 1.0 / tau - scores
            pos_mask = torch.nn.functional.one_hot(
                torch.tensor(positive_idx_per_question),
                num_classes=ctx_vectors.shape[0]).to(scores.device)
            align_scores = torch.sum(scores * pos_mask, dim=-1)
            uniform_scores = torch.mean((1.0 - pos_mask) * torch.exp(-scores), dim=-1)
            loss = torch.mean(align_scores + alpha * uniform_scores)
        elif donot_use_as_negative is not None:
            pos_mask = torch.nn.functional.one_hot(
                torch.tensor(positive_idx_per_question),
                num_classes=ctx_vectors.shape[0]).to(scores.device)

            pos_scores = torch.sum(scores * pos_mask, dim=-1)

            flags = donot_use_as_negative == -100
            masks = 1.0 - flags.float()
            masks.unsqueeze_(-1)

            donot_use_as_negative[flags] = 0

            donot_use_as_neg_mask, _ = torch.max(
                torch.nn.functional.one_hot(
                    donot_use_as_negative, num_classes=ctx_vectors.shape[0]
                    ).to(scores.device) * masks, 1)
            donot_use_as_neg_mask = 1.0 - (donot_use_as_neg_mask - pos_mask)

            neg_scores = torch.logsumexp(scores + torch.log(donot_use_as_neg_mask), dim=-1)

            loss = torch.mean(pos_scores - neg_scores)
        else:
            loss = F.nll_loss(
                softmax_scores,
                torch.tensor(positive_idx_per_question).to(softmax_scores.device),
                reduction="mean",
            )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
            max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)
        ).sum()

        if use_sec:
            q_norm = q_vectors.norm(dim=1, p=2)
            ctx_norm = ctx_vectors.norm(dim=1, p=2)
            mu_q = torch.mean(q_norm).detach()
            mu_ctx = torch.mean(ctx_norm).detach()
            sec_q_loss = torch.mean((q_norm - mu_q).norm(dim=0, p=2))
            sec_ctx_loss = torch.mean((ctx_norm - mu_ctx).norm(dim=0, p=2))
            sec_loss = (sec_q_loss + sec_ctx_loss) / 2.0
            loss += gamma * sec_loss

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T, sim_method: str = "dot", tau: float = 1.0) -> T:
        f = BiEncoderNllLoss.get_similarity_function(sim_method)
        return f(q_vector, ctx_vectors) / tau

    @staticmethod
    def get_similarity_function(sim_method: str = "cos"):
        if sim_method == "dot":
            return dot_product_scores
        elif sim_method == "cos":
            return cosine_scores
        elif sim_method == "gaussian":
            return cosine_scores 
        raise ValueError("Unknown sim_method=%s" % sim_method)


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
