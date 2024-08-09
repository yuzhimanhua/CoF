#!/usr/bin/env python

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import BertLayer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertIntermediate,
    BertOutput,
    BertEmbeddings,
    BertPooler,
    BertPreTrainedModel,
    BertSelfOutput,
)
from transformers.modeling_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging

logger = logging.get_logger(__name__)

instruct_to_func = {
    "mod2": lambda layer_idx: (layer_idx + 1) % 2 == 0,
    "mod3": lambda layer_idx: (layer_idx + 1) % 3 == 0,
    "mod4": lambda layer_idx: (layer_idx + 1) % 4 == 0,
    "mod6": lambda layer_idx: (layer_idx + 1) % 6 == 0,
    "mod12": lambda layer_idx: (layer_idx + 1) % 12 == 0,
    "ge6mod3": lambda layer_idx: (layer_idx + 1) >= 6 and (layer_idx + 1) % 3 == 0,
    "le7mod3": lambda layer_idx: layer_idx <= 6 and (layer_idx % 3 == 0),
    "ge6": lambda layer_idx: layer_idx >= 6,
}        


class InstructBertSelfAttention(nn.Module):
    def __init__(self, config, use_attn_gate=False):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.use_attn_gate = False
        self.gate = 1.0
        if config.use_attn_gate:
            logger.info("Using gating for instruction hidden attention!")
            self.use_attn_gate = True
            self.gate = torch.nn.Parameter(torch.zeros(1, config.num_attention_heads, 1, 1))

        self.proj_adaptor = False
        if config.proj_adaptor:
            logger.info("Add extra adaptor weights!")
            self.proj_adaptor = True
            self.adaptor = nn.Linear(config.hidden_size, config.hidden_size)

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

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
        instruct_hidden_states=None,
        instruct_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        if instruct_hidden_states is not None:
            if self.proj_adaptor:
                instruct_hidden_states = self.adaptor(instruct_hidden_states)

            if self.use_attn_gate:
                instruct_key_layer = self.transpose_for_scores(self.key(instruct_hidden_states))
                instruct_val_layer = self.transpose_for_scores(self.value(instruct_hidden_states))
            else:
                bz, seq_len, hidden_dim = hidden_states.size()
                instruct_bz = instruct_hidden_states.size()[0]
                if instruct_bz != bz:
                    if instruct_bz == 1:
                        instruct_hidden_states = instruct_hidden_states.repeat((bz, 1, 1))
                    else:
                        raise ValueError("The batch size is not match for instruct hiddens and input hiddens.")

                hidden_states = torch.cat([instruct_hidden_states, hidden_states], dim=1)
                attention_mask = torch.cat([instruct_attention_mask, attention_mask], dim=-1)

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

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
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

        if self.use_attn_gate and instruct_hidden_states is not None:
            # When using gating, this part is identical to normal attention scoring.
            instruct_attention_scores = torch.matmul(query_layer, instruct_key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
            if instruct_attention_mask is not None:
                instruct_attention_scores = instruct_attention_scores + instruct_attention_mask
            instruct_attention_probs = nn.Softmax(dim=-1)(instruct_attention_scores)
            instruct_attention_probs = self.dropout(instruct_attention_probs)
            instruct_ctx_layer = torch.matmul(instruct_attention_probs, instruct_val_layer)

            context_layer += self.gate.tanh() * instruct_ctx_layer

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class InstructBertAttention(nn.Module):
    def __init__(self, config, use_attn_gate=False):
        super().__init__()
        self.self = InstructBertSelfAttention(config, use_attn_gate=use_attn_gate)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        instruct_hidden_states=None,
        instruct_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):

        self_inputs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "past_key_value": past_key_value,
            "instruct_hidden_states": instruct_hidden_states,
            "instruct_attention_mask": instruct_attention_mask,
            "output_attentions": output_attentions,
        }
        # self_outputs = self.self(
        #     hidden_states,
        #     attention_mask,
        #     head_mask,
        #     encoder_hidden_states,
        #     encoder_attention_mask,
        #     instruct_hidden_states,
        #     instruct_attention_mask,
        #     past_key_value,
        #     output_attentions,
        # )
        self_outputs = self.self(**self_inputs)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class InstructBertLayer(nn.Module):
    def __init__(self, config, use_attn_gate=False):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.use_attn_gate = use_attn_gate

        self.attention = InstructBertAttention(config, use_attn_gate=use_attn_gate)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config)


        self.output = BertOutput(config)
        self.intermediate = BertIntermediate(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        instruct_hidden_states=None,
        instruct_attention_mask=None,
        expert_id=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        batch_size, seq_len, hidden_dim = hidden_states.shape

        attn_inputs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "instruct_hidden_states": instruct_hidden_states,
            "instruct_attention_mask": instruct_attention_mask,
            "output_attentions": output_attentions,
            "past_key_value": self_attn_past_key_value,
        }

        # if self.use_attn_moe:
        #     attn_inputs["expert_idx"] = expert_id
        self_attention_outputs = self.attention(**attn_inputs)
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # if self.is_tok_moe:
        #     # Reshapes inputs for token level MoE.
        #     attention_output = attention_output.view(batch_size * seq_len, 1, hidden_dim)
        #     self.expert_id = select_expert(batch_size * seq_len, self.num_expert, expert_id=expert_id)
        # else:
        #     self.expert_id = select_expert(batch_size, self.num_expert, expert_id=expert_id)

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # if self.use_fwd_moe:
        #     layer_output = self.moe_layer(attention_output, self.expert_id)
        # else:
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output



class InstructBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        instruct_type = config.instruct_type.split(":")[0]

        if instruct_type == "all":
            is_instruct_layer = lambda x: True
        elif instruct_type == "embed":
            is_instruct_layer = lambda x: x == 0
        else:
            is_instruct_layer = instruct_to_func[instruct_type]

        print("\n\n")
        print(f"Using instruct type {instruct_type}")
        print("\n\n")
        self.layer = nn.ModuleList([
            InstructBertLayer(config) if is_instruct_layer(layer_id) else BertLayer(config)
            for layer_id in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        instruct_hidden_states=None,
        instruct_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        has_instruct = False if instruct_hidden_states is None else True

        layerwise_instruct = True if type(instruct_hidden_states) is tuple else False

        instruct_hiddens = instruct_hidden_states if has_instruct and not layerwise_instruct else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            instruct_hiddens = instruct_hidden_states[i] if layerwise_instruct else instruct_hiddens

            if isinstance(layer_module, InstructBertLayer):
                layer_inputs = {
                    "hidden_states": hidden_states,
                    "attention_mask": attention_mask,
                    "head_mask": layer_head_mask,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": encoder_attention_mask,
                    "instruct_hidden_states": instruct_hiddens,
                    "instruct_attention_mask": instruct_attention_mask,
                    "output_attentions": output_attentions,
                }
            else:
                layer_inputs = {
                    "hidden_states": hidden_states,
                    "attention_mask": attention_mask,
                    "head_mask": layer_head_mask,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": encoder_attention_mask,
                    "output_attentions": output_attentions,
                }

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                # layer_outputs = torch.utils.checkpoint.checkpoint(
                #     create_custom_forward(layer_module),
                #     hidden_states,
                #     attention_mask,
                #     layer_head_mask,
                #     encoder_hidden_states,
                #     encoder_attention_mask,
                #     instruct_hiddens,
                #     instruct_attention_mask,
                # )
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    **layer_inputs,
                )

            else:
                layer_inputs["past_key_value"] = past_key_value
                layer_outputs = layer_module(**layer_inputs)
                # layer_outputs = layer_module(
                #     hidden_states,
                #     attention_mask,
                #     layer_head_mask,
                #     encoder_hidden_states,
                #     encoder_attention_mask,
                #     instruct_hiddens,
                #     instruct_attention_mask,
                #     past_key_value,
                #     output_attentions,
                # )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class InstructBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config, use_attn_gate=False):
        super().__init__(config)
        self.config = config

        if config.use_attn_gate:
            print("\n\n")
            print("Using attention gating for instructions.")
            print("\n\n")
        self.embeddings = BertEmbeddings(config)
        self.encoder = InstructBertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        input_embeds=None,
        instruct_hidden_states=None,
        instruct_attention_mask=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        elif input_embeds is not None:
            input_shape = input_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if input_embeds is not None:
            device = input_embeds.device
        else:
            device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if instruct_attention_mask is None:
            extended_instruct_attention_mask = None
        else:
            if type(instruct_hidden_states) is tuple:
                instruct_batch_size, instruct_sequence_length, _ = instruct_hidden_states[0].size()
            else:
                instruct_batch_size, instruct_sequence_length, _ = instruct_hidden_states.size()
            instruct_input_shape = (instruct_batch_size, instruct_sequence_length)
            extended_instruct_attention_mask: torch.Tensor = self.get_extended_attention_mask(instruct_attention_mask, instruct_input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if input_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
            )
        else:
            embedding_output = input_embeds
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            instruct_hidden_states=instruct_hidden_states,
            instruct_attention_mask=extended_instruct_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            past_key_values=encoder_outputs.past_key_values,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

        # outputs = (sequence_output, pooled_output,) + encoder_outputs[
        #     1:
        # ] + (embedding_output,)  # add hidden_states and attentions if they are here
        # return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
