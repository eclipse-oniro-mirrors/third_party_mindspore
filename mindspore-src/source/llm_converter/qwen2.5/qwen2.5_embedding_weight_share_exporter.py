import copy
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
import math
from torch import nn
import numpy
import transformers
from typing import List, Optional, Tuple, Union
import warnings
from importlib.metadata import version
import os
import logging
import argparse
import json
import numpy as np
import onnx
from onnxslim import slim
import torch
from torch.onnx import OperatorExportTypes
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from utils.duplicate_shared_initializer import duplicate_shared_initializers
from onnx import helper, shape_inference, TensorProto
from utils.quant.quantize import (
    apply_quant,
    quantize_weight_g128_4bit_nz,
    apply_shared_weight,
    quantize_weight_g32_4bit_nd,
)
from utils.quant.quant_config import ModelConfig, QuantizationConfig, LiteTurboConfig
device = "cpu"
dtype = torch.float16


def apply_shared_weight(model, is_quant=False):
    '''Apply WordEmbedding & LmHead weight shared'''
    graph = model.graph
    for node in graph.node:
        if '/lm_head/MatMul' not in node.name:
            continue

        weight_name = node.input[1]
        for init in graph.initializer:
            if init.name != weight_name:
                continue

            weight_data = onnx.numpy_helper.to_array(init)

            if is_quant:
                embedding_input = helper.make_tensor_value_info(
                    'embedding_weight', TensorProto.INT8, weight_data.shape)
            else:
                embedding_input = helper.make_tensor_value_info('embedding_weight', TensorProto.FLOAT16,
                                                                weight_data.T.shape)

            node_input = 'embedding_weight'
            if not is_quant:
                transpose_node = helper.make_node(
                    "Transpose",
                    inputs=['embedding_weight'],
                    outputs=['embedding_weight_transpose'],
                    perm=[1, 0],
                    name='embedding_weight_transpose'
                )
                node_input = 'embedding_weight_transpose'
                model.graph.node.append(transpose_node)

            node.input[1] = node_input
            model.graph.input.insert(6, embedding_input)
            model.graph.initializer.remove(init)
            print("apply lm head weight shared.")
            return
    return


def get_initializers_info(model):
    init_usage_map = {init.name: [] for init in model.graph.initializer}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name in init_usage_map:
                init_usage_map[input_name].append(node)

    name_2_init = {init.name: init for init in model.graph.initializer}

    return init_usage_map, name_2_init


def get_fusion_outputs(users):
    '''get fusion node outputs'''
    if len(users) == 1:
        if users[0].op_type == 'MsRmsNorm':
            return users[0], None
        return None, None

    if len(users) == 2:
        rms_node = None
        add_node = None
        for i, user in enumerate(users):
            if user.op_type == 'MsRmsNorm':
                rms_node = user
                add_node = users[1 - i]
                break
        return rms_node, add_node
    return None, None


def fuse_add_rmsnorm(model_path, output_path):
    '''Add RmsNorm operator fusion'''
    model = onnx.load(model_path)

    node_map = {}
    for node in model.graph.node:
        for node_input in node.input:
            if node_input in node_map:
                node_map[node_input].append(node)
            else:
                node_map[node_input] = [node]

    for node in model.graph.node:
        if node.op_type == 'Add':
            add_users = node_map[node.output[0]]
            rms_node, add_node = get_fusion_outputs(add_users)
            if rms_node is None:
                continue

            node_name = rms_node.name.replace('rmsnorm', 'addrmsnorm')
            outputs = [rms_node.output[0]] if add_node is None else [
                rms_node.output[0], node.output[0]]
            fused_rmsnorm = helper.make_node(
                'MsAddRmsNorm',
                name=node_name,
                inputs=[node.input[0], node.input[1], rms_node.input[1]],
                outputs=outputs,
                epsilon=rms_node.attribute[0].f
            )

            print(f'apply Add Rmsnorm fusion: {node.name}')
            model.graph.node.append(fused_rmsnorm)
            model.graph.node.remove(node)
            model.graph.node.remove(rms_node)

    if model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
        onnx.save(model, output_path, save_as_external_data=True,
                  location=os.path.basename(output_path) + ".data")
    else:
        onnx.save(model, output_path)
    print(f"Fused model saved to {output_path}")
    return model


'''
part 1: patches contents.
'''


def repeat_kv_4_40(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = torch.cat([hidden_states.reshape(-1, 1, slen, head_dim)] * n_rep, 1) \
        .reshape(batch, -1, slen, head_dim)
    return hidden_states


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb


def apply_rotary_pos_emb_4_40(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MsRotaryPosEmb(torch.autograd.Function):
    def forward(ctx, q, k, cos, sin, rope_q_shape, rope_k_shape, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q = q.transpose(2, 1)
        k = k.transpose(2, 1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed

    def symbolic(g, q, k, cos, sin, rope_q_shape, rope_k_shape, unsqueeze_dim=1):
        rope_q, rope_k = g.op('MsRotaryPosEmb', q, k, cos, sin, outputs=2)
        rope_q.setType(q.type().with_sizes(
            rope_q_shape).with_dtype(torch.float16))
        rope_k.setType(k.type().with_sizes(
            rope_k_shape).with_dtype(torch.float16))
        return rope_q, rope_k


class KVCacheUpdate(torch.autograd.Function):
    def forward(ctx, past, state, pos, kvcache_mask, shape):
        if state.shape[1] > 1:
            pass
        else:
            past = past + (kvcache_mask * state)
        return past

    @staticmethod
    def symbolic(g: torch.Graph, past, state, pos, kvcache_mask, shape):
        past = g.op("MsScatterND", past, pos, state, layout_s="BNSD")
        past.setType(past.type().with_sizes(shape).with_dtype(torch.float16))
        return past


def update_kvcache(past_key, past_value, key_states, value_states, current_pos, kvcache_mask):
    key_states = key_states.to(torch.float16)
    past_key = KVCacheUpdate.apply(
        past_key, key_states, current_pos, kvcache_mask, past_key.shape)

    value_states = value_states.to(torch.float16)
    past_value = KVCacheUpdate.apply(
        past_value, value_states, current_pos, kvcache_mask, past_value.shape)

    return past_key, past_value


class RmsNormCustom(torch.autograd.Function):
    def forward(ctx, hidden, weight, eps, shape):
        old_dtype = hidden.dtype
        variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
        return hidden * weight

    def symbolic(g: torch.Graph, hidden, weight, eps, shape):
        norm = g.op("MsRmsNorm", hidden, weight, epsilon_f=eps)
        norm.setType(norm.type().with_sizes(shape).with_dtype(torch.float16))
        return norm


def rms_forward(self, hidden):
    shape = hidden.shape
    return RmsNormCustom.apply(hidden, self.weight, self.variance_epsilon, shape)


class MsGroupMatmul(torch.autograd.Function):
    def forward(ctx, x1, x2, n_rep, trans_b, shape):
        x2 = repeat_kv_4_40(x2, n_rep)
        x2 = x2.transpose(2, 3) if trans_b == 'True' else x2
        y = torch.matmul(x1, x2)
        return y

    def symbolic(g: torch.Graph, x1, x2, n_rep, trans_b, shape):
        y = g.op("MsGroupMatmul", x1, x2, trans_b_s=trans_b)
        y.setType(y.type().with_sizes(shape).with_dtype(torch.float16))
        return y


class MsAddSoftmax(torch.autograd.Function):
    def forward(ctx, attn_weights, attention_mask, shape):
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        return attn_weights

    def symbolic(g: torch.Graph, attn_weights, attention_mask, shape):
        attn_weights = g.op("MsAddSoftmax", attn_weights, attention_mask)
        attn_weights.setType(attn_weights.type().with_sizes(
            shape).with_dtype(torch.float16))
        return attn_weights


def qwen2_attention_forward_4_40(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    kvcache_mask: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    past_key_value: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    valid_seq_len: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. " +
            "Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)
    key_states = key_states.view(
        bsz, -1, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(
        bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    rope_q_shape = [bsz, self.num_heads, q_len, self.head_dim]
    rope_k_shape = [bsz, self.num_key_value_heads, q_len, self.head_dim]
    query_states, key_states = MsRotaryPosEmb.apply(query_states, key_states, rope_cos, rope_sin,
                                                    rope_q_shape, rope_k_shape)

    key_states, value_states = update_kvcache(past_key_value[0], past_key_value[1], key_states, value_states,
                                              valid_seq_len, kvcache_mask)

    shape = list(query_states.shape)
    shape[-1] = key_states.shape[2]
    attn_weights = MsGroupMatmul.apply(
        query_states, key_states, self.num_key_value_groups, 'True', shape)
    attn_weights = attn_weights / math.sqrt(self.head_dim)

    # upcast attention to fp32
    attn_weights = MsAddSoftmax.apply(
        attn_weights, attention_mask, attn_weights.shape)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = MsGroupMatmul.apply(
        attn_weights, value_states, self.num_key_value_groups, 'False', query_states.shape)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, -1, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, [key_states, value_states]


def qwen2_decoder_layer_forward_4_40(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    kvcache_mask: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    past_key_value: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    valid_seq_len: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, sequence_length)` where padding elements are indicated by 0.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        kvcache_mask=kvcache_mask,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        valid_seq_len=valid_seq_len,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def qwen2_model_forward_4_40(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    kvcache_mask: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    past_key_values: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    valid_seq_len: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0

    max_seq_len = seq_length
    if seq_length == 1:
        position_ids = (valid_seq_len - 1).unsqueeze(0)
    else:
        position_ids = torch.arange(0, max_seq_len).unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = []
    layer_idx = 0

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kvcache_mask=kvcache_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                past_key_value=past_key_values[layer_idx],
                output_attentions=output_attentions,
                use_cache=use_cache,
                valid_seq_len=valid_seq_len,
            )

        hidden_states = layer_outputs[0]
        if use_cache:
            past_key_value = layer_outputs[2 if output_attentions else 1]
            next_decoder_cache.append(past_key_value)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
        layer_idx += 1
    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def qwen2_causal_model_forward_4_40(
    self,
    input_ids: torch.LongTensor = None,
    valid_seq_len: Optional[torch.LongTensor] = None,
    lmhead_idx: torch.LongTensor = None,
    kvcache_mask: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.Tensor]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

    >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    print(f'--> valid_seq_len is not None: {valid_seq_len is not None}')
    if valid_seq_len is not None:
        print("valid_seq_len:", valid_seq_len)

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        kvcache_mask=kvcache_mask,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        valid_seq_len=valid_seq_len
    )

    hidden_states = outputs[0]

    # hack: get one valid token hidden states
    # if valid_seq_len is not None:
    hidden_states = hidden_states[:, lmhead_idx]  # 消除prefill模型的sub算子

    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        print("not return_dict")
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


'''
part 2: applying patches.
'''


def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version


def apply_qwen2_patch():
    transformers_version = check_version()
    version_list = ['4.40']
    ok = False
    for version in version_list:
        if version in transformers_version:
            ok = True
            break
    if not ok:
        raise RuntimeError(
            f"Transformers version {transformers_version} is NOT compatible with qwen_export. qwen_export is ONLY tested with Transformers version {version_list}.")

    print("Applied qwen2 patches.")
    transformers.models.qwen2.modeling_qwen2.repeat_kv = repeat_kv_4_40
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.forward = qwen2_causal_model_forward_4_40
    transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb_4_40
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attention_forward_4_40
    transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer.forward = qwen2_decoder_layer_forward_4_40
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen2_model_forward_4_40
    transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm.forward = rms_forward


class Qwen2Onnx:
    """Qwen2 export onnx model"""

    def load(self, model_path, layers):
        """Load huggingface model"""
        self.config = AutoConfig.from_pretrained(model_path)
        if self.config.model_type != "qwen2" or \
                self.config.num_hidden_layers != 24 or \
                self.config.intermediate_size != 4864 or \
                self.config.max_position_embeddings != 32768 or \
                self.config.hidden_size != 896:
            raise ValueError(
                f"Error: The model at '{model_path}' is not a Qwen2.5-0.5B model. ")
        self.config._attn_implementation = "eager"  # pylint: disable=W0212
        self.config.num_hidden_layers = layers
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            config=self.config,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation="eager",
        )

        self.model = self.model.eval()
        logging.info("model loaded: %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        self.num_kv_heads = self.model.config.num_key_value_heads

    def export(self, model_path, max_seq_len=256, chunk_size=128):
        """export onnx model"""
        head_dim = (
            self.model.config.hidden_size // self.model.config.num_attention_heads
        )

        input_names = [
            "valid_seq_len",
            "lmhead_idx",
            "kvcache_mask",
            "rope_cos",
            "rope_sin",
            "inputs_embeds",
            "attention_mask",
        ]
        kv_names = [(f"past_key_{i}", f"past_val_{i}")
                    for i in range(self.num_layers)]
        kv_names = [name for kv in kv_names for name in kv]
        input_names = input_names + kv_names
        out_kv_names = [
            (f"out_key_{i}", f"out_val_{i}") for i in range(self.num_layers)
        ]
        out_kv_names = [name for kv in out_kv_names for name in kv]

        valid_seq_len = torch.tensor([1], dtype=torch.int32).to(device)
        lmhead_idx = torch.tensor([0], dtype=torch.int32).to(device)
        kvcache_mask = torch.zeros(
            1, max_seq_len, self.num_kv_heads, head_dim, dtype=dtype
        ).to(device)

        rope_cos = torch.zeros((1, chunk_size, head_dim),
                               device=device, dtype=dtype)
        rope_sin = torch.zeros((1, chunk_size, head_dim),
                               device=device, dtype=dtype)

        past_key_or_value = torch.zeros(
            (1, self.num_kv_heads, max_seq_len, head_dim), device=device, dtype=dtype
        )
        past_key_values = [[past_key_or_value] * 2] * self.num_layers

        inputs_embeds = torch.zeros(
            (1, chunk_size, self.hidden_size), device=device, dtype=dtype
        )
        attention_mask = torch.zeros(
            1, 1, chunk_size, max_seq_len, dtype=dtype)

        inputs = (
            None,
            valid_seq_len,
            lmhead_idx,
            kvcache_mask,
            rope_cos,
            rope_sin,
            inputs_embeds,
            attention_mask,
            past_key_values,
        )
        print("export beign.")
        torch.onnx.export(
            self.model,
            inputs,
            model_path,
            input_names=input_names,
            do_constant_folding=True,
            output_names=["logits", *out_kv_names],
            opset_version=18,
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
        )

        print("export end.")
        print("Slim begin.")
        new_model = slim(model_path, skip_fusion_patterns=["FusionGemm"])
        print("Slim end.")

        print("Apply add rmsnorm fusion begin.")
        new_model = fuse_add_rmsnorm(model_path, model_path)
        print("Apply add rmsnorm fusion end.")

        duplicate_shared_initializers(new_model)
        if new_model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
            onnx.save(
                new_model,
                model_path,
                save_as_external_data=True,
                location=os.path.basename(model_path) + ".data",
            )
        else:
            onnx.save(new_model, model_path)
        print(
            f"max_seq_len: {max_seq_len}, num_kv_heads: {self.num_kv_heads}, num_layers: {self.num_layers} \
                \nhidden_size: {self.hidden_size}"
        )

    def embedding_weight_save(
        self, embedding_weight_save_path=None, embedding_quantize_config=None
    ):
        embedding_layer = self.model.get_input_embeddings()
        weight = embedding_layer.weight
        embedding_weight = weight.detach().numpy().astype(np.float16)
        if embedding_quantize_config == "W4A8":
            weight_4bit = quantize_weight_g128_4bit_nz(embedding_weight.T)
            weight_4bit.tofile(embedding_weight_save_path)
        elif embedding_quantize_config == "W4A16":
            weight_4bit_gp32 = quantize_weight_g32_4bit_nd(embedding_weight.T)
            weight_4bit_gp32.tofile(embedding_weight_save_path)
        else:
            embedding_weight.flatten().tofile(embedding_weight_save_path)
        print(f"Save {embedding_weight_save_path}")

    def rope_sin_cos_save(self, cos_path, sin_path, seq_len):
        """save rope cos and sin constant"""
        rotary_layer = self.model.model.layers[0].self_attn.rotary_emb

        valid_seq_len = seq_len
        input_embed = torch.rand(
            1, valid_seq_len, self.hidden_size, dtype=torch.float16
        ).to(device)
        position_ids = torch.arange(0, valid_seq_len).unsqueeze(0)
        rope_cos, rope_sin = rotary_layer(input_embed, valid_seq_len)

        rope_cos = rope_cos[position_ids].detach().numpy().astype(np.float16)
        rope_sin = rope_sin[position_ids].detach().numpy().astype(np.float16)

        rope_cos.flatten().tofile(cos_path)
        rope_sin.flatten().tofile(sin_path)

    def attention_mask_save(self, attention_mask_path, max_seq_len):
        _, tgt_len = 1, max_seq_len
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min)
        mask_cond = torch.arange(mask.size(-1))
        mask.masked_fill_(mask_cond < (
            mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        attention_mask = mask[None, None, :, :].expand(1, 1, tgt_len, tgt_len)
        attention_mask.to(torch.float16).detach().numpy().tofile(
            attention_mask_path
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="export llm to various onnx.")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="/path/to/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument(
        "-o", "--onnx_model", type=str, required=True, help="onnx model path"
    )
    parser.add_argument(
        "--embedding_weight_save_path",
        type=str,
        required=True,
        help="path to embedding weight",
    )
    parser.add_argument(
        "--cos_weight_save_path",
        type=str,
        required=True,
        help="path to cos weight",
    )
    parser.add_argument(
        "--sin_weight_save_path",
        type=str,
        required=True,
        help="path to sin weight",
    )
    parser.add_argument(
        "--attention_mask_save_path",
        type=str,
        required=True,
        help="path to attention mask weight",
    )
    parser.add_argument(
        "--embedding_quantize_config",
        type=str,
        required=False,
        help="embedding/lm_head quantization config",
    )
    parser.add_argument(
        "--decoder_quantize_config",
        type=str,
        required=False,
        help="decoder layer quantization config",
    )
    args = parser.parse_args()
    max_length = 1024
    chunk_size = 128
    apply_qwen2_patch()
    qwen2onnx = Qwen2Onnx()

    # load torch ckpt
    logging.info("load model: %s...", args.model)
    qwen2onnx.load(args.model, 24)
    logging.info("loaded model: %s\n", args.model)

    embedding_quant_config = QuantizationConfig(
        args.embedding_quantize_config
    )
    decoder_quant_config = QuantizationConfig(
        args.decoder_quantize_config
    )
    model_config = ModelConfig(
        max_length=max_length,
        chunk_size=chunk_size,
        vocab_size=qwen2onnx.config.vocab_size,
        hidden_size=qwen2onnx.config.hidden_size,
        num_attention_heads=qwen2onnx.config.num_attention_heads,
        num_key_value_heads=qwen2onnx.config.num_key_value_heads,
        eos_id=qwen2onnx.config.eos_token_id,
        embedding_quant=embedding_quant_config,
        decoder_quant=decoder_quant_config,
    )
    lite_turbo_config = LiteTurboConfig(
        max_length=max_length,
        chunk_size=chunk_size,
        vocab_size=qwen2onnx.config.vocab_size,
        hidden_size=qwen2onnx.config.hidden_size,
        num_attention_heads=qwen2onnx.config.num_attention_heads,
        num_key_value_heads=qwen2onnx.config.num_key_value_heads,
        eos_id=qwen2onnx.config.eos_token_id,
        scale_gp_size=embedding_quant_config.group_size,
        embedding_quant=(
            True
            if args.embedding_quantize_config or args.decoder_quantize_config
            else False
        ),
        do_sample=True,
        temperature=0.3,
        top_k=50,
        top_p=0.9,
        typical_p=1.0,
        diversity_penalty=0.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        random_seed=42,
    )

    # export half precision model
    qwen2onnx.export(
        args.onnx_model,
        max_seq_len=max_length,
        chunk_size=chunk_size,
    )
    if args.embedding_quantize_config or args.decoder_quantize_config:
        path, file_name = os.path.split(args.onnx_model)
        file_name, ext = os.path.splitext(file_name)
        quant_model_path = os.path.join(path, file_name + "_quant" + ext)
        apply_quant(args.onnx_model, quant_model_path, model_config)
    else:
        model = onnx.load(args.onnx_model)
        apply_shared_weight(model)
        if model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
            onnx.save(
                model,
                args.onnx_model,
                save_as_external_data=True,
                location=os.path.basename(args.onnx_model) + ".data",
            )
        else:
            onnx.save(model, args.onnx_model)
    path, _ = os.path.split(args.onnx_model)
    qwen2onnx.embedding_weight_save(
        args.embedding_weight_save_path, args.embedding_quantize_config
    )
    qwen2onnx.rope_sin_cos_save(
        args.cos_weight_save_path, args.sin_weight_save_path, model_config.max_length)
    qwen2onnx.attention_mask_save(
        args.attention_mask_save_path, model_config.max_length)
    lite_turbo_json_str = json.dumps(
        lite_turbo_config.asdict(), indent=4, separators=(",", ": ")
    )
    lite_turbo_config_path = os.path.join(path, "lite_turbo_config.json")
    with open(lite_turbo_config_path, "w", encoding="utf-8") as f:
        f.write(lite_turbo_json_str)
        print(f"Save model config to {lite_turbo_config_path}")
