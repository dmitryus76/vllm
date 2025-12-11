# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GigaAM model implementation for vLLM.

GigaAM is a Conformer-based foundation model for automatic speech recognition (ASR),
developed by Salute-Developers. It uses self-supervised learning with HuBERT-CTC
objectives and supports both CTC and RNN-T decoders.

Based on the Whisper model implementation in vLLM.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, cast

import numpy as np
import torch
from torch import nn
from transformers import (
    BatchFeature,
    PretrainedConfig,
)

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention, MultiHeadAttention
from vllm.attention.layers.cross_attention import CrossAttention
from vllm.config import CacheConfig, ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils.jsontree import json_map_leaves
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsTranscription
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    cast_overflow_tensors,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

# GigaAM primarily supports Russian, but may support other languages
GIGAAM_SUPPORTED_LANGS = {
    "ru": "Russian",
    "en": "English",
}


class GigaAmAudioInputs(TensorSchema):
    """
    Audio inputs for GigaAM model.
    
    Dimensions:
        - b: Batch size
        - nmb: Number of mel bins
        - t: Time frames
    """

    input_features: Annotated[
        list[torch.Tensor] | None,
        TensorShape("b", "nmb", "t"),
    ]


class GigaAmEncoderAttention(MultiHeadAttention):
    """Multi-headed attention for GigaAM encoder with 2D tensor support."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input shape: batch_size x seq_len x hidden_size
                     or seq_len x hidden_size
        """
        is_2d = query.dim() == 2
        if is_2d:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

        out = super().forward(query, key, value)

        if is_2d:
            out = out.squeeze(0)

        return out


class GigaAmAttention(nn.Module):
    """Attention module for GigaAM Conformer blocks."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_type: AttentionType = AttentionType.DECODER,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        
        if self.total_num_heads >= tp_size:
            assert self.total_num_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_heads == 0
            
        self.num_kv_heads = max(1, self.total_num_heads // tp_size)
        self.head_dim = self.embed_dim // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_type = attn_type

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self._init_qkv(embed_dim, bias, quant_config, prefix=prefix)
        self.out_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        
        if attn_type == AttentionType.ENCODER:
            self.attn = GigaAmEncoderAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
            )
        elif self.attn_type == AttentionType.ENCODER_DECODER:
            self.attn = CrossAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                attn_type=self.attn_type,
            )
        else:  # AttentionType.DECODER
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                attn_type=self.attn_type,
            )

    def _init_qkv(
        self,
        embed_dim: int,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        attn_output = self.attn(q, k, v)

        output, _ = self.out_proj(attn_output)

        return output


class GigaAmCrossAttention(GigaAmAttention):
    """Cross-attention module for GigaAM decoder."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            attn_type=AttentionType.ENCODER_DECODER,
        )

    def _init_qkv(
        self,
        embed_dim: int,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.q_proj = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.kv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ):
        q, _ = self.q_proj(hidden_states)

        if encoder_hidden_states is not None:
            kv, _ = self.kv_proj(encoder_hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            k = v = None

        attn_output = self.attn(q, k, v)

        output, _ = self.out_proj(attn_output)

        return output


class GigaAmFeedForward(nn.Module):
    """Feed-forward module for GigaAM Conformer blocks."""

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        act_fn: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.activation_fn = get_act_fn(act_fn)
        self.fc1 = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            input_size=ffn_dim,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class GigaAmConformerBlock(nn.Module):
    """
    Conformer block for GigaAM encoder.
    
    The Conformer block combines multi-head self-attention and convolution modules
    for capturing both global and local dependencies in audio signals.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.embed_dim = getattr(config, "d_model", getattr(config, "hidden_size", 512))
        
        # Multi-head self-attention
        self.self_attn = GigaAmAttention(
            embed_dim=self.embed_dim,
            num_heads=getattr(config, "encoder_attention_heads", 
                            getattr(config, "num_attention_heads", 8)),
            attn_type=AttentionType.ENCODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Feed-forward module
        ffn_dim = getattr(config, "encoder_ffn_dim", 
                         getattr(config, "intermediate_size", 2048))
        act_fn = getattr(config, "activation_function", "relu")
        
        self.ffn = GigaAmFeedForward(
            embed_dim=self.embed_dim,
            ffn_dim=ffn_dim,
            act_fn=act_fn,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        # Convolution module (simplified for now)
        # In full Conformer, this would be a depthwise separable convolution
        kernel_size = getattr(config, "conv_kernel_size", 31)
        self.conv = nn.Conv1d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=self.embed_dim,
        )
        self.conv_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        # Convolution module
        residual = hidden_states
        hidden_states = self.conv_layer_norm(hidden_states)
        # Conv1d expects (batch, channels, seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = residual + hidden_states

        # Feed-forward
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = cast_overflow_tensors(hidden_states)

        return hidden_states


class GigaAmDecoderLayer(nn.Module):
    """Decoder layer for GigaAM model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        embed_dim = getattr(config, "d_model", getattr(config, "hidden_size", 512))
        num_heads = getattr(config, "decoder_attention_heads",
                          getattr(config, "num_attention_heads", 8))
        ffn_dim = getattr(config, "decoder_ffn_dim",
                         getattr(config, "intermediate_size", 2048))
        act_fn = getattr(config, "activation_function", "relu")

        self.self_attn = GigaAmAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_type=AttentionType.DECODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        
        self.encoder_attn = GigaAmCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)
        
        self.ffn = GigaAmFeedForward(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            act_fn=act_fn,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn",
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ):
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        # Cross-attention
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = residual + hidden_states

        # Feed-forward
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GigaAmEncoder(nn.Module):
    """
    GigaAM Conformer encoder.
    
    Processes audio features through Conformer blocks to produce
    high-level speech representations.
    """

    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = ""
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        
        embed_dim = getattr(config, "d_model", getattr(config, "hidden_size", 512))
        self.num_mel_bins = getattr(config, "num_mel_bins", 80)
        self.max_source_positions = getattr(config, "max_source_positions", 1500)

        # Audio feature projection
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        
        # Conformer blocks
        num_layers = getattr(config, "encoder_layers", 
                           getattr(config, "num_hidden_layers", 12))
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_layers,
            lambda prefix: GigaAmConformerBlock(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Positional embeddings
        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

    def forward(self, input_features: torch.Tensor | list[torch.Tensor]):
        hidden_states = []
        for features in input_features:
            # Apply convolutions
            embeds = nn.functional.gelu(self.conv1(features))
            embeds = nn.functional.gelu(self.conv2(embeds))
            embeds = embeds.transpose(-1, -2)
            
            # Add positional embeddings
            embeds = (embeds + self.embed_positions.weight[: embeds.size(-2), :]).to(
                embeds.dtype
            )
            hidden_states.append(embeds)
        
        hidden_states = torch.cat(hidden_states)

        # Pass through Conformer blocks
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class GigaAmDecoder(nn.Module):
    """GigaAM decoder for generating text from audio representations."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        
        embed_dim = getattr(config, "d_model", getattr(config, "hidden_size", 512))
        vocab_size = getattr(config, "vocab_size", 32000)
        padding_idx = getattr(config, "pad_token_id", 0)
        self.max_target_positions = getattr(config, "max_target_positions", 448)
        self.max_source_positions = getattr(config, "max_source_positions", 1500)

        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx)
        self.embed_positions = nn.Embedding(self.max_target_positions, embed_dim)
        
        num_layers = getattr(config, "decoder_layers",
                           getattr(config, "num_hidden_layers", 6))
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_layers,
            lambda prefix: GigaAmDecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ):
        inputs_embeds = self.embed_input_ids(input_ids)
        positions = self.embed_positions(positions)
        hidden_states = inputs_embeds + positions

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class GigaAmModel(nn.Module):
    """GigaAM encoder-decoder model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.encoder = GigaAmEncoder(
            vllm_config=vllm_config, prefix=f"{prefix}.encoder"
        )
        self.decoder = GigaAmDecoder(
            vllm_config=vllm_config, prefix=f"{prefix}.decoder"
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        assert len(encoder_outputs) in (0, 1)
        enc_states = encoder_outputs[0] if len(encoder_outputs) == 1 else None
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=enc_states,
        )
        return decoder_outputs

    def get_encoder_outputs(
        self,
        input_features: torch.Tensor | list[torch.Tensor] | None,
    ) -> torch.Tensor | None:
        if input_features is None:
            return None
        return self.encoder(input_features)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class GigaAmProcessingInfo(BaseProcessingInfo):
    """Processing info for GigaAM model."""

    def get_hf_config(self) -> PretrainedConfig:
        return self.ctx.get_hf_config(PretrainedConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_feature_extractor(self, **kwargs: object):
        """Get feature extractor for audio processing."""
        hf_processor = self.get_hf_processor(**kwargs)
        # GigaAM uses a custom feature extractor
        if hasattr(hf_processor, "feature_extractor"):
            return hf_processor.feature_extractor
        return hf_processor

    def get_num_audio_tokens(self) -> int:
        config = self.get_hf_config()
        return getattr(config, "max_source_positions", 1500)


class GigaAmDummyInputsBuilder(BaseDummyInputsBuilder[GigaAmProcessingInfo]):
    """Dummy inputs builder for GigaAM profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        # Use a simple token for audio input placeholder
        return "<audio>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        # Get sampling rate from feature extractor if available
        sampling_rate = getattr(feature_extractor, "sampling_rate", 16000)
        # Assume chunk length or use default
        chunk_length = getattr(feature_extractor, "chunk_length", 30)
        audio_len = chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }


class GigaAmMultiModalProcessor(EncDecMultiModalProcessor[GigaAmProcessingInfo]):
    """Multimodal processor for GigaAM."""

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        target_sr = getattr(feature_extractor, "sampling_rate", 16000)
        return MultiModalDataParser(target_sr=target_sr)

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
    ) -> str | list[int]:
        # GigaAM encoder only accepts audio features
        # Create dummy encoder prompt for profiling
        return [0]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
            # Get audios from mm_data, default to empty list if not present
            audios = mm_data.pop("audios", [])
            mm_data = dict(audio=audios)
            mm_kwargs = dict(
                **mm_kwargs,
                sampling_rate=getattr(feature_extractor, "sampling_rate", 16000),
            )
        
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        
        if "labels" in processed_outputs:
            processed_outputs["input_ids"] = processed_outputs.pop("labels")
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(input_features=MultiModalFieldConfig.batched("audio"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        num_tokens = self.info.get_num_audio_tokens()
        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=[0] * num_tokens,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    GigaAmMultiModalProcessor,
    info=GigaAmProcessingInfo,
    dummy_inputs=GigaAmDummyInputsBuilder,
)
class GigaAmForConditionalGeneration(
    nn.Module, SupportsTranscription, SupportsMultiModal
):
    """
    GigaAM model for conditional generation (speech-to-text).
    
    This is the main model class that wraps the encoder-decoder architecture
    and provides the interface for transcription and multimodal support.
    """
    
    packed_modules_mapping = {
        "self_attn.qkv_proj": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ],
        "encoder_attn.kv_proj": ["encoder_attn.k_proj", "encoder_attn.v_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".fc1.": ".ffn.fc1.", ".fc2.": ".ffn.fc2."}
    )

    # GigaAM only supports audio-conditioned generation
    supports_transcription_only = True
    supports_segment_timestamp = False
    supported_languages = GIGAAM_SUPPORTED_LANGS

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is None:
            logger.warning(
                "Defaulting to language='ru' for GigaAM. If you wish to transcribe "
                "audio in a different language, pass the `language` field "
                "in the TranscriptionRequest."
            )
            language = "ru"
        return super().validate_language(language)

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        if language is None:
            language = "ru"
        
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": (audio, stt_config.sample_rate),
                },
            },
            "decoder_prompt": (
                f"<s>{request_prompt}" if request_prompt else "<s>"
            ),
        }
        return cast(PromptType, prompt)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return None
        raise ValueError("Only audio modality is supported")

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)
        
        feature_extractor = (
            processor.feature_extractor
            if hasattr(processor, "feature_extractor")
            else processor
        )
        
        chunk_length = getattr(feature_extractor, "chunk_length", 30)
        sample_rate = getattr(feature_extractor, "sampling_rate", 16000)

        return SpeechToTextConfig(
            max_audio_clip_s=chunk_length,
            sample_rate=sample_rate,
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        processor = cached_processor_from_config(model_config)
        feature_extractor = (
            processor.feature_extractor
            if hasattr(processor, "feature_extractor")
            else processor
        )
        
        hop_length = getattr(feature_extractor, "hop_length", 160)
        return math.ceil(audio_duration_s * stt_config.sample_rate / hop_length)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype

        self.model = GigaAmModel(vllm_config=vllm_config, prefix=prefix)

        vocab_size = getattr(config, "vocab_size", 32000)
        embed_dim = getattr(config, "d_model", getattr(config, "hidden_size", 512))
        
        self.proj_out = ParallelLMHead(
            vocab_size,
            embed_dim,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "proj_out"),
        )
        self.proj_out = self.proj_out.tie_weights(self.model.decoder.embed_tokens)
        
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(vocab_size, scale=logit_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_outputs is None:
            encoder_outputs = []
        decoder_outputs = self.model(
            input_ids=input_ids,
            positions=positions,
            encoder_outputs=encoder_outputs,
        )
        return decoder_outputs

    def get_language_model(self) -> torch.nn.Module:
        return self.model.decoder

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        return [self.model.get_encoder_outputs(audio_input["input_features"])]

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        return self.model.decoder.embed_input_ids(input_ids)

    def _parse_and_validate_audio_input(self, **kwargs: object) -> GigaAmAudioInputs:
        input_features = kwargs.pop("input_features", None)

        if input_features is not None:
            input_features = json_map_leaves(lambda x: x.to(self.dtype), input_features)

        return GigaAmAudioInputs(input_features=input_features)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.proj_out, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["proj_out."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
