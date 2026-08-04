# SPDX-License-Identifier: Apache-2.0
"""Online NVFP4 quantization for diffusion transformers (FlashInfer).

Mirrors the online ``--quantization fp8`` path: load BF16/FP16 weights, convert
them to packed NVFP4 at ``process_weights_after_loading``, and quantize
activations dynamically at GEMM time. Boundary layers can stay unquantized via
``ignored_layers`` or by constructing linears with ``quant_config=None``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    _get_fp4_gemm_op,
    _get_fp4_quantize_op,
)
from sglang.multimodal_gen.runtime.models.parameter import ModelWeightParameter
from sglang.srt.layers.quantization.modelopt_quant import (
    pad_nvfp4_activation_for_cutlass,
    slice_nvfp4_output,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.layers.utils.common import copy_or_rebind_param
from sglang.srt.utils.common import is_flashinfer_available

logger = logging.getLogger(__name__)

_NVFP4_GROUP_SIZE = 16
_FP4_E2M1_MAX = 6.0


def _e4m3_max() -> float:
    return float(torch.finfo(torch.float8_e4m3fn).max)


def _amax_to_nvfp4_scale(amax: torch.Tensor) -> torch.Tensor:
    """Convert an amax to the ModelOpt-style NVFP4 per-tensor decode scale."""
    fp8_fp4_max = _e4m3_max() * _FP4_E2M1_MAX
    return torch.where(
        amax > 0,
        amax / fp8_fp4_max,
        torch.ones_like(amax),
    )


def _quantize_weight_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed NVFP4 weight, linear block scales, and per-tensor decode scale."""
    if not is_flashinfer_available():
        raise RuntimeError(
            "Online NVFP4 quantization requires flashinfer. "
            "Install flashinfer with Blackwell FP4 support."
        )
    from flashinfer import SfLayout, nvfp4_quantize

    if weight.ndim != 2:
        raise ValueError(
            f"Online NVFP4 expects 2D weights, got shape {tuple(weight.shape)}."
        )
    if weight.shape[-1] % _NVFP4_GROUP_SIZE != 0:
        raise ValueError(
            "Online NVFP4 requires weight K to be a multiple of "
            f"{_NVFP4_GROUP_SIZE}, got shape {tuple(weight.shape)}."
        )

    weight = weight.contiguous()
    weight_amax = weight.abs().nan_to_num().amax().to(
        device=weight.device, dtype=torch.float32
    )
    weight_scale_2 = _amax_to_nvfp4_scale(weight_amax)
    fp4_weight, weight_sf = nvfp4_quantize(
        weight,
        (1.0 / weight_scale_2).to(device=weight.device, dtype=torch.float32),
        sfLayout=SfLayout.layout_linear,
        backend="cute-dsl",
    )
    rows, cols = weight.shape
    weight_sf = weight_sf.view(torch.float8_e4m3fn).reshape(
        rows, cols // _NVFP4_GROUP_SIZE
    )
    return (
        fp4_weight.reshape(rows, cols // 2).contiguous(),
        weight_sf.contiguous(),
        weight_scale_2,
    )


class Nvfp4Config(QuantizationConfig):
    """Online NVFP4 config for diffusion DiTs.

    No-arg ``Nvfp4Config()`` selects post-load weight quantization with dynamic
    activation quantization through FlashInfer FP4 kernels.
    """

    def __init__(
        self,
        ignored_layers: Optional[List[str]] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
        group_size: int = _NVFP4_GROUP_SIZE,
    ) -> None:
        super().__init__()
        self.is_checkpoint_nvfp4_serialized = False
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}
        self.group_size = group_size

    @classmethod
    def get_name(cls) -> str:
        return "nvfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict) -> "Nvfp4Config":
        ignored_layers = config.get("ignored_layers") or config.get(
            "modules_to_not_convert"
        )
        if isinstance(ignored_layers, str):
            ignored_layers = [ignored_layers]
        return cls(
            ignored_layers=ignored_layers,
            packed_modules_mapping=config.get("packed_modules_mapping"),
            group_size=int(config.get("group_size", _NVFP4_GROUP_SIZE)),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix,
                self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                logger.debug(
                    "NVFP4: Keeping layer %s unquantized (in ignored_layers)", prefix
                )
                return UnquantizedLinearMethod()
            input_size = getattr(layer, "input_size", None)
            if input_size is not None and input_size % self.group_size != 0:
                logger.info(
                    "NVFP4: Keeping layer %s unquantized "
                    "(input_size=%s not divisible by group_size=%s)",
                    prefix,
                    input_size,
                    self.group_size,
                )
                return UnquantizedLinearMethod()
            return Nvfp4LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Nvfp4LinearMethod(LinearMethodBase):
    """Online NVFP4 linear method backed by FlashInfer FP4 GEMM."""

    def __init__(self, quant_config: Nvfp4Config):
        self.quant_config = quant_config
        # Reuse the serialized FlashInfer layout/shuffle path without claiming
        # the source checkpoint was already NVFP4 (avoids the load-time warning).
        self._layout = ModelOptFp4LinearMethod(
            ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=False,
                group_size=quant_config.group_size,
                exclude_modules=[],
            )
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "Online NVFP4 requires input features divisible by "
                f"{self.quant_config.group_size}, got {input_size_per_partition}."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight
        if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(
                "Online NVFP4 expected BF16/FP16/FP32 source weights, "
                f"got dtype {weight.dtype}."
            )

        was_on_cpu = weight.device.type == "cpu"
        weight_data = weight.data
        if was_on_cpu:
            weight_data = weight_data.cuda()

        fp4_weight, weight_scale, weight_scale_2 = _quantize_weight_nvfp4(
            weight_data.to(dtype=torch.bfloat16)
        )

        # Rebuild the serialized NVFP4 parameter layout expected by the shared
        # FlashInfer post-process / GEMM path.
        layer.output_size_per_partition = fp4_weight.shape[0]
        copy_or_rebind_param(layer, "weight", fp4_weight)
        copy_or_rebind_param(layer, "weight_scale", weight_scale)
        copy_or_rebind_param(
            layer,
            "weight_scale_2",
            weight_scale_2.detach().to(dtype=torch.float32).reshape(1),
        )
        # No calibrated activation scale: use a unit decode scale and let
        # FlashInfer block scales plus a dynamic global scale handle range.
        copy_or_rebind_param(
            layer,
            "input_scale",
            torch.ones(1, dtype=torch.float32, device=fp4_weight.device),
        )
        self._layout.process_weights_after_loading(layer)
        # Keep the static weight decode scale for dynamic alpha updates.
        copy_or_rebind_param(
            layer,
            "weight_scale_2",
            weight_scale_2.detach().to(dtype=torch.float32).reshape(1),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_dtype = x.dtype
        input_shape = x.shape
        x_2d = x.view(-1, input_shape[-1])

        output_size = layer.output_size_per_partition
        output_shape = list(input_shape[:-1]) + [output_size]

        fp4_quantize = _get_fp4_quantize_op()
        if fp4_quantize is None:
            raise RuntimeError(
                "No FP4 quantization kernel available. Install flashinfer."
            )

        # Dynamic per-tensor activation global scale (online FP8-style).
        x_amax = (
            x_2d.abs().nan_to_num().amax().to(device=x_2d.device, dtype=torch.float32)
        )
        input_scale = _amax_to_nvfp4_scale(x_amax)
        input_scale_inv = (1.0 / input_scale).to(dtype=torch.float32)
        weight_scale_2 = layer.weight_scale_2.to(device=x_2d.device, dtype=torch.float32)
        alpha = (input_scale * weight_scale_2).reshape(1).to(dtype=torch.float32)

        x_fp4, x_scale_interleaved = fp4_quantize(x_2d, input_scale_inv)
        weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
        x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

        w = layer.weight
        w_scale_interleaved = layer.weight_scale_interleaved
        if x_scale_interleaved.dtype == torch.uint8:
            x_scale_interleaved = x_scale_interleaved.view(torch.float8_e4m3fn)
        if w_scale_interleaved.dtype == torch.uint8:
            w_scale_interleaved = w_scale_interleaved.view(torch.float8_e4m3fn)

        fp4_gemm, flashinfer_backend = _get_fp4_gemm_op()
        if fp4_gemm is None:
            raise RuntimeError("No FP4 GEMM kernel available. Install flashinfer.")

        out = fp4_gemm(
            x_fp4,
            w.T,
            x_scale_interleaved,
            w_scale_interleaved.T,
            alpha,
            output_dtype,
            backend=flashinfer_backend,
        )
        out = slice_nvfp4_output(out, output_size)
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


__all__ = [
    "Nvfp4Config",
    "Nvfp4LinearMethod",
]
