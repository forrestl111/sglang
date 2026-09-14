#!/usr/bin/env python3
"""把 alibaba-pai 的 PDD LoRA 烘进 MiniMax-H3 的官方权重。

    python3 build_pdd_weights.py <底模transformer目录> <lora.safetensors> <输出目录>

产出两样东西：

* 一份完整的 transformer 权重，2D 的 LoRA 增量已经合并进去（sglang 和 VideoX-Fun
  都能当普通 checkpoint 直接加载）；
* `pdd_heads.safetensors`：32 份位置级输出头，外加 `pdd_config.json` 记下
  num_steps / block_size。这部分不能合并 —— 它是 PDD 的核心，见下。

## PDD 是什么，为什么输出头不能合并

Parallel Decoding Distillation（https://research.nvidia.com/labs/genair/pdd/）
把时间轴切成 N 个区间并按 L 个一组分块，"the parallel decoder utilizes the same
backbone, but with the final linear layer replicated N times"。推理时每次 backbone
求值推进一整块，即 N/L 次 forward —— 这里 32/4 = 8，就是 "8Step" 的来源。

所以 checkpoint 里的 `proj_out.weight` 是 (32, 96, 5376)：32 份最终线性层，比底模
的 (96, 5376) 多一个位置维。数值上它们都是底模那一层的小扰动（相对差 1~5%，本脚本
`--verify` 会打出来），这正是"复制 N 份再微调"的样子。

一个块内的 4 个头共享同一次 backbone 求值。论文说生成时可以"fuse the layers into a
single linear layer that directly predicts a step across the full block"；对 flow
matching 而言那就是按 Δσ 加权求和 —— 因为
    x_{n+L} = x_n + Σ_j Δσ_{n+j} · W_{n+j} h
中的 h 是同一个，求和可以提到 W 上。本脚本不做这个融合：调用方逐个应用 4 个头、走
4 个 scheduler 步，与融合逐位等价，却不需要我们自己复刻 σ 网格和权重推导。省下的
计算量也可以忽略（96x5376 的 GEMM）。

## 命名映射

LoRA 用 diffusers 命名，官方权重用它自己的一套，两边对不上。逐条对应：

    transformer_blocks.N.attn.to_q/k/v   -> blocks.N.attn.qkv_proj   （见下）
    transformer_blocks.N.attn.to_out.0   -> blocks.N.attn.out_proj
    transformer_blocks.N.ff.net.0.proj   -> blocks.N.mlp.fc1
    transformer_blocks.N.ff.net.2        -> blocks.N.mlp.fc2
    transformer_blocks.N.adaln_proj.linear -> blocks.N.adaln_proj.linear
    token_refiner.refiner_blocks.N.*     -> token_refiner.blocks.N.*
    proj_out / audio_proj_out            -> 位置级输出头，不合并

qkv 是唯一需要动布局的：官方权重按**头**交错存放，每个头一段
`[q(128) | k(128) | v(128)]`，而 LoRA 把 q/k/v 分成三张表。合并前必须把三份增量
按同样的交错顺序拼起来，否则每个头都会拿到别人的 K/V —— 产物不会报错，只会安静地
变成噪声。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HEAD_DIM = 128


def _lora_delta(down: torch.Tensor, up: torch.Tensor, scale: float) -> torch.Tensor:
    return (up.float() @ down.float()) * scale


def _interleave_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """[3 x (heads*head_dim), in] -> 官方权重的逐头交错布局。"""
    heads = q.shape[0] // HEAD_DIM
    in_dim = q.shape[1]
    grouped = torch.empty(heads, 3 * HEAD_DIM, in_dim, dtype=q.dtype)
    grouped[:, 0 * HEAD_DIM : 1 * HEAD_DIM] = q.view(heads, HEAD_DIM, in_dim)
    grouped[:, 1 * HEAD_DIM : 2 * HEAD_DIM] = k.view(heads, HEAD_DIM, in_dim)
    grouped[:, 2 * HEAD_DIM : 3 * HEAD_DIM] = v.view(heads, HEAD_DIM, in_dim)
    return grouped.reshape(heads * 3 * HEAD_DIM, in_dim)


def _target_of(lora_key: str) -> str | None:
    """LoRA 的模块名 -> 官方权重的模块名。返回 None 表示不由本函数处理。"""
    k = lora_key
    if k.startswith("token_refiner.refiner_blocks."):
        k = "token_refiner.blocks." + k[len("token_refiner.refiner_blocks.") :]
    elif k.startswith("transformer_blocks."):
        k = "blocks." + k[len("transformer_blocks.") :]
    else:
        return None
    for src, dst in (
        (".attn.to_out.0", ".attn.out_proj"),
        (".ff.net.0.proj", ".mlp.fc1"),
        (".ff.net.2", ".mlp.fc2"),
        (".adaln_proj.linear", ".adaln_proj.linear"),
    ):
        if k.endswith(src):
            return k[: -len(src)] + dst
    return k  # attn.to_q/k/v 保持原样，由调用方合并


def load_lora(path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(path, "pt") as f:
        meta = f.metadata() or {}
        return {k: f.get_tensor(k) for k in f.keys()}, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir")
    ap.add_argument("lora_path")
    ap.add_argument("out_dir")
    ap.add_argument("--verify", action="store_true", help="打印每类改动的相对变化幅度")
    args = ap.parse_args()

    lora, meta = load_lora(args.lora_path)
    rank = int(meta.get("lora_rank", 64))
    alpha = float(meta.get("lora_alpha", rank))
    scale = alpha / rank
    pdd = {"num_steps": int(meta.get("pdd_num_steps", 32)),
           "block_size": int(meta.get("pdd_block_size", 4))}
    print(f"LoRA: rank={rank} alpha={alpha} scale={scale} pdd={pdd}")

    # 把 lora_down/lora_up 按模块归堆；q/k/v 单独留着，最后一起交错。
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in lora.items():
        for suffix in (".lora_down", ".lora_up"):
            if key.endswith(suffix):
                pairs.setdefault(key[: -len(suffix)], {})[suffix[1:]] = value
                break

    deltas: dict[str, torch.Tensor] = {}          # 官方权重名 -> ΔW
    qkv: dict[str, dict[str, torch.Tensor]] = {}  # 模块前缀 -> {q,k,v: ΔW}
    for module, dv in pairs.items():
        if "lora_down" not in dv or "lora_up" not in dv:
            raise SystemExit(f"{module} 只有半边 LoRA")
        delta = _lora_delta(dv["lora_down"], dv["lora_up"], scale)
        target = _target_of(module)
        if target is None:
            raise SystemExit(f"没有映射规则：{module}")
        for role in ("to_q", "to_k", "to_v"):
            if target.endswith(f".attn.{role}"):
                qkv.setdefault(target[: -len(f".attn.{role}")], {})[role[-1]] = delta
                break
        else:
            deltas[target + ".weight"] = delta

    for prefix, parts in qkv.items():
        if set(parts) != {"q", "k", "v"}:
            raise SystemExit(f"{prefix} 的 q/k/v 不全：{sorted(parts)}")
        deltas[prefix + ".attn.qkv_proj.weight"] = _interleave_qkv(
            parts["q"], parts["k"], parts["v"]
        )
    print(f"待合并的 2D 增量：{len(deltas)} 个（其中 qkv {len(qkv)} 个）")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    applied, shards = 0, []
    for shard in sorted(glob.glob(os.path.join(args.base_dir, "*.safetensors"))):
        with safe_open(shard, "pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        for key in list(tensors):
            if key in deltas:
                base = tensors[key]
                merged = base.float() + deltas[key].to(base.device)
                if args.verify:
                    rel = ((merged - base.float()).norm() / base.float().norm()).item()
                    print(f"  {key}: 相对变化 {rel:.4f}")
                tensors[key] = merged.to(base.dtype)
                applied += 1
        name = os.path.basename(shard)
        save_file(tensors, str(out / name), metadata={"format": "pt"})
        shards.append(name)
    print(f"合并了 {applied}/{len(deltas)} 个张量")
    if applied != len(deltas):
        missing = sorted(set(deltas) - set())
        raise SystemExit(f"有 {len(deltas) - applied} 个增量没找到落点，别用这份产物")

    for extra in ("config.json", "diffusion_pytorch_model.safetensors.index.json"):
        src = os.path.join(args.base_dir, extra)
        if os.path.exists(src):
            shutil.copy(src, out / extra)

    heads = {k: lora[k] for k in ("proj_out.weight", "proj_out.bias",
                                  "audio_proj_out.weight", "audio_proj_out.bias")}
    save_file(heads, str(out / "pdd_heads.safetensors"), metadata={"format": "pt"})
    (out / "pdd_config.json").write_text(json.dumps(pdd, indent=2) + "\n")
    print(f"位置级输出头 -> {out/'pdd_heads.safetensors'}（{pdd['num_steps']} 份，"
          f"块大小 {pdd['block_size']}，即 {pdd['num_steps'] // pdd['block_size']} 次 forward）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
