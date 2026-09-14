#!/usr/bin/env python3
"""把 PDD 的 32 份位置级输出头融合成每块一份，共 8 份。

    python3 fuse_pdd_heads.py <pdd目录> [--video-shift 12.0 --audio-shift 3.0]

## 为什么可以融合

MiniMax-H3 的调度器名字里有 ancestral，但走的是 eta0 分支，不注噪：

    x_next = r·x + (1-r)·x0,  r = σ_next/σ_curr,  x0 = x - σ_curr·v
           = x - (σ_curr - σ_next)·v

纯 Euler，且对 v 线性。PDD 的一个块里 L 个头共享同一次 backbone 求值，所以

    x_{n+L} = x_n - Σ_j Δσ_{n+j} · W_{n+j} h
            = x_n - (Σ_j Δσ_{n+j}) · (Σ_j w_j W_{n+j}) h,   w_j = Δσ_{n+j} / Σ Δσ

括号里就是融合头 —— 论文所说的 "fuse the layers into a single linear layer that
directly predicts a step across the full block"。于是每块只剩一次前向、一次调度器
步进，去噪循环一行都不用改。

要注意这一步依赖 eta0：换成真正的 ancestral（每步注噪），块内第二步起的状态就不再是
块首状态，求和提不到 W 上，融合失效，必须老老实实做 L 次子步。

## σ 网格为什么对得上

H3 的 σ 是 `s·base/(1+(s-1)·base)`，`base = linspace(1, 0, num_steps)`。
`linspace(1,0,33)[::4] == linspace(1,0,9)`，而 shift 是逐点单调变换，所以 32 区间网格
每隔 4 取一个，恰好等于 8 区间网格。跑的时候给 `--num-inference-steps 9`（H3 的步数
是网格点数，实际去噪 8 步）即可，无需自定义调度。

video 和 audio 有各自的 shift（FL2VA 是 12.0 / 3.0），两个头分别用自己的网格加权。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sigma_grid(num_points: int, shift: float) -> torch.Tensor:
    base = torch.linspace(1.0, 0.0, num_points, dtype=torch.float64)
    return shift * base / (1 + (shift - 1) * base)


def fuse(heads: torch.Tensor, sigmas: torch.Tensor, block: int) -> torch.Tensor:
    """heads: (N, ...) -> (N//block, ...)，块内按 Δσ 加权平均。"""
    steps = heads.shape[0]
    if sigmas.numel() != steps + 1:
        raise SystemExit(f"σ 网格点数 {sigmas.numel()} 与头数 {steps} 对不上")
    dsigma = (sigmas[:-1] - sigmas[1:]).clamp(min=0)
    out = []
    for b in range(steps // block):
        w = dsigma[b * block : (b + 1) * block]
        if float(w.sum()) <= 0:
            raise SystemExit(f"第 {b} 块的 Δσ 全为 0")
        w = (w / w.sum()).to(heads.dtype)
        chunk = heads[b * block : (b + 1) * block].float()
        out.append((chunk * w.float().view(-1, *([1] * (chunk.dim() - 1)))).sum(0))
    return torch.stack(out).to(heads.dtype)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdd_dir")
    ap.add_argument("--video-shift", type=float, default=12.0)
    ap.add_argument("--audio-shift", type=float, default=3.0)
    args = ap.parse_args()

    d = Path(args.pdd_dir)
    cfg = json.loads((d / "pdd_config.json").read_text())
    n, block = cfg["num_steps"], cfg["block_size"]
    with safe_open(str(d / "pdd_heads.safetensors"), "pt") as f:
        heads = {k: f.get_tensor(k) for k in f.keys()}

    sv = sigma_grid(n + 1, args.video_shift)
    sa = sigma_grid(n + 1, args.audio_shift)
    fused = {
        "video_out.weight": fuse(heads["proj_out.weight"], sv, block),
        "video_out.bias": fuse(heads["proj_out.bias"], sv, block),
        "audio_out.weight": fuse(heads["audio_proj_out.weight"], sa, block),
        "audio_out.bias": fuse(heads["audio_proj_out.bias"], sa, block),
    }
    save_file(fused, str(d / "pdd_fused_heads.safetensors"), metadata={"format": "pt"})
    cfg["fused_steps"] = n // block
    cfg["num_inference_steps"] = n // block + 1   # H3 的步数是网格点数
    cfg["video_shift"], cfg["audio_shift"] = args.video_shift, args.audio_shift
    (d / "pdd_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    print(f"融合 {n} -> {n // block} 份（块大小 {block}）")
    for k, v in fused.items():
        print(f"  {k}: {tuple(v.shape)}")
    print(f"跑的时候用 --num-inference-steps {cfg['num_inference_steps']}")
    print("视频 σ 网格前 9 点:", [round(float(x), 4) for x in sv[::block]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
