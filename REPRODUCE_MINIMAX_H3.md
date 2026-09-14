# MiniMax-H3 方案5 复现说明

4x B200 (SM100)，MiniMax-H3 768p / 5s。方案5 = fp8 + cache-dit + SubBlock sparse
attention + PDD。

## 两份代码

| 仓库 | 分支 | 内容 |
|---|---|---|
| sglang | `minimax-h3-pdd-and-elementwise` | PDD 集成、fused silu_and_mul、uint32 relayout 索引、SubBlock 输出暂存消除 |
| flashinfer | `minimax-h3-bsa-output-permute` | blk64 输出 TMA 直写 BSHD（基于 upstream main） |

flashinfer 另有 `minimax-h3-bsa-legacy-cpp-kernel` 分支，针对的是被 main 重写掉的
旧版 C++ AOT kernel（Q/K/V staging 消除、V 的 MN-major）。main 的 CuTe-DSL 实现已经
用 `maybe_contiguous` 独立避开了 Q/K/V 拷贝，所以那个分支只在需要复现旧版数据时才用。

## 权重

```bash
hf download MiniMaxAI/MiniMax-H3            --local-dir $W/MiniMax-H3
hf download alibaba-pai/MiniMax-H3-Acc-LoRAs --local-dir $W/pdd-loras
```

FL2VA 分区的 `video_vae/` 权重在 `source/` 子目录里，而 sglang 的完整性检查只看
组件目录顶层，需要补一个软链：

```bash
ln -s source/model.safetensors $W/MiniMax-H3/FL2VA/video_vae/model.safetensors
```

## 构建 PDD checkpoint

```bash
python3 build_pdd_weights.py $W/MiniMax-H3/FL2VA/transformer \
    $W/pdd-loras/MiniMax-H3-FL2VA-Acc-8Step.safetensors $PDD/transformer --verify
mkdir -p $PDD/pdd_extra && mv $PDD/transformer/pdd_{heads.safetensors,config.json} $PDD/pdd_extra/
python3 fuse_pdd_heads.py $PDD/pdd_extra
```

32 份 position-level heads 必须移出 `transformer/`：sglang 按 glob 找 shard，会把它们
当成普通权重加载，然后在 `final_layer.audio_out.bias` 形状 (32,32) vs (32,) 上崩掉。

## 启动

```bash
export SGLANG_CACHE_DIT_ENABLED=1 SGLANG_CACHE_DIT_FN=1 SGLANG_CACHE_DIT_BN=0 \
       SGLANG_CACHE_DIT_WARMUP=4 SGLANG_CACHE_DIT_RDT=0.24 SGLANG_CACHE_DIT_MC=3
export SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS=$PDD/pdd_extra/pdd_fused_heads.safetensors

sglang serve --model-path $PDD --num-gpus 4 --ulysses-degree 4 --performance-mode speed \
  --quantization fp8 --attention-backend subblock_sparse_attn \
  --attention-backend-config '{"skip_first_steps": 1}' \
  --component-attention-backends text_encoder=fa
```

请求用 `--num-inference-steps 9`（8 次 PDD forward）。

**`skip_first_steps` 必须调低。** 默认是 10，而 PDD 只有 8 次 forward，于是每一步都
落在"前 N 步保持 dense"的窗口里，block-sparse kernel 一次都不会跑 —— nsys 里
`bsa_attn_fwd_kernel` 调用数为 0，flashinfer 侧的优化全部失效。代价是画质：把这个
cutoff 从 10 降到 5 会让 cosine-vs-dense 从 0.558 掉到 0.310 并明显改变构图，降到 1
只会更多，这是速度/质量的取舍，不是免费的。

## 实测（4x B200，warmup 后 5 个 prompt 取中位数）

方案1（t2va，49 步）denoise，逐项叠加：

| | denoise |
|---|---|
| stock flashinfer | 10.02 s |
| + host staging 合并 | 9.24 s |
| + V MN-major | 9.13 s |
| + fused silu_mul | 8.75 s |
| + uint32 relayout 索引 | 8.75 s |
| + K/V zero-copy TMA | 8.57 s |
| + Q zero-copy TMA | 8.49 s |

精度（对无损 bf16 dense baseline 逐帧）：PSNR 18.68 -> 18.65 dB，SSIM 0.7261 -> 0.7248。

注意这条流水线**不是逐次确定的**：未改动的 stock build 连跑两次，同一 prompt 的 PSNR
能差 30 dB（落到另一个同样合理的样本上）。所以端到端视频对比不能用来判断改动是否等价，
要用固定输入的 kernel 级对拍（`torch.equal`）。以上每一项改动都是这样验证的。
