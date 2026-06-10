// Fused-transpose Ulysses all-to-all over NVLink P2P (CUDA IPC).
//
// This implements the head-scatter / sequence-gather collective used by
// Ulysses sequence parallelism (see python/sglang/multimodal_gen/runtime/
// layers/usp.py), with the layout permutation folded directly into the
// cross-GPU write addresses. It reuses the Signal / multi_gpu_barrier
// machinery from the custom all-reduce kernel for inter-GPU synchronization.
//
// Push model: each rank reads from its *local* input tensor (no registration
// required) and writes the head block destined for each peer directly into
// that peer's IPC-registered output staging buffer. Only the output staging
// buffers and the signal buffers need to be IPC-shared.
//
// head_dim == 2 layout, uniform sequence splits. With
//   W        = ulysses world size
//   H_local  = H / W
//   S_global = S_local * W
//
//   mode == 0 (input  a2a): [B, S_local, H,       D] -> [B, S_global, H_local, D]
//       y_r[b, j*S_local + s, hl, d] = x_j[b, s, r*H_local + hl, d]
//   mode == 1 (output a2a): [B, S_global, H_local, D] -> [B, S_local, H,       D]
//       out_j[b, s, r*H_local + hl, d] = u_r[b, j*S_local + s, hl, d]
//
// In both modes the unit of transfer is a contiguous (H_local * D) block, so
// every cross-GPU store is fully coalesced.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>

#include "allreduce/custom_all_reduce.cuh"

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

namespace sglang {

// Handle holding the IPC-shared output staging buffers and signals for one
// Ulysses group. It does not own any device memory; buffers are passed in from
// Python (see UlyssesP2PAllToAll).
class UlyssesA2A {
 public:
  int rank_;
  int world_size_;
  bool full_nvlink_;

  RankSignals sg_;
  Signal* self_sg_;
  // ptrs[r] points to rank r's output staging buffer (device pointer, opened
  // via IPC for remote ranks). Passed to the kernel by value.
  RankData out_ptrs_;
  // Convenience copy of this rank's own output staging buffer base pointer.
  void* local_out_buf_;

  UlyssesA2A(Signal** signals, void** out_bufs, int rank, int world_size, bool full_nvlink)
      : rank_(rank), world_size_(world_size), full_nvlink_(full_nvlink), self_sg_(signals[rank]) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
      out_ptrs_.ptrs[i] = out_bufs[i];
    }
    local_out_buf_ = out_bufs[rank];
  }
};

template <typename T, int NGPUS>
__global__ void __launch_bounds__(kDefaultThreads, 1) ulysses_a2a_push_kernel(
    const T* __restrict__ local_in,
    RankData out_ptrs,
    RankSignals sg,
    Signal* self_sg,
    int rank,
    int B,
    int S_local,
    int H_local,
    int D,
    int mode) {
  const int W = NGPUS;
  const int64_t H = static_cast<int64_t>(H_local) * W;
  const int64_t S_global = static_cast<int64_t>(S_local) * W;
  const int64_t block_len = static_cast<int64_t>(H_local) * D;  // elements per copy block
  const int64_t block_bytes = block_len * static_cast<int64_t>(sizeof(T));
  const int64_t num_copy_blocks = static_cast<int64_t>(B) * S_local * W;

  // Ensure every rank has entered before we start writing into peer buffers.
  multi_gpu_barrier<NGPUS, true>(sg, self_sg, rank);

  for (int64_t cb = blockIdx.x; cb < num_copy_blocks; cb += gridDim.x) {
    const int64_t peer = cb % W;
    const int64_t tmp = cb / W;
    const int64_t s = tmp % S_local;
    const int64_t b = tmp / S_local;

    int64_t src_off;
    int64_t dst_off;
    if (mode == 0) {
      // input a2a: read local heads [peer*H_local, (peer+1)*H_local) for (b, s),
      // write to peer's buffer at global sequence (rank*S_local + s).
      src_off = ((b * S_local + s) * H + peer * H_local) * D;
      dst_off = (b * S_global + static_cast<int64_t>(rank) * S_local + s) * block_len;
    } else {
      // output a2a (inverse): read local global-seq block (peer*S_local + s),
      // write to peer's buffer at heads [rank*H_local, (rank+1)*H_local).
      src_off = (b * S_global + peer * S_local + s) * block_len;
      dst_off = ((b * S_local + s) * H + static_cast<int64_t>(rank) * H_local) * D;
    }

    const char* s8 = reinterpret_cast<const char*>(local_in + src_off);
    // ptrs[] is declared const void*; this is a destination we own and write to.
    char* d8 = reinterpret_cast<char*>((T*)out_ptrs.ptrs[peer] + dst_off);

    const bool aligned = ((reinterpret_cast<uintptr_t>(s8) | reinterpret_cast<uintptr_t>(d8)) & 0xF) == 0;
    if (aligned) {
      const int64_t vec_bytes = (block_bytes / 16) * 16;
      for (int64_t i = static_cast<int64_t>(threadIdx.x) * 16; i < vec_bytes;
           i += static_cast<int64_t>(blockDim.x) * 16) {
        *reinterpret_cast<int4*>(d8 + i) = *reinterpret_cast<const int4*>(s8 + i);
      }
      for (int64_t j = vec_bytes + threadIdx.x; j < block_bytes; j += blockDim.x) {
        d8[j] = s8[j];
      }
    } else {
      for (int64_t j = threadIdx.x; j < block_bytes; j += blockDim.x) {
        d8[j] = s8[j];
      }
    }
  }

  // Release-acquire barrier so all peer writes are visible before any rank
  // reads its own (now complete) output staging buffer.
  multi_gpu_barrier<NGPUS, false, true>(sg, self_sg, rank);
}

}  // namespace sglang

fptr_t init_ulysses_a2a(
    const std::vector<fptr_t>& out_ipc_ptrs,
    const std::vector<fptr_t>& signal_ipc_ptrs,
    int64_t rank,
    int64_t world_size,
    bool full_nvlink) {
  if (world_size > 8) throw std::invalid_argument("ulysses a2a world size > 8 is not supported");
  if (world_size != 2 && world_size != 4 && world_size != 6 && world_size != 8)
    throw std::invalid_argument("ulysses a2a only supports world size in (2, 4, 6, 8)");
  if (rank < 0 || rank >= world_size) throw std::invalid_argument("invalid rank passed in");
  if (static_cast<int64_t>(out_ipc_ptrs.size()) != world_size)
    throw std::invalid_argument("out_ipc_ptrs size must equal world_size");
  if (static_cast<int64_t>(signal_ipc_ptrs.size()) != world_size)
    throw std::invalid_argument("signal_ipc_ptrs size must equal world_size");

  sglang::Signal* signals[8];
  void* out_bufs[8];
  for (int i = 0; i < world_size; i++) {
    signals[i] = reinterpret_cast<sglang::Signal*>(signal_ipc_ptrs[i]);
    out_bufs[i] = reinterpret_cast<void*>(out_ipc_ptrs[i]);
  }
  return (fptr_t) new sglang::UlyssesA2A(signals, out_bufs, rank, world_size, full_nvlink);
}

void dispose_ulysses_a2a(fptr_t _fa) {
  delete reinterpret_cast<sglang::UlyssesA2A*>(_fa);
}

// Performs the fused-transpose all-to-all. The result for this rank lands in the
// local output staging buffer and is then copied into `out`.
//   mode == 0: inp [B, S_local, H, D]        -> out [B, S_global, H_local, D]
//   mode == 1: inp [B, S_global, H_local, D] -> out [B, S_local, H, D]
// where H here is the *global* head count and H_local = H / world_size.
void ulysses_a2a(
    fptr_t _fa,
    torch::Tensor& inp,
    torch::Tensor& out,
    int64_t B,
    int64_t S_local,
    int64_t H,
    int64_t D,
    int64_t mode) {
  auto fa = reinterpret_cast<sglang::UlyssesA2A*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(inp.is_cuda() && out.is_cuda(), "ulysses_a2a inputs must be CUDA tensors");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(), "ulysses_a2a inputs must be contiguous");
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(mode == 0 || mode == 1, "ulysses_a2a mode must be 0 or 1");

  const int W = fa->world_size_;
  TORCH_CHECK(H % W == 0, "global head count must be divisible by world size");
  const int H_local = static_cast<int>(H / W);

  const int64_t num_copy_blocks = B * S_local * W;
  const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(sglang::kMaxBlocks, num_copy_blocks)));
  const int threads = sglang::kDefaultThreads;

  const size_t out_bytes = out.numel() * out.element_size();

#define LAUNCH_ULYSSES_A2A(T, NG)                                                                      \
  sglang::ulysses_a2a_push_kernel<T, NG><<<blocks, threads, 0, stream>>>(                              \
      reinterpret_cast<const T*>(inp.data_ptr()),                                                      \
      fa->out_ptrs_,                                                                                   \
      fa->sg_,                                                                                         \
      fa->self_sg_,                                                                                    \
      fa->rank_,                                                                                       \
      static_cast<int>(B),                                                                             \
      static_cast<int>(S_local),                                                                       \
      H_local,                                                                                         \
      static_cast<int>(D),                                                                             \
      static_cast<int>(mode))

#define DISPATCH_NGPUS(T)                                                          \
  switch (W) {                                                                     \
    case 2:                                                                        \
      LAUNCH_ULYSSES_A2A(T, 2);                                                    \
      break;                                                                       \
    case 4:                                                                        \
      LAUNCH_ULYSSES_A2A(T, 4);                                                    \
      break;                                                                       \
    case 6:                                                                        \
      LAUNCH_ULYSSES_A2A(T, 6);                                                    \
      break;                                                                       \
    case 8:                                                                        \
      LAUNCH_ULYSSES_A2A(T, 8);                                                    \
      break;                                                                       \
    default:                                                                       \
      throw std::runtime_error("ulysses_a2a only supports world size in (2,4,6,8)"); \
  }

  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      DISPATCH_NGPUS(float);
      break;
    }
    case at::ScalarType::Half: {
      DISPATCH_NGPUS(half);
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
      DISPATCH_NGPUS(nv_bfloat16);
      break;
    }
#endif
    default:
      throw std::runtime_error("ulysses_a2a only supports float32, float16 and bfloat16");
  }

#undef DISPATCH_NGPUS
#undef LAUNCH_ULYSSES_A2A

  AT_CUDA_CHECK(cudaGetLastError());
  // Copy this rank's completed result out of the staging buffer.
  AT_CUDA_CHECK(
      cudaMemcpyAsync(out.data_ptr(), fa->local_out_buf_, out_bytes, cudaMemcpyDeviceToDevice, stream));
}
