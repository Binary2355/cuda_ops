#include <type_traits>
#include <thrust/tuple.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/block_reduce.cuh>

namespace at::native {

constexpr int kCUDANumThreads = 256;
constexpr int kReduceTileSize = 32;

template <typename T>
static void check_group_norm_inputs(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const T& C,
    int64_t num_groups) {
    TORCH_CHECK(
        num_groups > 0,
        "Expected num groups to be greater than 0, got ", num_groups);

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor")
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor")
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor")
    TORCH_CHECK(input.is_contiguous(at::MemoryFormat::Contiguous));
    TORCH_CHECK(weight.scalar_type() == input.scalar_type());
    TORCH_CHECK(bias.scalar_type() == input.scalar_type());

    TORCH_CHECK(C % num_groups == 0,
        "Expected number of channels in input to be divisible by ",
        "num_groups, but got input of shape ",
        input.sizes(),
        " and "
        "num_groups=",
        num_groups);
    TORCH_CHECK(weight.dim() == 1 && at::symint::numel<T>(weight) == C,
        "Expected weight to be a vector of size equal to the number of ",
        "channels in input, but got weight of shape ",
        weight.sizes(),
        " and input of shape ",
        input.sizes());
    TORCH_CHECK(bias.dim() == 1 && at::symint::numel<T>(bias) == C,
        "Expected bias to be a vector of size equal to the number of ",
        "channels in input, but got bias of shape ",
        weight.sizes(),
        " and input of shape ",
        input.sizes());
}

template <typename scalar_t, typename index_t>
struct vFxWelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  scalar_t nf;

  C10_HOST_DEVICE vFxWelfordData() : mean(0), m2(0), n(0), nf(0) {}

  C10_HOST_DEVICE vFxWelfordData(
      scalar_t mean,
      scalar_t m2,
      index_t n,
      scalar_t nf)
      : mean(mean), m2(m2), n(n), nf(nf) {}
};

template <typename scalar_t, typename acc_scalar_t, typename index_t, typename res_t>
struct vFxWelfordOps {
    acc_scalar_t correction;
    bool take_sqrt;
    public:
    using acc_t = vFxWelfordData<acc_scalar_t, index_t>;
    inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
        // We accumulate n in index_t to avoid cumulative rounding error, but still
        // need nf for use in combine where int32 may overflow.
        index_t new_n = acc.n + 1;
        acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);
        acc_scalar_t delta = data - acc.mean;
        acc_scalar_t new_mean = acc.mean + delta / new_nf;
        acc_scalar_t new_delta = data - new_mean;
        return {
        new_mean,
        acc.m2 + delta * new_delta,
        new_n,
        new_nf,
        };
    }
    inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
        if (a.nf == 0) {
        return b;
        }
        if (b.nf == 0) {
        return a;
        }
        acc_scalar_t delta = b.mean - a.mean;
        acc_scalar_t new_count = a.nf + b.nf;
        acc_scalar_t nb_over_n = b.nf / new_count;
        return {
        a.mean + delta * nb_over_n,
        a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
        // setting acc.n as -1 since acc.n might not be able to represent the count
        // correctly within its range, setting it to -1 to avoid confusion
        -1,
        new_count
        };
    }
    inline C10_DEVICE res_t project(acc_t acc) const __ubsan_ignore_float_divide_by_zero__ {
        const auto mean = static_cast<scalar_t>(acc.mean);
        const auto divisor = acc.nf > correction ? acc.nf - correction : 0;
        const auto var = acc.m2 / divisor;
        res_t results(take_sqrt ? device_sqrt(var) : var, mean, acc.nf);
        return results;
    }

    static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
        return acc;
    }

#if defined(__CUDACC__) || defined(__HIPCC__)
    inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
        return {
        WARP_SHFL_DOWN(acc.mean, offset)
        , WARP_SHFL_DOWN(acc.m2, offset)
        , WARP_SHFL_DOWN(acc.n, offset)
        , WARP_SHFL_DOWN(acc.nf, offset)
        };
    }
#endif
    C10_HOST_DEVICE vFxWelfordOps(acc_scalar_t correction, bool take_sqrt)
        : correction(correction), take_sqrt(take_sqrt) {}
};

template <typename T, int kSplit>
__global__ void veFusionXRowwiseMomentsCUDAKernel2(
    int64_t N,
    T eps,
    T* sum,
    T* sum2,
    T* num,
    T* mean,
    T* rstd)
{
    using T_ACC = acc_type<T, true>;
    const int64_t bid = blockIdx.x;
    T a_num = num[bid];
    T a_mean = sum[bid];
    T a_var = sum2[bid];
    for (int i=1; i < kSplit; i++) {
        T b_num = num[i*gridDim.x+bid];
        T b_mean = sum[i*gridDim.x+bid];
        T b_var = sum2[i*gridDim.x+bid];

        T total_num = a_num + b_num;
        T delta = b_mean - a_mean;
        T new_mean = a_mean + delta * (b_num / total_num);
        T delta2 = b_mean - new_mean;
        T new_var = (a_num * (a_var + (a_mean - new_mean) * (a_mean - new_mean)) + 
                    b_num * (b_var + delta2 * delta2)) / total_num;

        a_num = total_num;
        a_mean = new_mean;
        a_var = new_var;
    }
    mean[bid] = a_mean;
    rstd[bid] = c10::cuda::compat::rsqrt(a_var + eps);
}

template <typename T, int kSplit>
__global__ void veFusionXRowwiseMomentsCUDAKernel(
    int64_t N,
    int64_t dim_,
    const T* X,
    T* sum,
    T* sum2,
    T* num)
{
    using T_ACC = acc_type<T, true>;
    using vFxWelfordType = vFxWelfordData<T_ACC, int64_t>;
    using vFxWelfordOp =
      vFxWelfordOps<T_ACC, T_ACC, int64_t, thrust::tuple<T_ACC, T_ACC, T_ACC>>;

    const int64_t i = blockIdx.x;
    const int64_t start_col = dim_ * blockIdx.y;
    const int64_t end_col = (blockIdx.y == (kSplit-1)) ? N : (start_col + dim_);

    vFxWelfordOp vfxwelford_op = {0, false};
    vFxWelfordType val(0, 0, 0, 0);

    for (int64_t j = start_col+threadIdx.x; j < end_col; j += blockDim.x) {
        const int64_t index = i * N + j;
        val = vfxwelford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
    }

    if (blockDim.x <= C10_WARP_SIZE) {
        val = cuda_utils::WarpReduce(val, vfxwelford_op);
    } else {
        __shared__ typename std::aligned_storage<
            sizeof(vFxWelfordType),
            alignof(vFxWelfordType)>::type val_shared[C10_WARP_SIZE];
        vFxWelfordType* val_shared_ptr = reinterpret_cast<vFxWelfordType*>(val_shared);
        val = cuda_utils::BlockReduce(
            val,
            vfxwelford_op,
            vFxWelfordType(0, 0, 0, 0),
            val_shared_ptr);
    }
    if (threadIdx.x == 0) {
        T_ACC m1;
        T_ACC m2;
        T_ACC m3;
        thrust::tie(m2, m1, m3) = vfxwelford_op.project(val);
        sum[blockIdx.y*gridDim.x+i] = m1;
        sum2[blockIdx.y*gridDim.x+i] = m2;
        num[blockIdx.y*gridDim.x+i] = m3;
    }
}

template <typename T>
__global__ void RowwiseMomentsCUDAKernel(
    int64_t N,
    T eps,
    const T* X,
    T* mean,
    T* rstd)
{
    using T_ACC = acc_type<T, true>;
    using WelfordType = WelfordData<T_ACC, int64_t>;
    using WelfordOp =
        WelfordOps<T_ACC, T_ACC, int64_t, thrust::pair<T_ACC, T_ACC>>;

    const int64_t i = blockIdx.x;
    WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
    WelfordType val(0, 0, 0, 0);
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
        const int64_t index = i * N + j;
        val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
    }
    if (blockDim.x <= C10_WARP_SIZE) {
        val = cuda_utils::WarpReduce(val, welford_op);
    } else {
        // There will be a warning if we declare a __shared__ WelfordType array.
        // https://github.com/pytorch/pytorch/pull/13967
        __shared__ typename std::aligned_storage<
            sizeof(WelfordType),
            alignof(WelfordType)>::type val_shared[C10_WARP_SIZE];
        WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
        val = cuda_utils::BlockReduce(
            val,
            welford_op,
            /*identity_element=*/WelfordType(0, 0, 0, 0),
            val_shared_ptr);
    }
    if (threadIdx.x == 0) {
        T_ACC m1;
        T_ACC m2;
        thrust::tie(m2, m1) = welford_op.project(val);
        mean[i] = m1;
        rstd[i] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
    }
}

template <typename T>
__global__ void ComputeFusedParamsCUDAKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const T* beta,
    acc_type<T, true>* a,
    acc_type<T, true>* b)
{
    using T_ACC = acc_type<T, true>;
    const int64_t index = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < N * C) {
        const int64_t ng = index / (C / group);
        const int64_t c = index % C;
        const T_ACC scale = (gamma == nullptr)
            ? static_cast<T_ACC>(rstd[ng])
            : static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]);
        a[index] = scale;
        b[index] = -scale * static_cast<T_ACC>(mean[ng]) +
            ((beta == nullptr) ? 0 : static_cast<T_ACC>(beta[c]));
    }
}

template <typename T>
__global__ void Compute1dBackwardFusedParamsCUDAKernel(
    int64_t C,
    int64_t group,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    const T* gamma,
    acc_type<T, true>* c2,
    acc_type<T, true>* c3)
{
    using T_ACC = acc_type<T, true>;
    const int64_t G = group;
    const int64_t D = C / G;
    const int64_t n = blockIdx.x;
    const int64_t g = blockIdx.y;
    const int64_t ng = n * G + g;
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
        const int64_t index = ng * D + i;
        const int64_t c = g * D + i;
        const T_ACC gamma_v =
            gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[c]);
        sum1 += dY[index] * X[index] * gamma_v;
        sum2 += dY[index] * gamma_v;
    }
    if (blockDim.x <= C10_WARP_SIZE) {
        sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
        sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
    } else {
        __shared__ T_ACC ds_shared[C10_WARP_SIZE];
        __shared__ T_ACC db_shared[C10_WARP_SIZE];
        sum1 = cuda_utils::BlockReduceSum<T_ACC>(sum1, ds_shared);
        sum2 = cuda_utils::BlockReduceSum<T_ACC>(sum2, db_shared);
    }
    if (threadIdx.x == 0) {
        const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D);
        const T_ACC x = (sum2 * static_cast<T_ACC>(mean[ng]) - sum1) *
            static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(rstd[ng]) *
            static_cast<T_ACC>(rstd[ng]) * s;
        c2[ng] = x;
        c3[ng] = -x * static_cast<T_ACC>(mean[ng]) -
            sum2 * static_cast<T_ACC>(rstd[ng]) * s;
    }
}

template <typename T>
__global__ void GammaBeta1dBackwardCUDAKernel1(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    T* dgamma,
    T* dbeta) {
  using T_ACC = acc_type<T, true>;
  const int64_t c = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
  if (c < C) {
    const int64_t G = group;
    const int64_t D = C / G;
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t n = 0; n < N; ++n) {
      const int64_t nc = n * C + c;
      const int64_t ng = n * G + c / D;
      const T_ACC dy_acc = static_cast<T_ACC>(dY[nc]);
      const T_ACC x_acc = static_cast<T_ACC>(X[nc]);
      sum1 += (dgamma == nullptr)
          ? T_ACC(0)
          : ((dy_acc * x_acc - dy_acc * static_cast<T_ACC>(mean[ng])) *
             static_cast<T_ACC>(rstd[ng]));
      sum2 += (dbeta == nullptr) ? T_ACC(0) : dy_acc;
    }
    if (dgamma != nullptr) {
      dgamma[c] = sum1;
    }
    if (dbeta != nullptr) {
      dbeta[c] = sum2;
    }
  }
}

template <typename T>
__global__ void GammaBeta1dBackwardCUDAKernel2(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    T* dgamma,
    T* dbeta) {
  using T_ACC = acc_type<T, true>;
  __shared__ T_ACC g_shared[kReduceTileSize][kReduceTileSize + 1];
  __shared__ T_ACC b_shared[kReduceTileSize][kReduceTileSize + 1];
  const int64_t c = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
  T_ACC dg_sum1 = 0;
  T_ACC dg_sum2 = 0;
  T_ACC db_sum1 = 0;
  T_ACC db_sum2 = 0;
  if (c < C) {
    const int64_t G = group;
    const int64_t D = C / G;
    // Accumulate each 32 cols into a 32 * 32 tile.
    // Since the blockDim is (32, 16), accumulate twice for 1st and 2nd 16 rows
    // of a 32 contiguous elements.
    for (int64_t n = threadIdx.y; n < N; n += blockDim.y * 2) {
      const int64_t n1 = n;
      const int64_t n2 = n + blockDim.y;
      const int64_t nc1 = n1 * C + c;
      const int64_t nc2 = n2 * C + c;
      const int64_t ng1 = n1 * G + c / D;
      const int64_t ng2 = n2 * G + c / D;
      const T_ACC dy1_acc = static_cast<T_ACC>(dY[nc1]);
      const T_ACC x1_acc = static_cast<T_ACC>(X[nc1]);
      dg_sum1 += dgamma == nullptr
          ? T_ACC(0)
          : ((dy1_acc * x1_acc - dy1_acc * static_cast<T_ACC>(mean[ng1])) *
             static_cast<T_ACC>(rstd[ng1]));
      db_sum1 += dbeta == nullptr ? T_ACC(0) : dy1_acc;
      if (n2 < N) {
        const T_ACC dy2_acc = static_cast<T_ACC>(dY[nc2]);
        const T_ACC x2_acc = static_cast<T_ACC>(X[nc2]);
        dg_sum2 += dgamma == nullptr
            ? T_ACC(0)
            : ((dy2_acc * x2_acc - dy2_acc * static_cast<T_ACC>(mean[ng2])) *
               static_cast<T_ACC>(rstd[ng2]));
        db_sum2 += dbeta == nullptr ? T_ACC(0) : dy2_acc;
      }
    }
  }

  // Write accumulated tile to shared memory.
  g_shared[threadIdx.y][threadIdx.x] = dg_sum1;
  g_shared[threadIdx.y + blockDim.y][threadIdx.x] = dg_sum2;
  b_shared[threadIdx.y][threadIdx.x] = db_sum1;
  b_shared[threadIdx.y + blockDim.y][threadIdx.x] = db_sum2;
  __syncthreads();

  // Do warp reduce for the 1st 16 cols in the tile.
  T_ACC sum1 = g_shared[threadIdx.x][threadIdx.y];
  T_ACC sum2 = b_shared[threadIdx.x][threadIdx.y];
  sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
  sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
  if (threadIdx.x == 0) {
    const int64_t c = blockIdx.x * blockDim.x + threadIdx.y;
    if (c < C) {
      if (dgamma != nullptr) {
        dgamma[c] = sum1;
      }
      if (dbeta != nullptr) {
        dbeta[c] = sum2;
      }
    }
  }

  // Do warp reduce for the 2nd 16 cols in the tile.
  sum1 = g_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum2 = b_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
  sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
  if (threadIdx.x == 0) {
    const int64_t c = blockIdx.x * blockDim.x + threadIdx.y + blockDim.y;
    if (c < C) {
      if (dgamma != nullptr) {
        dgamma[c] = sum1;
      }
      if (dbeta != nullptr) {
        dbeta[c] = sum2;
      }
    }
  }
}

template <typename T>
__global__ void ComputeInternalGradientsCUDAKernel(
    int64_t HxW,
    const T* dY,
    const T* X,
    acc_type<T, true>* ds,
    acc_type<T, true>* db)
{
    using T_ACC = acc_type<T, true>;
    const int64_t nc = blockIdx.x;
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t hw = threadIdx.x; hw < HxW; hw += blockDim.x) {
        const int64_t index = nc * HxW + hw;
        sum1 += static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]);
        sum2 += static_cast<T_ACC>(dY[index]);
    }
    if (blockDim.x <= C10_WARP_SIZE) {
        sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
        sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
    } else {
        __shared__ T_ACC ds_shared[C10_WARP_SIZE];
        __shared__ T_ACC db_shared[C10_WARP_SIZE];
        sum1 = cuda_utils::BlockReduceSum<T_ACC>(sum1, ds_shared);
        sum2 = cuda_utils::BlockReduceSum<T_ACC>(sum2, db_shared);
    }
    if (threadIdx.x == 0) {
        ds[nc] = sum1;
        db[nc] = sum2;
    }
}

template <typename T>
__global__ void ComputeBackwardFusedParamsCUDAKernel(
    int64_t C,
    int64_t HxW,
    int64_t group,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const acc_type<T, true>* ds,
    const acc_type<T, true>* db,
    acc_type<T, true>* c2,
    acc_type<T, true>* c3)
{
    using T_ACC = acc_type<T, true>;
    const int64_t G = group;
    const int64_t D = C / G;
    const int64_t n = blockIdx.x;
    const int64_t g = blockIdx.y;
    const int64_t ng = n * G + g;
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
        const int64_t index = ng * D + i;
        const int64_t c = g * D + i;
        const T_ACC gamma_v =
            gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[c]);
        sum1 += ds[index] * gamma_v;
        sum2 += db[index] * gamma_v;
    }
    if (blockDim.x <= C10_WARP_SIZE) {
        sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
        sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
    } else {
        __shared__ T_ACC ds_shared[C10_WARP_SIZE];
        __shared__ T_ACC db_shared[C10_WARP_SIZE];
        sum1 = cuda_utils::BlockReduceSum<T_ACC>(sum1, ds_shared);
        sum2 = cuda_utils::BlockReduceSum<T_ACC>(sum2, db_shared);
    }
    if (threadIdx.x == 0) {
        const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D * HxW);
        const T_ACC x = (sum2 * static_cast<T_ACC>(mean[ng]) - sum1) *
            static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(rstd[ng]) *
            static_cast<T_ACC>(rstd[ng]) * s;
        c2[ng] = x;
        c3[ng] = -x * static_cast<T_ACC>(mean[ng]) -
            sum2 * static_cast<T_ACC>(rstd[ng]) * s;
    }
}

template <typename T>
__global__ void GammaBetaBackwardCUDAKernel1(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* mean,
    const T* rstd,
    const acc_type<T, true>* ds,
    const acc_type<T, true>* db,
    T* dgamma,
    T* dbeta)
{
    using T_ACC = acc_type<T, true>;
    const int64_t c = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
    if (c < C) {
        const int64_t G = group;
        const int64_t D = C / G;
        T_ACC sum1 = 0;
        T_ACC sum2 = 0;
        for (int64_t n = 0; n < N; ++n) {
        const int64_t nc = n * C + c;
        const int64_t ng = n * G + c / D;
        sum1 += (dgamma == nullptr)
            ? T_ACC(0)
            : ((ds[nc] - db[nc] * static_cast<T_ACC>(mean[ng])) *
                static_cast<T_ACC>(rstd[ng]));
        sum2 += (dbeta == nullptr) ? T_ACC(0) : db[nc];
        }
        if (dgamma != nullptr) {
        dgamma[c] = sum1;
        }
        if (dbeta != nullptr) {
        dbeta[c] = sum2;
        }
    }
}

template <typename T>
__global__ void GammaBetaBackwardCUDAKernel2(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* mean,
    const T* rstd,
    const acc_type<T, true>* ds,
    const acc_type<T, true>* db,
    T* dgamma,
    T* dbeta)
{
    using T_ACC = acc_type<T, true>;
    __shared__ T_ACC g_shared[kReduceTileSize][kReduceTileSize + 1];
    __shared__ T_ACC b_shared[kReduceTileSize][kReduceTileSize + 1];
    const int64_t c = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
    T_ACC dg_sum1 = 0;
    T_ACC dg_sum2 = 0;
    T_ACC db_sum1 = 0;
    T_ACC db_sum2 = 0;
    if (c < C) {
        const int64_t G = group;
        const int64_t D = C / G;
        // Accumulate each 32 cols into a 32 * 32 tile.
        // Since the blockDim is (32, 16), accumulate twice for 1st and 2nd 16 rows
        // of a 32 contiguous elements.
        for (int64_t n = threadIdx.y; n < N; n += blockDim.y * 2) {
        const int64_t n1 = n;
        const int64_t n2 = n + blockDim.y;
        const int64_t nc1 = n1 * C + c;
        const int64_t nc2 = n2 * C + c;
        const int64_t ng1 = n1 * G + c / D;
        const int64_t ng2 = n2 * G + c / D;
        dg_sum1 += dgamma == nullptr
            ? T_ACC(0)
            : ((ds[nc1] - db[nc1] * static_cast<T_ACC>(mean[ng1])) *
                static_cast<T_ACC>(rstd[ng1]));
        db_sum1 += dbeta == nullptr ? T_ACC(0) : db[nc1];
        if (n2 < N) {
            dg_sum2 += dgamma == nullptr
                ? T_ACC(0)
                : ((ds[nc2] - db[nc2] * static_cast<T_ACC>(mean[ng2])) *
                static_cast<T_ACC>(rstd[ng2]));
            db_sum2 += dbeta == nullptr ? T_ACC(0) : db[nc2];
        }
        }
    }

    // Write accumulated tile to shared memory.
    g_shared[threadIdx.y][threadIdx.x] = dg_sum1;
    g_shared[threadIdx.y + blockDim.y][threadIdx.x] = dg_sum2;
    b_shared[threadIdx.y][threadIdx.x] = db_sum1;
    b_shared[threadIdx.y + blockDim.y][threadIdx.x] = db_sum2;
    __syncthreads();

    // Do warp reduce for the 1st 16 cols in the tile.
    T_ACC sum1 = g_shared[threadIdx.x][threadIdx.y];
    T_ACC sum2 = b_shared[threadIdx.x][threadIdx.y];
    sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
    sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
    if (threadIdx.x == 0) {
        const int64_t c = blockIdx.x * blockDim.x + threadIdx.y;
        if (c < C) {
        if (dgamma != nullptr) {
            dgamma[c] = sum1;
        }
        if (dbeta != nullptr) {
            dbeta[c] = sum2;
        }
        }
    }

    // Do warp reduce for the 2st 16 cols in the tile.
    sum1 = g_shared[threadIdx.x][threadIdx.y + blockDim.y];
    sum2 = b_shared[threadIdx.x][threadIdx.y + blockDim.y];
    sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
    sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
    if (threadIdx.x == 0) {
        const int64_t c = blockIdx.x * blockDim.x + threadIdx.y + blockDim.y;
        if (c < C) {
        if (dgamma != nullptr) {
            dgamma[c] = sum1;
        }
        if (dbeta != nullptr) {
            dbeta[c] = sum2;
        }
        }
    }
}

template <typename T>
void GroupNorm1dForward(
    const torch::Tensor& X,
    const torch::Tensor& mean,
    const torch::Tensor& rstd,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t group,
    torch::Tensor& Y)
{
    using T_ACC = acc_type<T, true>;
    const int64_t G = group;
    const int64_t D = C / G;
    if (gamma.defined() && beta.defined()){
        auto iter = TensorIteratorConfig()
                        .resize_outputs(false)
                        .add_owned_output(Y.view({N, G, D}))
                        .add_owned_const_input(X.view({N, G, D}))
                        .add_owned_input(mean.view({N, G, 1}))
                        .add_owned_input(rstd.view({N, G, 1}))
                        .add_owned_const_input(gamma.view({1, G, D}))
                        .add_owned_const_input(beta.view({1, G, D}))
                        .build();
        gpu_kernel(iter, [] GPU_LAMBDA(T x, T mean, T rstd, T gamma, T beta) -> T {
        return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
            static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma) +
            static_cast<T_ACC>(beta);
        });
    } else if (gamma.defined()) {
        auto iter = TensorIteratorConfig()
                        .resize_outputs(false)
                        .add_owned_output(Y.view({N, G, D}))
                        .add_owned_const_input(X.view({N, G, D}))
                        .add_owned_input(mean.view({N, G, 1}))
                        .add_owned_input(rstd.view({N, G, 1}))
                        .add_owned_const_input(gamma.view({1, G, D}))
                        .build();
        gpu_kernel(iter, [] GPU_LAMBDA(T x, T mean, T rstd, T gamma) -> T {
        return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
            static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma);
        });
    } else if (beta.defined()) {
        auto iter = TensorIteratorConfig()
                        .resize_outputs(false)
                        .add_owned_output(Y.view({N, G, D}))
                        .add_owned_const_input(X.view({N, G, D}))
                        .add_owned_input(mean.view({N, G, 1}))
                        .add_owned_input(rstd.view({N, G, 1}))
                        .add_owned_const_input(beta.view({1, G, D}))
                        .build();
        gpu_kernel(iter, [] GPU_LAMBDA(T x, T mean, T rstd, T beta) -> T {
        return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
            static_cast<T_ACC>(rstd) +
            static_cast<T_ACC>(beta);
        });
    } else {
        auto iter = TensorIteratorConfig()
                        .resize_outputs(false)
                        .add_owned_output(Y.view({N * G, D}))
                        .add_owned_const_input(X.view({N * G, D}))
                        .add_owned_input(mean.view({N * G, 1}))
                        .add_owned_input(rstd.view({N * G, 1}))
                        .build();
        gpu_kernel(iter, [] GPU_LAMBDA(T x, T mean, T rstd) -> T {
        return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
            static_cast<T_ACC>(rstd);
        });
    }
    AT_CUDA_CHECK(cudaGetLastError());
}

template <typename T, int kSplit>
void veFusionXRowwiseImpl(
    const int64_t num_threads,
    int64_t row_num,
    int64_t N,
    int64_t G,
    T eps,
    const torch::Tensor& X,
    const T* X_data,
    T* mean_data,
    T* rstd_data,
    cudaStream_t cuda_stream)
{
    std::cout << "kSplit for Group Norm: " << kSplit << std::endl;
    torch::Tensor sum_ = at::empty({kSplit, N, G}, X.options().dtype(c10::CppTypeToScalarType<T>::value));
    torch::Tensor sum2_ = at::empty({kSplit, N, G}, X.options().dtype(c10::CppTypeToScalarType<T>::value));
    torch::Tensor num_ = at::empty({kSplit, N, G}, X.options().dtype(c10::CppTypeToScalarType<T>::value));
    const int64_t dim_ = (row_num + kSplit - 1) / kSplit;
    dim3 grid(N * G, kSplit);
    dim3 block(num_threads);
    T* sum_data = sum_.mutable_data_ptr<T>();
    T* sum2_data = sum2_.mutable_data_ptr<T>();
    T* num_data = num_.mutable_data_ptr<T>();
    veFusionXRowwiseMomentsCUDAKernel<T,kSplit><<<grid, block, 0, cuda_stream>>>(
        row_num, dim_, X_data, sum_data, sum2_data, num_data);
    veFusionXRowwiseMomentsCUDAKernel2<T,kSplit><<<N * G, 1, 0, cuda_stream>>>(
        row_num, eps, sum_data, sum2_data, num_data, mean_data, rstd_data);
}

template <typename T>
void GroupNormKernelImplInternal(
    const torch::Tensor& X,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    T eps,
    torch::Tensor& Y,
    torch::Tensor& mean,
    torch::Tensor& rstd)
{
    using T_ACC = acc_type<T, true>;
    TORCH_CHECK(X.numel() == N * C * HxW);
    TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
    TORCH_CHECK(!beta.defined() || beta.numel() == C);
    if (N == 0) {
        return;
    }
    const int64_t G = group;
    const int64_t D = C / G;
    const T* X_data = X.const_data_ptr<T>();
    T* mean_data = mean.mutable_data_ptr<T>();
    T* rstd_data = rstd.mutable_data_ptr<T>();

    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
    int64_t row_num = D * HxW;
    if (row_num < 262144) { // torch原分支：1.00x
        const int64_t num_threads = D * HxW < cuda_utils::kCUDABlockReduceNumThreads
            ? at::cuda::warp_size()
            : cuda_utils::kCUDABlockReduceNumThreads;
        RowwiseMomentsCUDAKernel<T><<<N * G, num_threads, 0, cuda_stream>>>(
            D * HxW, eps, X_data, mean_data, rstd_data);
    } else if (row_num < 524288) { // A100: 1.07x ～ 2.77x; H20: 1.68x ~ 2.55x
        const int64_t num_threads = cuda_utils::kCUDABlockReduceNumThreads;
        veFusionXRowwiseImpl<T,2>(num_threads, row_num, N, G, eps, X, X_data, mean_data, rstd_data, cuda_stream);
    } else if (row_num < 1048576) { // A100: 2.77x ～ 4.40x; H20: 2.55x ~ 4.94x
        const int64_t num_threads = cuda_utils::kCUDABlockReduceNumThreads;
        veFusionXRowwiseImpl<T,4>(num_threads, row_num, N, G, eps, X, X_data, mean_data, rstd_data, cuda_stream);
    } else if (row_num < 2097152) { // A100: 4.40x ～ 6.62x; H20: 4.94x ~ 7.10x
        const int64_t num_threads = cuda_utils::kCUDABlockReduceNumThreads;
        veFusionXRowwiseImpl<T,8>(num_threads, row_num, N, G, eps, X, X_data, mean_data, rstd_data, cuda_stream);
    } else if (row_num < 4194304) { // A100: 6.62x ～ 7.26x; H20: 7.10x ~ 7.26x
        const int64_t num_threads = cuda_utils::kCUDABlockReduceNumThreads;
        veFusionXRowwiseImpl<T,16>(num_threads, row_num, N, G, eps, X, X_data, mean_data, rstd_data, cuda_stream);
    } else if (row_num < 8388608) { // A100: 7.26x ～ 9.50x; H20: 7.26x ~ 7.38x
        const int64_t num_threads = 256;
        veFusionXRowwiseImpl<T,32>(num_threads, row_num, N, G, eps, X, X_data, mean_data, rstd_data, cuda_stream);
    } else if (row_num < 16777216) { // A100: 8.62x ～ 9.14x; H20: 7.38x ~ 7.84x
        const int64_t num_threads = 256;
        veFusionXRowwiseImpl<T,64>(num_threads, row_num, N, G, eps, X, X_data, mean_data, rstd_data, cuda_stream);
    } else if (row_num < 33554432) { // A100: 9.14x ～ 9.37x; H20: 7.84x ~ 7.88x
        const int64_t num_threads = 256;
        veFusionXRowwiseImpl<T,128>(num_threads, row_num, N, G, eps, X, X_data, mean_data, rstd_data, cuda_stream);
    } else if (row_num < 67108864) { // A100: 9.37x ～ 9.47x; H20: 7.88x ~ 8.01x
        const int64_t num_threads = 256;
        veFusionXRowwiseImpl<T,256>(num_threads, row_num, N, G, eps, X, X_data, mean_data, rstd_data, cuda_stream);
    } else if (row_num < 134217728) { // A100: 9.47x ～ 10.06x; H20: 8.01x ~ 8.01x
        const int64_t num_threads = 256;
        veFusionXRowwiseImpl<T,512>(num_threads, row_num, N, G, eps, X, X_data, mean_data, rstd_data, cuda_stream);
    } else {
        const int64_t num_threads = 256;
        veFusionXRowwiseImpl<T,1024>(num_threads, row_num, N, G, eps, X, X_data, mean_data, rstd_data, cuda_stream);
    }

    if (HxW == 1) {
        GroupNorm1dForward<T>(X, mean, rstd, gamma, beta, N, C, G, Y);
    } else if (!gamma.defined() && !beta.defined()) {
        auto iter = TensorIteratorConfig()
                        .resize_outputs(false)
                        .add_owned_output(Y.view({N * G, D * HxW}))
                        .add_owned_const_input(X.view({N * G, D * HxW}))
                        .add_owned_input(mean.view({N * G, 1}))
                        .add_owned_input(rstd.view({N * G, 1}))
                        .build();
        gpu_kernel(iter, [] GPU_LAMBDA(T x, T mean, T rstd) -> T {
        return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
            static_cast<T_ACC>(rstd);
        });
    } else {
        const auto kAccType =
            (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
            ? kFloat
            : X.scalar_type();
        torch::Tensor a = at::empty({N, C}, X.options().dtype(kAccType));
        torch::Tensor b = at::empty({N, C}, X.options().dtype(kAccType));
        const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
        const T* beta_data = beta.defined() ? beta.const_data_ptr<T>() : nullptr;
        T_ACC* a_data = a.mutable_data_ptr<T_ACC>();
        T_ACC* b_data = b.mutable_data_ptr<T_ACC>();

        // TODO: Since there is some issues in gpu_kernel_multiple_outputs, we are
        // using manual kernel here. Make it using gpu_kernel_multiple_outputs once
        // the issue fixed.
        const int64_t B = (N * C + kCUDANumThreads - 1) / kCUDANumThreads;
        ComputeFusedParamsCUDAKernel<T><<<B, kCUDANumThreads, 0, cuda_stream>>>(
            N, C, G, mean_data, rstd_data, gamma_data, beta_data, a_data, b_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                        .resize_outputs(false)
                        .add_owned_output(Y.view({N * C, HxW}))
                        .add_owned_const_input(X.view({N * C, HxW}))
                        .add_owned_input(a.view({N * C, 1}))
                        .add_owned_input(b.view({N * C, 1}))
                        .build();
        gpu_kernel(iter, [] GPU_LAMBDA(T x, T_ACC a, T_ACC b) -> T {
        return a * static_cast<T_ACC>(x) + b;
        });
    }
    AT_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void GroupNorm1dBackward(
    const torch::Tensor dY,
    const torch::Tensor X,
    const torch::Tensor mean,
    const torch::Tensor rstd,
    const torch::Tensor gamma,
    int64_t N,
    int64_t C,
    int64_t group,
    torch::Tensor& dX,
    torch::Tensor& dgamma,
    torch::Tensor& dbeta)
{
    using T_ACC = acc_type<T, true>;
    const int64_t G = group;
    const int64_t D = C / G;
    const T* dY_data = dY.const_data_ptr<T>();
    const T* X_data = X.const_data_ptr<T>();
    const T* mean_data = mean.const_data_ptr<T>();
    const T* rstd_data = rstd.const_data_ptr<T>();

    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
    if (dX.defined()) {
        const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
        const auto kAccType =
            (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
            ? kFloat
            : X.scalar_type();
        torch::Tensor c2 = at::empty({N, G}, X.options().dtype(kAccType));
        torch::Tensor c3 = at::empty({N, G}, X.options().dtype(kAccType));
        T_ACC* c2_data = c2.mutable_data_ptr<T_ACC>();
        T_ACC* c3_data = c3.mutable_data_ptr<T_ACC>();
        const int64_t num_threads = (C / G) < cuda_utils::kCUDABlockReduceNumThreads
            ? at::cuda::warp_size()
            : cuda_utils::kCUDABlockReduceNumThreads;
        Compute1dBackwardFusedParamsCUDAKernel<T>
            <<<dim3(N, G), num_threads, 0, cuda_stream>>>(
                C,
                G,
                dY_data,
                X_data,
                mean_data,
                rstd_data,
                gamma_data,
                c2_data,
                c3_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        if (gamma.defined()) {
        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                        .resize_outputs(false)
                        .add_owned_output(dX.view({N, G, D}))
                        .add_owned_const_input(dY.view({N, G, D}))
                        .add_owned_const_input(X.view({N, G, D}))
                        .add_owned_const_input(rstd.view({N, G, 1}))
                        .add_owned_const_input(gamma.view({1, G, D}))
                        .add_owned_const_input(c2.view({N, G, 1}))
                        .add_owned_const_input(c3.view({N, G, 1}))
                        .build();
        gpu_kernel(
            iter,
            [] GPU_LAMBDA(T dy, T x, T rstd, T gamma, T_ACC c2, T_ACC c3) -> T {
                const T_ACC c1 =
                    static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma);
                return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) +
                    c3;
            });
        } else {
        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                        .resize_outputs(false)
                        .add_owned_output(dX.view({N * G, D}))
                        .add_owned_const_input(dY.view({N * G, D}))
                        .add_owned_const_input(X.view({N * G, D}))
                        .add_owned_const_input(rstd.view({N * G, 1}))
                        .add_owned_const_input(c2.view({N * G, 1}))
                        .add_owned_const_input(c3.view({N * G, 1}))
                        .build();
        gpu_kernel(
            iter, [] GPU_LAMBDA(T dy, T x, T rstd, T_ACC c2, T_ACC c3) -> T {
                const T_ACC c1 = static_cast<T_ACC>(rstd);
                return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) +
                    c3;
            });
        }
    }
    if (dgamma.defined() || dbeta.defined()) {
        T* dgamma_data = dgamma.defined() ? dgamma.mutable_data_ptr<T>() : nullptr;
        T* dbeta_data = dbeta.defined() ? dbeta.mutable_data_ptr<T>() : nullptr;
        if (N <= 128) {
        const int64_t B = (C + kCUDANumThreads - 1) / kCUDANumThreads;
        GammaBeta1dBackwardCUDAKernel1<T><<<B, kCUDANumThreads, 0, cuda_stream>>>(
            N,
            C,
            G,
            dY_data,
            X_data,
            mean_data,
            rstd_data,
            dgamma_data,
            dbeta_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
        const int64_t B = (C + kReduceTileSize - 1) / kReduceTileSize;
        // The algorithm for colwise reduction here is to accumulate each 32 cols
        // to a 32 * 32 tile and write the tile to shared memory. Then do warp
        // reduce for each col in the tile. So here the blockDim must be (32, 16).
        constexpr int kThreadX = kReduceTileSize;
        constexpr int kThreadY = kReduceTileSize / 2;
        GammaBeta1dBackwardCUDAKernel2<T>
            <<<B, dim3(kThreadX, kThreadY), 0, cuda_stream>>>(
                N,
                C,
                G,
                dY_data,
                X_data,
                mean_data,
                rstd_data,
                dgamma_data,
                dbeta_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
}

template <typename T>
void GroupNormBackwardKernelImplInternal(
    const torch::Tensor& dY,
    const torch::Tensor& X,
    const torch::Tensor& mean,
    const torch::Tensor& rstd,
    const torch::Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    torch::Tensor& dX,
    torch::Tensor& dgamma,
    torch::Tensor& dbeta)
{
    using T_ACC = acc_type<T, true>;
    const int64_t G = group;
    const int64_t D = C / G;
    TORCH_CHECK(dY.numel() == N * C * HxW);
    TORCH_CHECK(X.numel() == N * C * HxW);
    TORCH_CHECK(mean.numel() == N * G);
    TORCH_CHECK(rstd.numel() == N * G);
    TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

    if (N == 0) {
        if (dgamma.defined()) {
        dgamma.fill_(T(0));
        }
        if (dbeta.defined()) {
        dbeta.fill_(T(0));
        }
        return;
    }

    const T* dY_data = dY.const_data_ptr<T>();
    const T* X_data = X.const_data_ptr<T>();
    const T* mean_data = mean.const_data_ptr<T>();
    const T* rstd_data = rstd.const_data_ptr<T>();
    const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
    const auto kAccType =
        (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
        ? kFloat
        : X.scalar_type();
    torch::Tensor ds = at::empty({N, C}, X.options().dtype(kAccType));
    torch::Tensor db = at::empty({N, C}, X.options().dtype(kAccType));
    T_ACC* ds_data = ds.mutable_data_ptr<T_ACC>();
    T_ACC* db_data = db.mutable_data_ptr<T_ACC>();

    if (HxW == 1) {
        GroupNorm1dBackward<T>(
            dY, X, mean, rstd, gamma, N, C, G, dX, dgamma, dbeta);
        return;
    }

    int warp_size = at::cuda::warp_size();
    int64_t num_threads = HxW < cuda_utils::kCUDABlockReduceNumThreads
        ? warp_size
        : cuda_utils::kCUDABlockReduceNumThreads;
    ComputeInternalGradientsCUDAKernel<T><<<N * C, num_threads, 0, cuda_stream>>>(
        HxW, dY_data, X_data, ds_data, db_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (dX.defined()) {
        torch::Tensor c1 = at::empty({0}, X.options().dtype(kAccType));
        torch::Tensor c2 = at::empty({N, G}, X.options().dtype(kAccType));
        torch::Tensor c3 = at::empty({N, G}, X.options().dtype(kAccType));
        T_ACC* c2_data = c2.mutable_data_ptr<T_ACC>();
        T_ACC* c3_data = c3.mutable_data_ptr<T_ACC>();

        if (gamma.defined()) {
        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                        .add_output(c1)
                        .add_owned_const_input(rstd.view({N, G, 1}))
                        .add_owned_const_input(gamma.view({1, G, D}))
                        .build();
        gpu_kernel(iter, [] GPU_LAMBDA(T rstd, T gamma) -> T_ACC {
            return static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma);
        });
        }

        num_threads = (C / G) < cuda_utils::kCUDABlockReduceNumThreads
            ? warp_size
            : cuda_utils::kCUDABlockReduceNumThreads;
        ComputeBackwardFusedParamsCUDAKernel<T>
            <<<dim3(N, G), num_threads, 0, cuda_stream>>>(
                C,
                HxW,
                G,
                mean_data,
                rstd_data,
                gamma_data,
                ds_data,
                db_data,
                c2_data,
                c3_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        if (gamma.defined()) {
        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                        .resize_outputs(false)
                        .add_owned_output(dX.view({N * G, D, HxW}))
                        .add_owned_const_input(dY.view({N * G, D, HxW}))
                        .add_owned_const_input(X.view({N * G, D, HxW}))
                        .add_owned_const_input(c1.view({N * G, D, 1}))
                        .add_owned_const_input(c2.view({N * G, 1, 1}))
                        .add_owned_const_input(c3.view({N * G, 1, 1}))
                        .build();
        gpu_kernel(
            iter, [] GPU_LAMBDA(T dy, T x, T_ACC c1, T_ACC c2, T_ACC c3) -> T {
                return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) +
                    c3;
            });
        } else {
        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                        .resize_outputs(false)
                        .add_owned_output(dX.view({N * G, D * HxW}))
                        .add_owned_const_input(dY.view({N * G, D * HxW}))
                        .add_owned_const_input(X.view({N * G, D * HxW}))
                        .add_owned_const_input(rstd.view({N * G, 1}))
                        .add_owned_const_input(c2.view({N * G, 1}))
                        .add_owned_const_input(c3.view({N * G, 1}))
                        .build();
        gpu_kernel(
            iter, [] GPU_LAMBDA(T dy, T x, T_ACC c1, T_ACC c2, T_ACC c3) -> T {
                return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) +
                    c3;
            });
        }
    }
    if (dgamma.defined() || dbeta.defined()) {
        T* dgamma_data = dgamma.defined() ? dgamma.mutable_data_ptr<T>() : nullptr;
        T* dbeta_data = dbeta.defined() ? dbeta.mutable_data_ptr<T>() : nullptr;
        if (N <= 128) {
        // For small batch size, do colwise reduce directly.
        const int64_t B = (C + kCUDANumThreads - 1) / kCUDANumThreads;
        GammaBetaBackwardCUDAKernel1<T><<<B, kCUDANumThreads, 0, cuda_stream>>>(
            N,
            C,
            G,
            mean_data,
            rstd_data,
            ds_data,
            db_data,
            dgamma_data,
            dbeta_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
        const int64_t B = (C + kReduceTileSize - 1) / kReduceTileSize;
        // The algorithm for colwise reduction here is to accumulate each 32 cols
        // to a 32 * 32 tile and write the tile to shared memory. Then do warp
        // reduce for each col in the tile. So here the blockDim must be (32, 16).
        constexpr int kThreadX = kReduceTileSize;
        constexpr int kThreadY = kReduceTileSize / 2;
        GammaBetaBackwardCUDAKernel2<T>
            <<<B, dim3(kThreadX, kThreadY), 0, cuda_stream>>>(
                N,
                C,
                G,
                mean_data,
                rstd_data,
                ds_data,
                db_data,
                dgamma_data,
                dbeta_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> native_group_norm_(
    const torch::Tensor& X,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps)
{
    check_group_norm_inputs(X, gamma, beta, C, group);

    torch::Tensor Y = at::native::empty_like(
        X,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        at::MemoryFormat::Contiguous);
    const auto dtype = X.scalar_type();
    torch::Tensor mean = at::empty({N, group}, X.options().dtype(dtype));
    torch::Tensor rstd = at::empty({N, group}, X.options().dtype(dtype));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        X.scalar_type(),
        "native_group_norm_",
        [&]() {
            GroupNormKernelImplInternal<scalar_t>(
                X,
                gamma,
                beta,
                N,
                C,
                HxW,
                group,
                static_cast<scalar_t>(eps),
                Y,
                mean,
                rstd);
        });

    return std::make_tuple(Y, mean, rstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> native_group_norm_backward_(
    const torch::Tensor& dY,
    const torch::Tensor& X,
    const torch::Tensor& mean,
    const torch::Tensor& rstd,
    const torch::Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    std::array<bool, 3> grad_input_mask)
{
    TORCH_CHECK(X.scalar_type() == dY.scalar_type(), "Expected scalar types of X and dY are same.");
    TORCH_CHECK(X.is_cuda(), "Expect input is on cuda");
    TORCH_CHECK(X.scalar_type() == mean.scalar_type(), "Expected mean has the same dtype as input");
    TORCH_CHECK(X.scalar_type() == rstd.scalar_type(), "Expected rstd has the same dtype as input");
    auto memory_format = at::MemoryFormat::Contiguous;

    torch::Tensor dX;
    torch::Tensor dgamma;
    torch::Tensor dbeta;
    if (grad_input_mask[0]) {
        dX = at::native::empty_like(
            X,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            memory_format);
    }
    if (grad_input_mask[1]) {
        dgamma = at::native::empty_like(
            gamma,
            std::nullopt /* dtype */,
            std::nullopt /* layout */,
            std::nullopt /* device */,
            std::nullopt /* pin_memory */,
            at::MemoryFormat::Contiguous);
    }
    if (grad_input_mask[2]) {
        dbeta = at::native::empty_like(
            gamma,
            std::nullopt /* dtype */,
            std::nullopt /* layout */,
            std::nullopt /* device */,
            std::nullopt /* pin_memory */,
            at::MemoryFormat::Contiguous);
    }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "native_group_norm_backward_",
        [&]() {
            GroupNormBackwardKernelImplInternal<scalar_t>(
                dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
        });

    return std::make_tuple(dX, dgamma, dbeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("groupnorm_forward", &native_group_norm_, "GroupNorm forward");
    m.def("groupnorm_backward", &native_group_norm_backward_, "GroupNorm backward");
}

} // end of at::native