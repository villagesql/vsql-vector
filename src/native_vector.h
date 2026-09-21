// Copyright (c) 2026 VillageSQL Contributors
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License, version 2.0,
// as published by the Free Software Foundation.
//
// This program is designed to work with certain software (including
// but not limited to OpenSSL) that is licensed under separate terms,
// as designated in a particular file or component or in included license
// documentation.  The authors of MySQL hereby grant you an additional
// permission to link the program and your derivative works with the
// separately licensed software that they have either included with
// the program or referenced in the documentation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License, version 2.0, for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA

#ifndef VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_NATIVE_VECTOR_H
#define VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_NATIVE_VECTOR_H

#include <cassert>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>

#if defined(__GNUC__) || defined(__clang__)
#define V_FUNC_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define V_FUNC_ALWAYS_INLINE __forceinline
#else
#define V_FUNC_ALWAYS_INLINE inline
#endif

// SIMD intrinsic headers for the quantized-int16 dot product. The kernel picks
// AVX-512 > AVX2 > NEON > scalar at compile time. Guarded so the header still
// compiles on ISAs/toolchains without them (falls through to scalar).
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BW__)
#include <immintrin.h>
#define SVECTOR_QDOT_X86 1
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define SVECTOR_QDOT_NEON 1
#endif

namespace svector::native {

// Maximum supported vector dimension
constexpr uint32_t MAX_VECTOR_DIMENSION = 3072;

// Native Type Representation:
// The native form provides an optimized in-memory representation for vectors.
// This allows efficient processing without repeatedly parsing the encoded form.
//
// Memory layout:
//   [uint32_t dim][float][float]...[float]
struct Data {
  uint32_t dim;  // number of dimensions
  float data[];  // array of floats (flexible array member)
};

// Native length information
struct Length {
  size_t length;     // Size of the native representation in bytes
  size_t alignment;  // Alignment requirement for the native representation
                     // (power of 2)
};

// Helper to check if a pointer is properly aligned
static V_FUNC_ALWAYS_INLINE bool is_aligned(const void *ptr, size_t alignment) {
  // Alignment must be a power of 2
  assert(alignment > 0 && (alignment & (alignment - 1)) == 0);
  return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// Platform-independent float storage functions (little-endian format)
static V_FUNC_ALWAYS_INLINE void float4store(unsigned char *buffer,
                                             float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof bits);
  buffer[0] = (unsigned char)(bits);
  buffer[1] = (unsigned char)(bits >> 8);
  buffer[2] = (unsigned char)(bits >> 16);
  buffer[3] = (unsigned char)(bits >> 24);
}

static V_FUNC_ALWAYS_INLINE float float4get(const unsigned char *buffer) {
  uint32_t bits = ((uint32_t)buffer[0]) | ((uint32_t)buffer[1] << 8) |
                  ((uint32_t)buffer[2] << 16) | ((uint32_t)buffer[3] << 24);
  float value;
  memcpy(&value, &bits, sizeof value);
  return value;
}

// Aligned buffer helper:
// Provides hybrid stack/heap allocation with proper alignment.
// Uses stack buffer for small allocations, heap for large ones.
//
// Requirements:
//   - C++17 or later (uses std::align_val_t)
//   - Each instance allocates StackBytes on the stack
//   - Not thread-safe (caller must synchronize)
//
// IMPORTANT: Not movable or copyable due to stack buffer.
// Must be used as a local variable, cannot be returned or stored.
template <size_t StackBytes>
class AlignedBuffer {
 public:
  // Default constructor - leaves buffer uninitialized
  // Must call init() before use
  AlignedBuffer() = default;

  // Constructor with immediate initialization
  // Check is_initialized() after construction to verify success
  AlignedBuffer(size_t size, size_t alignment) noexcept {
    init(size, alignment);
  }

  // Initialize buffer with given size and alignment.
  // Returns false on allocation failure.
  // Safe to call multiple times - will free previous allocation.
  bool init(size_t size, size_t alignment) noexcept {
    // Double-init protection: free existing allocation if any
    reset();

    size_ = size;
    alignment_ = alignment;

    if (size <= StackBytes && alignment <= alignof(std::max_align_t)) {
      ptr_ = stack_;
      return true;
    }

    ptr_ = ::operator new(size, std::align_val_t(alignment), std::nothrow);
    if (!ptr_) {
      size_ = 0;
      alignment_ = 0;
      return false;
    }

    heap_ = true;
    return true;
  }

  ~AlignedBuffer() noexcept { reset(); }

  // Explicitly free resources
  // Safe to call multiple times
  void reset() noexcept {
    if (heap_ && ptr_) {
      ::operator delete(ptr_, std::align_val_t(alignment_));
    }
    ptr_ = nullptr;
    heap_ = false;
    size_ = 0;
    alignment_ = 0;
  }

  // Check if buffer has been initialized
  bool is_initialized() const noexcept { return ptr_ != nullptr; }

  // Get allocated size (0 if not initialized)
  size_t size() const noexcept { return size_; }

  // Get alignment (0 if not initialized)
  size_t alignment() const noexcept { return alignment_; }

  // Check if using heap allocation (false if stack or uninitialized)
  bool uses_heap() const noexcept { return heap_; }

  void *get() noexcept {
    assert(ptr_ && "AlignedBuffer not initialized");
    return ptr_;
  }

  const void *get() const noexcept {
    assert(ptr_ && "AlignedBuffer not initialized");
    return ptr_;
  }

  // Non-copyable and non-movable due to stack buffer
  AlignedBuffer(const AlignedBuffer &) = delete;
  AlignedBuffer &operator=(const AlignedBuffer &) = delete;
  AlignedBuffer(AlignedBuffer &&) = delete;
  AlignedBuffer &operator=(AlignedBuffer &&) = delete;

 private:
  alignas(std::max_align_t) unsigned char stack_[StackBytes];
  void *ptr_ = nullptr;
  size_t size_ = 0;
  size_t alignment_ = 0;
  bool heap_ = false;
};

// Get native representation length and alignment for a given dimension
Length length(uint32_t dimension);

// Convert encoded representation to native representation.
// The caller provides a properly aligned buffer of sufficient size.
// Returns true on error.
bool from_encoded(const unsigned char *encoded_data, size_t encoded_len,
                  void *native_buffer, size_t native_buf_len);

// Convert native representation to encoded representation.
// Returns true on error.
bool to_encoded(const void *native_data, unsigned char *encoded_buffer,
                size_t encoded_buf_len, size_t *encoded_len);

// The distance kernels accumulate in float (matching the stored element type)
// so the reduction vectorizes to a single SIMD lane-width under -O3 with the
// reassociation flags set in CMakeLists.txt. The public return type stays
// double to match the dispatch function-pointer signature; each kernel widens
// its float accumulator once at the return. Overflow of the float accumulator
// produces a well-ordered +inf (the reassociation flags do NOT assume finite
// math), so L2/inner-product ranking stays correct on pathological inputs.

// L1 distance (Manhattan distance) between two vectors
static V_FUNC_ALWAYS_INLINE double dist_l1(const Data *v1, const Data *v2) {
  float result = 0.0f;
  for (uint32_t i = 0; i < v1->dim; i++) {
    result += std::abs(v1->data[i] - v2->data[i]);
  }
  return result;
}

// Squared L2 distance between two vectors (without sqrt for efficiency)
static V_FUNC_ALWAYS_INLINE double dist_squared_l2(const Data *v1,
                                                   const Data *v2) {
  const float *a = v1->data;
  const float *b = v2->data;
  const float *end = a + v1->dim;

  // Use pointer iteration (helps vectorization)
  float result = 0.0f;
  for (; a != end; ++a, ++b) {
    float diff = *a - *b;
    result += diff * diff;
  }
  return result;
}

// L2 distance (Euclidean distance) between two vectors
static V_FUNC_ALWAYS_INLINE double dist_l2(const Data *v1, const Data *v2) {
  return std::sqrt(dist_squared_l2(v1, v2));
}

// Cosine distance between two vectors
static V_FUNC_ALWAYS_INLINE double dist_cosine(const Data *v1, const Data *v2) {
  float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
  for (uint32_t i = 0; i < v1->dim; i++) {
    float x = v1->data[i];
    float y = v2->data[i];
    dot += x * y;
    norm1 += x * x;
    norm2 += y * y;
  }
  // Widen to double for the division so the ratio keeps full precision even
  // when the float sums are large.
  double denom = std::sqrt(double(norm1)) * std::sqrt(double(norm2));
  // Divide only when denom is finite and positive. A zero vector gives
  // denom == 0; a large-magnitude vector can overflow the float norm
  // accumulator to +inf, which would make dot/denom evaluate to inf/inf ==
  // NaN and corrupt the comparator. Both fall through to max distance. This
  // guard is on the cold post-loop path, so it does not affect vectorization.
  if (denom > 0.0 && std::isfinite(denom)) {
    return 1.0 - (double(dot) / denom);
  }
  return 1.0;  // Maximum distance
}

// Inner product (dot product) between two vectors
static V_FUNC_ALWAYS_INLINE double dist_inner_product(const Data *v1,
                                                      const Data *v2) {
  float result = 0.0f;
  for (uint32_t i = 0; i < v1->dim; i++) {
    result += v1->data[i] * v2->data[i];
  }
  return result;
}

// --- Quantized (int16) L2 path -------------------------------------------
//
// A separate operand format for the HNSW node's inline vector copy, distinct
// from Data{dim; float[]} (the main SVECTOR column-store form). Each vector is
// scalar-quantized to int16 with a per-vector scale, so distances run as an
// int16 dot product -- twice the SIMD lane density of f32, and the graph is
// approximate anyway. Mirrors MariaDB MHNSW's FVector (sql/vector_mhnsw.cc).
//
// Squared-L2 identity used by the kernel:
//   ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
// With a[i] ~= sa*qa[i], b[i] ~= sb*qb[i], and abs2 storing 0.5*||.||^2 per
// vector, this becomes:
//   dist2(a,b) = 2*abs2_a + 2*abs2_b - 2*sa*sb*<qa,qb>
// The 0.5 factor is folded into abs2 at quantize time so the hot loop is a
// single int16 dot product plus a constant-time combine. Only the relative
// order of dist2 matters for KNN ranking, so the shared 2x is kept (it does
// not change the comparison) to match the plain-f32 dist_squared_l2 scale.
struct QData {
  uint32_t dim;      // number of dimensions
  float scale;       // original[i] ~= scale * dims[i]
  float abs2;        // 0.5 * scale^2 * <dims, dims>  (precomputed at quantize)
  int16_t dims[];    // quantized components (flexible array member)
};

// Byte length of a QData holding `dim` components.
static V_FUNC_ALWAYS_INLINE size_t qdata_length(uint32_t dim) {
  return sizeof(QData) + static_cast<size_t>(dim) * sizeof(int16_t);
}

// Quantize a float vector into a caller-provided QData buffer. Recipe matches
// MHNSW FVector::create: scale = max|v| / 32767, round to int16, precompute
// abs2. A zero vector maps to scale=1, all-zero dims, abs2=0.
//
// padded_dim (>= v->dim) zero-fills dims[v->dim .. padded_dim) so the SIMD
// distance kernel can read full blocks; the buffer must be >=
// qdata_length(padded_dim). out->dim stays the REAL dim (abs2 and the stored
// scale describe the real vector; the pad zeros contribute nothing).
static V_FUNC_ALWAYS_INLINE void quantize(const Data *v, QData *out,
                                          uint32_t padded_dim) {
  float max_abs = 0.0f;
  for (uint32_t i = 0; i < v->dim; i++)
    max_abs = std::max(max_abs, std::abs(v->data[i]));

  float scale = max_abs > 0.0f ? max_abs / 32767.0f : 1.0f;
  out->dim = v->dim;
  out->scale = scale;

  int64_t dot = 0;
  for (uint32_t i = 0; i < v->dim; i++) {
    int32_t q = static_cast<int32_t>(std::lround(v->data[i] / scale));
    if (q > 32767) q = 32767;
    if (q < -32768) q = -32768;
    out->dims[i] = static_cast<int16_t>(q);
    dot += static_cast<int32_t>(out->dims[i]) * static_cast<int32_t>(out->dims[i]);
  }
  for (uint32_t i = v->dim; i < padded_dim; i++) out->dims[i] = 0;
  out->abs2 = 0.5f * scale * scale * static_cast<float>(dot);
}

// int16 dot product of two vectors, returned as float. `dim` MUST be a
// multiple of 32 (the QVECTOR_DIM_PAD padding guarantee) so every SIMD path
// consumes whole blocks with no scalar remainder and no over-read. Loads are
// UNALIGNED (loadu / vld1q): the vectors live in column-store records placed at
// arbitrary page offsets, so absolute alignment cannot be assumed.
//
// Each block's int16xint16 products are reduced to int32 via the fused MAC
// (madd_epi16 on x86 -- adjacent pairs multiplied and added -- vmull on NEON),
// then converted to float and accumulated. Converting per block keeps the
// integer partials small (block sum <= 32 * 2^30 < 2^36, exact in the int32
// madd/int64 NEON reduce) while the float accumulator carries the running total
// without the int32 overflow a plain int32 accumulator would hit at high dim.
// Mirrors MariaDB MHNSW's FVector::dot_product per ISA.
static V_FUNC_ALWAYS_INLINE float qdot(const int16_t *a, const int16_t *b,
                                       uint32_t dim) {
#if defined(SVECTOR_QDOT_X86) && (defined(__AVX512BW__))
  // AVX-512: 32 int16 per _mm512_madd_epi16.
  __m512 acc = _mm512_setzero_ps();
  for (uint32_t i = 0; i < dim; i += 32) {
    __m512i va = _mm512_loadu_si512((const void *)(a + i));
    __m512i vb = _mm512_loadu_si512((const void *)(b + i));
    acc = _mm512_add_ps(acc, _mm512_cvtepi32_ps(_mm512_madd_epi16(va, vb)));
  }
  return _mm512_reduce_add_ps(acc);
#elif defined(SVECTOR_QDOT_X86)
  // AVX2: 16 int16 per _mm256_madd_epi16.
  __m256 acc = _mm256_setzero_ps();
  for (uint32_t i = 0; i < dim; i += 16) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
    acc = _mm256_add_ps(acc, _mm256_cvtepi32_ps(_mm256_madd_epi16(va, vb)));
  }
  // Horizontal add of the 8 float lanes.
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
#elif defined(SVECTOR_QDOT_NEON)
  // NEON: 8 int16 per block. Each int16xint16 product fits int32 (<2^30). The 4
  // low + 4 high products are pairwise-accumulated into an int64x2 register
  // across the whole loop (vpadalq widens int32->int64, so no int32 overflow
  // for any dim), reduced to a scalar ONCE at the end -- not per block.
  int64x2_t acc = vdupq_n_s64(0);
  for (uint32_t i = 0; i < dim; i += 8) {
    int16x8_t va = vld1q_s16(a + i);
    int16x8_t vb = vld1q_s16(b + i);
    acc = vpadalq_s32(acc, vmull_s16(vget_low_s16(va), vget_low_s16(vb)));
    acc = vpadalq_s32(acc, vmull_high_s16(va, vb));
  }
  return static_cast<float>(vgetq_lane_s64(acc, 0) + vgetq_lane_s64(acc, 1));
#else
  // Portable scalar fallback (float accumulate; still vectorizes under -O3).
  float dot = 0.0f;
  for (uint32_t i = 0; i < dim; ++i)
    dot += static_cast<float>(static_cast<int32_t>(a[i]) *
                              static_cast<int32_t>(b[i]));
  return dot;
#endif
}

// Squared L2 distance between two quantized vectors, field form. The operands
// are passed as raw fields (scale, precomputed abs2, int16 dims, dim) so a
// caller can feed a vector read in place from a record tail without first
// packing it into a QData. Same ranking scale as dist_squared_l2 (both carry
// the 2x), so callers can mix f32 and quantized indexes without rescaling
// thresholds. `dim` is the PADDED length (multiple of 32); the pad zeros add
// nothing to the dot product.
static V_FUNC_ALWAYS_INLINE double dist_squared_l2_q(float sa, float abs2a,
                                                     const int16_t *da,
                                                     float sb, float abs2b,
                                                     const int16_t *db,
                                                     uint32_t dim) {
  const float dot = qdot(da, db, dim);
  return 2.0 * (double(abs2a) + double(abs2b) -
                double(sa) * double(sb) * double(dot));
}

// Convenience overload for two packed QData operands.
static V_FUNC_ALWAYS_INLINE double dist_squared_l2_q(const QData *a,
                                                     const QData *b) {
  return dist_squared_l2_q(a->scale, a->abs2, a->dims, b->scale, b->abs2,
                           b->dims, a->dim);
}

// Calculate L2 norm (Euclidean norm) of a vector
static V_FUNC_ALWAYS_INLINE double norm_l2(const Data *v) {
  const float *a = v->data;
  const float *end = a + v->dim;

  // Use pointer iteration (helps vectorization)
  float sum_sq = 0.0f;
  for (; a != end; ++a) {
    sum_sq += *a * *a;
  }
  return std::sqrt(double(sum_sq));
}

// Canonical metric names and the single name -> kernel mapping. This is the one
// owner of "which distance name means which kernel": the SVECTOR function/index
// registration (vector.cc) binds these names, and index code resolves a bound
// helper name back to its kernel via dist_for_name(). Keep both sides on these
// symbols so a rename can never silently mismatch.
using DistFn = double (*)(const Data *, const Data *);

inline constexpr const char kDistL1[] = "l1_distance";
inline constexpr const char kDistL2Squared[] = "l2_squared_distance";
inline constexpr const char kDistCosine[] = "cosine_distance";
inline constexpr const char kDistInnerProduct[] = "inner_product";

// Map a metric name to its kernel; nullptr if the name is unknown.
inline DistFn dist_for_name(const char *name) {
  if (name == nullptr) return nullptr;
  if (std::strcmp(name, kDistL2Squared) == 0) return &dist_squared_l2;
  if (std::strcmp(name, kDistL1) == 0) return &dist_l1;
  if (std::strcmp(name, kDistCosine) == 0) return &dist_cosine;
  if (std::strcmp(name, kDistInnerProduct) == 0) return &dist_inner_product;
  return nullptr;
}

}  // namespace svector::native

#endif  // VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_NATIVE_VECTOR_H
