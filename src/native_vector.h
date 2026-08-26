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

// Distance kernels: portable, compiler-autovectorized reductions.
//
// Each kernel accumulates into LANES independent float32 partial sums and only
// combines them at the end. Two deliberate properties, both for portability:
//
//  1. Splitting the reduction into independent streams expresses the
//     reassociation IN THE SOURCE. A plain `sum += ...` reduction is vectorized
//     by Clang (which interleaves accumulators under strict IEEE) but NOT by GCC
//     at -O2 (GCC treats reassociating an FP reduction as illegal without
//     -fassociative-math and emits scalar code). With explicit streams, BOTH
//     compilers vectorize legally, with no fast-math flag and deterministic
//     results.
//
//  2. LANES = 16 lets the compiler fill the WIDEST SIMD register the target has
//     under -O2 -march=native: x86 AVX-512 -> zmm (16-wide), AVX2 -> ymm
//     (8-wide), arm64 NEON -> .4s (4-wide, its max), plain SSE/baseline -> xmm
//     (4-wide). A smaller count (e.g. 4) is optimal for NEON but caps x86 at
//     128-bit, wasting AVX. 16 is the widest useful width and degrades cleanly
//     to narrower targets (it just uses more register groups).
//
// float32 accumulation (not double) is deliberate: it keeps the whole loop in
// 32-bit lanes and avoids the per-element float->double widening (fcvtl on NEON)
// the double version pays. Summing over float32 embeddings in float32 is well
// within tolerance for ANN (recall is unaffected); streams combine and return as
// double so callers/graph are unchanged.
inline constexpr uint32_t DIST_LANES = 16;

// L1 distance (Manhattan distance) between two vectors.
static V_FUNC_ALWAYS_INLINE double dist_l1(const Data *v1, const Data *v2) {
  const float *a = v1->data;
  const float *b = v2->data;
  const uint32_t dim = v1->dim;

  float s[DIST_LANES] = {0.0f};
  uint32_t i = 0;
  const uint32_t nv = dim & ~(DIST_LANES - 1);
  for (; i < nv; i += DIST_LANES) {
    for (uint32_t j = 0; j < DIST_LANES; ++j) {
      s[j] += std::abs(a[i + j] - b[i + j]);
    }
  }
  float result = 0.0f;
  for (uint32_t j = 0; j < DIST_LANES; ++j) result += s[j];
  for (; i < dim; ++i) result += std::abs(a[i] - b[i]);  // scalar tail
  return result;
}

// Squared L2 distance between two vectors (without sqrt for efficiency).
static V_FUNC_ALWAYS_INLINE double dist_squared_l2(const Data *v1,
                                                   const Data *v2) {
  const float *a = v1->data;
  const float *b = v2->data;
  const uint32_t dim = v1->dim;

  float s[DIST_LANES] = {0.0f};
  uint32_t i = 0;
  const uint32_t nv = dim & ~(DIST_LANES - 1);
  for (; i < nv; i += DIST_LANES) {
    for (uint32_t j = 0; j < DIST_LANES; ++j) {
      const float d = a[i + j] - b[i + j];
      s[j] += d * d;
    }
  }
  float result = 0.0f;
  for (uint32_t j = 0; j < DIST_LANES; ++j) result += s[j];
  for (; i < dim; ++i) {  // scalar tail
    const float d = a[i] - b[i];
    result += d * d;
  }
  return result;
}

// L2 distance (Euclidean distance) between two vectors.
static V_FUNC_ALWAYS_INLINE double dist_l2(const Data *v1, const Data *v2) {
  return std::sqrt(dist_squared_l2(v1, v2));
}

// Cosine distance between two vectors. Three parallel reductions (dot, |v1|^2,
// |v2|^2); the final normalization is scalar double.
static V_FUNC_ALWAYS_INLINE double dist_cosine(const Data *v1, const Data *v2) {
  const float *a = v1->data;
  const float *b = v2->data;
  const uint32_t dim = v1->dim;

  float sd[DIST_LANES] = {0.0f}, s1[DIST_LANES] = {0.0f}, s2[DIST_LANES] = {0.0f};
  uint32_t i = 0;
  const uint32_t nv = dim & ~(DIST_LANES - 1);
  for (; i < nv; i += DIST_LANES) {
    for (uint32_t j = 0; j < DIST_LANES; ++j) {
      const float x = a[i + j], y = b[i + j];
      sd[j] += x * y;
      s1[j] += x * x;
      s2[j] += y * y;
    }
  }
  float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
  for (uint32_t j = 0; j < DIST_LANES; ++j) {
    dot += sd[j];
    norm1 += s1[j];
    norm2 += s2[j];
  }
  for (; i < dim; ++i) {  // scalar tail
    const float x = a[i], y = b[i];
    dot += x * y;
    norm1 += x * x;
    norm2 += y * y;
  }
  const double denom = std::sqrt(double(norm1)) * std::sqrt(double(norm2));
  if (denom > 0.0) {
    return 1.0 - (double(dot) / denom);
  }
  return 1.0;  // Maximum distance
}

// Inner product (dot product) between two vectors.
static V_FUNC_ALWAYS_INLINE double dist_inner_product(const Data *v1,
                                                      const Data *v2) {
  const float *a = v1->data;
  const float *b = v2->data;
  const uint32_t dim = v1->dim;

  float s[DIST_LANES] = {0.0f};
  uint32_t i = 0;
  const uint32_t nv = dim & ~(DIST_LANES - 1);
  for (; i < nv; i += DIST_LANES) {
    for (uint32_t j = 0; j < DIST_LANES; ++j) {
      s[j] += a[i + j] * b[i + j];
    }
  }
  float result = 0.0f;
  for (uint32_t j = 0; j < DIST_LANES; ++j) result += s[j];
  for (; i < dim; ++i) result += a[i] * b[i];  // scalar tail
  return result;
}

// Calculate L2 norm (Euclidean norm) of a vector.
static V_FUNC_ALWAYS_INLINE double norm_l2(const Data *v) {
  const float *a = v->data;
  const uint32_t dim = v->dim;

  float s[DIST_LANES] = {0.0f};
  uint32_t i = 0;
  const uint32_t nv = dim & ~(DIST_LANES - 1);
  for (; i < nv; i += DIST_LANES) {
    for (uint32_t j = 0; j < DIST_LANES; ++j) s[j] += a[i + j] * a[i + j];
  }
  float sum_sq = 0.0f;
  for (uint32_t j = 0; j < DIST_LANES; ++j) sum_sq += s[j];
  for (; i < dim; ++i) sum_sq += a[i] * a[i];  // scalar tail
  return std::sqrt(double(sum_sq));
}

}  // namespace svector::native

#endif  // VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_NATIVE_VECTOR_H
