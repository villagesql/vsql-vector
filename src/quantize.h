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

// Scalar int16 quantization for vectors, plus the SIMD squared-L2 kernel over
// the quantized form. Self-contained: no dependency on the HNSW storage layer.
// Intended for an in-memory resident quantized vector -- quantize a float
// vector once when it is materialized, then compare with dist_squared_l2_q.
//
// Recipe mirrors MariaDB MHNSW's FVector: per-vector scale = max|v| / 32767,
// components rounded to int16, and abs2 = 0.5 * ||v||^2 precomputed so squared
// L2 reduces to a single int16 dot product plus a constant-time combine.

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_QUANTIZE_H
#define VILLAGESQL_VSQL_VECTOR_SRC_QUANTIZE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#define SVECTOR_Q_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define SVECTOR_Q_ALWAYS_INLINE __forceinline
#else
#define SVECTOR_Q_ALWAYS_INLINE inline
#endif

// SIMD intrinsic headers for the int16 dot product. The kernel selects
// AVX-512 > AVX2 > NEON > scalar at compile time. Guarded so this header still
// compiles on toolchains/ISAs without them (falls through to scalar).
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BW__)
#include <immintrin.h>
#define SVECTOR_QDOT_X86 1
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define SVECTOR_QDOT_NEON 1
#endif

namespace svector::quant {

// The int16 dims array is zero-padded up to a multiple of this many components
// so the SIMD kernel processes only full-width blocks (no scalar remainder, no
// over-read). 32 covers every target: AVX-512 (32 int16/reg), AVX2 (16), and
// NEON (8) all divide it. The pad zeros contribute nothing to the dot product.
inline constexpr uint32_t QVECTOR_DIM_PAD = 32;

inline constexpr uint32_t qvector_padded_dim(uint32_t dim) {
  return (dim + QVECTOR_DIM_PAD - 1) / QVECTOR_DIM_PAD * QVECTOR_DIM_PAD;
}

// A quantized vector. dims holds the int16 components; a caller that will run
// the SIMD kernel must allocate dims to qvector_padded_dim(dim) and (via
// quantize) zero the pad. dim stays the REAL component count (scale/abs2
// describe the real vector; pad zeros add nothing).
struct QData {
  uint32_t dim;    // real component count
  float scale;     // original[i] ~= scale * dims[i]
  float abs2;      // 0.5 * scale^2 * <dims, dims>  (precomputed)
  int16_t dims[];  // quantized components (flexible array member)
};

// Byte length of a QData holding `dim` int16 components.
inline constexpr size_t qdata_length(uint32_t dim) {
  return sizeof(QData) + static_cast<size_t>(dim) * sizeof(int16_t);
}

// Quantize `dim` floats at `src` into a caller-provided QData `out` (buffer >=
// qdata_length(padded_dim)). padded_dim (>= dim) zero-fills dims[dim..padded).
// A zero vector maps to scale=1, all-zero dims, abs2=0. Not hot (runs once per
// vector materialization), so defined out-of-line in quantize.cc.
void quantize(const float *src, uint32_t dim, uint32_t padded_dim, QData *out);

// int16 dot product of two vectors, returned as float. `dim` MUST be a multiple
// of QVECTOR_DIM_PAD so every SIMD path consumes whole blocks with no scalar
// remainder and no over-read. Loads are UNALIGNED. Each block's int16xint16
// products go through the fused MAC (madd_epi16 on x86, vmull on NEON), reduced
// to int32/int64 then float-accumulated -- exact for int16 products, no int32
// overflow at high dim. Mirrors MHNSW FVector::dot_product per ISA.
static SVECTOR_Q_ALWAYS_INLINE float qdot(const int16_t *a, const int16_t *b,
                                          uint32_t dim) {
#if defined(SVECTOR_QDOT_X86) && defined(__AVX512BW__)
  __m512 acc = _mm512_setzero_ps();
  for (uint32_t i = 0; i < dim; i += 32) {
    __m512i va = _mm512_loadu_si512((const void *)(a + i));
    __m512i vb = _mm512_loadu_si512((const void *)(b + i));
    acc = _mm512_add_ps(acc, _mm512_cvtepi32_ps(_mm512_madd_epi16(va, vb)));
  }
  return _mm512_reduce_add_ps(acc);
#elif defined(SVECTOR_QDOT_X86)
  __m256 acc = _mm256_setzero_ps();
  for (uint32_t i = 0; i < dim; i += 16) {
    __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
    acc = _mm256_add_ps(acc, _mm256_cvtepi32_ps(_mm256_madd_epi16(va, vb)));
  }
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
#elif defined(SVECTOR_QDOT_NEON)
  int64x2_t acc = vdupq_n_s64(0);
  for (uint32_t i = 0; i < dim; i += 8) {
    int16x8_t va = vld1q_s16(a + i);
    int16x8_t vb = vld1q_s16(b + i);
    acc = vpadalq_s32(acc, vmull_s16(vget_low_s16(va), vget_low_s16(vb)));
    acc = vpadalq_s32(acc, vmull_high_s16(va, vb));
  }
  return static_cast<float>(vgetq_lane_s64(acc, 0) + vgetq_lane_s64(acc, 1));
#else
  float dot = 0.0f;
  for (uint32_t i = 0; i < dim; ++i)
    dot += static_cast<float>(static_cast<int32_t>(a[i]) *
                              static_cast<int32_t>(b[i]));
  return dot;
#endif
}

// Squared L2 distance between two quantized vectors, field form. `dim` is the
// PADDED length (multiple of QVECTOR_DIM_PAD); the pad zeros add nothing. The
// shared 2x is kept so this matches a plain-f32 squared-L2 ranking scale.
static SVECTOR_Q_ALWAYS_INLINE double dist_squared_l2_q(
    float sa, float abs2a, const int16_t *da, float sb, float abs2b,
    const int16_t *db, uint32_t dim) {
  const float dot = qdot(da, db, dim);
  return 2.0 * (double(abs2a) + double(abs2b) -
                double(sa) * double(sb) * double(dot));
}

// Convenience overload for two packed QData operands. Pass the padded dim.
static SVECTOR_Q_ALWAYS_INLINE double dist_squared_l2_q(const QData *a,
                                                        const QData *b,
                                                        uint32_t padded_dim) {
  return dist_squared_l2_q(a->scale, a->abs2, a->dims, b->scale, b->abs2,
                           b->dims, padded_dim);
}

}  // namespace svector::quant

#endif  // VILLAGESQL_VSQL_VECTOR_SRC_QUANTIZE_H
