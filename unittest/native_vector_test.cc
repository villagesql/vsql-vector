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

// Standalone, assert()-based unit test for the native distance kernels
// (svector::native::dist_*, norm_l2). Has no dependency on the VillageSQL SDK:
// the kernels are inline in native_vector.h and take only raw float buffers.
//
// The kernels use a 16-lane (DIST_LANES) explicit-accumulator form so the
// reduction autovectorizes portably (NEON .4s / AVX2 ymm / AVX-512 zmm). This
// test guards that rewrite two ways: (1) known-answer cases with hand-computed
// results, and (2) a cross-check against a trivial scalar reference over random
// vectors at a sweep of dims. The dim sweep deliberately straddles the 16-lane
// boundary (1, 15, 16, 17, 31, 33, ...) so a tail-handling or
// accumulator-combine bug is caught. Run it on each target arch (x86/arm) to
// confirm the vectorized kernels agree with the scalar reference everywhere.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "native_vector.h"

namespace {

using svector::native::Data;

// Allocate a Data (flexible array member) holding `vals`.
struct DataBuf {
  std::vector<unsigned char> bytes;
  Data *get() { return reinterpret_cast<Data *>(bytes.data()); }
};

DataBuf make_data(const std::vector<float> &vals) {
  DataBuf buf;
  buf.bytes.resize(sizeof(uint32_t) + vals.size() * sizeof(float));
  Data *d = reinterpret_cast<Data *>(buf.bytes.data());
  d->dim = static_cast<uint32_t>(vals.size());
  for (size_t i = 0; i < vals.size(); ++i)
    d->data[i] = vals[i];
  return buf;
}

bool approx(double a, double b, double eps) { return std::fabs(a - b) <= eps; }

// --- scalar reference implementations (obviously-correct, unvectorized) ------
// Accumulate in float to match the kernels' precision, so the cross-check tests
// the vectorization/tail logic, not a float-vs-double rounding difference.
double ref_l1(const std::vector<float> &a, const std::vector<float> &b) {
  float s = 0.0f;
  for (size_t i = 0; i < a.size(); ++i)
    s += std::fabs(a[i] - b[i]);
  return s;
}
double ref_l2sq(const std::vector<float> &a, const std::vector<float> &b) {
  float s = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}
double ref_ip(const std::vector<float> &a, const std::vector<float> &b) {
  float s = 0.0f;
  for (size_t i = 0; i < a.size(); ++i)
    s += a[i] * b[i];
  return s;
}
double ref_norm(const std::vector<float> &a) {
  float s = 0.0f;
  for (float x : a)
    s += x * x;
  return std::sqrt(static_cast<double>(s));
}
double ref_cosine(const std::vector<float> &a, const std::vector<float> &b) {
  float dot = 0.0f, n1 = 0.0f, n2 = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    n1 += a[i] * a[i];
    n2 += b[i] * b[i];
  }
  double denom =
      std::sqrt(static_cast<double>(n1)) * std::sqrt(static_cast<double>(n2));
  if (denom > 0.0)
    return 1.0 - (static_cast<double>(dot) / denom);
  return 1.0;
}

// --- known-answer cases ------------------------------------------------------
void test_known_answers() {
  // Hand-computed on tiny vectors.
  auto a = make_data({1.0f, 2.0f, 3.0f});
  auto b = make_data({4.0f, 6.0f, 3.0f});
  // diffs: 3, 4, 0
  assert(
      approx(svector::native::dist_squared_l2(a.get(), b.get()), 25.0, 1e-4));
  assert(approx(svector::native::dist_l2(a.get(), b.get()), 5.0, 1e-4));
  assert(approx(svector::native::dist_l1(a.get(), b.get()), 7.0, 1e-4));
  // dot = 1*4 + 2*6 + 3*3 = 25
  assert(approx(svector::native::dist_inner_product(a.get(), b.get()), 25.0,
                1e-4));
  // norm of a = sqrt(1+4+9) = sqrt(14)
  assert(approx(svector::native::norm_l2(a.get()), std::sqrt(14.0), 1e-4));

  // Identical vectors: L2 == 0, cosine == 0.
  auto c = make_data({0.5f, -1.5f, 2.25f, 7.0f});
  assert(approx(svector::native::dist_squared_l2(c.get(), c.get()), 0.0, 1e-5));
  assert(approx(svector::native::dist_cosine(c.get(), c.get()), 0.0, 1e-5));

  // Orthogonal vectors: cosine distance == 1 (dot == 0).
  auto e1 = make_data({1.0f, 0.0f});
  auto e2 = make_data({0.0f, 1.0f});
  assert(approx(svector::native::dist_cosine(e1.get(), e2.get()), 1.0, 1e-5));

  // Zero vector: cosine distance falls back to max (1.0) since denom == 0.
  auto z = make_data({0.0f, 0.0f, 0.0f});
  assert(approx(svector::native::dist_cosine(z.get(), e1.get()), 1.0, 1e-5));

  std::printf("  known-answer cases: ok\n");
}

// Deterministic pseudo-random floats (no <random> dependency / global RNG).
float rnd(uint32_t &state) {
  state = state * 1664525u + 1013904223u;
  // map to roughly [-1, 1)
  return static_cast<float>(state >> 8) / static_cast<float>(1u << 23) - 1.0f;
}

// --- vectorized-vs-scalar cross-check over a dim sweep straddling 16
// ----------
void test_matches_scalar_reference() {
  // Dims chosen around the DIST_LANES=16 boundary: below one block, exact
  // multiples, and each residue class of a partial tail.
  const uint32_t dims[] = {1,   2,   3,   4,   5,   6,   7,   8,   15,  16,
                           17,  31,  32,  33,  47,  48,  49,  63,  64,  65,
                           100, 127, 128, 129, 255, 256, 257, 784, 1024};
  // Multiple independent random vector pairs per dim, so value-dependent edge
  // cases (not just structural tail/lane bugs) are exercised.
  const int kRepsPerDim = 16;
  uint32_t state = 0x12345678u;
  for (uint32_t dim : dims) {
    for (int rep = 0; rep < kRepsPerDim; ++rep) {
      std::vector<float> va(dim), vb(dim);
      for (uint32_t i = 0; i < dim; ++i) {
        va[i] = rnd(state);
        vb[i] = rnd(state);
      }
      auto a = make_data(va);
      auto b = make_data(vb);

      // Tolerance is relative: float32 summation order differs between the
      // 16-lane kernel and the linear scalar reference, so results agree only
      // to ~float epsilon scaled by the magnitude of the sum.
      const double eps_l2 = 1e-3 * (ref_l2sq(va, vb) + 1.0);
      const double eps_l1 = 1e-3 * (ref_l1(va, vb) + 1.0);
      const double eps_ip = 1e-3 * (std::fabs(ref_ip(va, vb)) + 1.0);

      assert(approx(svector::native::dist_squared_l2(a.get(), b.get()),
                    ref_l2sq(va, vb), eps_l2));
      assert(approx(svector::native::dist_l2(a.get(), b.get()),
                    std::sqrt(ref_l2sq(va, vb)), std::sqrt(eps_l2) + 1e-4));
      assert(approx(svector::native::dist_l1(a.get(), b.get()), ref_l1(va, vb),
                    eps_l1));
      assert(approx(svector::native::dist_inner_product(a.get(), b.get()),
                    ref_ip(va, vb), eps_ip));
      assert(approx(svector::native::norm_l2(a.get()), ref_norm(va),
                    1e-3 * (ref_norm(va) + 1.0)));
      // Cosine is in [0, 2]; a small absolute tolerance is fine.
      assert(approx(svector::native::dist_cosine(a.get(), b.get()),
                    ref_cosine(va, vb), 1e-3));
    }
  }
  std::printf("  scalar-reference cross-check (dim sweep x %d reps): ok\n",
              kRepsPerDim);
}

// Explicitly exercise the all-tail case (dim < DIST_LANES => vectorized loop
// runs zero iterations, everything goes through the scalar tail).
void test_all_tail_dims() {
  uint32_t state = 0xdeadbeefu;
  for (uint32_t dim = 1; dim < 16; ++dim) {
    std::vector<float> va(dim), vb(dim);
    for (uint32_t i = 0; i < dim; ++i) {
      va[i] = rnd(state);
      vb[i] = rnd(state);
    }
    auto a = make_data(va);
    auto b = make_data(vb);
    assert(approx(svector::native::dist_squared_l2(a.get(), b.get()),
                  ref_l2sq(va, vb), 1e-3 * (ref_l2sq(va, vb) + 1.0)));
    assert(approx(svector::native::dist_inner_product(a.get(), b.get()),
                  ref_ip(va, vb), 1e-3 * (std::fabs(ref_ip(va, vb)) + 1.0)));
  }
  std::printf("  all-tail dims (< DIST_LANES): ok\n");
}

} // namespace

int main() {
  test_known_answers();
  test_all_tail_dims();
  test_matches_scalar_reference();
  std::printf("All native_vector distance tests passed.\n");
  return 0;
}
