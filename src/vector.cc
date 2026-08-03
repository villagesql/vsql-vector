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

// The VillageSQL SVECTOR extension provides a vector data type (SVECTOR) with
// external column storage for SVECTOR columns. External storage enables more
// efficient construction and traversal of HNSW indexes for ANN search.

#include <villagesql/preview/index_builder.h>
#include <villagesql/preview/storage_builder.h>
#include <villagesql/vsql.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <limits>

#include "index/hnsw/storage.h"
#include "native_vector.h"
#include "storage/storage.h"

using vsql::CustomArgWith;
using vsql::CustomResult;
using vsql::IntArg;
using vsql::IntResult;
using vsql::MaybeParams;
using vsql::RealResult;
using vsql::ResolvedTypeParams;
using vsql::Span;
using vsql::StringResult;
using vsql::preview_index_builder::Index;
using vsql::preview_index_builder::IndexProfileCapability;
using vsql::preview_index_builder::IndexTypeCapability;
using vsql::preview_index_builder::make_index_function;
using vsql::preview_index_builder::make_index_profile;
using vsql::preview_index_builder::make_index_type;
using vsql::preview_storage_builder::ColumnStoreCapability;
using vsql::preview_storage_builder::make_column_store;
using vsql::preview_storage_builder::StorageCapability;

namespace native = svector::native;

struct SVectorParams {
  int64_t dimension;
  static SVectorParams parse(const std::map<std::string, std::string> &params) {
    SVectorParams p;
    auto it = params.find("dimension");
    p.dimension = (it != params.end()) ? std::stoll(it->second) : 0;
    return p;
  }
  // Inverse of parse: render a typed SVectorParams back into canonical
  // key/value strings. Used by paths that produce a typed P at runtime (e.g.,
  // constant-string from_string) and need to publish the equivalent string-form
  // params back to the server.
  static void to_strings(const SVectorParams &p,
                         std::map<std::string, std::string> &out) {
    out["dimension"] = std::to_string(p.dimension);
  }
};

// Vector type with separate column storage.
static constexpr const char kSVectorTypeName[] = "SVECTOR";

// Enough for sign + decimal + exponent + round-trip precision + NaN/Inf
constexpr size_t MAX_FLOAT_STR_LENGTH = 16;

// Decode buffer for a N-element vector: '[' + float [',' float]* + ']' + '\0'
template <size_t N>
constexpr size_t DECODE_BUFFER_SIZE =
    2 + N * MAX_FLOAT_STR_LENGTH + (N - 1) + 1;

// Stack buffer size for AlignedBuffer (4KB - sized to handle typical vectors
// on stack while keeping total stack usage reasonable when multiple buffers
// are allocated. Larger vectors automatically use heap allocation.)
constexpr size_t VECTOR_STACK_BUFFER_SIZE = 4096;

// Verify at compile time that MAX_VECTOR_DIMENSION won't overflow
static_assert(native::MAX_VECTOR_DIMENSION <=
                  (INT64_MAX - 2) / (MAX_FLOAT_STR_LENGTH + 1),
              "MAX_VECTOR_DIMENSION would overflow decode_length()");
static_assert(native::MAX_VECTOR_DIMENSION <=
                  (SIZE_MAX - sizeof(native::Data)) / sizeof(float),
              "MAX_VECTOR_DIMENSION would overflow native vector allocation");

// When p is known, dimension is taken from p; the parsed element count must
// match p.value().dimension. When p is unknown, the element count from the
// string sets p.dimension. The loop is capped by what the output buffer can
// hold; expected-dimension mismatch is checked once at the end.
static void svector_from_string(MaybeParams<SVectorParams> &p,
                                std::string_view from, CustomResult out) {
  std::string_view sv = from;

  auto is_space = [](char c) {
    return std::isspace(static_cast<unsigned char>(c));
  };

  auto trim = [&](std::string_view &s) {
    while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
    while (!s.empty() && is_space(s.back())) s.remove_suffix(1);
  };

  auto skip_ws = [&](std::string_view &s) {
    while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
  };

  trim(sv);

  if (sv.size() < 2 || sv.front() != '[' || sv.back() != ']') {
    out.warning("svector_from_string: missing '[' or ']'");
    return;
  }

  std::string_view inner = sv.substr(1, sv.size() - 2);

  auto buf = out.buffer();
  if (buf.size() < sizeof(vef_storage_ref_t)) {
    out.warning("svector_from_string: buffer too small");
    return;
  }
  const size_t max_supportable =
      std::min((buf.size() - sizeof(vef_storage_ref_t)) / sizeof(float),
               static_cast<const size_t>(native::MAX_VECTOR_DIMENSION));

  // expected: known dimension when p is_known; otherwise SIZE_MAX (unbounded
  // until the buffer cap kicks in).
  size_t expected = SIZE_MAX;
  if (p.is_known()) {
    int64_t d = p.value().dimension;
    if (d <= 0 || d > native::MAX_VECTOR_DIMENSION) {
      out.warning("svector_from_string: invalid dimension");
      return;
    }
    expected = static_cast<size_t>(d);
    if (max_supportable < expected) {
      out.warning("svector_from_string: buffer too small");
      return;
    }
  } else {
    expected = max_supportable;
  }
  // expected is now either the "known" expected amount, or the max we can
  // accept for an unknown dimension.

  std::memset(buf.data(), 0, sizeof(vef_storage_ref_t));
  unsigned char *dst = buf.data() + sizeof(vef_storage_ref_t);

  std::string_view parse = inner;
  size_t count = 0;

  while (true) {
    skip_ws(parse);
    if (parse.empty()) {
      out.warning("svector_from_string: missing value");
      return;
    }

    if (count >= expected) {
      out.warning(p.is_known() ? "svector_from_string: too many elements"
                               : "svector_from_string: buffer too small");
      return;
    }

    float value;
    const char *begin = parse.data();

    char *next = nullptr;
    errno = 0;
    value = std::strtof(begin, &next);
    if (next == begin || errno == ERANGE) {
      out.warning("svector_from_string: parse error");
      return;
    }
    native::float4store(dst, value);
    dst += sizeof(float);
    ++count;

    parse.remove_prefix(static_cast<size_t>(next - begin));

    skip_ws(parse);
    if (parse.empty()) break;

    if (parse.front() != ',') {
      out.warning("svector_from_string: expected ','");
      return;
    }
    parse.remove_prefix(1);
  }

  if (p.is_known()) {
    if (count != expected) {
      out.warning("svector_from_string: dimension mismatch");
      return;
    }
  } else {
    if (count == 0 ||
        count > static_cast<size_t>(native::MAX_VECTOR_DIMENSION)) {
      out.warning("svector_from_string: invalid dimension");
      return;
    }
    p.set(SVectorParams{static_cast<int64_t>(count)});
  }

  out.set_length(sizeof(vef_storage_ref_t) + count * sizeof(float));
}

static bool svector_format_impl(const SVectorParams &p,
                                Span<const unsigned char> data,
                                std::chars_format fmt, int precision,
                                Span<char> out, size_t *out_len) {
  if (p.dimension <= 0 || p.dimension > native::MAX_VECTOR_DIMENSION)
    return true;

  const size_t expected = sizeof(vef_storage_ref_t) +
                          static_cast<size_t>(p.dimension) * sizeof(float);
  if (data.size() < expected) return true;

  const size_t count = static_cast<size_t>(p.dimension);

  size_t pos = 0;

  auto emit = [&](char c) {
    if (pos >= out.size()) return false;
    out[pos++] = c;
    return true;
  };
  auto emit_str = [&](const char *s, size_t n) {
    if (pos + n > out.size()) return false;
    std::memcpy(out.data() + pos, s, n);
    pos += n;
    return true;
  };

  if (!emit('[')) return true;

  const unsigned char *src = data.data() + sizeof(vef_storage_ref_t);

  for (size_t i = 0; i < count; ++i) {
    if (i > 0) {
      if (!emit(',')) return true;
    }
    float f = native::float4get(src + i * sizeof(float));

    char tmp[MAX_FLOAT_STR_LENGTH];
    auto [ptr, ec] = std::to_chars(tmp, tmp + sizeof(tmp), f, fmt, precision);

    if (ec != std::errc()) return true;
    if (!emit_str(tmp, static_cast<size_t>(ptr - tmp))) return true;
  }
  if (!emit(']')) return true;

  *out_len = pos;
  return false;
}

static void svector_to_string(CustomArgWith<SVectorParams> in,
                              StringResult out) {
  auto buf = out.buffer();
  size_t len;
  if (svector_format_impl(in.params(), in.value(), std::chars_format::general,
                          std::numeric_limits<float>::max_digits10, buf,
                          &len)) {
    return; // wrapper default ERROR
  }
  out.set_length(len);
}

static int svector_compare(CustomArgWith<SVectorParams> a,
                           CustomArgWith<SVectorParams> b) {
  const SVectorParams &p = a.params();
  assert(p.dimension > 0 && p.dimension <= native::MAX_VECTOR_DIMENSION);

  auto va = a.value();
  auto vb = b.value();
  const size_t expected = sizeof(vef_storage_ref_t) +
                          static_cast<size_t>(p.dimension) * sizeof(float);
  if (va.size() < expected || vb.size() < expected)
    return 0;

  const unsigned char *p_l = va.data() + sizeof(vef_storage_ref_t);
  const unsigned char *p_r = vb.data() + sizeof(vef_storage_ref_t);

  for (int64_t i = 0; i < p.dimension; ++i) {
    float f_l = native::float4get(p_l);
    float f_r = native::float4get(p_r);

    if (f_l < f_r) return -1;
    if (f_l > f_r) return 1;
    p_l += sizeof(float);
    p_r += sizeof(float);
  }
  return 0;
}

static bool svector_validate_dimension(int64_t dimension, char *error_msg) {
  if (dimension <= 0) {
    snprintf(error_msg, VEF_MAX_ERROR_LEN,
             "VECTOR: Dimension must be positive, got %" PRId64, dimension);
    return true;
  }
  if (dimension > native::MAX_VECTOR_DIMENSION) {
    snprintf(error_msg, VEF_MAX_ERROR_LEN,
             "VECTOR: Dimension %" PRId64 " exceeds maximum allowed (%d)",
             dimension, native::MAX_VECTOR_DIMENSION);
    return true;
  }
  return false;
}

static bool svector_get_params(int64_t dimension,
                               std::map<std::string, std::string> &params,
                               char *error_msg) {
  if (svector_validate_dimension(dimension, error_msg)) {
    return true;
  }
  params["dimension"] = std::to_string(dimension);
  return false;
}

static bool svector_resolve_params(
    const std::map<std::string, std::string> &params,
    ResolvedTypeParams *result, char *error_msg) {
  // Ensure only "dimension" is present
  if (params.size() != 1 || params.find("dimension") == params.end()) {
    snprintf(error_msg, VEF_MAX_ERROR_LEN,
             "SVECTOR: Only parameter 'dimension' is allowed");
    return true;
  }
  const std::string &dim_str = params.at("dimension");
  char *endptr = nullptr;
  errno = 0;
  int64_t dimension = strtoll(dim_str.c_str(), &endptr, 10);

  // Check for conversion errors
  if (endptr == dim_str.c_str() || *endptr != '\0' || errno == ERANGE) {
    snprintf(error_msg, VEF_MAX_ERROR_LEN,
             "SVECTOR: 'dimension' must be a "
             "valid integer, got '%s'",
             dim_str.c_str());
    return true;
  }
  if (svector_validate_dimension(dimension, error_msg)) {
    return true;
  }
  result->persisted_length =
      sizeof(vef_storage_ref_t) + dimension * sizeof(float);

  // Includes [], floats, commas and terminating '\0'
  result->max_decode_buffer_length =
      2 + dimension * MAX_FLOAT_STR_LENGTH + (dimension - 1) + 1;
  return false;
}

// Intrinsic default for SVECTOR: string representation
static std::string svector_default(const SVectorParams &p, char *error_msg) {
  if (svector_validate_dimension(p.dimension, error_msg)) {
    return "[]";
  }
  const int64_t n = p.dimension;
  std::string buf;

  assert(n > 0);
  size_t buf_len = 2 * n + 1;

  buf.resize(buf_len);

  char *ptr = buf.data();
  *ptr++ = '[';
  *ptr++ = '0';

  for (int64_t i = 1; i < n; ++i) {
    *ptr++ = ',';
    *ptr++ = '0';
  }
  *ptr++ = ']';

  assert(ptr == buf.data() + buf_len);
  return buf;
}

// SQL Functions (VDF) - Implementations

// Return the hex-encoded raw bytes of a vector (uppercase, no prefix)
void svector_hex(CustomArgWith<SVectorParams> vec, StringResult out) {
  if (vec.is_null()) {
    out.set_null();
    return;
  }
  static constexpr char kHexChars[] = "0123456789ABCDEF";
  static_assert(sizeof(kHexChars) == 17, "kHexChars must have 16 + NUL");

  auto data = vec.value();
  const size_t count = static_cast<size_t>(vec.params().dimension);
  const size_t byte_len = sizeof(vef_storage_ref_t) + count * sizeof(float);

  if (data.size() < byte_len) {
    out.error("encoded vector is too short");
    return;
  }

  auto buf = out.buffer();
  if (buf.size() < byte_len * 2) {
    out.error("output buffer too small");
    return;
  }
  auto *src = reinterpret_cast<const unsigned char *>(data.data());
  char *dst = buf.data();

  for (size_t i = 0; i < byte_len; ++i) {
    *dst++ = kHexChars[src[i] >> 4];
    *dst++ = kHexChars[src[i] & 0x0F];
  }

  out.set_length(byte_len * 2);
}

// Calculate euclidean norm (L2) of a vector
void svector_norm(CustomArgWith<SVectorParams> vec, RealResult out) {
  if (vec.is_null()) {
    out.set_null();
    return;
  }
  auto data = vec.value();
  const size_t count = static_cast<size_t>(vec.params().dimension);
  const size_t floats_len = count * sizeof(float);
  if (data.size() < sizeof(vef_storage_ref_t) + floats_len) {
    out.error("encoded vector is too short");
    return;
  }
  const unsigned char *floats = data.data() + sizeof(vef_storage_ref_t);
  native::Length native_len = native::length(count);
  native::AlignedBuffer<VECTOR_STACK_BUFFER_SIZE> buffer(native_len.length,
                                                         native_len.alignment);
  if (!buffer.is_initialized()) {
    out.error("buffer allocation failed");
    return;
  }
  if (native::from_encoded(floats, floats_len, buffer.get(), buffer.size())) {
    out.error("malformed vector");
    return;
  }
  const native::Data *v = static_cast<const native::Data *>(buffer.get());
  out.set(native::norm_l2(v));
}

// Get the dimension of a vector
void svector_dims(CustomArgWith<SVectorParams> vec, IntResult out) {
  if (vec.is_null()) {
    out.set_null();
    return;
  }
  out.set(vec.params().dimension);
}

// Get the maximum supported dimension for vectors
void svector_max_dims(IntResult out) { out.set(native::MAX_VECTOR_DIMENSION); }

// Format a vector as string with fixed-point notation and given precision
void svector_format(CustomArgWith<SVectorParams> vec, IntArg precision,
                    StringResult out) {
  if (vec.is_null()) {
    out.set_null();
    return;
  }
  if (precision.is_null()) {
    out.set_null();
    return;
  }
  int prec = static_cast<int>(precision.value());
  if (prec < 0) {
    out.error("precision must be non-negative");
    return;
  }
  auto buf = out.buffer();
  size_t out_len = 0;
  if (svector_format_impl(vec.params(), vec.value(), std::chars_format::general,
                          prec, buf, &out_len)) {
    out.error("failed to format vector");
    return;
  }
  out.set_length(out_len);
}

// Common implementation for vector distance SQL functions
static void svector_distance_impl(CustomArgWith<SVectorParams> vec1,
                                  CustomArgWith<SVectorParams> vec2,
                                  RealResult out,
                                  double (*dist_func)(const native::Data *,
                                                      const native::Data *)) {
  if (vec1.is_null() || vec2.is_null()) {
    out.set_null();
    return;
  }

  const size_t count1 = static_cast<size_t>(vec1.params().dimension);
  const size_t count2 = static_cast<size_t>(vec2.params().dimension);
  if (count1 != count2) {
    out.error("dimension mismatch");
    return;
  }

  auto data1 = vec1.value();
  auto data2 = vec2.value();

  const size_t floats_len = count1 * sizeof(float);
  if (data1.size() < sizeof(vef_storage_ref_t) + floats_len ||
      data2.size() < sizeof(vef_storage_ref_t) + floats_len) {
    out.error("encoded vector is too short");
    return;
  }

  const unsigned char *floats1 = data1.data() + sizeof(vef_storage_ref_t);
  const unsigned char *floats2 = data2.data() + sizeof(vef_storage_ref_t);

  native::Length native_len = native::length(count1);

  // Allocate aligned buffers for native representations
  native::AlignedBuffer<VECTOR_STACK_BUFFER_SIZE> buffer1(native_len.length,
                                                          native_len.alignment);
  native::AlignedBuffer<VECTOR_STACK_BUFFER_SIZE> buffer2(native_len.length,
                                                          native_len.alignment);
  if (!buffer1.is_initialized() || !buffer2.is_initialized()) {
    out.error("buffer allocation failed");
    return;
  }

  if (native::from_encoded(floats1, floats_len, buffer1.get(),
                           buffer1.size())) {
    out.error("malformed vector");
    return;
  }
  if (native::from_encoded(floats2, floats_len, buffer2.get(),
                           buffer2.size())) {
    out.error("malformed vector");
    return;
  }

  const native::Data *v1 = static_cast<const native::Data *>(buffer1.get());
  const native::Data *v2 = static_cast<const native::Data *>(buffer2.get());
  assert(v1->dim == v2->dim);

  out.set(dist_func(v1, v2));
}

// Calculate L1 distance between two vectors
void svector_distance_l1(CustomArgWith<SVectorParams> vec1,
                         CustomArgWith<SVectorParams> vec2, RealResult out) {
  svector_distance_impl(vec1, vec2, out, native::dist_l1);
}

// Calculate L2 distance between two vectors
void svector_distance_l2(CustomArgWith<SVectorParams> vec1,
                         CustomArgWith<SVectorParams> vec2, RealResult out) {
  svector_distance_impl(vec1, vec2, out, native::dist_l2);
}

// Squared L2 distance (skips sqrt; cheaper for comparisons inside HNSW)
void svector_distance_l2_squared(CustomArgWith<SVectorParams> vec1,
                                 CustomArgWith<SVectorParams> vec2,
                                 RealResult out) {
  svector_distance_impl(vec1, vec2, out, native::dist_squared_l2);
}

// Calculate cosine distance between two vectors
void svector_distance_cosine(CustomArgWith<SVectorParams> vec1,
                             CustomArgWith<SVectorParams> vec2,
                             RealResult out) {
  svector_distance_impl(vec1, vec2, out, native::dist_cosine);
}

// Calculate inner product between two vectors
void svector_inner_product(CustomArgWith<SVectorParams> vec1,
                           CustomArgWith<SVectorParams> vec2, RealResult out) {
  svector_distance_impl(vec1, vec2, out, native::dist_inner_product);
}

// Upper bound on SVECTOR's persisted byte size: storage-ref header plus
// MAX_VECTOR_DIMENSION float elements. Used only on the fix_fields-time
// constant-string inference path; row-time encoding uses the params-resolved
// persisted_length set by svector_resolve_params.
constexpr int64_t kSVectorMaxPersistedLength = static_cast<int64_t>(
    sizeof(vef_storage_ref_t) + native::MAX_VECTOR_DIMENSION * sizeof(float));

constexpr auto SVECTOR = vsql::make_type<kSVectorTypeName>()
                             // Data length related functions
                             .persisted_length(-1)
                             .max_decode_buffer_length(DECODE_BUFFER_SIZE<16>)
                             .max_persisted_length(kSVectorMaxPersistedLength)

                             // Data conversion and compare
                             .params<SVectorParams, &SVectorParams::parse,
                                     &SVectorParams::to_strings>()
                             .from_string<svector_from_string>()
                             .to_string<svector_to_string>()
                             .compare<svector_compare>()
                             .int_to_params<svector_get_params>()
                             .resolve_params<svector_resolve_params>()
                             .intrinsic_default_vdf("vector_default")
                             .build();

static constexpr auto kSVectorStorageIntf =
    make_column_store<svector::MultiColumnStore>(SVECTOR)
        .create<&svector::ColumnStorage::create>()
        .drop<&svector::ColumnStorage::drop>()
        .load<&svector::ColumnStorage::load>()
        .insert<&svector::ColumnStorage::insert>()
        .select<&svector::ColumnStorage::select>()
        .mark_delete<&svector::ColumnStorage::mark_delete>()
        .purge<&svector::ColumnStorage::purge>()
        .build();

static auto STORAGE = StorageCapability{};
static auto COLUMN_STORE = ColumnStoreCapability().column_store(kSVectorStorageIntf);

// ============================================================================
// HNSW index type, profiles, and capabilities
// ============================================================================

static constexpr const char kHNSWIndexName[] = "hnsw";
static constexpr const char kHNSWProfileL1[] = "hnsw_l1";
static constexpr const char kHNSWProfileL2[] = "hnsw_l2";
static constexpr const char kHNSWProfileCosine[] = "hnsw_cosine";
static constexpr const char kHNSWProfileInnerProduct[] = "hnsw_inner_product";

static constexpr const char kHNSWFuncL1Distance[] = "l1_distance";
static constexpr const char kHNSWFuncL2Distance[] = "l2_distance";
static constexpr const char kHNSWFuncCosineDistance[] = "cosine_distance";
static constexpr const char kHNSWFuncInnerProduct[] = "inner_product";
static constexpr const char kHNSWFuncL2SquaredDistance[] =
    "l2_squared_distance";

// clang-format off
static constexpr auto HNSW_INDEX_TYPE =
    make_index_type<kHNSWIndexName, svector::hnsw::IndexStore>()
        .lifecycle()
            .create<&svector::hnsw::create>()
            .load<&svector::hnsw::load>()
            .drop<&svector::hnsw::drop>()

        .dml()
            .insert<&svector::hnsw::insert>()
            .mark_delete<&svector::hnsw::mark_delete>()
            .purge<&svector::hnsw::purge>()

        .scan()
            .begin<&svector::hnsw::begin>()
            .position<&svector::hnsw::position>()
            .fetch<&svector::hnsw::fetch>()
            .save<&svector::hnsw::save>()
            .restore<&svector::hnsw::restore>()
            .end<&svector::hnsw::end>()

        .global()
            .capabilities(Index::Support::KNN)
            .storage_props(Index::Storage::HAS_COLUMN_REF |
                           Index::Storage::REF_LOOKUP)
            .options<svector::hnsw::Options, &svector::hnsw::Options::parse>()

        .build();
// clang-format on

static const auto HNSW_L1_FN =
    make_index_function<&svector_distance_l1>(kHNSWFuncL1Distance)
        .returns(vsql::REAL)
        .param(SVECTOR)
        .param(SVECTOR)
        .deterministic()
        .build();

static const auto HNSW_L2_FN =
    make_index_function<&svector_distance_l2>(kHNSWFuncL2Distance)
        .returns(vsql::REAL)
        .param(SVECTOR)
        .param(SVECTOR)
        .deterministic()
        .build();

static const auto HNSW_L2_SQUARED_FN =
    make_index_function<&svector_distance_l2_squared>(
        kHNSWFuncL2SquaredDistance)
        .returns(vsql::REAL)
        .param(SVECTOR)
        .param(SVECTOR)
        .deterministic()
        .build();

static const auto HNSW_COSINE_FN =
    make_index_function<&svector_distance_cosine>(kHNSWFuncCosineDistance)
        .returns(vsql::REAL)
        .param(SVECTOR)
        .param(SVECTOR)
        .deterministic()
        .build();

static const auto HNSW_IP_FN =
    make_index_function<&svector_inner_product>(kHNSWFuncInnerProduct)
        .returns(vsql::REAL)
        .param(SVECTOR)
        .param(SVECTOR)
        .deterministic()
        .build();

static const auto HNSW_L1_PROFILE = make_index_profile(kHNSWProfileL1)
                                        .for_type(kSVectorTypeName)
                                        .using_index(kHNSWIndexName)
                                        .with_function(1, HNSW_L1_FN)
                                        .with_helper(1, HNSW_L1_FN)
                                        .ordering(Index::Ordering::ASC)
                                        .build();

static const auto HNSW_L2_PROFILE = make_index_profile(kHNSWProfileL2)
                                        .for_type(kSVectorTypeName)
                                        .using_index(kHNSWIndexName)
                                        .with_function(1, HNSW_L2_FN)
                                        .with_helper(1, HNSW_L2_SQUARED_FN)
                                        .ordering(Index::Ordering::ASC)
                                        .default_for_type(true)
                                        .build();

static const auto HNSW_COSINE_PROFILE = make_index_profile(kHNSWProfileCosine)
                                            .for_type(kSVectorTypeName)
                                            .using_index(kHNSWIndexName)
                                            .with_function(1, HNSW_COSINE_FN)
                                            .with_helper(1, HNSW_COSINE_FN)
                                            .ordering(Index::Ordering::ASC)
                                            .build();

static const auto HNSW_IP_PROFILE = make_index_profile(kHNSWProfileInnerProduct)
                                        .for_type(kSVectorTypeName)
                                        .using_index(kHNSWIndexName)
                                        .with_function(1, HNSW_IP_FN)
                                        .with_helper(1, HNSW_IP_FN)
                                        .ordering(Index::Ordering::DESC)
                                        .build();

static auto HNSW_INDEX_CAPABILITY =
    IndexTypeCapability().index_type(HNSW_INDEX_TYPE);
static auto HNSW_PROFILE_CAPABILITY = IndexProfileCapability()
                                          .index_profile(HNSW_L1_PROFILE)
                                          .index_profile(HNSW_L2_PROFILE)
                                          .index_profile(HNSW_COSINE_PROFILE)
                                          .index_profile(HNSW_IP_PROFILE);

VEF_GENERATE_ENTRY_POINTS(
    make_extension()
        .with(STORAGE)
        .with(COLUMN_STORE)
        .with(HNSW_INDEX_CAPABILITY)
        .with(HNSW_PROFILE_CAPABILITY)
        .type(SVECTOR)

        // Hex encoding of raw vector float bytes (SQL)
        .func(make_func<&svector_hex>("vector_hex")
                  .returns(STRING)
                  .param(SVECTOR)
                  .deterministic()
                  .build())

        // Vector euclidean norm (L2) function (SQL)
        .func(make_func<&svector_norm>("vector_norm")
                  .returns(REAL)
                  .param(SVECTOR)
                  .build())

        // Vector dimension function (SQL)
        .func(make_func<&svector_dims>("vector_dimension")
                  .returns(INT)
                  .param(SVECTOR)
                  .deterministic()
                  .build())

        // Maximum supported dimension function (SQL)
        .func(make_func<&svector_max_dims>("vector_max_dimension")
                  .returns(INT)
                  .no_params()
                  .deterministic()
                  .build())

        // Fixed-point formatted string function (SQL)
        .func(make_func<&svector_format>("vector_format")
                  .returns(STRING)
                  .param(SVECTOR)
                  .param(INT)
                  .deterministic()
                  .build())

        // Vector distance functions (SQL)
        .func(make_func<&svector_distance_l1>("l1_distance")
                  .returns(REAL)
                  .param(SVECTOR)
                  .param(SVECTOR)
                  .deterministic()
                  .build())
        .func(make_func<&svector_distance_l2>("l2_distance")
                  .returns(REAL)
                  .param(SVECTOR)
                  .param(SVECTOR)
                  .deterministic()
                  .build())
        .func(make_func<&svector_distance_cosine>("cosine_distance")
                  .returns(REAL)
                  .param(SVECTOR)
                  .param(SVECTOR)
                  .deterministic()
                  .build())
        .func(make_func<&svector_inner_product>("inner_product")
                  .returns(REAL)
                  .param(SVECTOR)
                  .param(SVECTOR)
                  .deterministic()
                  .build())
        .func(make_intrinsic_default<&svector_default>("vector_default")))
