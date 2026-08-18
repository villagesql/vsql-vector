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

#include "hnsw_layout.h"

#include <sstream>

namespace svector {
namespace tool {

namespace {
constexpr uint8_t STORAGE_META_VERSION = 1;
constexpr uint8_t ENTRY_POINT_LEN = 8;
constexpr char OVERFLOW_NAME_SUFFIX[] = "-OV";

// Mirrors DataPage::SLOT_INDEX_BITS/MAX_SLOT_INDEX: the width of the slot
// index field packed into bits 32-45 of a Column::Ref.
constexpr uint8_t SLOT_INDEX_BITS = 14;
constexpr uint64_t SLOT_INDEX_MASK = (uint64_t{1} << SLOT_INDEX_BITS) - 1;
} // namespace

bool HnswIndexMetadata::is_overflow() const {
  const size_t suffix_len = sizeof(OVERFLOW_NAME_SUFFIX) - 1;
  return name.size() >= suffix_len &&
         name.compare(name.size() - suffix_len, suffix_len,
                      OVERFLOW_NAME_SUFFIX) == 0;
}

bool parse_hnsw_index_metadata(const std::string &raw, HnswIndexMetadata &meta,
                               std::string &error) {
  const auto *p = reinterpret_cast<const uint8_t *>(raw.data());
  size_t rem = raw.size();

  if (rem < 1) {
    error = "empty metadata";
    return false;
  }
  meta.version = *p++;
  --rem;
  if (meta.version != STORAGE_META_VERSION) {
    error = "unexpected StorageMeta version " + std::to_string(meta.version);
    return false;
  }

  if (rem < 1) {
    error = "truncated metadata: missing name length";
    return false;
  }
  uint8_t name_len = *p++;
  --rem;
  if (rem < name_len) {
    error = "truncated metadata: name";
    return false;
  }
  meta.name.assign(reinterpret_cast<const char *>(p), name_len);
  p += name_len;
  rem -= name_len;

  if (rem < 2) {
    error = "truncated metadata: level/entry_level";
    return false;
  }
  meta.level = *p++;
  meta.entry_level = *p++;
  rem -= 2;

  if (rem < 1) {
    error = "truncated metadata: entry point count";
    return false;
  }
  uint8_t num_eps = *p++;
  --rem;

  const size_t eps_bytes = static_cast<size_t>(num_eps) * ENTRY_POINT_LEN;
  if (rem < eps_bytes) {
    error = "truncated metadata: entry points";
    return false;
  }

  meta.entry_points.clear();
  meta.entry_points.reserve(num_eps);
  for (uint8_t i = 0; i < num_eps; ++i) {
    uint64_t v = 0;
    for (int b = 0; b < 8; ++b)
      v = (v << 8) | static_cast<uint64_t>(*p++);
    meta.entry_points.push_back(v);
  }
  rem -= eps_bytes;

  return true;
}

uint32_t hnsw_max_neighbours(uint16_t column_size, bool has_lower_level) {
  // storage_size = VID + (has_lower ? NID : 0) + max_n * (NID + VID) + NID
  constexpr uint32_t NODE_SIZE = 2 * HNSW_ID_SIZE;
  const uint32_t fixed = HNSW_ID_SIZE * (has_lower_level ? 3 : 2);
  if (column_size < fixed)
    return 0;
  return (column_size - fixed) / NODE_SIZE;
}

uint32_t hnsw_overflow_capacity(uint16_t column_size) {
  // storage_size = capacity * NID + NID
  if (column_size < HNSW_ID_SIZE)
    return 0;
  return (column_size / HNSW_ID_SIZE) - 1;
}

HnswColumnRef hnsw_decode_ref(uint64_t value) {
  HnswColumnRef ref;
  ref.page_ref = static_cast<uint32_t>(value & 0xFFFFFFFFull);
  ref.slot_index = static_cast<uint16_t>((value >> 32) & SLOT_INDEX_MASK);
  return ref;
}

std::string format_hnsw_ref(uint64_t value) {
  HnswColumnRef ref = hnsw_decode_ref(value);
  std::ostringstream out;
  out << "Page #" << ref.page_ref << ", Slot #" << ref.slot_index;
  return out.str();
}

} // namespace tool
} // namespace svector
