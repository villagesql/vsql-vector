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

#ifndef VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_HNSW_LAYOUT_H_
#define VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_HNSW_LAYOUT_H_

#include <cstdint>
#include <string>
#include <vector>

namespace svector {
namespace tool {

// This tool reimplements the on-disk layouts defined by
// svector::hnsw::StorageMeta and svector::hnsw::LevelStore (see
// src/index/hnsw/storage.h and src/index/hnsw/hnsw.h) rather than linking
// against the HNSW index sources, so the two must be kept in sync by hand.

// Which of an HNSW level's two per-level stores a data page's records belong
// to. A store's kind is recovered from its root page's decoded
// HnswIndexMetadata::name, not passed in independently.
enum class HnswRecordKind {
  None,      // Not an HNSW store; records are plain SVECTOR column data.
  Neighbour, // svector::hnsw::NeighbourEntry records (the primary store).
  Overflow,  // svector::hnsw::OverflowEntry records (the overflow store).
};

// On-disk width, in bytes, of a single NID/VID (svector::hnsw::Id<Tag>
// ::STORAGE_SIZE).
constexpr uint32_t HNSW_ID_SIZE = 6;

// NID's "incoming" flag bit and the mask that recovers the underlying
// Column::Ref from a NID value (svector::hnsw::Id<NIDTag>::INCOMING_BIT /
// COLUMN_REF_MASK).
constexpr uint64_t HNSW_NID_INCOMING_BIT = uint64_t{1} << 46;
constexpr uint64_t HNSW_NID_REF_MASK = HNSW_NID_INCOMING_BIT - 1;

// A NID/VID's value is itself a Column::Ref (see
// DataPage::encode_column_ref/decode_column_ref): the low 32 bits are the
// data page's Page::Ref, and the next 14 bits (DataPage::SLOT_INDEX_BITS)
// are the slot index within that page. Bit 46 -- one above the slot field
// -- is what HNSW_NID_INCOMING_BIT repurposes.
struct HnswColumnRef {
  uint32_t page_ref;
  uint16_t slot_index;
};

// Splits a NID/VID value into the data page and slot its record lives at.
// Pass the raw NID/VID value; the incoming flag (if any) sits above the
// slot field decoded here and does not need to be masked off first.
HnswColumnRef hnsw_decode_ref(uint64_t value);

// Formats value (a NID or VID) as "Page #<p>, Slot #<s>" for display.
std::string format_hnsw_ref(uint64_t value);

// Decoded svector::hnsw::StorageMeta -- the metadata every HNSW root page
// carries, identifying which level and which of its two stores (primary or
// overflow) the page belongs to.
struct HnswIndexMetadata {
  uint8_t version = 0;
  std::string name;
  uint8_t level = 0;
  uint8_t entry_level = 0;
  // KeyPartRef values; 0 (VEF_STORAGE_EMPTY_COLUMN_REF) marks an unset entry
  // point. Only the level-0 primary store's metadata carries a real one.
  std::vector<uint64_t> entry_points;

  // True when name identifies an overflow store ("HNSW-L<n>-OV") rather than
  // the primary NeighbourEntry store ("HNSW-L<n>"), per
  // IndexStore::build_storage_specs()'s naming.
  bool is_overflow() const;
};

// Decodes a root page's raw storage-metadata bytes (the full metadata_len
// bytes read verbatim, not NUL-truncated) as an HNSW StorageMeta blob.
// Returns true on success; false if the bytes are not a well-formed
// encoding, with error set to why.
bool parse_hnsw_index_metadata(const std::string &raw, HnswIndexMetadata &meta,
                               std::string &error);

// Recovers max_neighbours from a primary store's column_size and whether its
// level has a lower level (level > 0), by inverting
// NeighbourEntry::storage_size().
uint32_t hnsw_max_neighbours(uint16_t column_size, bool has_lower_level);

// Recovers overflow capacity from an overflow store's column_size, by
// inverting OverflowEntry::storage_size().
uint32_t hnsw_overflow_capacity(uint16_t column_size);

} // namespace tool
} // namespace svector

#endif // VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_HNSW_LAYOUT_H_
