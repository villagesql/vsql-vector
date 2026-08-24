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

#ifndef VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_DATA_PAGE_PARSER_H_
#define VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_DATA_PAGE_PARSER_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../data_page.h"
#include "hnsw_layout.h"

namespace svector {
namespace tool {

// Parser for SVECTOR data page
class DataPageParser {
 public:
  // Record status
  struct RecordStatus {
    bool is_free;
    bool is_deleted;
    uint64_t trx_ref;
    std::vector<float> vector_data; // Decoded vector (assuming float32);
                                    // valid when index_kind == None

    // Valid only when index_kind == HnswRecordKind::Neighbour, decoding the
    // record per svector::hnsw::NeighbourEntry's layout.
    uint64_t owner_vid = 0;
    uint64_t lower_level_nid = 0; // valid only when has_lower_level
    std::vector<std::pair<uint64_t, uint64_t>> neighbours; // (nid, vid)
    uint64_t overflow_nid = 0;

    // Valid only when index_kind == HnswRecordKind::Overflow, decoding the
    // record per svector::hnsw::OverflowEntry's layout. overflow_nid above
    // is shared with the Neighbour case -- both records end in the same
    // trailing overflow-chain NID.
    std::vector<uint64_t> incoming;
  };

  // Parsed data page data
  struct DataPageInfo {
    uint8_t version;
    uint8_t page_type;
    uint16_t free_slot_number;
    uint32_t root_page_ref;
    uint32_t prev_free_page;
    uint32_t next_free_page;
    uint16_t max_num_recs;
    uint16_t num_free_recs;
    // InnoDB page header fields
    uint32_t fil_page_prev;  // Previous page in list
    uint32_t fil_page_next;  // Next page in list
    std::vector<RecordStatus> records;

    // Set from parse()'s index_kind/has_lower_level arguments; tells
    // display() which of RecordStatus's field groups holds real data.
    HnswRecordKind index_kind = HnswRecordKind::None;
    bool has_lower_level = false;

    // Calculated statistics
    uint16_t num_allocated() const {
      uint16_t count = 0;
      for (const auto &rec : records) {
        if (!rec.is_free) count++;
      }
      return count;
    }

    uint16_t num_deleted() const {
      uint16_t count = 0;
      for (const auto &rec : records) {
        if (!rec.is_free && rec.is_deleted) count++;
      }
      return count;
    }

    uint16_t num_active() const {
      uint16_t count = 0;
      for (const auto &rec : records) {
        if (!rec.is_free && !rec.is_deleted) count++;
      }
      return count;
    }

    float utilization_percent() const {
      if (max_num_recs == 0) return 0.0f;
      return (100.0f * num_allocated()) / max_num_recs;
    }

    float free_percent() const {
      if (max_num_recs == 0) return 0.0f;
      return (100.0f * num_free_recs) / max_num_recs;
    }
  };

  // Parse data page from raw page data. index_kind selects how each record's
  // column bytes are decoded: None decodes them as a plain SVECTOR float
  // vector (the existing behavior); Neighbour/Overflow decode them per HNSW's
  // NeighbourEntry/OverflowEntry layout instead, with has_lower_level
  // selecting whether a Neighbour record carries a LowerLevel field (see
  // svector::hnsw::LevelStore::LevelId::has_lower_level()).
  static bool parse(const std::vector<uint8_t> &page_data, uint16_t column_size,
                    DataPageInfo &info, std::string &error,
                    HnswRecordKind index_kind = HnswRecordKind::None,
                    bool has_lower_level = false);

  // Display data page information
  static void display(const DataPageInfo &info, bool verbose = false,
                      bool show_records = false, uint32_t record_start = 0,
                      uint32_t record_count = 10);

 private:
  // Helper to read uint8_t from buffer
  static uint8_t read_uint8(const std::vector<uint8_t> &data, uint32_t offset);

  // Helper to read uint16_t from buffer (big-endian)
  static uint16_t read_uint16(const std::vector<uint8_t> &data,
                              uint32_t offset);

  // Helper to read uint32_t from buffer (big-endian)
  static uint32_t read_uint32(const std::vector<uint8_t> &data,
                              uint32_t offset);

  // Helper to read uint64_t from buffer (big-endian)
  static uint64_t read_uint64(const std::vector<uint8_t> &data,
                              uint32_t offset);

  // Helper to read float from buffer (using float4store little-endian format)
  static float read_float(const std::vector<uint8_t> &data, uint32_t offset);

  // Helper to read a 6-byte big-endian NID/VID (HNSW_ID_SIZE) from buffer.
  static uint64_t read_id48(const std::vector<uint8_t> &data, uint32_t offset);

  // Get record status bits from bitmap
  static void get_record_bits(const std::vector<uint8_t> &data,
                              uint32_t bitmap_offset, uint16_t rec_index,
                              bool &is_free, bool &is_deleted);
};

}  // namespace tool
}  // namespace svector

#endif  // VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_DATA_PAGE_PARSER_H_
