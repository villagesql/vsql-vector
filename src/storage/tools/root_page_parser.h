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

#ifndef VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_ROOT_PAGE_PARSER_H_
#define VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_ROOT_PAGE_PARSER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "../root_page.h"

namespace svector {
namespace tool {

// Parser for SVECTOR root page
class RootPageParser {
 public:
  // Parsed root page data
  struct RootPageInfo {
    uint8_t version;
    uint8_t page_type;
    std::string storage_metadata;
    uint8_t num_root_pages;
    std::vector<uint32_t> other_root_page_refs;
    uint16_t column_size;
    uint32_t all_slot_head;
    uint32_t all_slot_tail;
    uint32_t total_data_pages;
    uint32_t total_free_pages;
    uint16_t free_slot_array_max_size;
    uint16_t free_slot_array_cur_size;
    std::vector<uint32_t> free_slots;
  };

  // Parse root page from raw page data
  static bool parse(const std::vector<uint8_t> &page_data, RootPageInfo &info,
                    std::string &error);

  // Display root page information
  static void display(const RootPageInfo &info, bool verbose = false);

 private:
  // Helper to read uint8_t from buffer
  static uint8_t read_uint8(const std::vector<uint8_t> &data, uint32_t offset);

  // Helper to read uint16_t from buffer (big-endian)
  static uint16_t read_uint16(const std::vector<uint8_t> &data,
                              uint32_t offset);

  // Helper to read uint32_t from buffer (big-endian)
  static uint32_t read_uint32(const std::vector<uint8_t> &data,
                              uint32_t offset);

  // Helper to read string from buffer
  static std::string read_string(const std::vector<uint8_t> &data,
                                 uint32_t offset, uint32_t length);
};

}  // namespace tool
}  // namespace svector

#endif  // VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_ROOT_PAGE_PARSER_H_
