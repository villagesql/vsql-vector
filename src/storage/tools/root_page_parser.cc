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

#include "root_page_parser.h"

#include <iomanip>
#include <iostream>

#include "page_reader.h"

namespace svector {
namespace tool {

uint8_t RootPageParser::read_uint8(const std::vector<uint8_t> &data,
                                   uint32_t offset) {
  return data[offset];
}

uint16_t RootPageParser::read_uint16(const std::vector<uint8_t> &data,
                                     uint32_t offset) {
  // Big-endian (network byte order)
  return (static_cast<uint16_t>(data[offset]) << 8) |
         static_cast<uint16_t>(data[offset + 1]);
}

uint32_t RootPageParser::read_uint32(const std::vector<uint8_t> &data,
                                     uint32_t offset) {
  // Big-endian (network byte order)
  return (static_cast<uint32_t>(data[offset]) << 24) |
         (static_cast<uint32_t>(data[offset + 1]) << 16) |
         (static_cast<uint32_t>(data[offset + 2]) << 8) |
         static_cast<uint32_t>(data[offset + 3]);
}

std::string RootPageParser::read_string(const std::vector<uint8_t> &data,
                                        uint32_t offset, uint32_t length) {
  std::string result;
  result.reserve(length);
  for (uint32_t i = 0; i < length && data[offset + i] != '\0'; ++i) {
    result.push_back(static_cast<char>(data[offset + i]));
  }
  return result;
}

bool RootPageParser::parse(const std::vector<uint8_t> &page_data,
                           RootPageInfo &info, std::string &error) {
  constexpr uint32_t PAGE_HEADER_SIZE = PageReader::FIL_PAGE_DATA;

  if (page_data.size() <= PAGE_HEADER_SIZE) {
    error = "Page too small to contain root page header";
    return false;
  }

  RootPage rp;
  uint8_t num_segments = read_uint8(page_data, PAGE_HEADER_SIZE);
  // Three-stage bootstrap: each stage unlocks the next offset function.
  // Stage 1: set N only (enables storage_metadata_len_off()).
  rp.set_layout(num_segments, 0, 1);

  // Validate page size
  if (page_data.size() <= rp.storage_metadata_len_off()) {
    error = "Page too small to contain root page header";
    return false;
  }

  // Stage 2: read metadata length (enables num_other_root_pages_off()).
  uint8_t metadata_len = read_uint8(page_data, rp.storage_metadata_len_off());
  rp.set_layout(num_segments, metadata_len, 1);

  if (page_data.size() <= rp.num_other_root_pages_off()) {
    error = "Page too small to contain root page header";
    return false;
  }

  // Stage 3: read K and finalize layout.
  uint8_t num_other_root_pages =
      read_uint8(page_data, rp.num_other_root_pages_off());
  rp.set_layout(num_segments, metadata_len, num_other_root_pages);

  if (page_data.size() < rp.free_slot_array_off()) {
    error = "Page too small to contain root page header";
    return false;
  }

  // Parse fields using offsets from RootPage
  info.version = read_uint8(page_data, rp.version_off());
  info.page_type = read_uint8(page_data, rp.page_type_off());
  info.storage_metadata =
      read_string(page_data, rp.storage_metadata_off(), metadata_len);
  info.num_root_pages = num_other_root_pages + 1;
  info.other_root_page_refs.clear();
  for (uint8_t i = 0; i < num_other_root_pages; ++i) {
    uint32_t off = rp.other_root_pages_off() + i * RootPage::ROOT_PAGE_REF_LEN;
    info.other_root_page_refs.push_back(read_uint32(page_data, off));
  }
  info.column_size = read_uint16(page_data, rp.column_size_off());
  info.all_slot_head = read_uint32(page_data, rp.all_slot_head_off());
  info.all_slot_tail = read_uint32(page_data, rp.all_slot_tail_off());
  info.total_data_pages = read_uint32(page_data, rp.total_data_pages_off());
  info.total_free_pages = read_uint32(page_data, rp.total_free_pages_off());
  info.free_slot_array_max_size =
      read_uint16(page_data, rp.free_slot_array_max_size_off());
  info.free_slot_array_cur_size =
      read_uint16(page_data, rp.free_slot_array_cur_size_off());

  // Validate page type
  if (info.page_type != static_cast<uint8_t>(ColumnPageType::ROOT_PAGE)) {
    error = "Invalid page type: expected ROOT_PAGE (1), got " +
            std::to_string(info.page_type);
    return false;
  }

  // Parse free slot array
  info.free_slots.clear();
  info.free_slots.reserve(info.free_slot_array_cur_size);

  for (uint16_t i = 0; i < info.free_slot_array_cur_size; ++i) {
    uint32_t offset = rp.free_slot_array_off() + (i * RootPage::FREE_SLOT_LEN);
    if (offset + RootPage::FREE_SLOT_LEN > page_data.size()) {
      error = "Free slot array extends beyond page boundary";
      return false;
    }
    uint32_t slot_value = read_uint32(page_data, offset);
    info.free_slots.push_back(slot_value);
  }

  return true;
}

void RootPageParser::display(const RootPageInfo &info, bool verbose) {
  std::cout << info.storage_metadata << " Root Page\n";
  std::cout << "=================\n\n";

  std::cout << "Version:           " << static_cast<int>(info.version) << "\n";
  std::cout << "Page Type:         " << static_cast<int>(info.page_type)
            << " (ROOT_PAGE)\n";
  std::cout << "Num Root Pages:    " << static_cast<int>(info.num_root_pages)
            << "\n";
  bool has_assigned_root = false;
  for (uint32_t ref : info.other_root_page_refs) {
    if (ref != RootPage::NULL_FREE_PAGE_REF) {
      has_assigned_root = true;
      break;
    }
  }
  if (has_assigned_root) {
    std::cout << "Other Root Pages:";
    for (uint32_t ref : info.other_root_page_refs) {
      if (ref != RootPage::NULL_FREE_PAGE_REF) {
        std::cout << " Page #" << ref;
      }
    }
    std::cout << "\n";
  }

  std::cout << "Column Size:       " << info.column_size << " bytes";

  // Calculate vector dimensions (assuming float32)
  if (info.column_size % 4 == 0) {
    std::cout << " (" << (info.column_size / 4) << "-dim float vector)";
  }
  std::cout << "\n\n";

  std::cout << "Data Pages:\n";
  std::cout << "  Total:           " << info.total_data_pages << "\n";
  std::cout << "  Free:            " << info.total_free_pages << "\n";
  std::cout << "  Head:            Page #" << info.all_slot_head;
  if (info.all_slot_head == RootPage::NULL_FREE_PAGE_REF) {
    std::cout << " (NULL)";
  }
  std::cout << "\n";
  std::cout << "  Tail:            Page #" << info.all_slot_tail;
  if (info.all_slot_tail == RootPage::NULL_FREE_PAGE_REF) {
    std::cout << " (NULL)";
  }
  std::cout << "\n\n";

  std::cout << "Free Slot Array:\n";
  std::cout << "  Max Capacity:    " << info.free_slot_array_max_size
            << " slots\n";
  std::cout << "  Current Size:    " << info.free_slot_array_cur_size
            << " slots\n";

  if (verbose && !info.free_slots.empty()) {
    std::cout << "\n  Slot List:\n";
    for (size_t i = 0; i < info.free_slots.size(); ++i) {
      std::cout << "    Slot[" << std::setw(3) << i << "] -> ";
      if (info.free_slots[i] == RootPage::NULL_FREE_PAGE_REF) {
        std::cout << "(empty)\n";
      } else {
        std::cout << "Page #" << info.free_slots[i] << "\n";
      }
    }
  } else if (!info.free_slots.empty()) {
    // Count non-empty slots
    uint32_t non_empty = 0;
    for (uint32_t slot : info.free_slots) {
      if (slot != RootPage::NULL_FREE_PAGE_REF) {
        non_empty++;
      }
    }
    std::cout << "  Non-empty Slots: " << non_empty << "\n";
  }

  std::cout << "\n";
}

}  // namespace tool
}  // namespace svector
