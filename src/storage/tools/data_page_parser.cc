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

#include "data_page_parser.h"

#include <cstring>
#include <iomanip>
#include <iostream>

#include "../root_page.h"

namespace svector {
namespace tool {

uint8_t DataPageParser::read_uint8(const std::vector<uint8_t> &data,
                                   uint32_t offset) {
  return data[offset];
}

uint16_t DataPageParser::read_uint16(const std::vector<uint8_t> &data,
                                     uint32_t offset) {
  // Big-endian (network byte order)
  return (static_cast<uint16_t>(data[offset]) << 8) |
         static_cast<uint16_t>(data[offset + 1]);
}

uint32_t DataPageParser::read_uint32(const std::vector<uint8_t> &data,
                                     uint32_t offset) {
  // Big-endian (network byte order)
  return (static_cast<uint32_t>(data[offset]) << 24) |
         (static_cast<uint32_t>(data[offset + 1]) << 16) |
         (static_cast<uint32_t>(data[offset + 2]) << 8) |
         static_cast<uint32_t>(data[offset + 3]);
}

uint64_t DataPageParser::read_uint64(const std::vector<uint8_t> &data,
                                     uint32_t offset) {
  // Big-endian (network byte order)
  return (static_cast<uint64_t>(data[offset]) << 56) |
         (static_cast<uint64_t>(data[offset + 1]) << 48) |
         (static_cast<uint64_t>(data[offset + 2]) << 40) |
         (static_cast<uint64_t>(data[offset + 3]) << 32) |
         (static_cast<uint64_t>(data[offset + 4]) << 24) |
         (static_cast<uint64_t>(data[offset + 5]) << 16) |
         (static_cast<uint64_t>(data[offset + 6]) << 8) |
         static_cast<uint64_t>(data[offset + 7]);
}

float DataPageParser::read_float(const std::vector<uint8_t> &data,
                                 uint32_t offset) {
  // SVECTOR uses float4store which stores 32-bit floats in little-endian format
  // Read 4 bytes in little-endian order
  uint32_t val = static_cast<uint32_t>(data[offset]) |
                 (static_cast<uint32_t>(data[offset + 1]) << 8) |
                 (static_cast<uint32_t>(data[offset + 2]) << 16) |
                 (static_cast<uint32_t>(data[offset + 3]) << 24);
  float f;
  std::memcpy(&f, &val, sizeof(float));
  return f;
}

void DataPageParser::get_record_bits(const std::vector<uint8_t> &data,
                                     uint32_t bitmap_offset, uint16_t rec_index,
                                     bool &is_free, bool &is_deleted) {
  // Each record uses 2 bits: DELETE_MARK_BIT and FREE_BIT
  uint32_t bit_pos = rec_index * DataPage::BITS_PER_RECORD;
  uint32_t byte_offset = bitmap_offset + (bit_pos / 8);
  uint8_t bit_in_byte = bit_pos % 8;

  uint8_t bitmap_byte = data[byte_offset];

  // Bit 0: Delete mark (0 = active, 1 = deleted)
  // Bit 1: Free (0 = free slot, 1 = occupied)
  is_deleted =
      (bitmap_byte & (1 << (bit_in_byte + DataPage::DELETE_MARK_BIT))) != 0;
  is_free = (bitmap_byte & (1 << (bit_in_byte + DataPage::FREE_BIT))) == 0;
}

bool DataPageParser::parse(const std::vector<uint8_t> &page_data,
                           uint16_t column_size, DataPageInfo &info,
                           std::string &error) {
  // Validate page size
  if (page_data.size() < DataPage::FREE_BITMAP_OFF) {
    error = "Page too small to contain data page header";
    return false;
  }

  // Parse InnoDB page header fields
  // From storage/innobase/include/fil0types.h
  constexpr uint32_t FIL_PAGE_PREV = 8;   // Previous page in page list
  constexpr uint32_t FIL_PAGE_NEXT = 12;  // Next page in page list
  info.fil_page_prev = read_uint32(page_data, FIL_PAGE_PREV);
  info.fil_page_next = read_uint32(page_data, FIL_PAGE_NEXT);

  // Parse SVECTOR data page header fields
  info.version = read_uint8(page_data, DataPage::VERSION_OFF);
  info.page_type = read_uint8(page_data, DataPage::PAGE_TYPE_OFF);
  info.free_slot_number =
      read_uint16(page_data, DataPage::FREE_SLOT_NUMBER_OFF);
  info.prev_free_page = read_uint32(page_data, DataPage::PREV_FREE_PAGE_OFF);
  info.next_free_page = read_uint32(page_data, DataPage::NEXT_FREE_PAGE_OFF);
  info.max_num_recs = read_uint16(page_data, DataPage::MAX_NUM_RECS_OFF);
  info.num_free_recs = read_uint16(page_data, DataPage::NUM_FREE_RECS_OFF);

  // Validate page type
  if (info.page_type != static_cast<uint8_t>(ColumnPageType::DATA_PAGE)) {
    error = "Invalid page type: expected DATA_PAGE (2), got " +
            std::to_string(info.page_type);
    return false;
  }

  // Calculate bitmap size
  uint32_t bitmap_size_bits = info.max_num_recs * DataPage::BITS_PER_RECORD;
  uint32_t bitmap_size_bytes = (bitmap_size_bits + 7) / 8;

  // Calculate first record offset
  uint32_t first_rec_offset = DataPage::FREE_BITMAP_OFF + bitmap_size_bytes;
  uint32_t rec_size = DataPage::TRX_REF_SIZE + column_size;

  // Parse records
  info.records.clear();
  info.records.reserve(info.max_num_recs);

  uint32_t vector_dim = (column_size) / sizeof(float);

  for (uint16_t i = 0; i < info.max_num_recs; ++i) {
    RecordStatus rec;

    // Get record status from bitmap
    get_record_bits(page_data, DataPage::FREE_BITMAP_OFF, i, rec.is_free,
                    rec.is_deleted);

    // Read record data if allocated
    uint32_t rec_offset = first_rec_offset + (i * rec_size);

    if (!rec.is_free && rec_offset + rec_size <= page_data.size()) {
      // Read transaction reference
      rec.trx_ref = read_uint64(page_data, rec_offset);

      // Read vector data (assuming float32 for display)
      rec.vector_data.reserve(vector_dim);
      for (uint32_t j = 0; j < vector_dim; ++j) {
        uint32_t float_offset =
            rec_offset + DataPage::TRX_REF_SIZE + (j * sizeof(float));
        if (float_offset + sizeof(float) <= page_data.size()) {
          rec.vector_data.push_back(read_float(page_data, float_offset));
        }
      }
    } else {
      rec.trx_ref = 0;
    }

    info.records.push_back(rec);
  }

  return true;
}

void DataPageParser::display(const DataPageInfo &info, bool verbose,
                             bool show_records, uint32_t record_start,
                             uint32_t record_count) {
  std::cout << "SVECTOR Data Page\n";
  std::cout << "=================\n\n";

  std::cout << "Version:           " << static_cast<int>(info.version) << "\n";
  std::cout << "Page Type:         " << static_cast<int>(info.page_type)
            << " (DATA_PAGE)\n";
  std::cout << "Free Slot Number:  " << info.free_slot_number;
  if (info.free_slot_number == DataPage::INVALID_SLOT) {
    std::cout << " (not in free list)";
  }
  std::cout << "\n\n";

  std::cout << "SVECTOR Data Page Links:\n";
  std::cout << "  Previous:        Page #" << info.fil_page_prev;
  if (info.fil_page_prev == 0xFFFFFFFF) {
    std::cout << " (NULL)";
  }
  std::cout << "\n";
  std::cout << "  Next:            Page #" << info.fil_page_next;
  if (info.fil_page_next == 0xFFFFFFFF) {
    std::cout << " (NULL)";
  }
  std::cout << "\n\n";

  std::cout << "SVECTOR Free Page Links:\n";
  std::cout << "  Previous:        Page #" << info.prev_free_page;
  if (info.prev_free_page == 0xFFFFFFFF) {
    std::cout << " (NULL)";
  }
  std::cout << "\n";
  std::cout << "  Next:            Page #" << info.next_free_page;
  if (info.next_free_page == 0xFFFFFFFF) {
    std::cout << " (NULL)";
  }
  std::cout << "\n\n";

  std::cout << "Capacity:\n";
  std::cout << "  Max Records:     " << info.max_num_recs << "\n";
  std::cout << "  Free Records:    " << info.num_free_recs << " (" << std::fixed
            << std::setprecision(1) << info.free_percent() << "%)\n";
  std::cout << "  Allocated:       " << info.num_allocated() << " ("
            << std::fixed << std::setprecision(1) << info.utilization_percent()
            << "%)\n";
  std::cout << "    Active:        " << info.num_active() << "\n";
  std::cout << "    Deleted:       " << info.num_deleted() << "\n\n";

  if (verbose || show_records) {
    std::cout << "Record Bitmap:\n  ";
    for (size_t i = 0; i < info.records.size(); ++i) {
      if (i > 0 && i % 40 == 0) {
        std::cout << "\n  ";
      }
      const auto &rec = info.records[i];
      if (rec.is_free) {
        std::cout << ".";  // Free
      } else if (rec.is_deleted) {
        std::cout << "D";  // Deleted
      } else {
        std::cout << "A";  // Active
      }
    }
    std::cout << "\n  (. = Free, A = Active, D = Deleted)\n\n";
  }

  if (show_records) {
    std::cout << "Records (showing from slot " << record_start << ", up to "
              << record_count << " records):\n";
    uint32_t shown = 0;
    uint32_t skipped = 0;
    for (size_t i = 0; i < info.records.size() && shown < record_count; ++i) {
      const auto &rec = info.records[i];
      if (!rec.is_free) {
        if (skipped < record_start) {
          skipped++;
          continue;
        }
        std::cout << "  [" << std::setw(3) << i << "] ";
        std::cout << "Trx ID:" << std::setw(12) << rec.trx_ref;
        if (rec.is_deleted) {
          std::cout << " (DELETED)";
        }
        std::cout << " Data:[";
        for (size_t j = 0; j < rec.vector_data.size(); ++j) {
          if (j > 0) std::cout << ", ";
          std::cout << std::fixed << std::setprecision(2) << rec.vector_data[j];
        }
        std::cout << "]\n";
        shown++;
      }
    }
    if (skipped + shown < info.num_allocated()) {
      std::cout << "  ... (" << (info.num_allocated() - skipped - shown)
                << " more records)\n";
    }
    std::cout << "\n";
  }
}

}  // namespace tool
}  // namespace svector
