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

#include "data_page.h"

#include <cassert>
#include <limits>
#include <tuple>

#include "root_page.h"

namespace svector {

using vsql::preview_storage::Error;

// Helper function to convert bits to bytes (ceiling division)
inline uint32_t bits_to_bytes_ceil(uint32_t bits) { return (bits + 7) >> 3; }

void DataPage::init(Space::Ref space_ref, uint16_t col_len) {
  // Calculate record size: transaction ref + column data
  m_rec_size = TRX_REF_SIZE + col_len;

  // Calculate maximum number of records that can fit in the page
  // This requires an iterative approach since bitmap size depends on num_recs
  uint32_t page_size = Page::get_size(space_ref);
  uint32_t available_space = page_size - FREE_BITMAP_OFF - Page::TRAILER_SIZE;

  // Validate we have minimum space for at least one record
  if (available_space < m_rec_size) {
    m_max_num_recs = 0;
    m_free_bitmap_size = 0;
    m_first_rec_offset = FREE_BITMAP_OFF;
    return;
  }

  // Start with conservative estimate (no bitmap)
  uint16_t estimated_max_recs =
      static_cast<uint16_t>(available_space / m_rec_size);

  // Iteratively refine to account for bitmap space with safety counter
  uint16_t prev_max_recs = 0;
  uint16_t iterations = 0;
  constexpr uint16_t MAX_ITERATIONS = 10;

  while (estimated_max_recs != prev_max_recs && iterations < MAX_ITERATIONS) {
    prev_max_recs = estimated_max_recs;
    iterations++;

    // Calculate bitmap size for current estimate
    // Bitmap uses BITS_PER_RECORD (2) bits per record
    m_free_bitmap_size = static_cast<Page::Offset>(
        bits_to_bytes_ceil(estimated_max_recs * BITS_PER_RECORD));

    // Recalculate max records with bitmap space accounted for
    uint32_t space_for_records = available_space - m_free_bitmap_size;
    estimated_max_recs = static_cast<uint16_t>(space_for_records / m_rec_size);
  }

  m_max_num_recs = estimated_max_recs;
  m_first_rec_offset = FREE_BITMAP_OFF + m_free_bitmap_size;

#ifndef NDEBUG
  // Validate final layout fits in page
  uint32_t total_used = m_first_rec_offset +
                        (static_cast<uint32_t>(m_max_num_recs) * m_rec_size) +
                        Page::TRAILER_SIZE;
  assert(total_used <= page_size);
#endif  // NDEBUG
}

void DataPage::format(Page &data_page, MtrCtx::Ref mtr, uint8_t format_version,
                      Page::Ref root_page_ref) {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE));

  assert(has_capacity());

  // Write version
  data_page.write_integer_1(VERSION_OFF, format_version, mtr);

  // Write page type
  data_page.write_integer_1(
      PAGE_TYPE_OFF, static_cast<uint8_t>(ColumnPageType::DATA_PAGE), mtr);

  // Write free slot number (initially invalid)
  data_page.write_integer_2(FREE_SLOT_NUMBER_OFF, INVALID_SLOT, mtr);

  // Write the owning root page reference
  data_page.write_integer_4(ROOT_PAGE_REF_OFF, root_page_ref, mtr);

  // Write previous free page (initially invalid)
  data_page.write_integer_4(PREV_FREE_PAGE_OFF, Page::INVALID_REF, mtr);

  // Write next free page (initially invalid)
  data_page.write_integer_4(NEXT_FREE_PAGE_OFF, Page::INVALID_REF, mtr);

  // Write maximum number of records
  data_page.write_integer_2(MAX_NUM_RECS_OFF, m_max_num_recs, mtr);

  // Write number of free records (initially all records are free)
  data_page.write_integer_2(NUM_FREE_RECS_OFF, m_max_num_recs, mtr);
}

Column::Ref DataPage::encode_column_ref(Page::Ref page_ref,
                                        uint16_t slot_index) {
  // Encoding: Lower 32 bits = page ref, Upper 16 bits = slot index
  return static_cast<uint64_t>(page_ref) |
         (static_cast<uint64_t>(slot_index) << 32);
}

void DataPage::decode_column_ref(Column::Ref col_ref, Page::Ref &page_ref,
                                 uint16_t &slot_index) {
  // Decoding: Extract page ref from lower 32 bits, slot from upper 16 bits
  page_ref = static_cast<Page::Ref>(col_ref & 0xFFFFFFFF);
  slot_index = static_cast<uint16_t>((col_ref >> 32) & 0xFFFF);
}

void DataPage::set_free_slot_number(Page &data_page, uint16_t slot_number,
                                    MtrCtx::Ref mtr) const {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE));
  data_page.write_integer_2(FREE_SLOT_NUMBER_OFF, slot_number, mtr);
}

uint16_t DataPage::get_free_slot_number(Page &data_page) const {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE) ||
         data_page.is_loaded(Page::Latch::SHARED));
  return data_page.read_integer_2(FREE_SLOT_NUMBER_OFF);
}

Page::Ref DataPage::get_root_page_ref(Page &data_page) const {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE) ||
         data_page.is_loaded(Page::Latch::SHARED));
  return data_page.read_integer_4(ROOT_PAGE_REF_OFF);
}

void DataPage::set_free_links(Page &data_page, Page::Ref *prev_free_page,
                              Page::Ref *next_free_page,
                              MtrCtx::Ref mtr) const {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE));

  // Write previous free page reference if provided
  if (prev_free_page != nullptr) {
    data_page.write_integer_4(PREV_FREE_PAGE_OFF, *prev_free_page, mtr);
  }

  // Write next free page reference if provided
  if (next_free_page != nullptr) {
    data_page.write_integer_4(NEXT_FREE_PAGE_OFF, *next_free_page, mtr);
  }
}

void DataPage::get_free_links(Page &data_page, Page::Ref *prev_free_page,
                              Page::Ref *next_free_page) const {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE) ||
         data_page.is_loaded(Page::Latch::SHARED));

  // Read previous free page reference if requested
  if (prev_free_page != nullptr) {
    *prev_free_page = data_page.read_integer_4(PREV_FREE_PAGE_OFF);
  }

  // Read next free page reference if requested
  if (next_free_page != nullptr) {
    *next_free_page = data_page.read_integer_4(NEXT_FREE_PAGE_OFF);
  }
}

bool DataPage::has_last_free_slot(Page &data_page) const {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE) ||
         data_page.is_loaded(Page::Latch::SHARED));

  // Read the number of free records from the data page
  uint16_t num_free_recs = data_page.read_integer_2(NUM_FREE_RECS_OFF);

  // Return true if this is the last free slot (exactly 1 free record remaining)
  // After inserting into this slot, the page will be full and needs to be
  // removed from the free list.
  return (num_free_recs == 1);
}

bool DataPage::needs_add_to_free_list(Page &data_page, bool pre_purge) const {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE) ||
         data_page.is_loaded(Page::Latch::SHARED));

  // Check if the page is already in the free list
  uint16_t free_slot_number = get_free_slot_number(data_page);
  if (free_slot_number != INVALID_SLOT) {
    return false;  // Already in the free list
  }

  // Calculate free percentage
  uint16_t max_num_recs = data_page.read_integer_2(MAX_NUM_RECS_OFF);
  uint16_t num_free_recs = data_page.read_integer_2(NUM_FREE_RECS_OFF);

  // Validate that the page has capacity
  assert(max_num_recs > 0);

  // Avoid division by zero (should not happen if assert passes)
  if (max_num_recs == 0) {
    return false;
  }

  // Latch ordering requires root to be latched before data. If this page will
  // need to be added to the free list after purge (which requires writing the
  // root page), the caller must release the data latch, acquire root, then
  // re-acquire data — before the purge happens. pre_purge accounts for the
  // record that is about to be freed so this pre-check can answer "will the
  // page qualify post-purge?" without actually purging yet. The caller then
  // re-checks without pre_purge after the purge to make the final decision.
  if (pre_purge && num_free_recs < max_num_recs) {
    ++num_free_recs;
  }

  // Calculate free percentage: (num_free_recs * 100) / max_num_recs
  uint32_t free_percent = (static_cast<uint32_t>(num_free_recs) * 100) /
                          static_cast<uint32_t>(max_num_recs);

  // Return true if free percentage exceeds the threshold
  return (free_percent > MIN_FREE_PERCENT_FOR_FREE_LIST);
}

std::pair<bool, bool> DataPage::get_record_status(Page &data_page,
                                                  uint16_t slot_index) const {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE) ||
         data_page.is_loaded(Page::Latch::SHARED));

#ifndef NDEBUG
  // Validate slot index
  uint16_t max_num_recs = data_page.read_integer_2(MAX_NUM_RECS_OFF);
  assert(slot_index < max_num_recs);
#endif  // NDEBUG

  // Calculate bitmap byte and bit offsets
  Page::Offset bitmap_byte_offset =
      FREE_BITMAP_OFF + ((slot_index * BITS_PER_RECORD) >> 3);
  uint8_t bitmap_bit_offset = (slot_index * BITS_PER_RECORD) & 7;

  // Read bitmap byte
  uint8_t bitmap_byte = data_page.read_integer_1(bitmap_byte_offset);

  // Extract delete mark and free bits
  bool delete_marked =
      (bitmap_byte >> (bitmap_bit_offset + DELETE_MARK_BIT)) & 1;
  bool is_allocated = (bitmap_byte >> (bitmap_bit_offset + FREE_BIT)) & 1;

  return std::make_pair(delete_marked, !is_allocated);
}

void DataPage::set_record_delete(Page &data_page, uint16_t slot_index,
                                 MtrCtx::Ref mtr) const {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE));

#ifndef NDEBUG
  // Get current status
  auto [delete_marked, is_free] = get_record_status(data_page, slot_index);

  // Assert that the record is not free
  assert(!is_free && "Cannot mark a free record as deleted");

  // Assert that the record is not already delete marked
  assert(!delete_marked && "Record is already marked as deleted");

  // Calculate bitmap byte and bit offsets
  uint16_t max_num_recs = data_page.read_integer_2(MAX_NUM_RECS_OFF);
  assert(slot_index < max_num_recs);
#endif  // NDEBUG

  Page::Offset bitmap_byte_offset =
      FREE_BITMAP_OFF + ((slot_index * BITS_PER_RECORD) >> 3);
  uint8_t bitmap_bit_offset = (slot_index * BITS_PER_RECORD) & 7;

  // Read current bitmap byte
  uint8_t bitmap_byte = data_page.read_integer_1(bitmap_byte_offset);

  // Set DELETE_MARK_BIT to 1
  bitmap_byte |= (1 << (bitmap_bit_offset + DELETE_MARK_BIT));

  // Write updated bitmap byte
  data_page.write_integer_1(bitmap_byte_offset, bitmap_byte, mtr);
}

void DataPage::set_record_undelete(Page &data_page, uint16_t slot_index,
                                   bool trx_match, MtrCtx::Ref mtr) const {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE));
  // Used only in debug assert
  std::ignore = trx_match;

#ifndef NDEBUG
  // Get current status
  auto [delete_marked, is_free] = get_record_status(data_page, slot_index);
#endif  // NDEBUG

  // Assert that the record is not free
  assert(!is_free && "Cannot undelete a free record");

  // Assert that the record is delete marked. After a crash the operation could
  // be reattempted and it is fine if the delete mark is already removed. In
  // such case, the transaction ID must match.
  assert(trx_match || (delete_marked && "Record is not marked as deleted"));

#ifndef NDEBUG
  // Calculate bitmap byte and bit offsets
  uint16_t max_num_recs = data_page.read_integer_2(MAX_NUM_RECS_OFF);
  assert(slot_index < max_num_recs);
#endif  // NDEBUG

  Page::Offset bitmap_byte_offset =
      FREE_BITMAP_OFF + ((slot_index * BITS_PER_RECORD) >> 3);
  uint8_t bitmap_bit_offset = (slot_index * BITS_PER_RECORD) & 7;

  // Read current bitmap byte
  uint8_t bitmap_byte = data_page.read_integer_1(bitmap_byte_offset);

  // Clear DELETE_MARK_BIT (0 = not deleted)
  bitmap_byte &= ~(1 << (bitmap_bit_offset + DELETE_MARK_BIT));

  // Write updated bitmap byte
  data_page.write_integer_1(bitmap_byte_offset, bitmap_byte, mtr);
}

void DataPage::set_record_free(Page &data_page, uint16_t slot_index,
                               MtrCtx::Ref mtr) const {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE));

#ifndef NDEBUG
  // Get current status
  bool is_free = true;
  std::tie(std::ignore, is_free) = get_record_status(data_page, slot_index);

  // Assert that the record is not already free
  assert(!is_free && "Record is already free");

  // Calculate bitmap byte and bit offsets
  uint16_t max_num_recs = data_page.read_integer_2(MAX_NUM_RECS_OFF);
  assert(slot_index < max_num_recs);
#endif  // NDEBUG

  Page::Offset bitmap_byte_offset =
      FREE_BITMAP_OFF + ((slot_index * BITS_PER_RECORD) >> 3);
  uint8_t bitmap_bit_offset = (slot_index * BITS_PER_RECORD) & 7;

  // Read current bitmap byte
  uint8_t bitmap_byte = data_page.read_integer_1(bitmap_byte_offset);

  // Clear both FREE_BIT (free means bit = 0) and DELETE_MARK_BIT
  bitmap_byte &= ~(1 << (bitmap_bit_offset + FREE_BIT));
  bitmap_byte &= ~(1 << (bitmap_bit_offset + DELETE_MARK_BIT));

  // Write updated bitmap byte
  data_page.write_integer_1(bitmap_byte_offset, bitmap_byte, mtr);
}

void DataPage::set_record_allocated(Page &data_page, uint16_t slot_index,
                                    MtrCtx::Ref mtr) const {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE));

#ifndef NDEBUG
  // Get current status
  bool is_free = true;
  std::tie(std::ignore, is_free) = get_record_status(data_page, slot_index);

  // Assert that the record is free
  assert(is_free && "Can only allocate a free record");

  // Calculate bitmap byte and bit offsets
  uint16_t max_num_recs = data_page.read_integer_2(MAX_NUM_RECS_OFF);
  assert(slot_index < max_num_recs);
#endif  // NDEBUG

  Page::Offset bitmap_byte_offset =
      FREE_BITMAP_OFF + ((slot_index * BITS_PER_RECORD) >> 3);
  uint8_t bitmap_bit_offset = (slot_index * BITS_PER_RECORD) & 7;

  // Read current bitmap byte
  uint8_t bitmap_byte = data_page.read_integer_1(bitmap_byte_offset);

  // Set FREE_BIT = 1 (allocated) and clear DELETE_MARK_BIT = 0 (not deleted)
  bitmap_byte |= (1 << (bitmap_bit_offset + FREE_BIT));
  bitmap_byte &= ~(1 << (bitmap_bit_offset + DELETE_MARK_BIT));

  // Write updated bitmap byte
  data_page.write_integer_1(bitmap_byte_offset, bitmap_byte, mtr);
}

bool DataPage::free_bit_in_byte(uint8_t bitmap_byte,
                                uint8_t &bitmap_bit) const {
  // Mask to extract all FREE_BITs in a byte: 0b10101010
  // Each 2-bit group has FREE_BIT at position 1, so we check bits 1, 3, 5, 7
  constexpr uint8_t FREE_BIT_MASK = 0xAA;  // Binary: 10101010

  // Fast path: If all FREE_BITs are set (all slots occupied), skip this byte
  if ((bitmap_byte & FREE_BIT_MASK) == FREE_BIT_MASK) {
    return false;  // All 4 slots in this byte are occupied
  }

  // At least one free slot exists in this byte, find it
  for (uint8_t bit_offset = 0; bit_offset < 8; bit_offset += BITS_PER_RECORD) {
    // Check FREE_BIT (bit 1 of the 2-bit group): 0 = free, 1 = occupied
    if (((bitmap_byte >> (bit_offset + FREE_BIT)) & 1) == 0) {
      bitmap_bit = bit_offset;
      return true;
    }
  }
  assert(false);
  return false;
}

bool DataPage::find_free_slot(Page &data_page, uint16_t &slot_index,
                              uint16_t hint_index) const {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE) ||
         data_page.is_loaded(Page::Latch::SHARED));

  // Fast path: Check if hint is valid and points to a free slot
  if (hint_index < m_max_num_recs) {
    bool is_free = true;
    std::tie(std::ignore, is_free) = get_record_status(data_page, hint_index);

    if (is_free) {
      slot_index = hint_index;
      return true;
    }
  }

  // Slow path: Scan bitmap byte by byte
  // Calculate number of bytes needed (2 bits per record, 4 records per byte)
  uint16_t num_bitmap_bytes =
      bits_to_bytes_ceil(m_max_num_recs * BITS_PER_RECORD);

  // Calculate start byte index from the hint.
  Page::Offset start_idx = (hint_index * BITS_PER_RECORD) >> 3;

  // Scan bitmap byte by byte from start index.
  for (uint16_t byte_idx = start_idx; byte_idx < num_bitmap_bytes; byte_idx++) {
    uint8_t bitmap_byte = data_page.read_integer_1(FREE_BITMAP_OFF + byte_idx);
    uint8_t bitmap_bit = 0;

    if (free_bit_in_byte(bitmap_byte, bitmap_bit)) {
      slot_index = byte_idx * 4 + (bitmap_bit >> 1);  // BITS_PER_RECORD is 2
      assert(slot_index < m_max_num_recs);
      return true;
    }
  }

  // Wrap around and search from the beginning.
  for (uint16_t byte_idx = 0; byte_idx < start_idx; byte_idx++) {
    uint8_t bitmap_byte = data_page.read_integer_1(FREE_BITMAP_OFF + byte_idx);
    uint8_t bitmap_bit = 0;

    if (free_bit_in_byte(bitmap_byte, bitmap_bit)) {
      slot_index = byte_idx * 4 + (bitmap_bit >> 1);  // BITS_PER_RECORD is 2
      assert(slot_index < m_max_num_recs);
      return true;
    }
  }

  return false;
}

void DataPage::insert(Page &data_page, MtrCtx::Ref mtr, Segment::TrxRef trx_ref,
                      Column::Data col_data, Column::Ref &col_ref) {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE));
  assert(col_data.data != nullptr);
  assert(col_data.length > 0);

  uint16_t &hint_index = RootPage::s_last_slot_info.next_free_rec_hint;

  // Step 1: Check if there are free slots available
  uint16_t num_free_recs = data_page.read_integer_2(NUM_FREE_RECS_OFF);
  assert(num_free_recs > 0);

  // Step 2: Find the first free slot using hint
  uint16_t slot_index = 0;
  [[maybe_unused]]
  bool found_free_slot = find_free_slot(data_page, slot_index, hint_index);
  assert(found_free_slot);

  // Step 3: Calculate the record offset and write transaction reference
  Page::Offset rec_offset = get_record_offset(slot_index);
  data_page.write_integer_8(rec_offset, trx_ref, mtr);

  // Step 4: Write column data (validated at function entry)
  Page::Offset col_data_offset = rec_offset + TRX_REF_SIZE;
  data_page.write_string(col_data_offset, col_data.data, col_data.length, mtr);

  // Step 5: Mark record as allocated
  set_record_allocated(data_page, slot_index, mtr);

  // Step 6: Decrement the number of free records
  num_free_recs--;
  data_page.write_integer_2(NUM_FREE_RECS_OFF, num_free_recs, mtr);

  // Step 7: Update hint to next slot.
  hint_index = (slot_index + 1 >= m_max_num_recs) ? 0 : slot_index + 1;

  // Step 8: Encode column reference (page ref in lower 32 bits, slot in upper
  // 16 bits)
  Page::Ref page_ref = data_page.get_ref();
  col_ref = encode_column_ref(page_ref, slot_index);
}

bool DataPage::purge(Page &data_page, MtrCtx::Ref mtr, uint16_t slot_index,
                     Segment::TrxRef trx_ref, bool &purged) {
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE));

  // Initialize output parameter
  purged = false;

  // Step 1: Validate slot index is within bounds
  if (slot_index >= m_max_num_recs) {
    return true;
  }

  // Step 2: Validate record state
  auto [delete_marked, is_free] = get_record_status(data_page, slot_index);

  if (is_free) {
    // Might happen in case of a server crash and restart, where purge is
    // re-attempted.
    return false;
  }

  // Step 3: Verify that the transaction reference matches
  Page::Offset rec_offset = get_record_offset(slot_index);
  Segment::TrxRef stored_trx_ref = data_page.read_integer_8(rec_offset);

  if (stored_trx_ref != trx_ref) {
    // Transaction reference mismatch - do not purge
    // This is not an error, just means the record was modified by another
    // transaction and should not be purged by this transaction.
    return false;
  }

  // Step 4: Mark record as free (clears both FREE_BIT and DELETE_MARK_BIT)
  if (!delete_marked) {
    // Rollback path: purging a non-deleted slot.
  }
  set_record_free(data_page, slot_index, mtr);

  // Step 5: Increment the number of free records
  uint16_t num_free_recs = data_page.read_integer_2(NUM_FREE_RECS_OFF);
  num_free_recs++;
  data_page.write_integer_2(NUM_FREE_RECS_OFF, num_free_recs, mtr);

  // Step 6: Set output parameter to indicate successful purge
  purged = true;

  return false;
}

}  // namespace svector
