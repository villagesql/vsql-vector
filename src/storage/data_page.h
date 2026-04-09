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

#ifndef VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_DATA_PAGE_H
#define VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_DATA_PAGE_H

#include <cassert>
#include <cstdint>
#include <utility>

#include <villagesql/storage_api.h>

namespace svector {

using villagesql::storage::Column;
using villagesql::storage::MtrCtx;
using villagesql::storage::Page;
using villagesql::storage::Segment;
using villagesql::storage::Space;

// Data Page structure for SVECTOR column storage
struct DataPage {
  // Data Page format: Version-1.
  // [Page Header] [Version] [Type] [Free Slot Number] [Previous free Page]
  // |-----38-----|----1---|---1---|--------2---------|----------4--------|
  //
  // [Next free Page] [Number of records (M)] [Number of free records]
  // |-------4-------|-----------2-----------|-----------2-----------|
  //
  // [Free bitmap : Delete Mark and Free bit]
  // |-----------1 + (M * 2) / 8------------|
  //
  // [M Records : Transaction Ref, Column Data] [Left Over] [Page Trailer]
  // |-------M * (8 + Fixed column size)-------|-----L-----|-----8-------|
  //
  // Left over space L is less than one record size.

 public:
  // Transaction reference size in bytes stored with each record
  static constexpr Page::Offset TRX_REF_SIZE = 8;

  // Minimum free percentage threshold for adding page to free list
  static constexpr uint8_t MIN_FREE_PERCENT_FOR_FREE_LIST = 20;

  // Bitmap bit usage (2 bits per record):
  // - Bit 0: Delete mark (0 = active , 1 = deleted)
  // - Bit 1: Free (0 = free slot, 1 = occupied)
  static constexpr uint8_t DELETE_MARK_BIT = 0;
  static constexpr uint8_t FREE_BIT = 1;
  static constexpr uint8_t BITS_PER_RECORD = 2;

  static constexpr Page::Offset VERSION_OFF = Page::HEADER_SIZE;
  static constexpr Page::Offset VERSION_LEN = 1;

  // Page Type
  static constexpr Page::Offset PAGE_TYPE_OFF = VERSION_OFF + VERSION_LEN;
  static constexpr Page::Offset PAGE_TYPE_LEN = 1;

  // Free slot number in root page (INVALID_SLOT if not in free list)
  static constexpr Page::Offset FREE_SLOT_NUMBER_OFF =
      PAGE_TYPE_OFF + PAGE_TYPE_LEN;
  static constexpr Page::Offset FREE_SLOT_NUMBER_LEN = 2;

  static constexpr Page::Offset PREV_FREE_PAGE_OFF =
      FREE_SLOT_NUMBER_OFF + FREE_SLOT_NUMBER_LEN;
  static constexpr Page::Offset PREV_FREE_PAGE_LEN = 4;

  static constexpr Page::Offset NEXT_FREE_PAGE_OFF =
      PREV_FREE_PAGE_OFF + PREV_FREE_PAGE_LEN;
  static constexpr Page::Offset NEXT_FREE_PAGE_LEN = 4;

  // Maximum number of records page can hold : Total capacity
  static constexpr Page::Offset MAX_NUM_RECS_OFF =
      NEXT_FREE_PAGE_OFF + NEXT_FREE_PAGE_LEN;
  static constexpr Page::Offset MAX_NUM_RECS_LEN = 2;

  // Current number of free records : Insert capacity
  static constexpr Page::Offset NUM_FREE_RECS_OFF =
      MAX_NUM_RECS_OFF + MAX_NUM_RECS_LEN;
  static constexpr Page::Offset NUM_FREE_RECS_LEN = 2;

  static constexpr Page::Offset FREE_BITMAP_OFF =
      NUM_FREE_RECS_OFF + NUM_FREE_RECS_LEN;

  // Compile-time validation of offset calculations
  // Sentinel value for invalid slot number
  static constexpr uint16_t INVALID_SLOT = 0xFFFF;

  static_assert(VERSION_OFF == Page::HEADER_SIZE,
                "VERSION_OFF must start after page header");
  static_assert(PAGE_TYPE_OFF == VERSION_OFF + VERSION_LEN,
                "PAGE_TYPE_OFF must follow VERSION");
  static_assert(FREE_SLOT_NUMBER_OFF == PAGE_TYPE_OFF + PAGE_TYPE_LEN,
                "FREE_SLOT_NUMBER_OFF must follow PAGE_TYPE");
  static_assert(PREV_FREE_PAGE_OFF ==
                    FREE_SLOT_NUMBER_OFF + FREE_SLOT_NUMBER_LEN,
                "PREV_FREE_PAGE_OFF must follow FREE_SLOT_NUMBER");
  static_assert(NEXT_FREE_PAGE_OFF == PREV_FREE_PAGE_OFF + PREV_FREE_PAGE_LEN,
                "NEXT_FREE_PAGE_OFF must follow PREV_FREE_PAGE");
  static_assert(MAX_NUM_RECS_OFF == NEXT_FREE_PAGE_OFF + NEXT_FREE_PAGE_LEN,
                "MAX_NUM_RECS_OFF must follow NEXT_FREE_PAGE");
  static_assert(NUM_FREE_RECS_OFF == MAX_NUM_RECS_OFF + MAX_NUM_RECS_LEN,
                "NUM_FREE_RECS_OFF must follow MAX_NUM_RECS");
  static_assert(FREE_BITMAP_OFF == NUM_FREE_RECS_OFF + NUM_FREE_RECS_LEN,
                "FREE_BITMAP_OFF must follow NUM_FREE_RECS");

  // Initialize data page parameters (calculate capacity).
  void init(Space::Ref space_ref, uint16_t col_len);

  // Format data page.
  void format(Page &data_page, MtrCtx::Ref mtr, uint8_t format_version);

  // Get record offset by index
  Page::Offset get_record_offset(uint16_t rec_index) const {
    assert(rec_index < m_max_num_recs);
    return m_first_rec_offset + (rec_index * m_rec_size);
  }

  // Check if page has capacity
  bool has_capacity() const { return m_max_num_recs > 0; }

  // Encode column reference from page reference and slot index
  static Column::Ref encode_column_ref(Page::Ref page_ref, uint16_t slot_index);

  // Decode column reference to get page reference and slot index
  static void decode_column_ref(Column::Ref col_ref, Page::Ref &page_ref,
                                uint16_t &slot_index);

  // Set the free slot number in root page.
  void set_free_slot_number(Page &data_page, uint16_t slot_number,
                            MtrCtx::Ref mtr) const;

  // Get the free slot number in root page.
  uint16_t get_free_slot_number(Page &data_page) const;

  // Set the free page links (PREV_FREE_PAGE_OFF and NEXT_FREE_PAGE_OFF).
  // Pass nullptr to leave a link unchanged.
  void set_free_links(Page &data_page, Page::Ref *prev_free_page,
                      Page::Ref *next_free_page, MtrCtx::Ref mtr) const;

  // Get the free page links (PREV_FREE_PAGE_OFF and NEXT_FREE_PAGE_OFF).
  // Reads the links from the data page into the provided pointers.
  void get_free_links(Page &data_page, Page::Ref *prev_free_page,
                      Page::Ref *next_free_page) const;

  // Check if the data page has at least one free slot for insert.
  // The page must be X latched (will be asserted).
  // Returns true if there is space for a new record, false otherwise.
  bool has_last_free_slot(Page &data_page) const;

  // Check if the page needs to be added to the free list.
  // Returns true if:
  // 1. The page is not in the free list (free_slot_number == INVALID_SLOT)
  // 2. The free percentage is greater than MIN_FREE_PERCENT_FOR_FREE_LIST (20%)
  // The page must be latched (S or X).
  bool needs_add_to_free_list(Page &data_page) const;

  // Get the delete mark and free status for a record.
  // Returns pair<delete_marked, is_free>
  // The page must be latched (S or X).
  std::pair<bool, bool> get_record_status(Page &data_page,
                                          uint16_t slot_index) const;

  // Set the delete mark bit for a record (mark as deleted).
  // The page must be X latched.
  void set_record_delete(Page &data_page, uint16_t slot_index,
                         MtrCtx::Ref mtr) const;

  // Clear the delete mark bit for a record (unmark as deleted).
  // The page must be X latched.
  void set_record_undelete(Page &data_page, uint16_t slot_index, bool trx_match,
                           MtrCtx::Ref mtr) const;

  // Insert column data into the data page.
  void insert(Page &data_page, MtrCtx::Ref mtr, Segment::TrxRef trx_ref,
              Column::Data col_data, Column::Ref &col_ref);

  // Purge a record from the data page (physically remove it).
  // The page must be X latched.
  // Only purges if the record's stored trx_ref matches the input trx_ref.
  // Sets purged to true if the record was purged, false otherwise.
  // Returns false on success, true on error.
  bool purge(Page &data_page, MtrCtx::Ref mtr, uint16_t slot_index,
             Segment::TrxRef trx_ref, bool &purged);

 private:
  // Set the free bit for a record (mark as free/unallocated).
  // The page must be X latched.
  void set_record_free(Page &data_page, uint16_t slot_index,
                       MtrCtx::Ref mtr) const;

  // Mark a record as allocated (set FREE_BIT=1, DELETE_MARK_BIT=0).
  // The page must be X latched.
  void set_record_allocated(Page &data_page, uint16_t slot_index,
                            MtrCtx::Ref mtr) const;

  // Find the first free slot in the data page by scanning the bitmap.
  // Returns true if a free slot is found, false otherwise.
  // The slot_index is set to the found slot index if successful.
  // The page must be latched (S or X).
  bool find_free_slot(Page &data_page, uint16_t &slot_index,
                      uint16_t hint_index) const;

  // Find free BIT within a single byte combined bitmap.
  bool free_bit_in_byte(uint8_t bitmap_byte, uint8_t &bitmap_bit) const;

  // Record size: transaction ref + column data
  Page::Offset m_rec_size = 0;

  // Size in bytes for free bitmap - 2 BITs per record
  Page::Offset m_free_bitmap_size = 0;

  // Maximum number of records a data page can accommodate.
  Page::Offset m_max_num_recs = 0;

  // Offset of the first record in the page.
  Page::Offset m_first_rec_offset = 0;
};

}  // namespace svector

#endif  // VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_DATA_PAGE_H
