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

#ifndef VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_ROOT_PAGE_H
#define VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_ROOT_PAGE_H

#include <cstdint>
#include <string>
#include <string_view>

#include <villagesql/preview/storage_api.h>

namespace svector {

using vsql::preview_storage::MtrCtx;
using vsql::preview_storage::Page;
using vsql::preview_storage::Segment;
using vsql::preview_storage::Space;

// Forward declarations
struct DataPage;

// Page type identifiers for SVECTOR column storage
enum class ColumnPageType : uint8_t {
  ROOT_PAGE = 1,
  DATA_PAGE = 2,
};

// Root Page structure for SVECTOR column storage
struct RootPage {
  // Thread-local struct to remember the last chosen slot for optimization
  struct LastSlotInfo {
    Space::Ref space_ref;
    Page::Ref page_ref;
    uint16_t slot_number;
    uint16_t next_free_rec_hint;  // Hint for next free slot in data page

    // Constructor to initialize with invalid values
    LastSlotInfo()
        : space_ref(0),
          page_ref(Page::INVALID_REF),
          slot_number(0),
          next_free_rec_hint(0) {}
  };

  // Thread-local storage for last slot information
  static thread_local LastSlotInfo s_last_slot_info;

public:
  // Root Page format - Version-1
  // [Page Header] [N] [Segment Header] [Version] [Type]
  // |-----38-----|-1-|-----N x 10-----|----1----|--1--|
  //
  // [Metadata Len(M)] [Storage Metadata]
  // |--------1-------|--------M---------|
  //
  // [Number of root pages (K)] [Other root page REFs]
  // |------------1------------|-------4 x K---------|
  //
  // [Column Size] [Data Page Head] [Data Page Tail]
  // |-----2------|-------4--------|-------4-------|
  //
  // [Total Data Pages] [Total Free Pages]
  // |--------4--------|--------4--------|
  //
  // [Free slot array size (M)] [Free slot current size]
  // |------------2------------|----------2------------|
  //
  // [Free Slot Array: Max 2k slots] [Left Over] [Page Trailer]
  // |------------M * 4-------------|-----L-----|-----8-------|

  // Fixed-size field lengths. All offsets are dynamic based on number of
  // segments (N) and Number of other root pages(K).
  static constexpr Page::Offset VERSION_LEN = 1;
  static constexpr Page::Offset PAGE_TYPE_LEN = 1;
  static constexpr Page::Offset STORAGE_METADATA_LEN_SIZE = 1;
  static constexpr Page::Offset NUM_ROOT_PAGES_LEN = 1;
  static constexpr Page::Offset ROOT_PAGE_REF_LEN = 4;
  static constexpr Page::Offset COLUMN_SIZE_LEN = 2;
  static constexpr Page::Offset ALL_SLOT_HEAD_LEN = 4;
  static constexpr Page::Offset ALL_SLOT_TAIL_LEN = 4;
  static constexpr Page::Offset TOTAL_DATA_PAGES_LEN = 4;
  static constexpr Page::Offset TOTAL_FREE_PAGES_LEN = 4;
  static constexpr Page::Offset FREE_SLOT_ARRAY_MAX_SIZE_LEN = 2;
  static constexpr Page::Offset FREE_SLOT_ARRAY_CUR_SIZE_LEN = 2;
  // Each free slot holds the free (partially full) page reference.
  static constexpr Page::Offset FREE_SLOT_LEN = 4;

  inline Page::Offset version_off() const {
    return Page::HEADER_SIZE + Segment::NUM_SEGMENTS_SIZE +
           static_cast<Page::Offset>(Segment::HEADER_SIZE) * m_num_segments;
  }
  inline Page::Offset page_type_off() const {
    return version_off() + VERSION_LEN;
  }
  inline Page::Offset storage_metadata_len_off() const {
    return page_type_off() + PAGE_TYPE_LEN;
  }
  inline Page::Offset storage_metadata_off() const {
    return storage_metadata_len_off() + STORAGE_METADATA_LEN_SIZE;
  }
  inline Page::Offset num_other_root_pages_off() const {
    return storage_metadata_off() + m_storage_metadata_len;
  }
  inline Page::Offset other_root_pages_off() const {
    return num_other_root_pages_off() + NUM_ROOT_PAGES_LEN;
  }

  inline Page::Offset column_size_off() const {
    return other_root_pages_off() +
           ROOT_PAGE_REF_LEN * (m_num_other_root_pages);
  }
  inline Page::Offset all_slot_head_off() const {
    return column_size_off() + COLUMN_SIZE_LEN;
  }
  inline Page::Offset all_slot_tail_off() const {
    return all_slot_head_off() + ALL_SLOT_HEAD_LEN;
  }
  inline Page::Offset total_data_pages_off() const {
    return all_slot_tail_off() + ALL_SLOT_TAIL_LEN;
  }
  inline Page::Offset total_free_pages_off() const {
    return total_data_pages_off() + TOTAL_DATA_PAGES_LEN;
  }
  inline Page::Offset free_slot_array_max_size_off() const {
    return total_free_pages_off() + TOTAL_FREE_PAGES_LEN;
  }
  inline Page::Offset free_slot_array_cur_size_off() const {
    return free_slot_array_max_size_off() + FREE_SLOT_ARRAY_MAX_SIZE_LEN;
  }
  inline Page::Offset free_slot_array_off() const {
    return free_slot_array_cur_size_off() + FREE_SLOT_ARRAY_CUR_SIZE_LEN;
  }

  // Start with a single free slot and increase dynamically.
  static constexpr uint16_t NUM_FREE_SLOTS_INITIAL = 1;

  // NULL reference for free page list. Use same value as Page::INVALID_REF.
  static constexpr uint32_t NULL_FREE_PAGE_REF = Page::INVALID_REF;

  // Set N, M, and K without full initialization. Used during load() to
  // bootstrap offset functions before col_len is known. Must be followed by
  // init(). Call in three stages during load:
  //   1. set_layout(N, 0, 1)  — enables storage_metadata_len_off()
  //   2. set_layout(N, M, 1)  — enables num_other_root_pages_off()
  //   3. set_layout(N, M, K)  — final layout
  void set_layout(uint8_t num_segments, uint8_t storage_metadata_len,
                  uint8_t num_other_root_pages) {
    m_num_segments = num_segments;
    m_storage_metadata_len = storage_metadata_len;
    m_num_other_root_pages = num_other_root_pages;
  }

  // Initialize root page parameters. num_segments (N), num_root_pages (K), and
  // storage_metadata_len are fixed at creation time and must not change.
  void init(Space::Ref space_ref, uint16_t col_len, uint8_t num_segments,
            uint8_t num_root_pages, uint8_t storage_metadata_len);

  // Format root page.
  bool format(Page &root_page, MtrCtx::Ref mtr, uint8_t format_version,
              std::string_view metadata);

  // Read the storage metadata from a formatted root page.
  std::string read_metadata(const Page &root_page) const;

  // Select a data page for insert with latch on root. This function examines
  // the root page to find a suitable data page.
  // Sets data_page_ref to INVALID_REF if no page found.
  // Sets cur_free_slots to the current number of free slots in the array.
  void page_select(Page &root_page, Space::Ref space_ref,
                   Page::Ref &data_page_ref, uint16_t &cur_free_slots);

  // Get a free slot from the root page to add a free page. First search
  // for an empty slot. If none, then return a random free slot to apend
  // to. The root page must be S or X latched.
  uint16_t get_free_slot(Page &root_page);

  // Grow the free slots array to accommodate higher concurrency.
  // The root page must be X latched. Increase by 2x of current size
  // limiting the increment to MAX_FREE_SLOT_STEP. The total slot array
  // size is limited to m_max_free_slots.
  void grow_free_slots(Page &root_page, MtrCtx::Ref mtr);

  // Maximum step size for growing free slots array.
  static constexpr uint16_t MAX_FREE_SLOT_STEP = 16;

  // Add a new data page to the free slot list in the root page.
  // The root page must be X latched. The data page must be formatted.
  // The slot_number parameter specifies which slot to use.
  // Returns false on success, true on error.
  bool add_free_page(Page &root_page, Page &data_page, DataPage *data_page_info,
                     Space::Ref space_ref, uint16_t slot_number,
                     MtrCtx::Ref mtr);

  // Remove a data page from the free slot array in the root page.
  // The root page must be X latched. Called when a page becomes full or
  // completely free and we decide to return the page to Segment.
  // Returns false on success, true on error.
  bool remove_free_page(Page &root_page, Page &data_page,
                        DataPage *data_page_info, Space::Ref space_ref,
                        MtrCtx::Ref mtr);

  // Add a new data page to the data page list in the root page.
  // The root page must be X latched.
  // Returns false on success, true on error.
  bool add_data_page(Page &root_page, Page &data_page, Space::Ref space_ref,
                     MtrCtx::Ref mtr);

  // Remove a data page from the data page list in the root page.
  // The root page must be X latched. Called when a page becomes completely free
  // and we decide to return the page to Segment.
  // Returns false on success, true on error.
  bool remove_data_page(Page &root_page, Page &data_page, Space::Ref space_ref,
                        MtrCtx::Ref mtr);

  // Getters for member variables
  uint8_t get_num_segments() const { return m_num_segments; }
  uint16_t get_column_size() const { return m_column_size; }
  uint16_t get_max_free_slots() const { return m_max_free_slots; }

 private:
  // Randomly select a slot index between 0 and max_size-1 (inclusive).
  // Returns 0 if max_size is 0.
  static uint16_t get_random_slot(uint16_t max_size);

  // Validate and retrieve cached slot number from thread-local storage.
  // Returns true if cached slot is valid (space_ref, page_ref match and
  // slot_number < max_slots), false otherwise.
  static bool get_cached_slot_number(Space::Ref space_ref, Page::Ref page_ref,
                                     uint16_t max_slots, uint16_t &slot_number);

  // Set by init() and fixed for the lifetime of this RootPage. Must be
  // initialized before any offset function is called.
  uint8_t m_num_segments = 0;
  uint8_t m_num_other_root_pages = 0;
  uint8_t m_storage_metadata_len = 0;

  // Maximum number of free slots for concurrent insert. Might need to reduce
  // based on page size.
  uint16_t m_max_free_slots = 2048;

  // Initialized by init() and not changed later.
  uint16_t m_column_size = 0;
};

}  // namespace svector

#endif  // VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_ROOT_PAGE_H
