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

#include "root_page.h"

#include <cassert>
#include <random>

#include "data_page.h"

namespace svector {

using vsql::preview_storage::Error;

// Define the thread_local static member
thread_local RootPage::LastSlotInfo RootPage::s_last_slot_info;

void RootPage::init(Space::Ref space_ref, uint16_t col_len) {
  m_column_size = col_len;

  // Calculate maximum free slots based on available page space
  uint32_t page_size = Page::get_size(space_ref);
  uint32_t available_space =
      page_size - FREE_SLOT_ARRAY_OFF - Page::TRAILER_SIZE;
  uint16_t max_slots = static_cast<uint16_t>(available_space / FREE_SLOT_LEN);

  // Cap at the configured maximum
  if (max_slots > m_max_free_slots) {
    max_slots = m_max_free_slots;
  }

  m_max_free_slots = max_slots;
}

bool RootPage::format(Page &root_page, MtrCtx::Ref mtr,
                      uint8_t format_version) {
  assert(root_page.is_loaded(Page::Latch::EXCLUSIVE));

  if (m_column_size == 0 || m_max_free_slots == 0) {
    return true;
  }

  // Write version
  root_page.write_integer_1(VERSION_OFF, format_version, mtr);

  // Write page type
  root_page.write_integer_1(
      PAGE_TYPE_OFF, static_cast<uint8_t>(ColumnPageType::ROOT_PAGE), mtr);

  // Write creator name
  root_page.write_string(CREATOR_NAME_OFF, CREATOR, CREATOR_NAME_LEN, mtr);

  // Write column size
  root_page.write_integer_2(COLUMN_SIZE_OFF, m_column_size, mtr);

  // Write data page head (initially invalid - no pages yet)
  root_page.write_integer_4(ALL_SLOT_HEAD_OFF, Page::INVALID_REF, mtr);

  // Write data page tail (initially invalid - no pages yet)
  root_page.write_integer_4(ALL_SLOT_TAIL_OFF, Page::INVALID_REF, mtr);

  // Total data pages counter is 0. Page data is zero filled by default.
  assert(0 == root_page.read_integer_4(TOTAL_DATA_PAGES_OFF));

  // Total free pages counter is 0.
  assert(0 == root_page.read_integer_4(TOTAL_FREE_PAGES_OFF));

  // Write free slot array max size
  root_page.write_integer_2(FREE_SLOT_ARRAY_MAX_SIZE_OFF, m_max_free_slots,
                            mtr);

  // Write free slot array current size
  root_page.write_integer_2(FREE_SLOT_ARRAY_CUR_SIZE_OFF,
                            NUM_FREE_SLOTS_INITIAL, mtr);

  // Initialize NUM_FREE_SLOTS_INITIAL slots to NULL_FREE_PAGE_REF
  for (uint16_t i = 0; i < NUM_FREE_SLOTS_INITIAL; i++) {
    Page::Offset slot_offset = FREE_SLOT_ARRAY_OFF + (i * FREE_SLOT_LEN);
    root_page.write_integer_4(slot_offset, NULL_FREE_PAGE_REF, mtr);
  }

  return false;
}

uint16_t RootPage::get_random_slot(uint16_t max_size) {
  if (max_size == 0) {
    return 0;
  }

  // Use thread_local random generator for thread-safety and performance
  thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());

  // Generate random number in range [0, max_size-1]
  std::uniform_int_distribution<uint16_t> dist(0, max_size - 1);
  return dist(gen);
}

bool RootPage::get_cached_slot_number(Space::Ref space_ref, Page::Ref page_ref,
                                      uint16_t max_slots,
                                      uint16_t &slot_number) {
  // Check if space_ref matches
  if (s_last_slot_info.space_ref != space_ref) {
    return false;
  }

  // Check if page_ref (root page) matches
  if (s_last_slot_info.page_ref != page_ref) {
    return false;
  }

  // Check if cached slot_number is within valid range
  if (s_last_slot_info.slot_number >= max_slots) {
    return false;
  }

  // All checks passed - return the cached slot number
  slot_number = s_last_slot_info.slot_number;
  return true;
}

void RootPage::grow_free_slots(Page &root_page, MtrCtx::Ref mtr) {
  assert(root_page.is_loaded(Page::Latch::EXCLUSIVE));

  // Step 1: Read current free slots count
  uint16_t cur_free_slots =
      root_page.read_integer_2(FREE_SLOT_ARRAY_CUR_SIZE_OFF);

  // Step 2: We should only be called if we can grow
  assert(cur_free_slots < m_max_free_slots);

  // Step 3: Calculate new size
  // Increase by min(cur_free_slots, MAX_FREE_SLOT_STEP) to double the size
  uint16_t growth = std::min(cur_free_slots, MAX_FREE_SLOT_STEP);
  uint16_t new_free_slots = cur_free_slots + growth;

  // Cap at maximum
  if (new_free_slots > m_max_free_slots) {
    new_free_slots = m_max_free_slots;
  }

  // Step 4: Initialize new slots to NULL_FREE_PAGE_REF
  for (uint16_t i = cur_free_slots; i < new_free_slots; i++) {
    Page::Offset slot_offset = FREE_SLOT_ARRAY_OFF + (i * FREE_SLOT_LEN);
    root_page.write_integer_4(slot_offset, NULL_FREE_PAGE_REF, mtr);
  }

  // Step 5: Update current free slots count
  root_page.write_integer_2(FREE_SLOT_ARRAY_CUR_SIZE_OFF, new_free_slots, mtr);
}

void RootPage::page_select(Page &root_page, Space::Ref space_ref,
                           Page::Ref &data_page_ref, uint16_t &cur_free_slots) {
  assert(root_page.is_loaded(Page::Latch::EXCLUSIVE) ||
         root_page.is_loaded(Page::Latch::SHARED));

  // Step 1: Extract current size of free slot array from root page
  cur_free_slots = root_page.read_integer_2(FREE_SLOT_ARRAY_CUR_SIZE_OFF);

  // Free slot array size should always be at least 1
  assert(cur_free_slots > 0);

  // Step 2: Try to get cached slot number, or select random slot
  uint16_t slot_number;
  Page::Ref root_page_ref = root_page.get_ref();

  if (!get_cached_slot_number(space_ref, root_page_ref, cur_free_slots,
                              slot_number)) {
    // No valid cached slot - select a random slot
    slot_number = get_random_slot(cur_free_slots);

    // Update cached slot number for next insert
    s_last_slot_info.space_ref = space_ref;
    s_last_slot_info.page_ref = root_page_ref;
    s_last_slot_info.slot_number = slot_number;

    // We have switched to a different page. The hint points to the beginning.
    s_last_slot_info.next_free_rec_hint = 0;
  }

  // Step 3: Read data page reference from the slot
  Page::Offset slot_offset =
      FREE_SLOT_ARRAY_OFF + (slot_number * FREE_SLOT_LEN);
  data_page_ref = root_page.read_integer_4(slot_offset);

  if (data_page_ref == NULL_FREE_PAGE_REF) {
    // Slot is empty or invalid - need pessimistic path to allocate new page
    data_page_ref = Page::INVALID_REF;
  }
}

bool RootPage::add_free_page(Page &root_page, Page &data_page,
                             DataPage *data_page_info, Space::Ref space_ref,
                             uint16_t slot_number, MtrCtx::Ref mtr) {
  assert(root_page.is_loaded(Page::Latch::EXCLUSIVE));
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE));

  Page::Ref data_page_ref = data_page.get_ref();
  Page::Ref root_page_ref = root_page.get_ref();

  // Step 1: Validate slot_number
  assert(slot_number < root_page.read_integer_2(FREE_SLOT_ARRAY_CUR_SIZE_OFF));

  Page::Offset slot_offset =
      FREE_SLOT_ARRAY_OFF + (slot_number * FREE_SLOT_LEN);
  Page::Ref slot_page_ref = root_page.read_integer_4(slot_offset);

  // Step 2: Setup links for the current slot page.
  if (slot_page_ref != NULL_FREE_PAGE_REF) {
    Page slot_page;
    Error error =
        slot_page.load(space_ref, slot_page_ref, Page::Latch::EXCLUSIVE, mtr);
    if (error != Error::SUCCESS) {
      // It is fine to return error before making any page changes.
      return true;
    }
    assert(slot_number == data_page_info->get_free_slot_number(slot_page));
    data_page_info->set_free_links(slot_page, &data_page_ref, nullptr, mtr);
  }

  // Step 3: Add to the head of the free slot.
  root_page.write_integer_4(slot_offset, data_page_ref, mtr);

  // Step 4: Setup links for the added free page.
  data_page_info->set_free_links(data_page, &root_page_ref, &slot_page_ref,
                                 mtr);
  data_page_info->set_free_slot_number(data_page, slot_number, mtr);

  // Step 5: Increment total free pages counter
  uint32_t total_free_pages = root_page.read_integer_4(TOTAL_FREE_PAGES_OFF);
  root_page.write_integer_4(TOTAL_FREE_PAGES_OFF, total_free_pages + 1, mtr);

  return false;
}

bool RootPage::remove_free_page(Page &root_page, Page &data_page,
                                DataPage *data_page_info, Space::Ref space_ref,
                                MtrCtx::Ref mtr) {
  assert(root_page.is_loaded(Page::Latch::EXCLUSIVE));
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE));

  Page::Ref root_page_ref = root_page.get_ref();

  // Step 1: Get free slot number from data page.
  uint16_t slot_number = data_page_info->get_free_slot_number(data_page);
  assert(slot_number != DataPage::INVALID_SLOT);
  assert(slot_number < root_page.read_integer_2(FREE_SLOT_ARRAY_CUR_SIZE_OFF));

  Page::Offset slot_offset =
      FREE_SLOT_ARRAY_OFF + (slot_number * FREE_SLOT_LEN);

#ifndef NDEBUG
  Page::Ref data_page_ref = data_page.get_ref();
  Page::Ref slot_page_ref = root_page.read_integer_4(slot_offset);
#endif  // NDEBUG
  assert(slot_page_ref != NULL_FREE_PAGE_REF);

  Page::Ref prev_ref, next_ref;
  data_page_info->get_free_links(data_page, &prev_ref, &next_ref);

  Page prev_page;
  Page next_page;

  // 1. Load previous page and return error before making any changes.
  if (prev_ref != root_page_ref) {
    Error error =
        prev_page.load(space_ref, prev_ref, Page::Latch::EXCLUSIVE, mtr);
    if (error != Error::SUCCESS) {
      return true;
    }
  }

  // 2. Load next page and return error before making any changes.
  if (next_ref != NULL_FREE_PAGE_REF) {
    Error error =
        next_page.load(space_ref, next_ref, Page::Latch::EXCLUSIVE, mtr);
    if (error != Error::SUCCESS) {
      return true;
    }
  }

  // 3. Set the links for the previous page
  if (prev_ref == root_page_ref) {
    // Removing the head of the free page list.
    assert(slot_page_ref == data_page_ref);
    root_page.write_integer_4(slot_offset, next_ref, mtr);

  } else {
    assert(slot_number == data_page_info->get_free_slot_number(prev_page));
    data_page_info->set_free_links(prev_page, nullptr, &next_ref, mtr);
  }

  // 4. Set the links for the next page.
  if (next_ref != NULL_FREE_PAGE_REF) {
    assert(slot_number == data_page_info->get_free_slot_number(next_page));
    data_page_info->set_free_links(next_page, &prev_ref, nullptr, mtr);
  }

  // Step 5: Reset free links for the data page being removed.
  prev_ref = NULL_FREE_PAGE_REF;
  data_page_info->set_free_links(data_page, &prev_ref, &prev_ref, mtr);

  // Step 6: Decrement total free pages counter
  uint32_t total_free_pages = root_page.read_integer_4(TOTAL_FREE_PAGES_OFF);
  assert(total_free_pages > 0);
  root_page.write_integer_4(TOTAL_FREE_PAGES_OFF, total_free_pages - 1, mtr);

  return false;
}

bool RootPage::add_data_page(Page &root_page, Page &data_page,
                             Space::Ref space_ref, MtrCtx::Ref mtr) {
  assert(root_page.is_loaded(Page::Latch::EXCLUSIVE));
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE));

  // Insert data page at the head of the list
  Page::Ref data_page_ref = data_page.get_ref();
  Page::Ref invalid_ref = Page::INVALID_REF;

  Page::Ref head_ref = root_page.read_integer_4(ALL_SLOT_HEAD_OFF);
#ifndef NDEBUG
  Page::Ref tail_ref = root_page.read_integer_4(ALL_SLOT_TAIL_OFF);
#endif  // NDEBUG

  // 1. Set Current head page links.
  if (head_ref == Page::INVALID_REF) {
    // Current head is empty. It is the first page being inserted.
    assert(tail_ref == Page::INVALID_REF);
    root_page.write_integer_4(ALL_SLOT_TAIL_OFF, data_page_ref, mtr);
  } else {
    Page head_page;
    Error error =
        head_page.load(space_ref, head_ref, Page::Latch::EXCLUSIVE, mtr);
    if (error != Error::SUCCESS) {
      // It is fine to return error before making any page changes.
      return true;
    }
    head_page.write_prev_link(data_page_ref, mtr);
  }

  // 2. Set Data page links : The new head.
  data_page.write_links(invalid_ref, head_ref, mtr);

  // 3. Set root page head and tails links.
  root_page.write_integer_4(ALL_SLOT_HEAD_OFF, data_page_ref, mtr);

  // 4. Increment total data pages counter
  uint32_t total_data_pages = root_page.read_integer_4(TOTAL_DATA_PAGES_OFF);
  root_page.write_integer_4(TOTAL_DATA_PAGES_OFF, total_data_pages + 1, mtr);

  return false;
}

bool RootPage::remove_data_page(Page &root_page, Page &data_page,
                                Space::Ref space_ref, MtrCtx::Ref mtr) {
  assert(root_page.is_loaded(Page::Latch::EXCLUSIVE));
  assert(data_page.is_loaded(Page::Latch::EXCLUSIVE));

  // Get the previous and next page references from the data page
  Page::Ref prev_ref = Page::INVALID_REF;
  Page::Ref next_ref = Page::INVALID_REF;
  data_page.read_links(prev_ref, next_ref);

#ifndef NDEBUG
  Page::Ref data_page_ref = data_page.get_ref();
  Page::Ref head_ref = root_page.read_integer_4(ALL_SLOT_HEAD_OFF);
  Page::Ref tail_ref = root_page.read_integer_4(ALL_SLOT_TAIL_OFF);
#endif  // NDEBUG

  Page prev_page;
  Page next_page;

  // 1. Load previous page and return error before making any changes.
  if (prev_ref != Page::INVALID_REF) {
    Error error =
        prev_page.load(space_ref, prev_ref, Page::Latch::EXCLUSIVE, mtr);
    if (error != Error::SUCCESS) {
      return true;
    }
  }

  // 2. Load next page and return error before making any changes.
  if (next_ref != Page::INVALID_REF) {
    Error error =
        next_page.load(space_ref, next_ref, Page::Latch::EXCLUSIVE, mtr);
    if (error != Error::SUCCESS) {
      return true;
    }
  }

  // 3. Update the previous page's next link (or head if this is the first page)
  if (prev_ref == Page::INVALID_REF) {
    // This is the head page, update root's head pointer
    assert(head_ref == data_page_ref);
    root_page.write_integer_4(ALL_SLOT_HEAD_OFF, next_ref, mtr);

  } else {
    // Update the previous page's next link
    prev_page.write_next_link(next_ref, mtr);
  }

  // 4. Update the next page's previous link (or tail if this is the last page)
  if (next_ref == Page::INVALID_REF) {
    // This is the tail page, update root's tail pointer
    assert(tail_ref == data_page_ref);
    root_page.write_integer_4(ALL_SLOT_TAIL_OFF, prev_ref, mtr);

  } else {
    // Update the next page's previous link
    next_page.write_prev_link(prev_ref, mtr);
  }

  // 5. Clear the data page's links
  data_page.write_links(Page::INVALID_REF, Page::INVALID_REF, mtr);

  // 6. Decrement total data pages counter
  uint32_t total_data_pages = root_page.read_integer_4(TOTAL_DATA_PAGES_OFF);
  assert(total_data_pages > 0);
  root_page.write_integer_4(TOTAL_DATA_PAGES_OFF, total_data_pages - 1, mtr);

  return false;
}

uint16_t RootPage::get_free_slot(Page &root_page) {
  assert(root_page.is_loaded(Page::Latch::EXCLUSIVE) ||
         root_page.is_loaded(Page::Latch::SHARED));

  // Step 1: Get current number of free slots
  uint16_t cur_free_slots =
      root_page.read_integer_2(FREE_SLOT_ARRAY_CUR_SIZE_OFF);
  assert(cur_free_slots > 0);

  // Step 2: Search for an empty slot (NULL_FREE_PAGE_REF)
  for (uint16_t slot_idx = 0; slot_idx < cur_free_slots; slot_idx++) {
    Page::Offset slot_offset = FREE_SLOT_ARRAY_OFF + (slot_idx * FREE_SLOT_LEN);
    Page::Ref slot_page_ref = root_page.read_integer_4(slot_offset);

    if (slot_page_ref == NULL_FREE_PAGE_REF) {
      // Found an empty slot
      return slot_idx;
    }
  }

  // Step 3: No empty slot found, return a random slot to append to
  return get_random_slot(cur_free_slots);
}

}  // namespace svector
