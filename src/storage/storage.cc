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

#include "storage.h"

#include <cassert>
#include <cstdio>
#include <limits>
#include <new>
#include <tuple>

namespace svector {

using villagesql::storage::Error;

// RAII guard to automatically track insert concurrency
class ConcurrencyGuard {
 public:
  explicit ConcurrencyGuard(std::atomic<uint32_t> &counter)
      : m_counter(counter) {
    m_counter.fetch_add(1, std::memory_order_relaxed);
  }

  ~ConcurrencyGuard() { m_counter.fetch_sub(1, std::memory_order_relaxed); }

  // Delete copy and move operations
  ConcurrencyGuard(const ConcurrencyGuard &) = delete;
  ConcurrencyGuard &operator=(const ConcurrencyGuard &) = delete;
  ConcurrencyGuard(ConcurrencyGuard &&) = delete;
  ConcurrencyGuard &operator=(ConcurrencyGuard &&) = delete;

 private:
  std::atomic<uint32_t> &m_counter;
};

void ColumnStorageContext::fill_error(const char *info, char *msg, uint32_t len,
                                      bool local) {
  if (local) {
    snprintf(msg, len, "SVECTOR: %s", info);
  } else {
    auto detail = villagesql::storage::last_error();
    snprintf(msg, len, "SVECTOR: %s: %.*s", info,
             static_cast<int>(detail.size()), detail.data());
  }
}

bool ColumnStorageContext::create(Space::Ref space_ref, Segment::TrxRef trx_ref,
                                  uint16_t col_len, char *error_msg,
                                  uint32_t error_msg_len) {
  m_root.init(space_ref, col_len);
  m_data.init(space_ref, col_len);

  Page::Ref root_page_ref;

  if (Segment::create(space_ref, RootPage::NUM_SEGMENTS, trx_ref,
                      root_page_ref) != Error::SUCCESS) {
    fill_error("create: failed to create segment", error_msg, error_msg_len,
               false);
    return true;
  }

  {
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();

    Page root_page;
    if (root_page.load(space_ref, root_page_ref, Page::Latch::EXCLUSIVE, mtr) !=
        Error::SUCCESS) {
      fill_error("create: failed to load root page", error_msg, error_msg_len,
                 false);
      mtr_ctx.commit();
      return true;
    }

    bool err = m_root.format(root_page, mtr, FORMAT_VERSION);
    mtr_ctx.commit();

    if (err) {
      fill_error("create: failed to format root page", error_msg, error_msg_len,
                 true);
      return true;
    }
  }

  encode_ref(space_ref, root_page_ref);
  return false;
}

bool ColumnStorageContext::drop(Segment::TrxRef trx_ref, char *error_msg,
                                uint32_t error_msg_len) {
  Space::Ref space_ref;
  Page::Ref page_ref;
  decode_ref(space_ref, page_ref);

  if (Segment::drop(space_ref, trx_ref, page_ref) != Error::SUCCESS) {
    fill_error("drop: failed to drop segment", error_msg, error_msg_len, false);
    return true;
  }
  return false;
}

bool ColumnStorageContext::insert(MtrCtx::Ref mctx, Segment::TrxRef trx_ref,
                                  Column::Data col_data, Column::Ref &col_ref,
                                  char *error_msg, uint32_t error_msg_len) {
  // Track concurrency: increment on entry, decrement on exit
  ConcurrencyGuard concurrency_guard(m_insert_concurrency_counter);

  // Validate input: column data must be valid
  if (col_data.data == nullptr) {
    fill_error("insert: NULL column data", error_msg, error_msg_len, true);
    return true;
  }
  if (col_data.length != m_root.get_column_size()) {
    char info[64];
    snprintf(info, sizeof(info),
             "insert: data size mismatch: size=%u, expected=%u",
             col_data.length, m_root.get_column_size());
    fill_error(info, error_msg, error_msg_len, true);
    return true;
  }

  // Step 1: Decode Space and Page reference (root page)
  Space::Ref space_ref;
  Page::Ref root_page_ref;
  decode_ref(space_ref, root_page_ref);

  // Step 2: Use the provided mtr context
  auto mtr = static_cast<MtrCtx::Ref>(mctx);

  // Step 3: Load and latch the root page (SHARED latch for optimistic insert)
  Page root_page;
  if (root_page.load(space_ref, root_page_ref, Page::Latch::SHARED, mtr) !=
      Error::SUCCESS) {
    fill_error("insert: failed to load root page", error_msg, error_msg_len,
               false);
    return true;
  }

  // Step 4: Try optimistic page selection
  Page::Ref data_page_ref;
  uint16_t cur_free_slots = 0;
  m_root.page_select(root_page, space_ref, data_page_ref, cur_free_slots);

  // Step 4a: Check if concurrency exceeds available slots and we can grow
  uint32_t concurrency =
      m_insert_concurrency_counter.load(std::memory_order_relaxed);
  bool need_grow = (concurrency > cur_free_slots) &&
                   (cur_free_slots < m_root.get_max_free_slots());

  // Step 5: Optimistic path - check if we got a valid page
  Page data_page;
  bool need_pessimistic = need_grow;

  if (!need_pessimistic && data_page_ref != Page::INVALID_REF) {
    // Step 5a: Load data page with X latch
    if (data_page.load(space_ref, data_page_ref, Page::Latch::EXCLUSIVE, mtr) !=
        Error::SUCCESS) {
      fill_error("insert: failed to load data page", error_msg, error_msg_len,
                 false);
      return true;
    }

    // Step 5b: Check if data page has its last free slot
    // If the page has its last free slot, it will become full after insert
    // and needs to be removed from the root page's free list. This requires
    // X latch on root, so we must fall back to pessimistic path.
    if (m_data.has_last_free_slot(data_page)) {
      if (data_page.release(mtr) != Error::SUCCESS) {
        fill_error("insert: failed to release data page", error_msg,
                   error_msg_len, false);
        return true;
      }
      need_pessimistic = true;
    }
  } else {
    need_pessimistic = true;
  }

  // Step 6: Pessimistic path if needed
  if (need_pessimistic) {
    // Release S latch on root page
    if (root_page.release(mtr) != Error::SUCCESS) {
      fill_error("insert: failed to release root page", error_msg,
                 error_msg_len, false);
      return true;
    }

    // Acquire X latch on root page
    if (root_page.load(space_ref, root_page_ref, Page::Latch::EXCLUSIVE, mtr) !=
        Error::SUCCESS) {
      fill_error("insert: failed to load root page", error_msg, error_msg_len,
                 false);
      return true;
    }

    // Step 6a: Grow free slots if needed
    if (need_grow) {
      m_root.grow_free_slots(root_page, mtr);
    }

    // Re-read the cached slot under X latch. Between releasing S and acquiring
    // X, another thread may have allocated a page in this slot or removed it.
    // We must re-check to get the current state.
    m_root.page_select(root_page, space_ref, data_page_ref, cur_free_slots);

    if (data_page_ref == Page::INVALID_REF) {
      Segment::Ref seg_head = Segment::get_header(root_page, 0);
      if (data_page.load_new(seg_head, mtr) != Error::SUCCESS) {
        fill_error("insert: failed to allocate new data page", error_msg,
                   error_msg_len, false);
        return true;
      }

      m_data.format(data_page, mtr, FORMAT_VERSION);

      if (m_root.add_data_page(root_page, data_page, space_ref, mtr) ||
          m_root.add_free_page(root_page, data_page, &m_data, space_ref,
                               RootPage::s_last_slot_info.slot_number, mtr)) {
        fill_error("insert: failed to register new data page", error_msg,
                   error_msg_len, true);
        return true;
      }
    } else {
      if (data_page.load(space_ref, data_page_ref, Page::Latch::EXCLUSIVE,
                         mtr) != Error::SUCCESS) {
        fill_error("insert: failed to load data page", error_msg, error_msg_len,
                   false);
        return true;
      }

      if (!m_data.has_last_free_slot(data_page)) {
        if (m_root.remove_free_page(root_page, data_page, &m_data, space_ref,
                                    mtr)) {
          fill_error("insert: failed to remove full page from free list",
                     error_msg, error_msg_len, true);
          return true;
        }
      }
    }
  } else {
    // Step 7: Release root page latch. We haven't modified root page.
    if (root_page.release(mtr) != Error::SUCCESS) {
      fill_error("insert: failed to release root page", error_msg,
                 error_msg_len, false);
      return true;
    }
  }

  // Step 8: Insert into data page (hint will be updated)
  assert(data_page.is_loaded());
  m_data.insert(data_page, mtr, trx_ref, col_data, col_ref);

  return false;
}

bool ColumnStorageContext::fetch(MtrCtx::Ref mctx, Column::Ref col_ref,
                                 Column::Data &col_data,
                                 Column::Data &rowid_prefix,
                                 Segment::TrxRef &trx_ref, bool &delete_marked,
                                 char *error_msg, uint32_t error_msg_len) {
  // Step 1: Decode column reference to get page and slot
  Page::Ref data_page_ref;
  uint16_t slot_index;
  DataPage::decode_column_ref(col_ref, data_page_ref, slot_index);

  // Step 2: Get space reference from storage context
  Space::Ref space_ref;
  Page::Ref root_page_ref;
  decode_ref(space_ref, root_page_ref);

  // Step 3: Use the provided mtr context
  auto mtr = static_cast<MtrCtx::Ref>(mctx);

  // Step 4: Load data page with S latch
  Page data_page;
  if (data_page.load(space_ref, data_page_ref, Page::Latch::SHARED, mtr) !=
      Error::SUCCESS) {
    fill_error("fetch: failed to load data page", error_msg, error_msg_len,
               false);
    return true;
  }

  // Step 5: Get record status (delete_marked and is_free)
  bool is_free = true;
  std::tie(delete_marked, is_free) =
      m_data.get_record_status(data_page, slot_index);

  if (is_free) {
    char info[64];
    snprintf(info, sizeof(info), "fetch: slot %u is free", slot_index);
    fill_error(info, error_msg, error_msg_len, true);
    return true;
  }

  // Step 7: Read transaction reference from record
  Page::Offset rec_offset = m_data.get_record_offset(slot_index);
  trx_ref = data_page.read_integer_8(rec_offset);

  // Step 8: Set column data pointer to the column data in the page
  Page::Offset col_data_offset = rec_offset + DataPage::TRX_REF_SIZE;
  col_data.data = data_page.get_data() + col_data_offset;
  col_data.length = m_root.get_column_size();

  // Step 9: We don't store rowid_prefix for SVECTOR
  rowid_prefix.data = nullptr;
  rowid_prefix.length = 0;

  return false;
}

bool ColumnStorageContext::mark_delete(MtrCtx::Ref mctx,
                                       Segment::TrxRef trx_ref,
                                       Column::Ref col_ref, bool delete_mark,
                                       char *error_msg,
                                       uint32_t error_msg_len) {
  // Step 1: Decode column reference to get page and slot
  Page::Ref data_page_ref;
  uint16_t slot_index;
  DataPage::decode_column_ref(col_ref, data_page_ref, slot_index);

  // Step 2: Get space reference from storage context
  Space::Ref space_ref;
  Page::Ref root_page_ref;
  decode_ref(space_ref, root_page_ref);

  // Step 3: Use the provided mtr context
  auto mtr = static_cast<MtrCtx::Ref>(mctx);

  // Step 4: Load data page with X latch
  Page data_page;
  if (data_page.load(space_ref, data_page_ref, Page::Latch::EXCLUSIVE, mtr) !=
      Error::SUCCESS) {
    fill_error("mark_delete: failed to load data page", error_msg,
               error_msg_len, false);
    return true;
  }

  // Step 5: Mark or unmark the record as deleted based on delete_mark parameter
  Page::Offset rec_offset = m_data.get_record_offset(slot_index);
  Segment::TrxRef old_trx_ref = data_page.read_integer_8(rec_offset);
  bool trx_id_match = (old_trx_ref == trx_ref);

  if (delete_mark) {
    m_data.set_record_delete(data_page, slot_index, mtr);
  } else {
    m_data.set_record_undelete(data_page, slot_index, trx_id_match, mtr);
  }

  // Step 6: Update the transaction reference
  if (!trx_id_match) {
    data_page.write_integer_8(rec_offset, trx_ref, mtr);
  }

  return false;
}

bool ColumnStorageContext::purge(MtrCtx::Ref mctx, Segment::TrxRef trx_ref,
                                 Column::Ref col_ref, char *error_msg,
                                 uint32_t error_msg_len) {
  // Step 1: Decode column reference to get page and slot
  Page::Ref data_page_ref;
  uint16_t slot_index;
  DataPage::decode_column_ref(col_ref, data_page_ref, slot_index);

  // Step 2: Get space reference from storage context
  Space::Ref space_ref;
  Page::Ref root_page_ref;
  decode_ref(space_ref, root_page_ref);

  // Step 3: Use the provided mtr context
  auto mtr = static_cast<MtrCtx::Ref>(mctx);

  // Step 4: Load data page with X latch
  Page data_page;
  if (data_page.load(space_ref, data_page_ref, Page::Latch::EXCLUSIVE, mtr) !=
      Error::SUCCESS) {
    fill_error("purge: failed to load data page", error_msg, error_msg_len,
               false);
    return true;
  }

  // Step 5: Check if page will need to be added to free list after purge
  bool need_pessimistic = m_data.needs_add_to_free_list(data_page);
  Page root_page;

  // Step 6: If page needs to be added to free list, follow pessimistic path
  if (need_pessimistic) {
    // Release data page X latch
    if (data_page.release(mtr) != Error::SUCCESS) {
      fill_error("purge: failed to release data page", error_msg, error_msg_len,
                 false);
      return true;
    }

    // Load root page with X latch
    if (root_page.load(space_ref, root_page_ref, Page::Latch::EXCLUSIVE, mtr) !=
        Error::SUCCESS) {
      fill_error("purge: failed to load root page", error_msg, error_msg_len,
                 false);
      return true;
    }

    // Re-load data page with X latch
    if (data_page.load(space_ref, data_page_ref, Page::Latch::EXCLUSIVE, mtr) !=
        Error::SUCCESS) {
      fill_error("purge: failed to reload data page", error_msg, error_msg_len,
                 false);
      return true;
    }
  }

  // Step 7: Purge the record
  bool purged = false;
  if (m_data.purge(data_page, mtr, slot_index, trx_ref, purged)) {
    char info[64];
    snprintf(info, sizeof(info), "purge: failed on page %u slot %u",
             data_page_ref, slot_index);
    fill_error(info, error_msg, error_msg_len, true);
    return true;
  }

  // Step 8: Add page to free list, after Re-check
  if (need_pessimistic && purged && m_data.needs_add_to_free_list(data_page)) {
    // Get a free slot from root page to add this page
    uint16_t slot_number = m_root.get_free_slot(root_page);
    if (m_root.add_free_page(root_page, data_page, &m_data, space_ref,
                             slot_number, mtr)) {
      char info[64];
      snprintf(info, sizeof(info), "purge: failed to add page %u to free list",
               data_page_ref);
      fill_error(info, error_msg, error_msg_len, true);
      return true;
    }
  }

  return false;
}

// ColumnStorage implementation: top-level entry points called via ABI wrappers
// in storage_builder.h. Each method retrieves the user context via
// storage->user() and delegates to its methods.

bool ColumnStorage::create(Ctx *storage, Space::Ref space,
                           Segment::TrxRef trx_ref, uint32_t col_len,
                           char *error_msg, uint32_t error_msg_len) {
  if (col_len >= Page::get_size(space) || col_len < sizeof(Column::Ref)) {
    snprintf(error_msg, error_msg_len,
             "SVECTOR: create: column length %u out of range for page size %u",
             col_len, Page::get_size(space));
    return true;
  }

  // Subtract the storage_ref prefix from the column length stored per record.
  uint16_t store_len = static_cast<uint16_t>(col_len - sizeof(Column::Ref));

  auto *col_store = storage->user();
  bool err =
      col_store->create(space, trx_ref, store_len, error_msg, error_msg_len);
  if (!err) storage->set_ref(col_store->m_ref);
  return err;
}

bool ColumnStorage::drop(Ctx *storage, Segment::TrxRef trx_ref, char *error_msg,
                         uint32_t error_msg_len) {
  return storage->user()->drop(trx_ref, error_msg, error_msg_len);
}

bool ColumnStorage::load(Ctx *storage, Column::StorageRef storage_ref,
                         char *error_msg, uint32_t error_msg_len) {
  auto *col_store = storage->user();
  col_store->m_ref = storage_ref;

  Space::Ref space_ref;
  Page::Ref root_page_ref;
  col_store->decode_ref(space_ref, root_page_ref);

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();

  Page root_page;
  if (root_page.load(space_ref, root_page_ref, Page::Latch::SHARED, mtr) !=
      Error::SUCCESS) {
    snprintf(error_msg, error_msg_len,
             "SVECTOR: load: failed to load root page %u", root_page_ref);
    mtr_ctx.commit();
    return true;
  }

  uint16_t col_len = root_page.read_integer_2(RootPage::COLUMN_SIZE_OFF);
  mtr_ctx.commit();

  col_store->m_root.init(space_ref, col_len);
  col_store->m_data.init(space_ref, col_len);

  return false;
}

bool ColumnStorage::insert(Ctx *storage, MtrCtx::Ref mctx,
                           Segment::TrxRef trx_ref, Column::Data col_data,
                           Column::Data rowid_prefix, Column::Ref *col_ref,
                           char *error_msg, uint32_t error_msg_len) {
  // Ignore rowid prefix. Currently we don't support fetching the record back
  // from column reference.
  (void)rowid_prefix;
  return storage->user()->insert(mctx, trx_ref, col_data, *col_ref, error_msg,
                                 error_msg_len);
}

bool ColumnStorage::select(Ctx *storage, MtrCtx::Ref mctx, Column::Ref col_ref,
                           Column::Data *col_data, Column::Data *rowid_prefix,
                           Segment::TrxRef *trx_ref, bool *delete_marked,
                           char *error_msg, uint32_t error_msg_len) {
  return storage->user()->fetch(mctx, col_ref, *col_data, *rowid_prefix,
                                *trx_ref, *delete_marked, error_msg,
                                error_msg_len);
}

bool ColumnStorage::mark_delete(Ctx *storage, MtrCtx::Ref mctx,
                                Segment::TrxRef trx_ref, Column::Ref col_ref,
                                bool delete_mark, char *error_msg,
                                uint32_t error_msg_len) {
  return storage->user()->mark_delete(mctx, trx_ref, col_ref, delete_mark,
                                      error_msg, error_msg_len);
}

bool ColumnStorage::purge(Ctx *storage, MtrCtx::Ref mctx,
                          Segment::TrxRef trx_ref, Column::Ref col_ref,
                          char *error_msg, uint32_t error_msg_len) {
  return storage->user()->purge(mctx, trx_ref, col_ref, error_msg,
                                error_msg_len);
}

}  // namespace svector
