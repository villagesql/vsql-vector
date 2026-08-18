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

using vsql::preview_storage::Error;

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

void MultiColumnStore::fill_error(const char *info, char *msg, uint32_t len,
                                  bool local) {
  // A caller with nothing to report passes (nullptr, 0) -- see the note on
  // the declaration. Checked here rather than left to snprintf's tolerance
  // of a null buffer with a zero length, so that any error path added below
  // is safe by construction.
  if (msg == nullptr || len == 0)
    return;

  if (local) {
    snprintf(msg, len, "SVECTOR: %s", info);
  } else {
    auto detail = vsql::preview_storage::last_error();
    snprintf(msg, len, "SVECTOR: %s: %.*s", info,
             static_cast<int>(detail.size()), detail.data());
  }
}

void ColumnStore::fill_error(const char *info, char *msg, uint32_t len,
                             bool local) {
  MultiColumnStore::fill_error(info, msg, len, local);
}

bool MultiColumnStore::create(Space::Ref space_ref, Segment::TrxRef trx_ref,
                              const std::vector<Storage_spec> &storages,
                              uint8_t num_segments, char *error_msg,
                              uint32_t error_msg_len) {
  assert(!storages.empty());
  size_t num_storages = storages.size();
  assert(m_stores.empty());
  m_stores.resize(num_storages);

  ColumnStore &primary = m_stores[0];

  assert(num_storages >= (size_t)num_segments);
  Page::Ref root_page_ref;

  if (Segment::create(space_ref, num_segments, trx_ref, root_page_ref) !=
      Error::SUCCESS) {
    fill_error("create: failed to create segment", error_msg, error_msg_len,
               false);
    return true;
  }

  auto num_root_pages = static_cast<uint8_t>(num_storages);
  primary.init(space_ref, root_page_ref, storages[0].col_len, num_segments,
               num_root_pages, storages[0].metadata);
  // The primary store owns the segment page and uses segment 0.
  primary.m_primary_root_page_ref = root_page_ref;
  primary.m_segment_index = 0;

  // Additional stores do not have root pages yet. Their root pages will be
  // allocated and assigned on first use.
  for (size_t i = 1; i < storages.size(); ++i) {
    m_stores[i].init(space_ref, Page::INVALID_REF, storages[i].col_len,
                     /*num_segments=*/0, /*num_root_pages=*/1,
                     storages[i].metadata);
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

    bool err = primary.m_root.format(root_page, mtr, FORMAT_VERSION,
                                     primary.m_metadata);
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

bool MultiColumnStore::drop(Segment::TrxRef trx_ref, char *error_msg,
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

bool MultiColumnStore::init_root_page(uint8_t seg_idx, uint8_t root_idx,
                                      char *error_msg, uint32_t error_msg_len) {
  assert(root_idx > 0 && root_idx < m_stores.size());

  ColumnStore &primary = m_stores[0];
  ColumnStore &target = m_stores[root_idx];
  Space::Ref space_ref = primary.m_space_ref;

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();

  Page primary_root;
  if (primary_root.load(space_ref, primary.m_root_page_ref,
                        Page::Latch::EXCLUSIVE, mtr) != Error::SUCCESS) {
    fill_error("init_root_page: failed to load primary root page", error_msg,
               error_msg_len, false);
    mtr_ctx.commit();
    return true;
  }

  Segment::Ref seg_head = Segment::get_header(primary_root, seg_idx);

  Page new_page;
  if (new_page.load_new(seg_head, mtr) != Error::SUCCESS) {
    fill_error("init_root_page: failed to allocate page", error_msg,
               error_msg_len, false);
    mtr_ctx.commit();
    return true;
  }

  if (target.m_root.format(new_page, mtr, FORMAT_VERSION, target.m_metadata)) {
    fill_error("init_root_page: failed to format root page", error_msg,
               error_msg_len, true);
    mtr_ctx.commit();
    return true;
  }

  Page::Ref new_ref = new_page.get_ref();
  Page::Offset ref_off =
      primary.m_root.other_root_pages_off() +
      static_cast<Page::Offset>(root_idx - 1) * RootPage::ROOT_PAGE_REF_LEN;
  primary_root.write_integer_4(ref_off, new_ref, mtr);

  mtr_ctx.commit();

  target.m_root_page_ref = new_ref;
  // Record where this store's segment lives so insert() can allocate data pages
  // from the correct segment on the primary root page (segments exist only
  // there, not on this store's own root page).
  target.m_primary_root_page_ref = primary.m_root_page_ref;
  target.m_segment_index = seg_idx;
  return false;
}

bool MultiColumnStore::load(Column::StorageRef storage_ref,
                            const std::vector<Storage_spec> &specs,
                            char *error_msg, uint32_t error_msg_len) {
  m_ref = storage_ref;
  Space::Ref space_ref;
  Page::Ref root_page_ref;
  decode_ref(space_ref, root_page_ref);

  // Bootstraps a RootPage from a latched page and extracts its metadata.
  // Returns the bootstrapped RootPage so the caller can use it to compute
  // further offsets before the MTR is committed.
  struct PageInfo {
    uint8_t num_segs;
    uint8_t num_other_root_pages;
    uint16_t col_len;
    std::string metadata;
    RootPage rp;
  };
  auto read_page_info = [](const Page &page) -> PageInfo {
    RootPage rp;
    // Stage 1: set N only (enables storage_metadata_len_off()).
    uint8_t ns = page.read_integer_1(Page::HEADER_SIZE);
    rp.set_layout(ns, 0, 1);
    // Stage 2: read metadata length (enables num_other_root_pages_off()).
    uint8_t ml = page.read_integer_1(rp.storage_metadata_len_off());
    rp.set_layout(ns, ml, 1);
    // Stage 3: read K and finalize layout.
    uint8_t nrp = page.read_integer_1(rp.num_other_root_pages_off());
    rp.set_layout(ns, ml, nrp);
    return {ns, nrp, page.read_integer_2(rp.column_size_off()),
            rp.read_metadata(page), std::move(rp)};
  };

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();
  Page primary_root;
  if (primary_root.load(space_ref, root_page_ref, Page::Latch::SHARED, mtr) !=
      Error::SUCCESS) {
    fill_error("load: failed to load primary root page", error_msg,
               error_msg_len, false);
    mtr_ctx.commit();
    return true;
  }
  auto info0 = read_page_info(primary_root);
  uint8_t num_root_pages = info0.num_other_root_pages + 1;

  // Collect all K other root page refs before releasing the latch.
  std::vector<Page::Ref> other_refs(
      static_cast<size_t>(info0.num_other_root_pages));

  for (uint8_t i = 0; i < info0.num_other_root_pages; ++i) {
    Page::Offset off =
        info0.rp.other_root_pages_off() +
        static_cast<Page::Offset>(i) * RootPage::ROOT_PAGE_REF_LEN;
    other_refs[i] = primary_root.read_integer_4(off);
  }
  mtr_ctx.commit();

  // Single resize to avoid triggering the assert(false) move constructor.
  assert(m_stores.empty());
  m_stores.resize(num_root_pages);
  m_stores[0].init(space_ref, root_page_ref, info0.col_len, info0.num_segs,
                   num_root_pages, info0.metadata);
  // The primary store owns the segment page and uses segment 0 -- same as
  // create(). insert() consults these to source its segment header, so they
  // must be restored on reload or the first insert after reopen reads a
  // segment from an invalid page (m_primary_root_page_ref stays INVALID_REF).
  m_stores[0].m_primary_root_page_ref = root_page_ref;
  m_stores[0].m_segment_index = 0;

  assert(num_root_pages == specs.size());

  for (uint8_t i = 1; i < num_root_pages; ++i) {
    Page::Ref ref = other_refs[i - 1];
    if (ref == Page::INVALID_REF) {
      m_stores[i].init(space_ref, Page::INVALID_REF, specs[i].col_len,
                       /*num_segments=*/0, /*num_root_pages=*/1,
                       specs[i].metadata);
      continue;
    }

    MtrCtx sec_mtr_ctx;
    auto sec_mtr = sec_mtr_ctx.start();
    Page sec_root;
    if (sec_root.load(space_ref, ref, Page::Latch::SHARED, sec_mtr) !=
        Error::SUCCESS) {
      fill_error("load: failed to load secondary root page", error_msg,
                 error_msg_len, false);
      sec_mtr_ctx.commit();
      return true;
    }
    auto info = read_page_info(sec_root);
    sec_mtr_ctx.commit();

    assert(info.num_segs == 0);
    assert(info.num_other_root_pages == 0);

    m_stores[i].init(space_ref, ref, info.col_len, info.num_segs,
                     info.num_other_root_pages + 1, info.metadata);
    // A secondary store's segment lives on the primary root page; record that
    // here. Its segment index is not persisted in the root page (it is owned by
    // the layer that laid out the segments, e.g. the HNSW IndexStore's
    // level->segment mapping), so the owner restores m_segment_index when it
    // re-drives init_root_page() on reload. Until then this store must not be
    // inserted into.
    m_stores[i].m_primary_root_page_ref = root_page_ref;
  }

  encode_ref(space_ref, root_page_ref);
  return false;
}

bool MultiColumnStore::get_root_index(MtrCtx::Ref mctx, Column::Ref col_ref,
                                      uint8_t &root_idx, char *error_msg,
                                      uint32_t error_msg_len) {
  // Step 1: Decode column reference to get the owning data page
  Page::Ref data_page_ref;
  uint16_t slot_index;
  DataPage::decode_column_ref(col_ref, data_page_ref, slot_index);

  // Step 2: Load the data page with a SHARED latch; we only need to read its
  // root page ref.
  Space::Ref space_ref = m_stores[0].m_space_ref;
  Page data_page;
  if (data_page.load(space_ref, data_page_ref, Page::Latch::SHARED, mctx) !=
      Error::SUCCESS) {
    fill_error("get_root_index: failed to load data page", error_msg,
               error_msg_len, false);
    return true;
  }

  // Step 3: Read the root page ref this data page belongs to, and match it
  // against each store's root page ref.
  Page::Ref root_page_ref = m_stores[0].m_data.get_root_page_ref(data_page);
  for (size_t i = 0; i < m_stores.size(); ++i) {
    if (m_stores[i].m_root_page_ref == root_page_ref) {
      root_idx = static_cast<uint8_t>(i);
      return false;
    }
  }

  fill_error("get_root_index: no store matches root page ref", error_msg,
             error_msg_len, true);
  return true;
}

void ColumnStore::init(Space::Ref space_ref, Page::Ref root_page_ref,
                       uint16_t col_len, uint8_t num_segments,
                       uint8_t num_root_pages, std::string_view metadata) {
  m_space_ref = space_ref;
  m_root_page_ref = root_page_ref;
  m_metadata = metadata;
  m_root.init(space_ref, col_len, num_segments, num_root_pages,
              static_cast<uint8_t>(metadata.size()));
  m_data.init(space_ref, col_len);
}

bool ColumnStore::insert(MtrCtx::Ref mctx, Segment::TrxRef trx_ref,
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

  // Step 1: Use the provided mtr context
  auto mtr = static_cast<MtrCtx::Ref>(mctx);

  // Step 2: Load and latch the root page (SHARED latch for optimistic insert)
  Page root_page;
  if (root_page.load(m_space_ref, m_root_page_ref, Page::Latch::SHARED, mtr) !=
      Error::SUCCESS) {
    fill_error("insert: failed to load root page", error_msg, error_msg_len,
               false);
    return true;
  }

  // Step 3: Try optimistic page selection
  Page::Ref data_page_ref;
  uint16_t cur_free_slots = 0;
  m_root.page_select(root_page, m_space_ref, data_page_ref, cur_free_slots);

  // Step 3a: Check if concurrency exceeds available slots and we can grow
  uint32_t concurrency =
      m_insert_concurrency_counter.load(std::memory_order_relaxed);
  bool need_grow = (concurrency > cur_free_slots) &&
                   (cur_free_slots < m_root.get_max_free_slots());

  // Step 4: Optimistic path - check if we got a valid page
  Page data_page;
  bool need_pessimistic = need_grow;

  if (!need_pessimistic && data_page_ref != Page::INVALID_REF) {
    // Step 4a: Load data page with X latch
    if (data_page.load(m_space_ref, data_page_ref, Page::Latch::EXCLUSIVE,
                       mtr) != Error::SUCCESS) {
      fill_error("insert: failed to load data page", error_msg, error_msg_len,
                 false);
      return true;
    }

    // Step 4b: Check if data page has its last free slot
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

  // Step 5: Pessimistic path if needed
  if (need_pessimistic) {
    // Release S latch on root page
    if (root_page.release(mtr) != Error::SUCCESS) {
      fill_error("insert: failed to release root page", error_msg,
                 error_msg_len, false);
      return true;
    }

    // Acquire X latch on root page
    if (root_page.load(m_space_ref, m_root_page_ref, Page::Latch::EXCLUSIVE,
                       mtr) != Error::SUCCESS) {
      fill_error("insert: failed to load root page", error_msg, error_msg_len,
                 false);
      return true;
    }

    // Step 5a: Grow free slots if needed
    if (need_grow) {
      m_root.grow_free_slots(root_page, mtr);
    }

    // Re-read the cached slot under X latch. Between releasing S and acquiring
    // X, another thread may have allocated a page in this slot or removed it.
    // We must re-check to get the current state.
    m_root.page_select(root_page, m_space_ref, data_page_ref, cur_free_slots);

    if (data_page_ref == Page::INVALID_REF) {
      // Segments live only on the segment-owning ("primary") root page. This
      // store allocates data pages from its assigned segment there. For the
      // primary store the segment page is its own root (already X-latched as
      // root_page); any other store loads the primary root to reach its
      // segment header.
      //
      // LATCH ORDERING: this is currently the ONLY place that holds latches on
      // two root pages at once. The order is: this store's own root page FIRST
      // (root_page, latched above), THEN the owning/primary root page. Any future
      // code that takes both must follow this same order to avoid deadlock.
      //
      // The owning root is taken EXCLUSIVE only because Segment::get_header()
      // asserts an EXCLUSIVE latch (storage_api.h). We do not modify the owning
      // root page here -- we just read its segment header to allocate from that
      // segment -- so SHARED_EXCLUSIVE (which still serializes segment allocation)
      // is the intended latch.
      // TODO(villagesql-indexing): switch to Page::Latch::SHARED_EXCLUSIVE once
      // the get_header() EXCLUSIVE assertion is relaxed in the storage API.
      //
      // The latch is released immediately after load_new(), since it is not
      // needed for the remainder of the insert.
      Segment::Ref seg_head;
      Page primary_root_holder;
      Page *seg_root = &root_page;
      const bool own_owning_root = (m_primary_root_page_ref != m_root_page_ref);
      if (own_owning_root) {
        if (primary_root_holder.load(m_space_ref, m_primary_root_page_ref,
                                     Page::Latch::EXCLUSIVE,
                                     mtr) != Error::SUCCESS) {
          fill_error("insert: failed to load primary root page for segment",
                     error_msg, error_msg_len, false);
          return true;
        }
        seg_root = &primary_root_holder;
      }
      seg_head = Segment::get_header(*seg_root, m_segment_index);
      if (data_page.load_new(seg_head, mtr) != Error::SUCCESS) {
        fill_error("insert: failed to allocate new data page", error_msg,
                   error_msg_len, false);
        return true;
      }

      // The owning root latch is no longer needed once the page is allocated;
      // release it immediately rather than holding it for the rest of the insert.
      if (own_owning_root &&
          primary_root_holder.release(mtr) != Error::SUCCESS) {
        fill_error("insert: failed to release primary root page", error_msg,
                   error_msg_len, false);
        return true;
      }

      m_data.format(data_page, mtr, MultiColumnStore::FORMAT_VERSION,
                    m_root_page_ref);

      if (m_root.add_data_page(root_page, data_page, m_space_ref, mtr)) {
        fill_error("insert: failed to register new data page", error_msg,
                   error_msg_len, true);
        return true;
      }

      // Skip adding to free list if the page is already at its last slot.
      // This handles large vector dimensions (e.g. SVECTOR(3072)) where a
      // single vector occupies the whole page; the page will be full after
      // this insert and must never appear in the free list.
      if (!m_data.has_last_free_slot(data_page) &&
          m_root.add_free_page(root_page, data_page, &m_data, m_space_ref,
                               RootPage::s_last_slot_info.slot_number, mtr)) {
        fill_error("insert: failed to register new data page", error_msg,
                   error_msg_len, true);
        return true;
      }
    } else {
      if (data_page.load(m_space_ref, data_page_ref, Page::Latch::EXCLUSIVE,
                         mtr) != Error::SUCCESS) {
        fill_error("insert: failed to load data page", error_msg, error_msg_len,
                   false);
        return true;
      }

      if (m_data.has_last_free_slot(data_page)) {
        if (m_root.remove_free_page(root_page, data_page, &m_data, m_space_ref,
                                    mtr)) {
          fill_error("insert: failed to remove full page from free list",
                     error_msg, error_msg_len, true);
          return true;
        }
      }
    }
  } else {
    // Step 6: Release root page latch. We haven't modified root page.
    if (root_page.release(mtr) != Error::SUCCESS) {
      fill_error("insert: failed to release root page", error_msg,
                 error_msg_len, false);
      return true;
    }
  }

  // Step 7: Insert into data page (hint will be updated)
  assert(data_page.is_loaded());
  m_data.insert(data_page, mtr, trx_ref, col_data, col_ref);

  return false;
}

bool ColumnStore::fetch(MtrCtx::Ref mctx, Column::Ref col_ref, bool for_update,
                        Column::Data &col_data, Column::Data &rowid_prefix,
                        Segment::TrxRef &trx_ref, bool &delete_marked,
                        char *error_msg, uint32_t error_msg_len) {
  // Step 1: Decode column reference to get page and slot
  Page::Ref data_page_ref;
  uint16_t slot_index;
  DataPage::decode_column_ref(col_ref, data_page_ref, slot_index);

  // Step 2: Use the provided mtr context
  auto mtr = static_cast<MtrCtx::Ref>(mctx);

  // Step 3: Load data page with S latch, or X if the caller intends to
  // update the record.
  Page data_page;
  auto latch = for_update ? Page::Latch::EXCLUSIVE : Page::Latch::SHARED;
  if (data_page.load(m_space_ref, data_page_ref, latch, mtr) !=
      Error::SUCCESS) {
    fill_error("fetch: failed to load data page", error_msg, error_msg_len,
               false);
    return true;
  }

  // Step 4: Get record status (delete_marked and is_free)
  bool is_free = true;
  std::tie(delete_marked, is_free) =
      m_data.get_record_status(data_page, slot_index);

  if (is_free) {
    char info[64];
    snprintf(info, sizeof(info), "fetch: slot %u is free", slot_index);
    fill_error(info, error_msg, error_msg_len, true);
    return true;
  }

  // Step 5: Read transaction reference from record
  Page::Offset rec_offset = m_data.get_record_offset(slot_index);
  trx_ref = data_page.read_integer_8(rec_offset);

  // Step 6: Set column data pointer to the column data in the page
  Page::Offset col_data_offset = rec_offset + DataPage::TRX_REF_SIZE;
  col_data.data = data_page.get_data() + col_data_offset;
  col_data.length = m_root.get_column_size();

  // Step 7: We don't store rowid_prefix for SVECTOR
  rowid_prefix.data = nullptr;
  rowid_prefix.length = 0;

  return false;
}

bool ColumnStore::update(MtrCtx::Ref mctx, Column::Ref col_ref,
                         const Column::Data col_data,
                         std::span<const uint16_t> chunk_ids,
                         uint16_t chunk_size, char *error_msg,
                         uint32_t error_msg_len) {
  if (chunk_ids.empty()) {
    return false;
  }

  // Step 1: Check the caller supplied arguments. All the conditions below are
  // caller bugs: assert in debug builds, and in release builds raise an error
  // instead of writing anything. This must be done outside the write loop:
  // page writes under an mtr cannot be undone, so failing part way through the
  // loop would leave the record half updated.
  assert(col_data.data != nullptr);
  assert(chunk_size > 0);
  assert(col_data.length == m_root.get_column_size());
  if (col_data.data == nullptr || chunk_size == 0 ||
      col_data.length != m_root.get_column_size()) {
    char info[96];
    snprintf(info, sizeof(info),
             "update: bad arguments: data=%p, chunk_size=%u, length=%u, "
             "column_size=%u",
             static_cast<const void *>(col_data.data), chunk_size,
             col_data.length, m_root.get_column_size());
    fill_error(info, error_msg, error_msg_len, true);
    return true;
  }

  // Step 2: Validate all the chunk indexes, for the same reason: they must be
  // in strictly increasing order and within the record, and both conditions
  // have to be known before the first write. Both are caller bugs too.
  for (size_t i = 0; i < chunk_ids.size(); ++i) {
    uint16_t chunk_index = chunk_ids[i];

    assert(i == 0 || chunk_index > chunk_ids[i - 1]);
    if (i > 0 && chunk_index <= chunk_ids[i - 1]) {
      char info[96];
      snprintf(info, sizeof(info),
               "update: chunk indexes not in strictly increasing order: "
               "%u follows %u",
               chunk_index, chunk_ids[i - 1]);
      fill_error(info, error_msg, error_msg_len, true);
      return true;
    }

    // The column data length is already known to match the record width, so
    // this only guards against an out of range chunk index.
    uint64_t offset = static_cast<uint64_t>(chunk_index) * chunk_size;
    assert(offset + chunk_size <= col_data.length);
    if (offset + chunk_size > col_data.length) {
      char info[96];
      snprintf(info, sizeof(info),
               "update: chunk %u out of bounds: end=%llu, length=%u",
               chunk_index,
               static_cast<unsigned long long>(offset + chunk_size),
               col_data.length);
      fill_error(info, error_msg, error_msg_len, true);
      return true;
    }
  }

  // Step 3: Decode column reference to get page and slot
  Page::Ref data_page_ref;
  uint16_t slot_index;
  DataPage::decode_column_ref(col_ref, data_page_ref, slot_index);

  // Step 4: Use the provided mtr context
  auto mtr = static_cast<MtrCtx::Ref>(mctx);

  // Step 5: Load data page with X latch
  Page data_page;
  if (data_page.load(m_space_ref, data_page_ref, Page::Latch::EXCLUSIVE, mtr) !=
      Error::SUCCESS) {
    fill_error("update: failed to load data page", error_msg, error_msg_len,
               false);
    return true;
  }

  // Step 6: Refuse to write into a free slot. The delete mark is deliberately
  // ignored: this API is non-transactional and leaves record visibility to the
  // caller, so a delete marked record is still updated in place.
  bool is_free = true;
  std::tie(std::ignore, is_free) =
      m_data.get_record_status(data_page, slot_index);
  if (is_free) {
    char info[64];
    snprintf(info, sizeof(info), "update: slot %u is free", slot_index);
    fill_error(info, error_msg, error_msg_len, true);
    return true;
  }

  // Step 7: Write the requested chunks. Everything is validated by now, so
  // this loop cannot fail: either all the chunks are written or none of them
  // is, and the caller never sees a partially updated record. The indexes are
  // in strictly increasing order, so a run of consecutive indexes covers one
  // contiguous byte range and is written with a single call, keeping the
  // number of redo log records down.
  Page::Offset col_offset =
      m_data.get_record_offset(slot_index) + DataPage::TRX_REF_SIZE;
  for (size_t i = 0; i < chunk_ids.size();) {
    // Extend the run as long as the next index follows the current one.
    size_t end = i + 1;
    while (end < chunk_ids.size() && chunk_ids[end] == chunk_ids[end - 1] + 1) {
      ++end;
    }

    // Step 2 has established that the run lies within the column size, so the
    // cast to the narrower page offset cannot truncate.
    uint32_t offset = static_cast<uint32_t>(chunk_ids[i]) * chunk_size;
    size_t length = (end - i) * chunk_size;
    data_page.write_string(col_offset + static_cast<Page::Offset>(offset),
                           col_data.data + offset, length, mtr);
    i = end;
  }

  return false;
}

bool ColumnStore::mark_delete(MtrCtx::Ref mctx, Segment::TrxRef trx_ref,
                              Column::Ref col_ref, bool delete_mark,
                              char *error_msg, uint32_t error_msg_len) {
  // Step 1: Decode column reference to get page and slot
  Page::Ref data_page_ref;
  uint16_t slot_index;
  DataPage::decode_column_ref(col_ref, data_page_ref, slot_index);

  // Step 2: Use the provided mtr context
  auto mtr = static_cast<MtrCtx::Ref>(mctx);

  // Step 3: Load data page with X latch
  Page data_page;
  if (data_page.load(m_space_ref, data_page_ref, Page::Latch::EXCLUSIVE, mtr) !=
      Error::SUCCESS) {
    fill_error("mark_delete: failed to load data page", error_msg,
               error_msg_len, false);
    return true;
  }

  // Step 4: Mark or unmark the record as deleted based on delete_mark parameter
  Page::Offset rec_offset = m_data.get_record_offset(slot_index);
  Segment::TrxRef old_trx_ref = data_page.read_integer_8(rec_offset);
  bool trx_id_match = (old_trx_ref == trx_ref);

  if (delete_mark) {
    m_data.set_record_delete(data_page, slot_index, mtr);
  } else {
    m_data.set_record_undelete(data_page, slot_index, trx_id_match, mtr);
  }

  // Step 5: Update the transaction reference
  if (!trx_id_match) {
    data_page.write_integer_8(rec_offset, trx_ref, mtr);
  }

  return false;
}

bool ColumnStore::purge(MtrCtx::Ref mctx, Segment::TrxRef trx_ref,
                        Column::Ref col_ref, char *error_msg,
                        uint32_t error_msg_len) {
  // Step 1: Decode column reference to get page and slot
  Page::Ref data_page_ref;
  uint16_t slot_index;
  DataPage::decode_column_ref(col_ref, data_page_ref, slot_index);

  // Step 2: Use the provided mtr context
  auto mtr = static_cast<MtrCtx::Ref>(mctx);

  // Step 3: Load data page with X latch
  Page data_page;
  if (data_page.load(m_space_ref, data_page_ref, Page::Latch::EXCLUSIVE, mtr) !=
      Error::SUCCESS) {
    fill_error("purge: failed to load data page", error_msg, error_msg_len,
               false);
    return true;
  }

  // Step 4: Check if page will need to be added to free list after purge
  bool need_pessimistic = m_data.needs_add_to_free_list(data_page, true);
  Page root_page;

  // Step 5: If page needs to be added to free list, follow pessimistic path
  if (need_pessimistic) {
    // Release data page X latch
    if (data_page.release(mtr) != Error::SUCCESS) {
      fill_error("purge: failed to release data page", error_msg, error_msg_len,
                 false);
      return true;
    }

    // Load root page with X latch
    if (root_page.load(m_space_ref, m_root_page_ref, Page::Latch::EXCLUSIVE,
                       mtr) != Error::SUCCESS) {
      fill_error("purge: failed to load root page", error_msg, error_msg_len,
                 false);
      return true;
    }

    // Re-load data page with X latch
    if (data_page.load(m_space_ref, data_page_ref, Page::Latch::EXCLUSIVE,
                       mtr) != Error::SUCCESS) {
      fill_error("purge: failed to reload data page", error_msg, error_msg_len,
                 false);
      return true;
    }
  }

  // Step 6: Purge the record
  bool purged = false;
  if (m_data.purge(data_page, mtr, slot_index, trx_ref, purged)) {
    char info[64];
    snprintf(info, sizeof(info), "purge: failed on page %u slot %u",
             data_page_ref, slot_index);
    fill_error(info, error_msg, error_msg_len, true);
    return true;
  }

  // Step 7: Add page to free list, after Re-check
  if (need_pessimistic && purged && m_data.needs_add_to_free_list(data_page)) {
    // Get a free slot from root page to add this page
    uint16_t slot_number = m_root.get_free_slot(root_page);
    if (m_root.add_free_page(root_page, data_page, &m_data, m_space_ref,
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

bool ColumnStore::update_metadata(MtrCtx::Ref mctx, std::string_view metadata,
                                  char *error_msg, uint32_t error_msg_len) {
  auto mtr = static_cast<MtrCtx::Ref>(mctx);

  Page root_page;
  if (root_page.load(m_space_ref, m_root_page_ref, Page::Latch::EXCLUSIVE,
                     mtr) != Error::SUCCESS) {
    fill_error("update_metadata: failed to load root page", error_msg,
               error_msg_len, false);
    return true;
  }

  if (m_root.update_header(root_page, mtr, metadata)) {
    char info[96];
    snprintf(info, sizeof(info),
             "update_metadata: size mismatch: got=%zu, expected=%u",
             metadata.size(), m_root.get_metadata_len());
    fill_error(info, error_msg, error_msg_len, true);
    return true;
  }

  m_metadata = metadata;
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
  constexpr uint8_t NUM_SEGMENTS = 1;

  auto *col_store = storage->user();
  bool err = col_store->create(space, trx_ref, {{store_len, "SVECTOR"}},
                               NUM_SEGMENTS, error_msg, error_msg_len);
  if (!err) storage->set_ref(col_store->m_ref);
  return err;
}

bool ColumnStorage::drop(Ctx *storage, Segment::TrxRef trx_ref, char *error_msg,
                         uint32_t error_msg_len) {
  return storage->user()->drop(trx_ref, error_msg, error_msg_len);
}

bool ColumnStorage::load(Ctx *storage, Column::StorageRef storage_ref,
                         char *error_msg, uint32_t error_msg_len) {
  return storage->user()->load(storage_ref, {{0, ""}}, error_msg,
                               error_msg_len);
}

bool ColumnStorage::insert(Ctx *storage, MtrCtx::Ref mctx,
                           Segment::TrxRef trx_ref, Column::Data col_data,
                           Column::Data rowid_prefix, Column::Ref *col_ref,
                           char *error_msg, uint32_t error_msg_len) {
  // Ignore rowid prefix. Currently we don't support fetching the record back
  // from column reference.
  (void)rowid_prefix;
  return storage->user()->m_stores[0].insert(mctx, trx_ref, col_data, *col_ref,
                                             error_msg, error_msg_len);
}

bool ColumnStorage::select(Ctx *storage, MtrCtx::Ref mctx, Column::Ref col_ref,
                           Column::Data *col_data, Column::Data *rowid_prefix,
                           Segment::TrxRef *trx_ref, bool *delete_marked,
                           char *error_msg, uint32_t error_msg_len) {
  return storage->user()->m_stores[0].fetch(
      mctx, col_ref, /*for_update=*/false, *col_data, *rowid_prefix, *trx_ref,
      *delete_marked, error_msg, error_msg_len);
}

bool ColumnStorage::mark_delete(Ctx *storage, MtrCtx::Ref mctx,
                                Segment::TrxRef trx_ref, Column::Ref col_ref,
                                bool delete_mark, char *error_msg,
                                uint32_t error_msg_len) {
  return storage->user()->m_stores[0].mark_delete(
      mctx, trx_ref, col_ref, delete_mark, error_msg, error_msg_len);
}

bool ColumnStorage::purge(Ctx *storage, MtrCtx::Ref mctx,
                          Segment::TrxRef trx_ref, Column::Ref col_ref,
                          char *error_msg, uint32_t error_msg_len) {
  return storage->user()->m_stores[0].purge(mctx, trx_ref, col_ref, error_msg,
                                            error_msg_len);
}

}  // namespace svector
