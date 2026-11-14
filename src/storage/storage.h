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

#ifndef VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_H
#define VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_H

#include <atomic>
#include <cassert>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <villagesql/preview/storage_api.h>

#include "data_page.h"
#include "root_page.h"

namespace svector {

using vsql::preview_storage::Column;
using vsql::preview_storage::MtrCtx;
using vsql::preview_storage::Page;
using vsql::preview_storage::Segment;
using vsql::preview_storage::Space;

/*
 * Latching Protocol for Column Storage
 * ====================================
 *
 * All page latches are taken through mini-transactions (mtr) and are held
 * until the mtr commits. The rules below define the order in which latches may
 * be acquired so that concurrent operations cannot deadlock.
 *
 * Terminology note: "primary root page" (e.g. in init_root_page) is the first
 * root page of a multi-root column store. It is unrelated to the "Primary"
 * column store of an index level (see 3.2), which is paired with a "Secondary"
 * column store.
 *
 * 1. Across column stores
 * -----------------------
 * 1.1 A column store operation never acquires latches in more than one column
 *     store. Ordering across column stores is the caller's responsibility.
 *
 * 1.2 Exception: initializing a new column store. The primary root page and
 *     the newly allocated root page are latched at the same time. This is safe
 *     because the new root page is not yet visible to any other thread.
 *
 * 2. Within a column store
 * ------------------------
 * Two kinds of pages are involved: the root page and the data pages.
 *
 * 2.1 Single data page. If an operation latches exactly one data page (S or
 *     X), it does not need to latch the root page.
 *
 * 2.2 Root page before data pages. If an operation needs the root page and
 *     one or more data pages, it must latch the root page first.
 *
 * 2.3 Multiple data pages. If an operation latches more than one data page,
 *     it must hold the root page latch first (see 2.2). The root page latch
 *     mode then decides the order for the data pages:
 *       - Root X latch: data pages may be latched in any order.
 *       - Root S latch: data pages must be latched in ascending page ID order.
 *         (Reserved for future use. No current code path latches multiple
 *         data pages under an S root latch.)
 *
 * 3. Rules for callers
 * --------------------
 * 3.1 Reusing an mtr within one column store. The same mtr must not be passed
 *     to several APIs of the same column store, unless every call after the
 *     first operates only on pages the first call already latched. Example:
 *     fetch-for-update followed by update of the same record.
 *
 * 3.2 Reusing an mtr across column stores. The same mtr may be used for
 *     operations on different column stores. The caller must then keep a
 *     consistent latch order across the column stores. Example: the HNSW index
 *     keeps a top-down order across levels, since each level is a separate
 *     column store. Within one level, the Primary column store is latched
 *     before the Secondary one.
 */

// Describes one column store to be created within a segment.
struct Storage_spec {
  uint16_t col_len;
  std::string metadata;
};

struct ColumnStore {
  // Atomic counter to measure insert concurrency for this storage context.
  // This counter is incremented when entering insert() and decremented when
  // leaving, allowing us to track the peak number of concurrent inserts.
  std::atomic<uint32_t> m_insert_concurrency_counter{0};

  RootPage m_root;
  DataPage m_data;

  // Space and root page for this store.
  Space::Ref m_space_ref{0};
  Page::Ref m_root_page_ref{Page::INVALID_REF};
  Page::Ref m_primary_root_page_ref{Page::INVALID_REF};
  uint8_t m_segment_index{0};

  std::string m_metadata;

  ColumnStore() = default;

  // std::atomic is not movable so std::vector<ColumnStore> requires a move
  // constructor to compile. Callers must reserve() before emplace_back() so
  // reallocation never happens and this constructor is never actually called.
  ColumnStore(ColumnStore &&) noexcept { assert(false); }

  ColumnStore(const ColumnStore &) = delete;
  ColumnStore &operator=(const ColumnStore &) = delete;
  ColumnStore &operator=(ColumnStore &&) = delete;

  void init(Space::Ref space_ref, Page::Ref root_page_ref, uint16_t col_len,
            uint8_t num_segments, uint8_t num_root_pages,
            std::string_view metadata);

  bool initialized() const { return (m_root_page_ref != Page::INVALID_REF); }

  // See MultiColumnStore::fill_error, including the (nullptr, 0) convention
  // for a caller that has nothing to report.
  static void fill_error(const char *info, char *msg, uint32_t len, bool local);

  // Storage operations. All return false on success, true on error.
  bool insert(MtrCtx::Ref mctx, Segment::TrxRef trx_ref, Column::Data col_data,
              Column::Ref &col_ref, char *error_msg, uint32_t error_msg_len);

  // for_update selects the latch mode on the data page: EXCLUSIVE when true
  // (caller intends to modify the record), SHARED otherwise.
  bool fetch(MtrCtx::Ref mctx, Column::Ref col_ref, bool for_update,
             Column::Data &col_data, Column::Data &rowid_prefix,
             Segment::TrxRef &trx_ref, Segment::TrxRef &delete_trx_ref,
             char *error_msg, uint32_t error_msg_len);

  // Overwrites one or more fixed-size chunks of the record's column data,
  // leaving the rest of the record (including trx_ref) untouched.
  // Non-transactional: does not read, check, or write the record's trx_ref.
  // col_data holds the full-length column data; for each index in
  // chunk_indexes, the chunk_size bytes at offset (index * chunk_size) are
  // copied from col_data into the same offset in the record. chunk_indexes
  // must be given in strictly increasing order; violating this is an error.
  bool update(MtrCtx::Ref mctx, Column::Ref col_ref,
              const Column::Data col_data, std::span<const uint16_t> chunk_ids,
              uint16_t chunk_size, char *error_msg, uint32_t error_msg_len);

  bool mark_delete(MtrCtx::Ref mctx, Segment::TrxRef trx_ref,
                   Column::Ref col_ref, bool delete_mark, char *error_msg,
                   uint32_t error_msg_len);

  bool purge(MtrCtx::Ref mctx, Segment::TrxRef trx_ref, Column::Ref col_ref,
             char *error_msg, uint32_t error_msg_len);

  // Reports whether purge() with the same trx_ref would remove the record:
  // purgeable is false if the slot is already free or its stored trx ref
  // doesn't match trx_ref (see DataPage::is_purgeable). Neither is an error.
  // Takes only an S latch and modifies nothing.
  bool is_purgeable(MtrCtx::Ref mctx, Segment::TrxRef trx_ref,
                    Column::Ref col_ref, bool &purgeable, char *error_msg,
                    uint32_t error_msg_len);

  // Overwrites this store's metadata in the root page in place. metadata
  // must be the same length as the metadata this store was created/loaded
  // with -- every other field in the root page layout is offset relative to
  // that length, so it cannot change without reformatting the page. Updates
  // m_metadata on success.
  bool update_metadata(MtrCtx::Ref mctx, std::string_view metadata,
                       char *error_msg, uint32_t error_msg_len);
};

struct MultiColumnStore {
  static constexpr uint8_t FORMAT_VERSION = 1;

  // Persistent storage reference: encodes (space_ref, root_page_ref).
  // Set on create/load; used by decode_ref throughout DML operations.
  Column::StorageRef m_ref{0};

  // One entry per column store. First entry is the primary store whose ref is
  // encoded in m_ref; its root page also holds segments and root page refs
  // for any additional stores.
  std::vector<ColumnStore> m_stores;

  // Encode space_ref and root page ref into m_ref.
  void encode_ref(Space::Ref space_ref, Page::Ref page_ref) {
    m_ref = static_cast<uint64_t>(space_ref) |
            (static_cast<uint64_t>(page_ref) << 32);
  }

  // Decode space_ref and root page ref from m_ref.
  void decode_ref(Space::Ref &space_ref, Page::Ref &page_ref) const {
    space_ref = static_cast<Space::Ref>(m_ref & 0xFFFFFFFFU);
    page_ref = static_cast<Page::Ref>(m_ref >> 32);
  }

  // Storage operations. All return false on success, true on error.
  bool create(Space::Ref space_ref, Segment::TrxRef trx_ref,
              const std::vector<Storage_spec> &storages, uint8_t num_segments,
              char *error_msg, uint32_t error_msg_len);

  bool drop(Segment::TrxRef trx_ref, char *error_msg, uint32_t error_msg_len);

  // Allocate a new root page for the store at root_idx from segment seg_idx,
  // format it, and record its ref in the primary root page's other-refs array.
  // Reused when creating higher-level stores on demand.
  bool init_root_page(uint8_t seg_idx, uint8_t root_idx, char *error_msg,
                      uint32_t error_msg_len);

  bool load(Column::StorageRef storage_ref,
            const std::vector<Storage_spec> &specs, char *error_msg,
            uint32_t error_msg_len);

  // Look up which entry in m_stores a column reference belongs to, by
  // loading its data page and matching the page's root page ref against
  // each store's m_root_page_ref.
  bool get_root_index(MtrCtx::Ref mctx, Column::Ref col_ref, uint8_t &root_idx,
                      char *error_msg, uint32_t error_msg_len);

  // Format an error message into msg/len.
  // Writes "SVECTOR: <info>" when local is true, or
  // "SVECTOR: <info>: <last_error()>" when local is false.
  //
  // (msg, len) may be (nullptr, 0), which discards the message. Every
  // storage operation taking an error buffer accepts that pair, for callers
  // whose failure is not reportable -- a debug-only check, say, whose caller
  // asserts instead. Anything else must be a writable buffer of len bytes.
  static void fill_error(const char *info, char *msg, uint32_t len, bool local);
};

// ColumnStorage provides external storage for VECTOR columns.
// All methods are static and correspond to the VEF storage interface.
class ColumnStorage {
 public:
   using Ctx = Column::StorageCtx<MultiColumnStore>;

   static bool create(Ctx *storage, Space::Ref space, Segment::TrxRef trx_ref,
                      uint32_t col_len, char *error_msg,
                      uint32_t error_msg_len);

   static bool drop(Ctx *storage, Segment::TrxRef trx_ref, char *error_msg,
                    uint32_t error_msg_len);

   static bool load(Ctx *storage, Column::StorageRef storage_ref,
                    char *error_msg, uint32_t error_msg_len);

   static bool insert(Ctx *storage, MtrCtx::Ref mctx, Segment::TrxRef trx_ref,
                      Column::Data col_data, Column::Data rowid_prefix,
                      Column::Ref *col_ref, char *error_msg,
                      uint32_t error_msg_len);

   static bool select(Ctx *storage, MtrCtx::Ref mctx, Column::Ref col_ref,
                      Column::Data *col_data, Column::Data *rowid_prefix,
                      Segment::TrxRef *trx_ref, Segment::TrxRef *delete_trx_ref,
                      char *error_msg, uint32_t error_msg_len);

   static bool mark_delete(Ctx *storage, MtrCtx::Ref mctx,
                           Segment::TrxRef trx_ref, Column::Ref col_ref,
                           bool delete_mark, char *error_msg,
                           uint32_t error_msg_len);

   static bool purge(Ctx *storage, MtrCtx::Ref mctx, Segment::TrxRef trx_ref,
                     Column::Ref col_ref, char *error_msg,
                     uint32_t error_msg_len);
};

}  // namespace svector

#endif  // VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_H
