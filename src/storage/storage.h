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
#include <cstdint>

#include <villagesql/storage_api.h>

#include "data_page.h"
#include "root_page.h"

namespace svector {

using villagesql::storage::Column;
using villagesql::storage::MtrCtx;
using villagesql::storage::Page;
using villagesql::storage::Segment;
using villagesql::storage::Space;

struct ColumnStorageContext {
  static constexpr uint8_t FORMAT_VERSION = 1;

  // Atomic counter to measure insert concurrency for this storage context.
  // This counter is incremented when entering insert() and decremented when
  // leaving, allowing us to track the peak number of concurrent inserts.
  std::atomic<uint32_t> m_insert_concurrency_counter{0};

  // Persistent storage reference: encodes (space_ref, root_page_ref).
  // Set on create/load; used by decode_ref throughout DML operations.
  Column::StorageRef m_ref{0};

  RootPage m_root;
  DataPage m_data;

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
  bool create(Space::Ref space_ref, Segment::TrxRef trx_ref, uint16_t col_len,
              char *error_msg, uint32_t error_msg_len);

  bool drop(Segment::TrxRef trx_ref, char *error_msg, uint32_t error_msg_len);

  bool insert(MtrCtx::Ref mctx, Segment::TrxRef trx_ref, Column::Data col_data,
              Column::Ref &col_ref, char *error_msg, uint32_t error_msg_len);

  bool fetch(MtrCtx::Ref mctx, Column::Ref col_ref, Column::Data &col_data,
             Column::Data &rowid_prefix, Segment::TrxRef &trx_ref,
             bool &delete_marked, char *error_msg, uint32_t error_msg_len);

  bool mark_delete(MtrCtx::Ref mctx, Segment::TrxRef trx_ref,
                   Column::Ref col_ref, bool delete_mark, char *error_msg,
                   uint32_t error_msg_len);

  bool purge(MtrCtx::Ref mctx, Segment::TrxRef trx_ref, Column::Ref col_ref,
             char *error_msg, uint32_t error_msg_len);

 private:
  // Format an error message into msg/len.
  // Writes "SVECTOR: <info>" when local is true, or
  // "SVECTOR: <info>: <last_error()>" when local is false.
  static void fill_error(const char *info, char *msg, uint32_t len, bool local);
};

// ColumnStorage provides external storage for VECTOR columns.
// All methods are static and correspond to the VEF storage interface.
class ColumnStorage {
 public:
  using Ctx = Column::StorageCtx<ColumnStorageContext>;

  static bool create(Ctx *storage, Space::Ref space, Segment::TrxRef trx_ref,
                     uint32_t col_len, char *error_msg, uint32_t error_msg_len);

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
                     Segment::TrxRef *trx_ref, bool *delete_marked,
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
