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

#ifndef VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_COLUMN_STORAGE_ROWID_H
#define VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_COLUMN_STORAGE_ROWID_H

// Rowid-trailer variant of the SVECTOR column-storage adapter. This whole file
// is active only when SVECTOR_ROWID_TRAILER is defined; otherwise the baseline
// svector::ColumnStorage in storage.h/.cc is used and this file is empty.
//
// Layering: ColumnStorage is the SVECTOR-specific adapter that implements the
// VEF storage ABI. The generic engine underneath (MultiColumnStore/ColumnStore
// in storage.h) stores opaque fixed-size records and knows nothing about
// vectors or rowids. This variant keeps that engine untouched: it stores the
// row's clustered-key prefix (rowid_prefix) in a fixed trailer AFTER the vector
// within one opaque record, so an index hit -- which yields only a col_ref into
// this store -- can be resolved back to its owning row without the index
// keeping the primary key (REF_LOOKUP). The trailer is assembled on insert and
// split off on select, entirely here; the engine only ever sees a slightly
// larger opaque blob.
//
// Record layout (opaque to the engine):
//   [ vector : vector_len ][ rowid_len : 1 ][ rowid : ROWID_MAX ]
//
// The vector occupies the leading bytes, so the hot distance path -- which
// reads a vector via the server's key-data callback, sized to the registered
// column length (vector_len) -- never sees the trailer.

#include "storage.h"

#ifdef SVECTOR_ROWID_TRAILER

namespace svector {

class ColumnStorage {
 public:
  using Ctx = Column::StorageCtx<MultiColumnStore>;

  // Maximum rowid_prefix bytes stored per record, and the total trailer size
  // ([rowid_len:1][rowid:ROWID_MAX]) folded into each record.
  static constexpr uint8_t ROWID_MAX = 32;
  static constexpr uint16_t ROWID_TRAILER_LEN = 1 + ROWID_MAX;

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

#endif  // SVECTOR_ROWID_TRAILER

#endif  // VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_COLUMN_STORAGE_ROWID_H
