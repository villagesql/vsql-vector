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

// This file is compiled only when SVECTOR_ROWID_TRAILER is enabled (CMake adds
// it to the svector sources only then); otherwise the baseline ColumnStorage in
// storage.cc is used. The code below is not wrapped in an #ifdef -- the build
// system owns the gating -- but fail loudly if that gating is ever bypassed,
// since the ColumnStorage declaration this file defines is itself behind the
// macro in the header.
#ifndef SVECTOR_ROWID_TRAILER
#error "column_storage_rowid.cc must be compiled with SVECTOR_ROWID_TRAILER defined"
#endif

#include "column_storage_rowid.h"

#include <cstdio>
#include <cstring>
#include <vector>

// Rowid-trailer variant of the SVECTOR column-storage adapter. See
// column_storage_rowid.h for the layering rationale and record layout. Each
// method delegates to the generic MultiColumnStore/ColumnStore engine, which is
// unchanged from the baseline; the only difference from the baseline
// ColumnStorage (storage.cc) is that the vector and the rowid_prefix travel
// together as one opaque record, packed here and split here.

namespace svector {

bool ColumnStorage::create(Ctx *storage, Space::Ref space,
                           Segment::TrxRef trx_ref, uint32_t col_len,
                           char *error_msg, uint32_t error_msg_len) {
  if (col_len >= Page::get_size(space) || col_len < sizeof(Column::Ref)) {
    snprintf(error_msg, error_msg_len,
             "SVECTOR: create: column length %u out of range for page size %u",
             col_len, Page::get_size(space));
    return true;
  }

  // Subtract the storage_ref prefix from the column length stored per record,
  // then add the rowid trailer. The engine sizes its records to this total and
  // treats the whole thing as one opaque blob; only this adapter knows the
  // leading store_len bytes are the vector and the rest is the trailer. The
  // inflated size is persisted in the root page, so load() recovers it without
  // needing to know the trailer size.
  uint16_t vector_len = static_cast<uint16_t>(col_len - sizeof(Column::Ref));
  uint16_t store_len = static_cast<uint16_t>(vector_len + ROWID_TRAILER_LEN);
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
  // The engine recovers the (trailer-inflated) record size from the persisted
  // root page, so the spec's col_len is unused for the single primary store.
  return storage->user()->load(storage_ref, {{0, ""}}, error_msg,
                               error_msg_len);
}

bool ColumnStorage::insert(Ctx *storage, MtrCtx::Ref mctx,
                           Segment::TrxRef trx_ref, Column::Data col_data,
                           Column::Data rowid_prefix, Column::Ref *col_ref,
                           char *error_msg, uint32_t error_msg_len) {
  if (rowid_prefix.length > ROWID_MAX) {
    snprintf(error_msg, error_msg_len,
             "SVECTOR: insert: rowid_prefix too long: len=%u, max=%u",
             rowid_prefix.length, ROWID_MAX);
    return true;
  }

  // Pack [vector][rowid_len:1][rowid:ROWID_MAX] into one opaque record. The
  // trailer is zero-padded past the actual rowid so every record is the same
  // fixed size the engine was created with. Assembled in a local buffer (insert
  // is not the hot path -- search/distance is), so no shared state and no data
  // race between concurrent inserts.
  const uint16_t payload_len =
      static_cast<uint16_t>(col_data.length + ROWID_TRAILER_LEN);
  std::vector<unsigned char> record(payload_len);
  unsigned char *p = record.data();
  memcpy(p, col_data.data, col_data.length);
  p += col_data.length;
  *p++ = static_cast<unsigned char>(rowid_prefix.length);
  if (rowid_prefix.length > 0)
    memcpy(p, rowid_prefix.data, rowid_prefix.length);
  memset(p + rowid_prefix.length, 0, ROWID_MAX - rowid_prefix.length);

  return storage->user()->m_stores[0].insert(
      mctx, trx_ref, Column::Data{record.data(), payload_len}, *col_ref,
      error_msg, error_msg_len);
}

bool ColumnStorage::select(Ctx *storage, MtrCtx::Ref mctx, Column::Ref col_ref,
                           Column::Data *col_data, Column::Data *rowid_prefix,
                           Segment::TrxRef *trx_ref, bool *delete_marked,
                           char *error_msg, uint32_t error_msg_len) {
  // The engine returns the full opaque record ([vector][trailer]); split it
  // back into the vector (leading bytes) and the rowid (in the trailer). Both
  // point in-page, no copy.
  if (storage->user()->m_stores[0].fetch(mctx, col_ref, /*for_update=*/false,
                                         *col_data, *rowid_prefix, *trx_ref,
                                         *delete_marked, error_msg,
                                         error_msg_len))
    return true;

  if (col_data->length < ROWID_TRAILER_LEN) {
    snprintf(error_msg, error_msg_len,
             "SVECTOR: select: record too short for rowid trailer: len=%u",
             col_data->length);
    return true;
  }

  const unsigned char *rec = col_data->data;
  uint16_t vector_len =
      static_cast<uint16_t>(col_data->length - ROWID_TRAILER_LEN);

  // Vector is the leading bytes; trailer follows.
  const unsigned char *trailer = rec + vector_len;
  uint8_t rowid_len = trailer[0];

  col_data->length = vector_len;
  rowid_prefix->data = trailer + 1;
  rowid_prefix->length = rowid_len;
  return false;
}

bool ColumnStorage::mark_delete(Ctx *storage, MtrCtx::Ref mctx,
                                Segment::TrxRef trx_ref, Column::Ref col_ref,
                                bool delete_mark, char *error_msg,
                                uint32_t error_msg_len) {
  return storage->user()->m_stores[0].mark_delete(
      mctx, trx_ref, col_ref, delete_mark, error_msg, error_msg_len);
}

bool ColumnStorage::purge(Ctx *storage, MtrCtx::Ref mctx, Segment::TrxRef trx_ref,
                          Column::Ref col_ref, char *error_msg,
                          uint32_t error_msg_len) {
  return storage->user()->m_stores[0].purge(mctx, trx_ref, col_ref, error_msg,
                                            error_msg_len);
}

}  // namespace svector
