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

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_STORAGE_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_STORAGE_H

#include <algorithm>
#include <cstdint>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <vector>

#include "../../storage/storage.h"
#include "hnsw.h"
#include <villagesql/preview/index_builder.h>

namespace svector::hnsw {

using vsql::preview_index_builder::Index;
using vsql::preview_index_builder::IndexScanDesc;
using vsql::preview_index_builder::IndexScanKey;
using vsql::preview_storage::MtrCtx;
using vsql::preview_storage::Page;
using vsql::preview_storage::Segment;
using vsql::preview_storage::Space;

// Options parsed from WITH (...) at CREATE INDEX time.
struct Options {
  static constexpr uint32_t DEFAULT_M = 16;
  static constexpr uint32_t DEFAULT_EF_CONSTRUCTION = 200;

  static constexpr uint32_t MAX_M = 128;
  // Allow ef_construction up to 32 x M
  static constexpr uint32_t MAX_EF_CONSTRUCTION_FACTOR = 32;

  uint32_t M = DEFAULT_M;
  uint32_t ef_construction = DEFAULT_EF_CONSTRUCTION;

  bool validate(char *error_msg, uint32_t error_msg_len) const;

  static bool parse(const vef_index_param_t *params, uint32_t count,
                    Options *out, char *error_msg, uint32_t error_msg_len);
};

// Per-scan cursor (empty until traversal is implemented).
class Cursor {};

using svector::ColumnStore;
using svector::MultiColumnStore;

class LevelStore {
public:
  struct LevelId {
    constexpr LevelId() : value(0) {}
    constexpr explicit LevelId(uint8_t v) : value(v) {}
    constexpr bool has_lower_level() const { return value > 0; }
    constexpr LevelId lower() const {
      assert(value > 0);
      return value > 0 ? LevelId(value - 1) : LevelId(0);
    }
    constexpr bool operator==(const LevelId &) const = default;
    constexpr auto operator<=>(const LevelId &) const = default;
    uint8_t value;
  };

  LevelStore(LevelId level, ColumnStore &store, ColumnStore &overflow,
             uint32_t num_neighbours)
      : m_level(level), m_store(store), m_overflow(overflow),
        m_num_neighbours(num_neighbours) {}

  std::shared_mutex &mutex() { return m_mutex; }

  // Target neighbour degree (M) at a given level. Equal to the configured M
  // for every level today, but kept level-parameterized since the HNSW
  // paper treats it as a per-level quantity.
  static constexpr uint32_t target_neighbours(LevelId /*level*/, uint32_t M) {
    return M;
  }

  // Maximum neighbour degree permitted at a given level (Mmax, or
  // Mmax0 = 2*M at level 0, per the HNSW paper).
  static constexpr uint32_t max_neighbours(LevelId level, uint32_t M) {
    return level.value == 0 ? 2 * M : M;
  }

  // Maximum number of incoming-NID slots held by a single overflow entry
  // (chain link). Capped at MAX_OVERFLOW_LEN, but never larger than M itself
  // since it would be pointless to size an overflow link above the degree
  // it's compensating for.
  static constexpr uint32_t MAX_OVERFLOW_LEN = 8;
  static constexpr uint32_t overflow_capacity(LevelId /*level*/, uint32_t M) {
    return std::min(MAX_OVERFLOW_LEN, M);
  }

  uint32_t target_neighbours() const {
    return target_neighbours(m_level, m_num_neighbours);
  }
  uint32_t max_neighbours() const {
    return max_neighbours(m_level, m_num_neighbours);
  }
  uint32_t overflow_capacity() const {
    return overflow_capacity(m_level, m_num_neighbours);
  }

  // TODO(villagesql-indexing): Implement Insert
  // TODO(villagesql-indexing): Implement Search
  // TODO(villagesql-indexing): Implement Purge

private:
  mutable std::shared_mutex m_mutex;
  LevelId m_level;
  ColumnStore &m_store;
  ColumnStore &m_overflow;
  uint32_t m_num_neighbours = 0;
};

// Metadata stored in each store's root page.
//
// Binary layout - Version 1:
//
// [Version] [Name Len (L)] [Name] [Level]
// |---1----|------1--------|---L--|---1---|
//
// [Entry Level] [Num Entry Points (N)] [Entry Points]
// |-----1-------|----------1----------|---N * 8------|
struct StorageMeta {
  static constexpr uint8_t VERSION = 1;

  static constexpr size_t VERSION_LEN = 1;
  static constexpr size_t NAME_LEN_SIZE = 1;
  static constexpr size_t LEVEL_LEN = 1;
  static constexpr size_t ENTRY_LEVEL_LEN = 1;
  static constexpr size_t NUM_ENTRY_POINTS_LEN = 1;
  static constexpr size_t ENTRY_POINT_LEN = sizeof(IndexScanKey::KeyPartRef);

  // Minimum encoded size: fixed fields only, empty name, zero entry points.
  static constexpr size_t MIN_ENCODED_LEN = VERSION_LEN + NAME_LEN_SIZE +
                                            LEVEL_LEN + ENTRY_LEVEL_LEN +
                                            NUM_ENTRY_POINTS_LEN;

  std::string name;
  // Level of the root page this metadata is stored in.
  LevelStore::LevelId level{0};
  LevelStore::LevelId entry_level{0};
  std::vector<IndexScanKey::KeyPartRef> entry_points;

  // Serialize to a string suitable for Storage_spec::metadata.
  void encode(std::string *out) const;

  // Deserialize from a root-page metadata string into this object.
  // Returns false on success, true on malformed input.
  bool decode(std::string_view data);
};

class IndexStore {
  static constexpr uint8_t S_MAX_LEVEL = 32;
  static constexpr size_t S_NUM_STORES = static_cast<size_t>(S_MAX_LEVEL) * 2;

public:
  static constexpr size_t KEY_REF_SIZE = StorageMeta::ENTRY_POINT_LEN;
  bool create(Space::Ref space_ref, Segment::TrxRef trx_ref,
              const Options &opts, char *err, uint32_t err_len);

  bool drop(Segment::TrxRef trx_ref, char *err, uint32_t err_len);

  bool load(Index::StorageRef storage_ref, const Options &opts, char *err,
            uint32_t err_len);

  // TODO(villagesql-indexing): Implement DML insert, mark_delete, purge
  // TODO(villagesql-indexing): Implement cursor operations
  // begin, end, position, fetch, save and restore.

  Index::StorageRef storage_ref() const { return m_multi_store.m_ref; }

  bool is_initialized() const {
    std::shared_lock lock(m_mutex);
    return m_initialized;
  }

  bool is_empty() const {
    std::shared_lock lock(m_mutex);
    return m_entry_point == IndexScanKey::EMPTY_REF;
  }

  // Configured number of neighbours M.
  uint32_t num_neighbours() const { return m_num_neighbours; }

  // Graph exploration factor during insertion.
  uint32_t ef_construction() const { return m_ef_construction; }

  // Normalization factor mL used to draw an element's insertion level
  // (Algorithm 1, line 4), derived from M as 1/ln(M).
  double level_norm_factor() const { return m_level_norm_factor; }

  // Highest level a node may ever be inserted at, bounded by the fixed
  // number of per-level stores this index can hold.
  static constexpr LevelStore::LevelId max_level() {
    return LevelStore::LevelId{S_MAX_LEVEL - 1};
  }

  // Whole-graph lock, protecting graph-wide metadata (entry point/level).
  std::shared_mutex &mutex() { return m_mutex; }

  // Store for level, or nullptr if that level has not been created yet.
  LevelStore *level(LevelStore::LevelId level);

private:
  // Two segments: Primary for level-0, Secondary for the rest of the levels.
  enum class SegmentIndex : uint8_t {
    Primary = 0,
    Secondary = 1,
    Count = 2,
  };

  static constexpr uint8_t segment_total() noexcept {
    return static_cast<uint8_t>(SegmentIndex::Count);
  }

  struct RootId {
    enum class Type : uint8_t {
      Primary,
      Overflow,
    };
    LevelStore::LevelId level;
    Type type;
  };

  mutable std::shared_mutex m_mutex;

  // Get stored length of Index entry.
  uint16_t entry_len(LevelStore::LevelId level) const;
  uint16_t overflow_len(LevelStore::LevelId level) const;

  uint8_t segment_index(const RootId &root) const;
  uint8_t root_index(const RootId &root) const;

  void build_storage_specs(std::vector<Storage_spec> &specs);

  bool m_initialized = false;

  // Configured number of neighbours M.
  uint32_t m_num_neighbours = 0;
  // Graph exploration factor during insertion.
  uint32_t m_ef_construction = 0;

  // Normalization factor for level generation ML.
  double m_level_norm_factor = 0;

  // Current highest level and key to enter the graph.
  LevelStore::LevelId m_entry_level{0};
  IndexScanKey::KeyPartRef m_entry_point = IndexScanKey::EMPTY_REF;

  std::array<std::optional<LevelStore>, S_MAX_LEVEL> m_levels;
  MultiColumnStore m_multi_store;
};

using StorageCtx = Index::StorageCtx<IndexStore>;

// Lifecycle hooks
bool create(StorageCtx *ctx, const Index &index, Space::Ref space_ref,
            Segment::TrxRef trx_ref, char *err, uint32_t err_len);

bool drop(StorageCtx *ctx, const Index &index, Segment::TrxRef trx_ref,
          char *err, uint32_t err_len);

bool load(StorageCtx *ctx, const Index &index, Index::StorageRef storage_ref,
          char *err, uint32_t err_len);

// DML hooks
bool insert(StorageCtx *ctx, const Index &index, Segment::TrxRef trx_ref,
            IndexScanKey::KeyPartData *key_columns,
            IndexScanKey::KeyPartData *pkey_columns,
            IndexScanKey::KeyPartRef *key_ref, char *err, uint32_t err_len);

bool mark_delete(StorageCtx *ctx, const Index &index, Segment::TrxRef trx_ref,
                 IndexScanKey::KeyPartRef *key_ref,
                 IndexScanKey::KeyPartData *key_columns,
                 IndexScanKey::KeyPartData *pkey_columns, bool delete_mark,
                 char *err, uint32_t err_len);

bool purge(StorageCtx *ctx, const Index &index, Segment::TrxRef trx_ref,
           IndexScanKey::KeyPartRef *key_ref,
           IndexScanKey::KeyPartData *key_columns,
           IndexScanKey::KeyPartData *pkey_columns, char *err,
           uint32_t err_len);

// Scan hooks
bool begin(StorageCtx *ctx, const Index &index, MtrCtx::Ref mctx,
           const IndexScanDesc &scan_desc, Index::Cursor *cursor, bool *eof,
           char *err, uint32_t err_len);

bool position(Index::Cursor cursor, Index::CursorOp op, bool *eof, char *err,
              uint32_t err_len);

bool fetch(Index::Cursor cursor, IndexScanKey::KeyPartRef *key_ref,
           IndexScanKey::KeyPartData *key_columns,
           IndexScanKey::KeyPartData *pkey_columns, char *err,
           uint32_t err_len);

bool save(Index::Cursor cursor, char *err, uint32_t err_len);

bool restore(Index::Cursor cursor, MtrCtx::Ref mctx, bool *eof, char *err,
             uint32_t err_len);

void end(Index::Cursor *cursor);

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_STORAGE_H
