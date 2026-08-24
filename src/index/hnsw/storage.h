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

  // Level Lock: guards traversal and existence of this level's storage.
  // Levels are locked top-down and, during lock-coupled descent, released
  // bottom-up by IndexGraph::LockLevels. See the lock hierarchy comment in
  // graph.h for the acquisition order.
  std::shared_mutex &mutex() { return m_mutex; }

  // Level Operation Lock: serializes graph operations performed at this level,
  // such as linking or adjusting neighbours. Unlike mutex(), this lock does
  // not guard access to the level itself; it coordinates operations on the
  // graph structure within the level. Only one Level Operation Lock is held at
  // a time. See the lock hierarchy comment in graph.h for acquisition order
  // and per-operation S/X mode.
  std::shared_mutex &operation_mutex() { return m_op_mutex; }

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
  static constexpr uint32_t overflow_capacity(LevelId /*level*/, uint32_t M) {
    return std::min(MAX_OVERFLOW_LEN, M);
  }

  // Upper bound, over every level, on the number of chunk ids a single
  // update() can emit: the owner, the nid/vid pair for each neighbour slot,
  // the lower level link and the trailing overflow link. Level 0 dominates,
  // since it carries twice the degree and has no lower level field. Sizes the
  // caller's chunk id scratch buffer, so it is derived from the chunk layout
  // below rather than restated at the call site. Also covers the overflow
  // record, whose update() emits at most overflow_capacity() + 1 ids.
  static constexpr size_t max_update_chunks(uint32_t M) {
    return static_cast<size_t>(neighbour_overflow_chunk(
               /*has_lower=*/false, max_neighbours(LevelId{0}, M))) +
           1;
  }

  LevelId level() const { return m_level; }

  uint32_t target_neighbours() const {
    return target_neighbours(m_level, m_num_neighbours);
  }
  uint32_t max_neighbours() const {
    return max_neighbours(m_level, m_num_neighbours);
  }
  uint32_t overflow_capacity() const {
    return overflow_capacity(m_level, m_num_neighbours);
  }

  // Formats entry into buffer per this level's on-disk record layout and
  // inserts it as a new record in m_store. On success, out is set to the
  // NID of the newly inserted entry.
  bool insert(MtrCtx::Ref mtr, const NeighbourEntry &entry,
              Segment::TrxRef trx_ref, ScratchBytes &buffer, NID &out,
              char *err, uint32_t err_len);

  // Formats entry into buffer per this level's on-disk overflow-record
  // layout and inserts it as a new record in m_overflow. On success, out is
  // set to the NID of the newly inserted entry.
  bool insert(MtrCtx::Ref mtr, const OverflowEntry &entry,
              Segment::TrxRef trx_ref, ScratchBytes &buffer, NID &out,
              char *err, uint32_t err_len);

  // Physically removes the record identified by nid from m_store or
  // m_overflow, per kind.
  bool remove(MtrCtx::Ref mtr, StoreKind kind, NID nid, Segment::TrxRef trx_ref,
              char *err, uint32_t err_len);

  // Marks or unmarks for deletion the m_store record identified by nid.
  // m_overflow records have no row of their own to be deleted with, so this
  // is never needed for them.
  bool mark_delete(MtrCtx::Ref mtr, NID nid, Segment::TrxRef trx_ref,
                   bool delete_mark, char *err, uint32_t err_len);

  // Rewrites the fields of id's record set in mask from entry, leaving the
  // rest of the record untouched (see ColumnStore::update). When mask
  // includes Neighbours, slots[i] is the on-disk slot index that
  // entry.neighbours[i] is written to -- slots must be strictly increasing
  // and only its first entry.neighbours.size() entries are read; slots is
  // unused otherwise. buffer is scratch space for the encoded neighbour
  // bytes.
  bool update(MtrCtx::Ref mtr, NID id, const NeighbourEntry &entry,
              NodeField mask, const ScratchSlots &slots, ScratchBytes &buffer,
              ScratchChunkIds &chunk_ids, char *err, uint32_t err_len);

  // Same as above, for m_overflow's record layout: slots[i] pairs with
  // entry.incoming[i] when mask includes Incoming.
  bool update(MtrCtx::Ref mtr, NID id, const OverflowEntry &entry,
              OverflowField mask, const ScratchSlots &slots,
              ScratchBytes &buffer, ScratchChunkIds &chunk_ids, char *err,
              uint32_t err_len);

  // Fetches id's record from m_store (always a full-record read -- see
  // ColumnStore::fetch) and unmarshals into entry only the fields selected
  // by mask; fields outside mask leave entry untouched. for_update selects
  // the page latch mode (SHARED vs EXCLUSIVE) exactly as in
  // ColumnStore::fetch.
  //
  // When mask includes Neighbours: if slots is null, every slot up to
  // max_neighbours() is decoded; otherwise entry.neighbours[i] is decoded
  // from on-disk slot (*slots)[i] for each i (unlike update(), slots need
  // not be in any particular order for a fetch). num_valid is set to how
  // many of the decoded neighbours are valid (non-INVALID and, per filter
  // below, not excluded); fields outside mask leave it untouched.
  //
  // filter selects whether a decoded neighbour whose NID has the incoming
  // flag set (Id<NIDTag>::is_incoming()) counts as valid: All keeps it,
  // ExcludeIncoming treats it exactly like an INVALID slot for num_valid
  // and the for_update compaction described below.
  //
  // for_update distinguishes a read meant to feed a later slot-aligned
  // update() from a read-only lookup: when true, every requested slot is
  // written to entry.neighbours in place (including INVALID and
  // filtered-out ones), so positions still line up with slots for that
  // later update() call, and entry.neighbours must be sized to hold every
  // requested slot. When false, INVALID and filtered-out slots are skipped
  // and the remaining ones are compacted to the front of entry.neighbours,
  // which is then shrunk to entry.neighbours.first(num_valid) -- there's no
  // later update() to stay aligned with, so entry.neighbours only needs to
  // be sized for the number of slots requested, not exactly num_valid.
  bool fetch(MtrCtx::Ref mtr, NID id, bool for_update, NeighbourEntry &entry,
             size_t &num_valid, char *err, uint32_t err_len,
             NodeField mask = FieldAll,
             IncomingFilter filter = IncomingFilter::ExcludeIncoming,
             const ScratchChunkIds *slots = nullptr);

  // Same as above, for m_overflow's record layout: slots[i] identifies the
  // on-disk incoming-slot decoded into entry.incoming[i] when mask includes
  // Incoming (with the same null-means-everything, num_valid, and
  // for_update conventions).
  bool fetch(MtrCtx::Ref mtr, NID id, bool for_update, OverflowEntry &entry,
             size_t &num_valid, char *err, uint32_t err_len,
             OverflowField mask = OverflowFieldAll,
             const ScratchChunkIds *slots = nullptr);

  // Resolves nid to the Node it names -- nid itself, paired with the vid
  // read from its record's Owner field -- via a fetch in its own mtr. A
  // convenience for callers that only need a record's owning vector: nid
  // must already be known to belong to this level's store (e.g. via
  // IndexStore::locate(), or because the caller persisted it there itself).
  bool resolve_owner(NID nid, Node &out, char *err, uint32_t err_len);

private:
  static constexpr uint32_t MAX_OVERFLOW_LEN = 8;

  // Chunk-index layout for this level's two record types, in
  // ColumnStore::update()'s chunk_size units. Partial update and (future)
  // partial read must agree on these positions, so they're exposed here
  // rather than computed at each call site.
  //
  // NID and VID share the same on-disk width, so that width serves as the
  // chunk size for both the primary (NeighbourEntry) and overflow
  // (OverflowEntry) records.
  static constexpr uint16_t CHUNK_SIZE =
      static_cast<uint16_t>(NID::STORAGE_SIZE);
  static_assert(NID::STORAGE_SIZE == VID::STORAGE_SIZE);

  // -- NeighbourEntry (m_store) layout --
  static constexpr uint16_t owner_chunk() { return 0; }
  static constexpr uint16_t lower_level_chunk() { return owner_chunk() + 1; }
  static constexpr uint16_t neighbours_base_chunk(bool has_lower) {
    return lower_level_chunk() + (has_lower ? 1 : 0);
  }
  static constexpr uint16_t neighbour_nid_chunk(bool has_lower, uint16_t slot) {
    return neighbours_base_chunk(has_lower) + slot * 2;
  }
  static constexpr uint16_t neighbour_vid_chunk(bool has_lower, uint16_t slot) {
    return neighbour_nid_chunk(has_lower, slot) + 1;
  }
  static constexpr uint16_t neighbour_overflow_chunk(bool has_lower,
                                                     uint32_t max_n) {
    return neighbours_base_chunk(has_lower) + static_cast<uint16_t>(max_n) * 2;
  }

  uint16_t neighbours_base_chunk() const {
    return neighbours_base_chunk(m_level.has_lower_level());
  }
  uint16_t neighbour_nid_chunk(uint16_t slot) const {
    return neighbour_nid_chunk(m_level.has_lower_level(), slot);
  }
  uint16_t neighbour_vid_chunk(uint16_t slot) const {
    return neighbour_vid_chunk(m_level.has_lower_level(), slot);
  }
  uint16_t neighbour_overflow_chunk() const {
    return neighbour_overflow_chunk(m_level.has_lower_level(),
                                    max_neighbours());
  }

  // -- OverflowEntry (m_overflow) layout --
  static constexpr uint16_t incoming_chunk(uint16_t slot) { return slot; }
  static constexpr uint16_t overflow_chunk(uint32_t capacity) {
    return static_cast<uint16_t>(capacity);
  }

  uint16_t overflow_chunk() const {
    return overflow_chunk(overflow_capacity());
  }

  mutable std::shared_mutex m_mutex;
  mutable std::shared_mutex m_op_mutex;
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

  Index::StorageRef storage_ref() const { return m_multi_store.m_ref; }

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

  // Current entry point node, or an invalid node (nid unset) if the graph
  // has no entry point yet (Algorithm 1, lines 2-3), in which case
  // entry_level() is unspecified. Kept fully resolved (nid and vid) in
  // memory -- refreshed on load() and by set_entry_point() below -- so this
  // is a plain field read with no page access. Caller must hold mutex() --
  // entry point/level are graph-wide metadata that mutex() protects.
  const Node &entry_point() const { return m_entry_point; }

  // Level of the current entry point. Caller must hold mutex().
  LevelStore::LevelId entry_level() const { return m_entry_level; }

  // Registers node as the graph's new entry point at level (Algorithm 1,
  // line 19), persisting the change to the level-0 primary store's root
  // page metadata. node is an invalid node (nid unset) to clear the entry
  // point (the graph became empty). Caller must hold mutex() in exclusive
  // mode.
  bool set_entry_point(MtrCtx::Ref mtr, const Node &node,
                       LevelStore::LevelId level, char *err, uint32_t err_len);

  // Store for level, or nullptr if that level has not been created yet.
  LevelStore *get_level(LevelStore::LevelId level) {
    return level.value < S_MAX_LEVEL && m_levels[level.value].has_value()
               ? &*m_levels[level.value]
               : nullptr;
  }

  // Store owning nid's record, or nullptr if it cannot be resolved. On
  // success, kind is set to which of that level's two stores (Neighbour or
  // Overflow) the record lives in. A caller that will not report the failure
  // may pass (nullptr, 0) for err/err_len (see MultiColumnStore::fill_error).
  LevelStore *locate(NID nid, StoreKind &kind, char *err, uint32_t err_len);

  // Materializes every level's storage from m_entry_level (exclusive) up to
  // and including target, allocating and formatting each new level's
  // primary and overflow root pages. Levels 0..m_entry_level are always
  // already created -- an invariant maintained by every past call to this
  // function paired with the set_entry_point() that follows it -- so this
  // only needs to walk from m_entry_level+1. target must not exceed
  // max_level(). Calling this with a target at or below m_entry_level is a
  // no-op that just returns get_level(target).
  //
  // Caller must already hold mutex(): this walks, and may mutate, graph-wide
  // level storage and does not lock mutex() itself. Exclusive mode is
  // required whenever target exceeds m_entry_level, since that is the case
  // that mutates; shared mode suffices for the no-op case, which only reads
  // m_levels and cannot run concurrently with a mutating call anyway.
  //
  // Returns the store for target, or nullptr on error (err/err_len set).
  // Any levels created before the failing one remain created -- levels are
  // never dropped, so levels above m_entry_level are a valid state, and this
  // function is idempotent: a later call skips whatever already exists,
  // including a level left with only one of its two root pages.
  LevelStore *ensure_levels(LevelStore::LevelId target, char *err,
                            uint32_t err_len);

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

  // Current highest level and node to enter the graph.
  LevelStore::LevelId m_entry_level{0};
  Node m_entry_point{};

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
