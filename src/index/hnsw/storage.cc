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
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace svector::hnsw {

using svector::RootPage;
using vsql::preview_storage::Error;

static_assert(IndexStore::KEY_REF_SIZE == 8, "KeyPartRef must be 8 bytes");

static void write_u64_be(std::string &buf, uint64_t v) {
  for (int i = 7; i >= 0; --i)
    buf.push_back(static_cast<char>((v >> (i * 8)) & 0xFFu));
}

static uint64_t read_u64_be(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v = (v << 8) | static_cast<uint64_t>(p[i]);
  return v;
}

// Writes v's low 6 bytes big-endian at p and advances p past them -- the
// on-disk width of NID/VID (Id<Tag>::STORAGE_SIZE).
static inline void write_id48_be(std::byte *&p, uint64_t v) {
  for (int i = 5; i >= 0; --i)
    *p++ = static_cast<std::byte>((v >> (i * 8)) & 0xff);
}

// Reads a 6-byte big-endian id from p and advances p past it -- inverse of
// write_id48_be.
static inline uint64_t read_id48_be(const std::byte *&p) {
  uint64_t v = 0;
  for (int i = 0; i < 6; ++i)
    v = (v << 8) | static_cast<uint64_t>(*p++);
  return v;
}

void StorageMeta::encode(std::string *out) const {
  assert(name.size() <= UINT8_MAX);
  assert(entry_points.size() <= UINT8_MAX);

  out->clear();
  out->reserve(MIN_ENCODED_LEN + name.size() +
               entry_points.size() * ENTRY_POINT_LEN);
  out->push_back(static_cast<char>(VERSION));
  out->push_back(static_cast<char>(name.size()));
  out->append(name);
  out->push_back(static_cast<char>(level.value));
  out->push_back(static_cast<char>(entry_level.value));
  out->push_back(static_cast<char>(entry_points.size()));
  for (auto ref : entry_points)
    write_u64_be(*out, static_cast<uint64_t>(ref));
}

bool StorageMeta::decode(std::string_view data) {
  const auto *p = reinterpret_cast<const uint8_t *>(data.data());
  size_t rem = data.size();

  auto read1 = [&](uint8_t &v) -> bool {
    if (rem < 1)
      return true;
    v = *p++;
    --rem;
    return false;
  };

  uint8_t version;
  if (read1(version) || version != VERSION)
    return true;

  uint8_t name_len;
  if (read1(name_len) || rem < name_len)
    return true;
  name.assign(reinterpret_cast<const char *>(p), name_len);
  p += name_len;
  rem -= name_len;

  uint8_t lvl;
  if (read1(lvl))
    return true;
  level = LevelStore::LevelId{lvl};

  uint8_t el;
  if (read1(el))
    return true;
  entry_level = LevelStore::LevelId{el};

  uint8_t num_eps;
  if (read1(num_eps))
    return true;
  if (rem < static_cast<size_t>(num_eps) * ENTRY_POINT_LEN)
    return true;

  entry_points.resize(num_eps);
  for (uint8_t i = 0; i < num_eps; ++i) {
    entry_points[i] = static_cast<IndexScanKey::KeyPartRef>(read_u64_be(p));
    p += ENTRY_POINT_LEN;
    rem -= ENTRY_POINT_LEN;
  }
  return false;
}

static bool parse_u32(const char *str, uint32_t *out) {
  if (str == nullptr || *str == '\0' || *str == '-')
    return true;

  errno = 0;
  char *end = nullptr;
  unsigned long v = strtoul(str, &end, 10);

  if (end == str || *end != '\0' || errno == ERANGE ||
      v > std::numeric_limits<uint32_t>::max())
    return true;

  *out = static_cast<uint32_t>(v);
  return false;
}

bool Options::validate(char *error_msg, uint32_t error_msg_len) const {
  if (M < 2) {
    snprintf(error_msg, error_msg_len, "M must be at least 2, got %u", M);
    return true;
  }
  if (M > MAX_M) {
    snprintf(error_msg, error_msg_len, "M must not exceed %u", MAX_M);
    return true;
  }
  if (ef_construction < M) {
    snprintf(error_msg, error_msg_len, "ef_construction (%u) must be >= M (%u)",
             ef_construction, M);
    return true;
  }
  uint32_t max_ef_construction = M * MAX_EF_CONSTRUCTION_FACTOR;
  if (ef_construction > max_ef_construction) {
    snprintf(error_msg, error_msg_len,
             "ef_construction (%u) exceeds maximum allowed value (%u) for M=%u",
             ef_construction, max_ef_construction, M);
    return true;
  }
  return false;
}

bool Options::parse(const vef_index_param_t *params, uint32_t count,
                    Options *out, char *error_msg, uint32_t error_msg_len) {
  *out = Options{};
  bool seen_m = false;
  bool seen_ef_construction = false;

  for (uint32_t i = 0; i < count; ++i) {
    const char *key = params[i].key;
    const char *val = params[i].value;

    if (strcmp(key, "M") == 0) {
      if (seen_m) {
        snprintf(error_msg, error_msg_len, "duplicate option 'M'");
        return true;
      }
      if (parse_u32(val, &out->M)) {
        snprintf(error_msg, error_msg_len,
                 "M must be a positive integer, got '%s'", val);
        return true;
      }
      seen_m = true;

    } else if (strcmp(key, "ef_construction") == 0) {
      if (seen_ef_construction) {
        snprintf(error_msg, error_msg_len,
                 "duplicate option 'ef_construction'");
        return true;
      }
      if (parse_u32(val, &out->ef_construction)) {
        snprintf(error_msg, error_msg_len,
                 "ef_construction must be a positive integer, got '%s'", val);
        return true;
      }
      seen_ef_construction = true;

    } else {
      snprintf(error_msg, error_msg_len, "unknown option '%s'", key);
      return true;
    }
  }
  return out->validate(error_msg, error_msg_len);
}

uint16_t IndexStore::entry_len(LevelStore::LevelId level) const {
  auto max_neighbours = LevelStore::max_neighbours(level, m_num_neighbours);
  return static_cast<uint16_t>(
      NeighbourEntry::storage_size(max_neighbours, level.has_lower_level()));
}

uint16_t IndexStore::overflow_len(LevelStore::LevelId level) const {
  auto capacity = LevelStore::overflow_capacity(level, m_num_neighbours);
  return static_cast<uint16_t>(OverflowEntry::storage_size(capacity));
}

bool LevelStore::insert(MtrCtx::Ref mtr, const NeighbourEntry &entry,
                        Segment::TrxRef trx_ref, ScratchBytes &buffer, NID &out,
                        char *err, uint32_t err_len) {
  const bool has_lower = m_level.has_lower_level();
  const uint32_t max_n = max_neighbours();
  assert(entry.neighbours.size() <= max_n);

  const size_t len = NeighbourEntry::storage_size(max_n, has_lower);
  assert(buffer.size() >= len);

  std::byte *p = buffer.data();
  write_id48_be(p, entry.owner.value);
  if (has_lower)
    write_id48_be(p, entry.lower_level.value);

  for (uint32_t i = 0; i < max_n; ++i) {
    if (i < entry.neighbours.size()) {
      write_id48_be(p, entry.neighbours[i].nid.value);
      write_id48_be(p, entry.neighbours[i].vid.value);
    } else {
      write_id48_be(p, NID::INVALID);
      write_id48_be(p, VID::INVALID);
    }
  }
  write_id48_be(p, entry.overflow.value);
  assert(static_cast<size_t>(p - buffer.data()) == len);

  Column::Data col_data{reinterpret_cast<const unsigned char *>(buffer.data()),
                        static_cast<uint32_t>(len)};
  Column::Ref col_ref;
  if (m_store.insert(mtr, trx_ref, col_data, col_ref, err, err_len))
    return true;

  out = NID{static_cast<uint64_t>(col_ref)};
  return false;
}

bool LevelStore::insert(MtrCtx::Ref mtr, const OverflowEntry &entry,
                        Segment::TrxRef trx_ref, ScratchBytes &buffer, NID &out,
                        char *err, uint32_t err_len) {
  const uint32_t capacity = overflow_capacity();
  assert(entry.incoming.size() <= capacity);

  const size_t len = OverflowEntry::storage_size(capacity);
  assert(buffer.size() >= len);

  std::byte *p = buffer.data();
  for (uint32_t i = 0; i < capacity; ++i) {
    if (i < entry.incoming.size())
      write_id48_be(p, entry.incoming[i].value);
    else
      write_id48_be(p, NID::INVALID);
  }
  write_id48_be(p, entry.overflow.value);
  assert(static_cast<size_t>(p - buffer.data()) == len);

  Column::Data col_data{reinterpret_cast<const unsigned char *>(buffer.data()),
                        static_cast<uint32_t>(len)};
  Column::Ref col_ref;
  if (m_overflow.insert(mtr, trx_ref, col_data, col_ref, err, err_len))
    return true;

  out = NID{static_cast<uint64_t>(col_ref)};
  return false;
}

bool LevelStore::remove(MtrCtx::Ref mtr, StoreKind kind, NID nid,
                        Segment::TrxRef trx_ref, char *err, uint32_t err_len) {
  ColumnStore &store = (kind == StoreKind::Neighbour) ? m_store : m_overflow;
  return store.purge(mtr, trx_ref, nid.column_ref(), err, err_len);
}

bool LevelStore::update(MtrCtx::Ref mtr, NID id, const NeighbourEntry &entry,
                        NodeField mask, const ScratchSlots &slots,
                        ScratchBytes &buffer, ScratchChunkIds &chunk_ids,
                        char *err, uint32_t err_len) {
  const bool has_lower = m_level.has_lower_level();
  const uint32_t max_n = max_neighbours();
  assert(!has(mask, NodeField::LowerLevel) || has_lower);

  const uint16_t overflow_idx = neighbour_overflow_chunk();
  const size_t len = (static_cast<size_t>(overflow_idx) + 1) * CHUNK_SIZE;
  assert(buffer.size() >= len);

  size_t num_chunks = 0;
  auto write_chunk = [&](uint16_t idx, uint64_t value) {
    std::byte *p = buffer.data() + static_cast<size_t>(idx) * CHUNK_SIZE;
    write_id48_be(p, value);
    assert(num_chunks < chunk_ids.size());
    chunk_ids[num_chunks++] = idx;
  };

  if (has(mask, NodeField::Owner))
    write_chunk(owner_chunk(), entry.owner.value);

  if (has(mask, NodeField::LowerLevel))
    write_chunk(lower_level_chunk(), entry.lower_level.value);

  if (has(mask, NodeField::Neighbours)) {
    const size_t n = entry.neighbours.size();
    assert(n <= slots.size());
    for (size_t i = 0; i < n; ++i) {
      assert(slots[i].is_valid() && slots[i].value < max_n);
      assert(i == 0 || slots[i].value > slots[i - 1].value);
      write_chunk(neighbour_nid_chunk(slots[i].value),
                  entry.neighbours[i].nid.value);
      write_chunk(neighbour_vid_chunk(slots[i].value),
                  entry.neighbours[i].vid.value);
    }
  }

  if (has(mask, NodeField::Overflow))
    write_chunk(overflow_idx, entry.overflow.value);

  Column::Data col_data{reinterpret_cast<const unsigned char *>(buffer.data()),
                        static_cast<uint32_t>(len)};
  return m_store.update(mtr, id.column_ref(), col_data,
                        chunk_ids.span(num_chunks), CHUNK_SIZE, err, err_len);
}

bool LevelStore::update(MtrCtx::Ref mtr, NID id, const OverflowEntry &entry,
                        OverflowField mask, const ScratchSlots &slots,
                        ScratchBytes &buffer, ScratchChunkIds &chunk_ids,
                        char *err, uint32_t err_len) {
  const uint32_t capacity = overflow_capacity();

  const uint16_t overflow_idx = overflow_chunk();
  const size_t len = (static_cast<size_t>(overflow_idx) + 1) * CHUNK_SIZE;
  assert(buffer.size() >= len);

  size_t num_chunks = 0;
  auto write_chunk = [&](uint16_t idx, uint64_t value) {
    std::byte *p = buffer.data() + static_cast<size_t>(idx) * CHUNK_SIZE;
    write_id48_be(p, value);
    assert(num_chunks < chunk_ids.size());
    chunk_ids[num_chunks++] = idx;
  };

  if (has(mask, OverflowField::Incoming)) {
    const size_t n = entry.incoming.size();
    assert(n <= slots.size());
    for (size_t i = 0; i < n; ++i) {
      assert(slots[i].is_valid() && slots[i].value < capacity);
      assert(i == 0 || slots[i].value > slots[i - 1].value);
      write_chunk(incoming_chunk(slots[i].value), entry.incoming[i].value);
    }
  }

  if (has(mask, OverflowField::Overflow))
    write_chunk(overflow_idx, entry.overflow.value);

  Column::Data col_data{reinterpret_cast<const unsigned char *>(buffer.data()),
                        static_cast<uint32_t>(len)};
  return m_overflow.update(mtr, id.column_ref(), col_data,
                           chunk_ids.span(num_chunks), CHUNK_SIZE, err,
                           err_len);
}

bool LevelStore::fetch(MtrCtx::Ref mtr, NID id, bool for_update,
                       NeighbourEntry &entry, size_t &num_valid, char *err,
                       uint32_t err_len, NodeField mask, IncomingFilter filter,
                       const ScratchChunkIds *slots) {
  Column::Data col_data;
  Column::Data rowid_prefix;
  Segment::TrxRef trx_ref;
  bool delete_marked = false;
  if (m_store.fetch(mtr, id.column_ref(), for_update, col_data, rowid_prefix,
                    trx_ref, delete_marked, err, err_len))
    return true;

  const auto *base = reinterpret_cast<const std::byte *>(col_data.data);
  auto read_chunk = [&](uint16_t idx) -> uint64_t {
    const std::byte *p = base + static_cast<size_t>(idx) * CHUNK_SIZE;
    return read_id48_be(p);
  };

  if (has(mask, NodeField::Owner))
    entry.owner = VID{read_chunk(owner_chunk())};

  if (has(mask, NodeField::LowerLevel) && m_level.has_lower_level())
    entry.lower_level = NID{read_chunk(lower_level_chunk())};

  if (has(mask, NodeField::Neighbours)) {
    // for_update: a later update() will write back by slot, so every
    // requested slot -- INVALID and filtered-out ones included -- is
    // written in place to keep entry.neighbours aligned with slots.
    // Read-only: no such alignment is needed, so INVALID and filtered-out
    // slots are dropped and the rest are compacted to the front.
    size_t write_idx = 0;
    size_t valid_count = 0;
    auto handle_slot = [&](uint16_t slot) {
      const uint64_t nid_val = read_chunk(neighbour_nid_chunk(slot));
      const NID nid{nid_val};
      const bool include =
          nid.is_valid() &&
          !(filter == IncomingFilter::ExcludeIncoming && nid.is_incoming());
      if (include) {
        ++valid_count;
      } else if (!for_update) {
        return;
      }
      entry.neighbours[write_idx++] =
          Node{nid, VID{read_chunk(neighbour_vid_chunk(slot))}};
    };

    if (slots != nullptr) {
      assert(slots->size() <= entry.neighbours.size());
      for (size_t i = 0; i < slots->size(); ++i)
        handle_slot((*slots)[i]);
    } else {
      const uint32_t max_n = max_neighbours();
      assert(max_n <= entry.neighbours.size());
      for (uint32_t slot = 0; slot < max_n; ++slot)
        handle_slot(static_cast<uint16_t>(slot));
    }

    if (!for_update)
      entry.neighbours = entry.neighbours.first(write_idx);
    num_valid = valid_count;
  }

  if (has(mask, NodeField::Overflow))
    entry.overflow = NID{read_chunk(neighbour_overflow_chunk())};

  return false;
}

bool LevelStore::fetch(MtrCtx::Ref mtr, NID id, bool for_update,
                       OverflowEntry &entry, size_t &num_valid, char *err,
                       uint32_t err_len, OverflowField mask,
                       const ScratchChunkIds *slots) {
  Column::Data col_data;
  Column::Data rowid_prefix;
  Segment::TrxRef trx_ref;
  bool delete_marked = false;
  if (m_overflow.fetch(mtr, id.column_ref(), for_update, col_data, rowid_prefix,
                       trx_ref, delete_marked, err, err_len))
    return true;

  const auto *base = reinterpret_cast<const std::byte *>(col_data.data);
  auto read_chunk = [&](uint16_t idx) -> uint64_t {
    const std::byte *p = base + static_cast<size_t>(idx) * CHUNK_SIZE;
    return read_id48_be(p);
  };

  if (has(mask, OverflowField::Incoming)) {
    // Same for_update/read-only distinction as the NeighbourEntry overload
    // above: preserve slot alignment for a later update(), or compact away
    // INVALID entries for a read-only lookup.
    size_t write_idx = 0;
    size_t valid_count = 0;
    auto handle_slot = [&](uint16_t slot) {
      const uint64_t nid_val = read_chunk(incoming_chunk(slot));
      const bool valid = NID{nid_val}.is_valid();
      if (valid) {
        ++valid_count;
      } else if (!for_update) {
        return;
      }
      entry.incoming[write_idx++] = NID{nid_val};
    };

    if (slots != nullptr) {
      assert(slots->size() <= entry.incoming.size());
      for (size_t i = 0; i < slots->size(); ++i)
        handle_slot((*slots)[i]);
    } else {
      const uint32_t capacity = overflow_capacity();
      assert(capacity <= entry.incoming.size());
      for (uint32_t slot = 0; slot < capacity; ++slot)
        handle_slot(static_cast<uint16_t>(slot));
    }

    if (!for_update)
      entry.incoming = entry.incoming.first(write_idx);
    num_valid = valid_count;
  }

  if (has(mask, OverflowField::Overflow))
    entry.overflow = NID{read_chunk(overflow_chunk())};

  return false;
}

LevelStore *IndexStore::locate(NID nid, StoreKind &kind, char *err,
                               uint32_t err_len) {
  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();

  uint8_t root_idx = 0;
  bool failed = m_multi_store.get_root_index(mtr, nid.column_ref(), root_idx,
                                             err, err_len);
  mtr_ctx.commit();

  if (failed)
    return nullptr;

  kind = (root_idx & 1) ? StoreKind::Overflow : StoreKind::Neighbour;
  return get_level(LevelStore::LevelId{static_cast<uint8_t>(root_idx >> 1)});
}

LevelStore *IndexStore::ensure_levels(LevelStore::LevelId target, char *err,
                                      uint32_t err_len) {
  assert(target <= max_level());
  assert(m_levels[m_entry_level.value].has_value());

  // Levels above m_entry_level may already exist, wholly or in part: an
  // insert that created them and then failed leaves them behind, since
  // levels are never dropped. Every step below is therefore skipped when its
  // work is already done rather than asserted against. Re-running
  // init_root_page() for a half that exists would allocate a second root
  // page and overwrite the stored ref, orphaning the first page along with
  // any records already written into it.
  for (uint8_t l = m_entry_level.value + 1; l <= target.value; ++l) {
    RootId primary{LevelStore::LevelId{l}, RootId::Type::Primary};
    RootId overflow{LevelStore::LevelId{l}, RootId::Type::Overflow};

    // The two halves are created by separate calls, so a failure (or a crash
    // and restart) can leave a level with only its primary store: each half
    // is checked on its own, not inferred from the other.
    if (!m_multi_store.m_stores[root_index(primary)].initialized() &&
        m_multi_store.init_root_page(segment_index(primary),
                                     root_index(primary), err, err_len))
      return nullptr;
    if (!m_multi_store.m_stores[root_index(overflow)].initialized() &&
        m_multi_store.init_root_page(segment_index(overflow),
                                     root_index(overflow), err, err_len))
      return nullptr;

    if (!m_levels[l].has_value())
      m_levels[l].emplace(
          LevelStore::LevelId{l}, m_multi_store.m_stores[root_index(primary)],
          m_multi_store.m_stores[root_index(overflow)], m_num_neighbours);
  }
  return get_level(target);
}

bool IndexStore::set_entry_point(MtrCtx::Ref mtr, const Node &node,
                                 LevelStore::LevelId level, char *err,
                                 uint32_t err_len) {
  ColumnStore &primary = m_multi_store.m_stores[0];

  StorageMeta meta;
  if (meta.decode(primary.m_metadata)) {
    snprintf(err, err_len,
             "HNSW: set_entry_point: failed to decode level-0 metadata");
    return true;
  }
  meta.entry_level = level;
  meta.entry_points.assign(
      1, node.nid.is_valid()
             ? static_cast<IndexScanKey::KeyPartRef>(node.nid.value)
             : IndexScanKey::EMPTY_REF);

  std::string encoded;
  meta.encode(&encoded);
  if (primary.update_metadata(mtr, encoded, err, err_len))
    return true;

  m_entry_point = node;
  m_entry_level = level;
  return false;
}

uint8_t IndexStore::segment_index(const RootId &root) const {
  return root.level.value == 0 ? static_cast<uint8_t>(SegmentIndex::Primary)
                               : static_cast<uint8_t>(SegmentIndex::Secondary);
}

uint8_t IndexStore::root_index(const RootId &root) const {
  return root.level.value * 2 + (root.type == RootId::Type::Overflow ? 1 : 0);
}

void IndexStore::build_storage_specs(std::vector<Storage_spec> &specs) {
  specs.clear();
  specs.reserve(S_NUM_STORES);
  for (uint8_t l = 0; l < S_MAX_LEVEL; ++l) {
    LevelStore::LevelId level{l};
    std::string meta;

    // Level-0 primary store carries the global entry level and entry point.
    // Always encode one entry point (may be EMPTY_REF on an empty index).
    // All other stores use entry_level{0} and no entry points.
    std::vector<IndexScanKey::KeyPartRef> entry_pts;
    if (l == 0)
      entry_pts.push_back(
          m_entry_point.nid.is_valid()
              ? static_cast<IndexScanKey::KeyPartRef>(m_entry_point.nid.value)
              : IndexScanKey::EMPTY_REF);
    auto el = (l == 0) ? m_entry_level : LevelStore::LevelId{0};
    StorageMeta{("HNSW-L" + std::to_string(l)), level, el, entry_pts}.encode(
        &meta);
    specs.push_back({entry_len(level), std::move(meta)});

    StorageMeta{("HNSW-L" + std::to_string(l) + "-OV"),
                level,
                LevelStore::LevelId{0},
                {}}
        .encode(&meta);
    specs.push_back({overflow_len(level), std::move(meta)});
  }
}

bool IndexStore::create(Space::Ref space_ref, Segment::TrxRef trx_ref,
                        const Options &opts, char *err, uint32_t err_len) {
  m_num_neighbours = opts.M;
  m_ef_construction = opts.ef_construction;
  m_level_norm_factor = 1.0 / std::log(static_cast<double>(opts.M));

  std::vector<Storage_spec> specs;
  build_storage_specs(specs);

  // Formats only the primary (level-0 store) root page. Stores 1..63 are sized
  // and named but their root pages are INVALID_REF until the corresponding
  // level is created.
  if (m_multi_store.create(space_ref, trx_ref, specs, segment_total(), err,
                           err_len))
    return true;

  // Level-0 overflow root page: allocated from segment 0.
  RootId root{LevelStore::LevelId{0}, RootId::Type::Overflow};
  if (m_multi_store.init_root_page(segment_index(root), root_index(root), err,
                                   err_len))
    return true;

  m_levels[0].emplace(LevelStore::LevelId{0}, m_multi_store.m_stores[0],
                      m_multi_store.m_stores[1], opts.M);
  m_initialized = true;
  return false;
}

bool IndexStore::drop(Segment::TrxRef trx_ref, char *err, uint32_t err_len) {
  return m_multi_store.drop(trx_ref, err, err_len);
}

bool IndexStore::load(Index::StorageRef storage_ref, const Options &opts,
                      char *err, uint32_t err_len) {
  m_num_neighbours = opts.M;
  m_ef_construction = opts.ef_construction;
  m_level_norm_factor = 1.0 / std::log(static_cast<double>(opts.M));

  std::vector<Storage_spec> specs;
  build_storage_specs(specs);

  if (m_multi_store.load(storage_ref, specs, err, err_len))
    return true;

  StorageMeta level0_meta;
  if (level0_meta.decode(m_multi_store.m_stores[0].m_metadata)) {
    snprintf(err, err_len, "HNSW: load: failed to decode level-0 metadata");
    return true;
  }
  assert(level0_meta.entry_points.size() <= 1);
  assert(level0_meta.level == LevelStore::LevelId{0});
  m_entry_level = level0_meta.entry_level;
  IndexScanKey::KeyPartRef entry_ref = level0_meta.entry_points.empty()
                                           ? IndexScanKey::EMPTY_REF
                                           : level0_meta.entry_points[0];
  NID entry_nid = (entry_ref != IndexScanKey::EMPTY_REF)
                      ? NID{static_cast<uint64_t>(entry_ref)}
                      : NID{};

  // Reconstruct each level whose stores have been created.
  auto num_root_pages = static_cast<uint8_t>(m_multi_store.m_stores.size());
  for (uint8_t lvl = 0; lvl < S_MAX_LEVEL; ++lvl) {
    uint8_t si = lvl * 2;
    if (si + 1 >= num_root_pages || !m_multi_store.m_stores[si].initialized() ||
        !m_multi_store.m_stores[si + 1].initialized())
      break;

#ifndef NDEBUG
    StorageMeta primary_meta, overflow_meta;
    assert(!primary_meta.decode(m_multi_store.m_stores[si].m_metadata));
    assert(!overflow_meta.decode(m_multi_store.m_stores[si + 1].m_metadata));
    assert(primary_meta.level == LevelStore::LevelId{lvl});
    assert(overflow_meta.level == LevelStore::LevelId{lvl});
#endif // NDEBUG
    m_levels[lvl].emplace(LevelStore::LevelId{lvl}, m_multi_store.m_stores[si],
                          m_multi_store.m_stores[si + 1], m_num_neighbours);
  }

  // Only the entry point's NID is persisted; resolve its VID once here so
  // entry_point() is a plain field read with no page access thereafter.
  m_entry_point = Node{};
  if (entry_nid.is_valid()) {
    LevelStore *store = get_level(m_entry_level);
    assert(store != nullptr);

    NeighbourEntry entry;
    size_t num_valid = 0;
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    bool failed = store->fetch(mtr, entry_nid, /*for_update=*/false, entry,
                               num_valid, err, err_len, NodeField::Owner);
    mtr_ctx.commit();
    if (failed)
      return true;
    m_entry_point = Node{entry_nid, entry.owner};
  }

  m_initialized = true;
  return false;
}

bool create(StorageCtx *ctx, const Index &index, Space::Ref space_ref,
            Segment::TrxRef trx_ref, char *err, uint32_t err_len) {
  const auto *opts = index.options<Options>();
  assert(opts != nullptr);
  auto *store = ctx->user();
  if (store->create(space_ref, trx_ref, *opts, err, err_len))
    return true;
  ctx->set_ref(store->storage_ref());
  return false;
}

bool drop(StorageCtx *ctx, const Index & /*index*/, Segment::TrxRef trx_ref,
          char *err, uint32_t err_len) {
  return ctx->user()->drop(trx_ref, err, err_len);
}

bool load(StorageCtx *ctx, const Index &index, Index::StorageRef storage_ref,
          char *err, uint32_t err_len) {
  const auto *opts = index.options<Options>();
  assert(opts != nullptr);
  return ctx->user()->load(storage_ref, *opts, err, err_len);
}

bool insert(StorageCtx * /*ctx*/, const Index & /*index*/,
            Segment::TrxRef /*trx_ref*/,
            IndexScanKey::KeyPartData * /*key_columns*/,
            IndexScanKey::KeyPartData * /*pkey_columns*/,
            IndexScanKey::KeyPartRef * /*key_ref*/, char * /*err*/,
            uint32_t /*err_len*/) {
  return false;
}

bool mark_delete(StorageCtx * /*ctx*/, const Index & /*index*/,
                 Segment::TrxRef /*trx_ref*/,
                 IndexScanKey::KeyPartRef * /*key_ref*/,
                 IndexScanKey::KeyPartData * /*key_columns*/,
                 IndexScanKey::KeyPartData * /*pkey_columns*/,
                 bool /*delete_mark*/, char * /*err*/, uint32_t /*err_len*/) {
  return false;
}

bool purge(StorageCtx * /*ctx*/, const Index & /*index*/,
           Segment::TrxRef /*trx_ref*/, IndexScanKey::KeyPartRef * /*key_ref*/,
           IndexScanKey::KeyPartData * /*key_columns*/,
           IndexScanKey::KeyPartData * /*pkey_columns*/, char * /*err*/,
           uint32_t /*err_len*/) {
  return false;
}

bool begin(StorageCtx * /*ctx*/, const Index & /*index*/, MtrCtx::Ref /*mctx*/,
           const IndexScanDesc & /*scan_desc*/, Index::Cursor *cursor,
           bool *eof, char * /*err*/, uint32_t /*err_len*/) {
  *cursor = new Cursor{};
  *eof = true;
  return false;
}

bool position(Index::Cursor /*cursor*/, Index::CursorOp /*op*/, bool *eof,
              char * /*err*/, uint32_t /*err_len*/) {
  *eof = true;
  return false;
}

bool fetch(Index::Cursor /*cursor*/, IndexScanKey::KeyPartRef * /*key_ref*/,
           IndexScanKey::KeyPartData * /*key_columns*/,
           IndexScanKey::KeyPartData * /*pkey_columns*/, char * /*err*/,
           uint32_t /*err_len*/) {
  return false;
}

bool save(Index::Cursor /*cursor*/, char * /*err*/, uint32_t /*err_len*/) {
  return false;
}

bool restore(Index::Cursor /*cursor*/, MtrCtx::Ref /*mctx*/, bool *eof,
             char * /*err*/, uint32_t /*err_len*/) {
  *eof = true;
  return false;
}

void end(Index::Cursor *cursor) {
  delete static_cast<Cursor *>(*cursor);
  *cursor = nullptr;
}

} // namespace svector::hnsw
