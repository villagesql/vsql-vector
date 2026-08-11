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

#include "index.h"

#include <array>
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
      if (!parse_u32(val, &out->M)) {
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

// TODO(villagesql-indexing): To be implemented based on Index Node
// storage format.
uint16_t IndexStore::entry_len(LevelStore::LevelId /* level */) const {
  return 32;
}
uint16_t IndexStore::overflow_len(LevelStore::LevelId /* level */) const {
  return 32;
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
      entry_pts.push_back(m_entry_point);
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
  m_entry_point = level0_meta.entry_points.empty()
                      ? IndexScanKey::EMPTY_REF
                      : level0_meta.entry_points[0];

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
