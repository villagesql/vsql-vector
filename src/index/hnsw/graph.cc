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

#include "graph.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include <villagesql/abi/types.h>  // vef_vdf_result_t, VEF_RESULT_*

namespace svector::hnsw {

namespace {

// Level 0 has the largest Mmax (2*M, vs. M above it) and every level's
// overflow capacity is the same regardless of level, so level 0 sizes the
// worst case for both buffers.
GraphContext make_graph_context(IndexStore &store, size_t vector_buf_size,
                                std::span<char> err) {
  const uint32_t M = store.num_neighbours();
  const uint32_t mmax0 = LevelStore::max_neighbours(LevelStore::LevelId{0}, M);
  const uint32_t overflow_capacity =
      LevelStore::overflow_capacity(LevelStore::LevelId{0}, M);
  // Level 0 has no lower level, so its NeighbourEntry omits that field.
  const size_t neighbour_buf_size =
      NeighbourEntry::storage_size(mmax0, /*has_lower_level=*/false);
  const size_t overflow_buf_size =
      OverflowEntry::storage_size(overflow_capacity);
  return GraphContext(neighbour_buf_size, overflow_buf_size, vector_buf_size,
                      /*max_update_slots=*/mmax0, err);
}

} // namespace

#ifndef NDEBUG
bool IndexGraph::debug_check_level(const Node &node, LevelId level) const {
  StoreKind kind;
  LevelStore *located = m_store.locate(node.nid, kind, err(), err_len());
  return located != nullptr && located->level() == level;
}
#endif // NDEBUG

IndexGraph::IndexGraph(IndexStore &store, Index &index, Segment::TrxRef trx_ref,
                       size_t vector_buf_size, std::span<char> err)
    : m_store(store), m_index(index), m_trx_ref(trx_ref),
      m_ctx(make_graph_context(store, vector_buf_size, err)) {}

namespace {

// fn_id the distance helper is registered under for every HNSW profile
// (vector.cc: .with_helper(1, ...)); the single key column is at key_pos 0.
constexpr uint32_t kDistanceHelperFnId = 1;
constexpr uint32_t kKeyPos = 0;

} // namespace

bool IndexGraph::fetch_vector(const Node &node, ScratchBytes &scratch,
                              IndexScanKey::KeyPartData &out) {
  if (!node.vid.is_valid()) {
    snprintf(err(), err_len(), "HNSW: fetch_vector: node has no VID");
    return true;
  }

  // A VID's value is a plain Column::Ref (VIDs never carry the NID incoming
  // flag), so it maps directly to a KeyPartRef.
  const IndexScanKey::KeyPartRef ref =
      static_cast<IndexScanKey::KeyPartRef>(node.vid.value);
  IndexScanKey::KeyPartData raw;
  if (m_index.get_key_data(ref, &raw)) {
    // get_key_data writes to the SDK's thread-local error buffer; surface a
    // stable message into this call's err span.
    snprintf(err(), err_len(),
             "HNSW: fetch_vector: failed to resolve VID to vector data: %s",
             m_index.get_error());
    return true;
  }

  // The SVECTOR column store persists only the float payload (it strips the
  // 8-byte storage_ref prefix, storage.cc). The distance VDF, however, consumes
  // a full persisted SVECTOR value -- [8-byte prefix][floats] -- and skips the
  // prefix (vector.cc). Reconstruct that layout into caller scratch so the
  // resolved node vector is a valid VDF operand, matching the query key (whose
  // prefix is likewise zero-filled at FROM_STRING time). Also copies the data
  // out of the server's per-thread resolve buffer, so two fetch_vector() calls
  // into distinct scratch buffers can be held live at once.
  constexpr size_t kPrefix = sizeof(vef_storage_ref_t);
  const size_t total = kPrefix + raw.length;
  assert(total <= scratch.size());
  std::memset(scratch.data(), 0, kPrefix);
  std::memcpy(scratch.data() + kPrefix, raw.data, raw.length);
  out.data = reinterpret_cast<const unsigned char *>(scratch.data());
  out.length = static_cast<uint32_t>(total);
  return false;
}

bool IndexGraph::distance_bytes(const IndexScanKey::KeyPartData &a,
                                const IndexScanKey::KeyPartData &b,
                                DistanceType &out) {
  // The server's profile-helper dispatcher writes the REAL distance back as a
  // bare double (call_profile_binding: `*static_cast<double*>(result) =
  // vdf_result.real_value`), NOT a vef_vdf_result_t. So the result out-param is
  // a double*. Per-call VDF errors are surfaced only in the server log, not
  // through this channel.
  double result = 0.0;
  m_index.helper<double>(kKeyPos, kDistanceHelperFnId, &result, a, b);
  out = static_cast<DistanceType>(result);
  return false;
}

bool IndexGraph::distance(const Node &a, const Node &b, DistanceType &out) {
  // Both node vectors must be live at once. fetch_vector() writes each into a
  // distinct scratch buffer (buf_1, buf_2), so neither the server's per-thread
  // resolve buffer nor the other fetch clobbers it.
  IndexScanKey::KeyPartData a_bytes;
  IndexScanKey::KeyPartData b_bytes;
  if (fetch_vector(a, m_ctx.m_vector_buf_1, a_bytes) ||
      fetch_vector(b, m_ctx.m_vector_buf_2, b_bytes))
    return true;
  return distance_bytes(a_bytes, b_bytes, out);
}

bool IndexGraph::distance(const NodeData &a, const Node &b, DistanceType &out) {
  // a is a not-yet-inserted query/insert element whose bytes the caller already
  // holds (NodeData wraps the raw key column, already in [prefix][floats] form).
  // Only b needs resolving -- this is the hot path for search descent and
  // insertion.
  IndexScanKey::KeyPartData b_bytes;
  if (fetch_vector(b, m_ctx.m_vector_buf_1, b_bytes))
    return true;
  return distance_bytes(a.data, b_bytes, out);
}

bool IndexGraph::neighbours(const Node &node, LevelId level,
                            std::vector<Node> &out) {
  out.clear();

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();

  out.resize(store->max_neighbours());
  NeighbourEntry entry;
  entry.neighbours = out;
  size_t num_valid;
  bool failed =
      store->fetch(mtr, node.nid, /*for_update=*/false, entry, num_valid, err(),
                   err_len(), NodeField::Neighbours);
  mtr_ctx.commit();
  if (failed) {
    out.clear();
    return true;
  }

  out.resize(num_valid);
  return false;
}

// TODO(villagesql-indexing): Implement MVCC visibility check for the
// querying transaction instead of treating every node as visible.
bool IndexGraph::visible(const Node & /*node*/, bool &out) {
  out = true;
  return false;
}

IndexGraph::LevelId IndexGraph::get_insert_level() {
  thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());
  // uniform_real_distribution's range is half-open [0, 1); flip it to
  // (0, 1] so unif() is never exactly 0, which would make -ln(unif)
  // diverge below.
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double unif = 1.0 - dist(gen);

  // Algorithm 1, line 4:
  // l <- floor(-ln(unif(0,1)) * mL)
  // where mL = 1 / ln(M)
  double level = std::floor(-std::log(unif) * m_store.level_norm_factor());

  double max_level = static_cast<double>(IndexStore::max_level().value);
  level = std::min(level, max_level);
  return LevelId{static_cast<uint8_t>(level)};
}

bool IndexGraph::get_entry_point(std::vector<Node> &out, LevelId &out_level) {
  out.clear();
  const Node &entry = m_store.entry_point();
  if (!entry.nid.is_valid())
    return false;

  out.push_back(entry);
  out_level = m_store.entry_level();
  return false;
}

bool IndexGraph::set_entry_point(const std::vector<Node> &nodes,
                                 LevelId level) {
  assert(nodes.size() <= 1);
  Node node = nodes.empty() ? Node{} : nodes.front();

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();
  bool failed = m_store.set_entry_point(mtr, node, level, err(), err_len());
  mtr_ctx.commit();
  return failed;
}

// Reads the 8-byte big-endian column reference InnoDB stores in the extended
// vector field's prefix (mach_write_to_8, mirrored by mach_read_from_8).
static IndexScanKey::KeyPartRef read_col_ref_be(const unsigned char *p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v = (v << 8) | static_cast<uint64_t>(p[i]);
  return static_cast<IndexScanKey::KeyPartRef>(v);
}

bool IndexGraph::create_node(const std::optional<Node> &parent, LevelId level,
                             const NodeData &data,
                             std::vector<Node> &neighbours, Node &out) {
  // The extension's node reference (VID) is the SVECTOR column store's stable
  // col_ref for this vector. That ref cannot be derived from the vector bytes
  // (it is an opaque storage address produced only at column insert), so we
  // read it from the extended vector field's 8-byte prefix, which InnoDB has
  // already populated by the time the custom index is maintained (the clustered
  // index inserts first and writes the ref into the shared row buffer this
  // field aliases). data.data is laid out [8-byte col_ref][vector floats].
  if (data.data.length < sizeof(vef_storage_ref_t)) {
    snprintf(err(), err_len(),
             "HNSW: create_node: vector field too short to hold a column ref");
    return true;
  }
  const IndexScanKey::KeyPartRef owner_ref = read_col_ref_be(data.data.data);

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();
  auto on_error = [&] {
    mtr_ctx.commit();
    return true;
  };

  LevelStore *parent_store = nullptr;
  if (parent) {
    // Select parent node for update from upper level. Always exactly one
    // level above the current one -- GraphOperations::insert() descends one
    // level per create_node() call, and parent is the node it created at
    // the level just above.
    LevelId parent_level{static_cast<uint8_t>(level.value + 1)};
    parent_store = m_store.get_level(parent_level);
    assert(parent_store != nullptr);

    assert(debug_check_level(*parent, parent_level));

    // Fetch (and latch) the parent's record before touching anything at the
    // current level, so mtr always acquires page latches top-down and never
    // risks a lock-order inversion against a concurrent top-down search.
    NeighbourEntry parent_check;
    // Unused: only meaningful when mask includes NodeField::Neighbours.
    size_t unused_num_valid;
    if (parent_store->fetch(mtr, parent->nid, /*for_update=*/true, parent_check,
                            unused_num_valid, err(), err_len(),
                            NodeField::LowerLevel))
      return on_error();
    // The node one level down doesn't exist yet -- this call is the one
    // that creates it, below.
    assert(!parent_check.lower_level.is_valid());
  } else {
    // Create levels if needed. Only required on the first (topmost) call
    // for this insert -- every lower level visited afterwards already
    // exists once the topmost one does.
    if (m_store.ensure_levels(level, err(), err_len()) == nullptr)
      return on_error();
  }

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);

  // Insert node data at current level. lower_level is left invalid here --
  // the node one level down doesn't exist yet; once it does, the next
  // create_node() call for this insert fills it in via the "update parent's
  // LowerLevel NID" step below. A level 0 node has no lower_level field at
  // all, so this never applies there.
  NeighbourEntry entry;
  entry.owner = VID{owner_ref};
  entry.neighbours = neighbours;
  NID new_nid;
  if (store->insert(mtr, entry, m_trx_ref, m_ctx.m_neighbour_buf, new_nid,
                    err(), err_len()))
    return on_error();

  out.nid = new_nid;
  out.vid = entry.owner;

  // Update parent's LowerLevel NID, now that this node's NID is known.
  if (parent) {
    NeighbourEntry parent_entry;
    parent_entry.lower_level = new_nid;
    if (parent_store->update(mtr, parent->nid, parent_entry,
                             NodeField::LowerLevel, m_ctx.m_update_slots,
                             m_ctx.m_neighbour_buf, m_ctx.m_chunk_ids, err(),
                             err_len()))
      return on_error();
  }

  mtr_ctx.commit();
  return false;
}

bool IndexGraph::drop_overflow_nodes(LevelId level, const Node &node) {
  assert(debug_check_level(node, level));

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);

  // Read node's own Overflow field first, before the chain-walking loop
  // below, since it lives in a NeighbourEntry rather than an OverflowEntry.
  NID head;
  {
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    NeighbourEntry entry;
    size_t unused_num_valid;
    bool failed =
        store->fetch(mtr, node.nid, /*for_update=*/false, entry,
                     unused_num_valid, err(), err_len(), NodeField::Overflow);
    mtr_ctx.commit();
    if (failed)
      return true;
    head = entry.overflow;
  }

  // Walk the chain, collecting (overflow NID, parent NID) links on a stack
  // -- one fetch, in its own mtr, per chain entry -- so the loop below can
  // unwind them tail-to-head.
  std::stack<OverflowLink> links;
  if (head.is_valid())
    links.push({head, node.nid, StoreKind::Neighbour});

  for (NID cur = head; cur.is_valid();) {
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    OverflowEntry entry;
    size_t unused_num_valid;
    bool failed =
        store->fetch(mtr, cur, /*for_update=*/false, entry, unused_num_valid,
                     err(), err_len(), OverflowField::Overflow);
    mtr_ctx.commit();
    if (failed)
      return true;

    if (entry.overflow.is_valid())
      links.push({entry.overflow, cur, StoreKind::Overflow});
    cur = entry.overflow;
  }

  // Unwind tail-to-head: each popped link's nid is, at the time it's
  // processed, the last remaining entry in what's left of the chain, so
  // dropping it and pointing its parent at NID{} is always correct.
  while (!links.empty()) {
    if (drop_overflow_node(level, links.top()))
      return true;
    links.pop();
  }
  return false;
}

bool IndexGraph::drop_overflow_node(LevelId level, const OverflowLink &link) {
  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);
  assert(debug_check_level(Node{link.nid, VID{}}, level));
  assert(debug_check_level(Node{link.parent, VID{}}, level));

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();
  auto on_error = [&] {
    mtr_ctx.commit();
    return true;
  };

  if (link.parent_kind == StoreKind::Neighbour) {
    NeighbourEntry parent_entry;
    parent_entry.overflow = NID{};
    if (store->update(mtr, link.parent, parent_entry, NodeField::Overflow,
                      m_ctx.m_update_slots, m_ctx.m_neighbour_buf,
                      m_ctx.m_chunk_ids, err(), err_len()))
      return on_error();
  } else {
    OverflowEntry parent_entry;
    parent_entry.overflow = NID{};
    if (store->update(mtr, link.parent, parent_entry, OverflowField::Overflow,
                      m_ctx.m_update_slots, m_ctx.m_overflow_buf,
                      m_ctx.m_chunk_ids, err(), err_len()))
      return on_error();
  }

  if (store->remove(mtr, StoreKind::Overflow, link.nid, m_trx_ref, err(),
                    err_len()))
    return on_error();

  mtr_ctx.commit();
  return false;
}

bool IndexGraph::drop_node(const std::optional<Node> &parent, LevelId level,
                           const Node &node) {
  if (drop_overflow_nodes(level, node))
    return true;

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();
  auto on_error = [&] {
    mtr_ctx.commit();
    return true;
  };

  if (parent) {
    // Parent is always exactly one level above the current one -- mirrors
    // create_node()'s parent/level relationship. Unlike create_node(), no
    // fetch is needed first: LowerLevel is being cleared unconditionally,
    // not spliced into a value that depends on the record's prior state.
    LevelId parent_level{static_cast<uint8_t>(level.value + 1)};
    LevelStore *parent_store = m_store.get_level(parent_level);
    assert(parent_store != nullptr);
    assert(debug_check_level(*parent, parent_level));

    NeighbourEntry parent_entry;
    parent_entry.lower_level = NID{};
    if (parent_store->update(mtr, parent->nid, parent_entry,
                             NodeField::LowerLevel, m_ctx.m_update_slots,
                             m_ctx.m_neighbour_buf, m_ctx.m_chunk_ids, err(),
                             err_len()))
      return on_error();
  }

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);

  if (store->remove(mtr, StoreKind::Neighbour, node.nid, m_trx_ref, err(),
                    err_len()))
    return on_error();

  mtr_ctx.commit();
  return false;
}

bool IndexGraph::get_next_level_node(const Node &node, LevelId level,
                                     Node &out) {
  assert(level.has_lower_level());
  assert(debug_check_level(node, level));

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();

  NeighbourEntry entry;
  size_t unused_num_valid;
  bool failed =
      store->fetch(mtr, node.nid, /*for_update=*/false, entry, unused_num_valid,
                   err(), err_len(), NodeField::LowerLevel);
  mtr_ctx.commit();
  if (failed) {
    out = Node{};
    return true;
  }

  assert(entry.lower_level.is_valid());
  out = Node{entry.lower_level, node.vid};
  assert(debug_check_level(out, level.lower()));
  return false;
}

bool IndexGraph::link_neighbours(const Node &node, LevelId level,
                                 std::vector<Node> &out) {
  out.clear();

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);

  // node's forward edges were just written by create_node(); read them back to
  // learn which neighbours need a reciprocal (back) edge to node.
  std::vector<Node> fwd(store->max_neighbours());
  {
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    NeighbourEntry entry;
    entry.neighbours = fwd;
    size_t num_valid = 0;
    bool failed =
        store->fetch(mtr, node.nid, /*for_update=*/false, entry, num_valid,
                     err(), err_len(), NodeField::Neighbours);
    mtr_ctx.commit();
    if (failed)
      return true;
    fwd.resize(num_valid);
  }

  const uint32_t max_n = store->max_neighbours();
  for (const Node &nb : fwd) {
    // Fetch nb's neighbour list slot-aligned (for_update): every slot,
    // INVALID included, is read in place so slot indices line up for the
    // single-slot update() below.
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    auto on_error = [&] {
      mtr_ctx.commit();
      return true;
    };

    std::vector<Node> nb_slots(max_n);
    NeighbourEntry nb_entry;
    nb_entry.neighbours = nb_slots;
    size_t nb_valid = 0;
    if (store->fetch(mtr, nb.nid, /*for_update=*/true, nb_entry, nb_valid, err(),
                     err_len(), NodeField::Neighbours, IncomingFilter::All))
      return on_error();

    // Find the first free (INVALID) slot. MINIMAL scope: if nb's neighbour
    // list is full, skip it -- no reciprocal edge is added.
    // TODO(villagesql-indexing): on a full list, withhold the edge and append
    // nb to `out` so GraphOperations::shrink_neighbours() reselects nb's full
    // neighbour set (Algorithm 1, lines 14-15). Overflow-chain handling is
    // likewise deferred.
    uint16_t free_slot = 0;
    bool found = false;
    for (uint16_t slot = 0; slot < max_n; ++slot) {
      if (!nb_entry.neighbours[slot].nid.is_valid()) {
        free_slot = slot;
        found = true;
        break;
      }
    }
    if (!found) {
      // nb is full; minimal scope skips it (see TODO above).
      mtr_ctx.commit();
      continue;
    }

    // Write the reciprocal edge (nb -> node) into the free slot as a plain
    // neighbour (no incoming flag): LayerOperations::search traverses via
    // IndexGraph::neighbours(), whose default IncomingFilter::ExcludeIncoming
    // would hide an incoming-flagged edge, so a search-traversable back-edge
    // must be stored unflagged -- symmetric with create_node()'s forward edges.
    std::array<Node, 1> one{node};
    NeighbourEntry upd;
    upd.neighbours = std::span<Node>(one);
    m_ctx.m_update_slots[0] = SlotIndex{free_slot};
    if (store->update(mtr, nb.nid, upd, NodeField::Neighbours,
                      m_ctx.m_update_slots, m_ctx.m_neighbour_buf,
                      m_ctx.m_chunk_ids, err(), err_len()))
      return on_error();

    mtr_ctx.commit();
  }

  return false;
}

bool IndexGraph::replace_neighbours(const Node & /*node*/, LevelId /*level*/,
                                    const std::vector<Node> & /*neighbours*/) {
  return false;
}

bool IndexGraph::unlink_neighbours(const Node & /*node*/, LevelId /*level*/,
                                   UnlinkOrphans /*orphans*/,
                                   std::vector<Node> &out) {
  out.clear();
  return false;
}

uint32_t IndexGraph::M() const { return m_store.num_neighbours(); }
uint32_t IndexGraph::ef_construction() const {
  return m_store.ef_construction();
}
uint32_t IndexGraph::Mmax(LevelId level) const {
  auto *store = m_store.get_level(level);
  assert(store != nullptr);
  return store->max_neighbours();
}

// Lock primitives backing LockGraph/LockLevels below.
void IndexGraph::lock_graph(LockMode mode) {
  if (mode == LockMode::Shared)
    m_store.mutex().lock_shared();
  else
    m_store.mutex().lock();
}

void IndexGraph::unlock_graph(LockMode mode) {
  if (mode == LockMode::Shared)
    m_store.mutex().unlock_shared();
  else
    m_store.mutex().unlock();
}

void IndexGraph::lock_level(LevelId level, LockMode mode) {
  auto *store = m_store.get_level(level);
  // TEMPORARY(villagesql-indexing): create-on-demand stopgap for the
  // level-lock-before-create ordering bug. GraphOperations::insert() constructs
  // LockLevels(start = max(entry_level, insert_level)) before create_node()/
  // ensure_levels() materializes a new top level, so this locks a level whose
  // store does not exist yet (see unittest/graph_ops_test.cc repro). The proper
  // fix is to ensure_levels(insert_level) in GraphOperations::insert() before
  // constructing LockLevels, within its S->X upgrade dance -- Deb's concurrency
  // model to own. Until then, materialize the level here. This is only ever
  // reached on the new-top-level path, where insert() has already upgraded the
  // graph lock to X (the mode ensure_levels() requires); the greedy descent
  // never locks a non-existent level.
  if (store == nullptr) {
    store = m_store.ensure_levels(level, err(), err_len());
  }
  assert(store != nullptr);
  if (mode == LockMode::Shared)
    store->mutex().lock_shared();
  else
    store->mutex().lock();
}

void IndexGraph::unlock_level(LevelId level, LockMode mode) {
  auto *store = m_store.get_level(level);
  assert(store != nullptr);
  if (mode == LockMode::Shared)
    store->mutex().unlock_shared();
  else
    store->mutex().unlock();
}

IndexGraph::LockGraph::LockGraph(IndexGraph &graph, LockMode mode)
    : m_graph(graph), m_mode(mode) {
  m_graph.lock_graph(m_mode);
}

void IndexGraph::LockGraph::relock(LockMode mode) {
  if (m_mode == mode) {
    return;
  }
  m_graph.unlock_graph(m_mode);
  m_graph.lock_graph(mode);
  m_mode = mode;
}

IndexGraph::LockGraph::~LockGraph() { m_graph.unlock_graph(m_mode); }

IndexGraph::LockLevels::LockLevels(IndexGraph &graph, LockMode mode,
                                   LevelId start, DescendPolicy policy)
    : m_graph(graph), m_mode(mode), m_policy(policy) {
  m_graph.lock_level(start, m_mode);
  m_stack.push(start);
}

IndexGraph::LockLevels::~LockLevels() {
  // Release level locks in LIFO order.
  while (!m_stack.empty()) {
    m_graph.unlock_level(m_stack.top(), m_mode);
    m_stack.pop();
  }
}

IndexGraph::LevelId IndexGraph::LockLevels::descend() {
  auto upper_level = level();
  if (!upper_level.has_lower_level()) {
    return upper_level;
  }
  auto lower_level = upper_level.lower();
  // Lock the lower level before releasing the current one (if any), so
  // this guard never momentarily holds zero level locks.
  m_graph.lock_level(lower_level, m_mode);

  if (m_policy == DescendPolicy::Release) {
    m_graph.unlock_level(upper_level, m_mode);
    m_stack.pop();
  }
  m_stack.push(lower_level);
  return lower_level;
}

IndexGraph::LevelId IndexGraph::LockLevels::level() const {
  assert(!m_stack.empty());
  return m_stack.top();
}

void IndexGraph::LockLevels::update_policy(DescendPolicy policy) {
  m_policy = policy;
}

} // namespace svector::hnsw
