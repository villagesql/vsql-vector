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

#include <cmath>
#include <random>

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
                      /*max_update_slots=*/mmax0,
                      LevelStore::max_update_chunks(M), err);
}

} // namespace

#ifndef NDEBUG
bool IndexGraph::debug_check_level(const Node &node, LevelId level) const {
  // Nothing here is reportable -- the assert at every call site aborts
  // instead -- so locate() is given no error buffer rather than the caller's,
  // which a debug-only check has no business overwriting.
  StoreKind kind;
  LevelStore *located =
      m_store.locate(node.nid, kind, /*err=*/nullptr, /*err_len=*/0);
  // Separate assert so an abort names the cause: locate() failing to resolve
  // the NID at all, rather than resolving it to a different level.
  assert(located != nullptr);
  return located != nullptr && located->level() == level;
}
#endif // NDEBUG

IndexGraph::IndexGraph(IndexStore &store, Index &index, Segment::TrxRef trx_ref,
                       size_t vector_buf_size, std::span<char> err)
    : m_store(store), m_index(index), m_trx_ref(trx_ref),
      m_ctx(make_graph_context(store, vector_buf_size, err)) {}

bool IndexGraph::distance(const Node & /*a*/, const Node & /*b*/,
                          DistanceType &out) {
  out = 0.0;
  return false;
}

bool IndexGraph::distance(const NodeData & /*a*/, const Node & /*b*/,
                          DistanceType &out) {
  out = 0.0;
  return false;
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
  size_t num_valid = 0;
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

bool IndexGraph::ensure_levels(LevelId level) {
  return m_store.ensure_levels(level, err(), err_len()) == nullptr;
}

bool IndexGraph::create_node(const std::optional<Node> &parent, LevelId level,
                             const NodeData &data,
                             std::vector<Node> &neighbours, Node &out) {
  // Resolved up front, before opening the mtr below, so a failure here
  // never touches storage.
  IndexScanKey::KeyPartRef owner_ref;
  if (m_index.get_key_ref(data.data, &owner_ref))
    return true;

  // ~MtrCtx() commits, and commit() is idempotent, so an early return below
  // needs nothing beyond the return itself; the explicit commit at the end
  // marks the success path.
  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();

  // Every level this insert touches was materialized by
  // GraphOperations::insert() before it locked them, so get_level() below
  // always resolves.
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
    size_t num_valid = 0;
    if (parent_store->fetch(mtr, parent->nid, /*for_update=*/true, parent_check,
                            num_valid, err(), err_len(), NodeField::LowerLevel))
      return true;
    // The node one level down doesn't exist yet -- this call is the one
    // that creates it, below.
    assert(!parent_check.lower_level.is_valid());
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
    return true;

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
      return true;
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
    size_t num_valid = 0;
    bool failed =
        store->fetch(mtr, node.nid, /*for_update=*/false, entry, num_valid,
                     err(), err_len(), NodeField::Overflow);
    mtr_ctx.commit();
    if (failed)
      return true;
    head = entry.overflow;
  }

  // Walk the chain, collecting (overflow NID, prev NID) links on a stack
  // -- one fetch, in its own mtr, per chain entry -- so the loop below can
  // unwind them tail-to-head.
  std::stack<OverflowLink> links;
  if (head.is_valid())
    links.push({head, node.nid, StoreKind::Neighbour});

  for (NID cur = head; cur.is_valid();) {
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    OverflowEntry entry;
    size_t num_valid = 0;
    bool failed = store->fetch(mtr, cur, /*for_update=*/false, entry, num_valid,
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
  // dropping it and clearing its prev's Overflow field is always correct.
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
  assert(debug_check_level(Node{link.prev, VID{}}, level));

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();

  if (link.prev_kind == StoreKind::Neighbour) {
    NeighbourEntry prev_entry;
    prev_entry.overflow = NID{};
    if (store->update(mtr, link.prev, prev_entry, NodeField::Overflow,
                      m_ctx.m_update_slots, m_ctx.m_neighbour_buf,
                      m_ctx.m_chunk_ids, err(), err_len()))
      return true;
  } else {
    OverflowEntry prev_entry;
    prev_entry.overflow = NID{};
    if (store->update(mtr, link.prev, prev_entry, OverflowField::Overflow,
                      m_ctx.m_update_slots, m_ctx.m_overflow_buf,
                      m_ctx.m_chunk_ids, err(), err_len()))
      return true;
  }

  if (store->remove(mtr, StoreKind::Overflow, link.nid, m_trx_ref, err(),
                    err_len()))
    return true;

  mtr_ctx.commit();
  return false;
}

bool IndexGraph::drop_node(const std::optional<Node> &parent, LevelId level,
                           const Node &node) {
  if (drop_overflow_nodes(level, node))
    return true;

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();

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
      return true;
  }

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);

  if (store->remove(mtr, StoreKind::Neighbour, node.nid, m_trx_ref, err(),
                    err_len()))
    return true;

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
  size_t num_valid = 0;
  bool failed =
      store->fetch(mtr, node.nid, /*for_update=*/false, entry, num_valid, err(),
                   err_len(), NodeField::LowerLevel);
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

bool IndexGraph::link_neighbours(const Node & /*node*/, LevelId /*level*/,
                                 std::vector<Node> &out) {
  out.clear();
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
