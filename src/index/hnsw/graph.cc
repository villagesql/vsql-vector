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

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <random>
#include <shared_mutex>

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

IndexGraph::IndexGraph(IndexStore &store, const Index &index,
                       Segment::TrxRef trx_ref, size_t vector_buf_size,
                       std::span<char> err)
    : m_store(store), m_index(index), m_trx_ref(trx_ref),
      m_ctx(make_graph_context(store, vector_buf_size, err)) {}

bool IndexGraph::resolve_node_data(VID vid, ScratchBytes &buf, NodeData &out) {
  assert(vid.is_valid());

  // The server fills the vector data into the buffer it is handed, so out
  // arrives pointing at buf, sized from the indexed column's maximum length --
  // which every stored vector fits in by construction.
  out.data.data = reinterpret_cast<const unsigned char *>(buf.data());
  out.data.length = static_cast<uint32_t>(buf.size());
  return m_index.get_key_data(static_cast<IndexScanKey::KeyPartRef>(vid.value),
                              &out.data);
}

bool IndexGraph::distance(const NodeData &a, const NodeData &b,
                          DistanceType &out) {
  // The helper rather than the profile function proper: it is the
  // index-internal variant of the same distance, which for L2 is the squared
  // one -- it orders nodes identically while skipping the sqrt.
  //
  // Profile functions are infallible, so there is no error to report past the
  // operand resolution the callers below do.
  m_index.helper(VECTOR_KEY_POS, DISTANCE_HELPER_FN_ID, &out, a.data, b.data);
  return false;
}

bool IndexGraph::distance(const Node &a, const Node &b, DistanceType &out) {
  NodeData a_data;
  NodeData b_data;
  if (resolve_node_data(a.vid, m_ctx.m_vector_buf_1, a_data) ||
      resolve_node_data(b.vid, m_ctx.m_vector_buf_2, b_data))
    return true;

  return distance(a_data, b_data, out);
}

bool IndexGraph::distance(const NodeData &a, const Node &b, DistanceType &out) {
  // a is the caller's own vector -- the one being inserted or queried for,
  // which is not in the graph and so has no vid to resolve. Only b needs a
  // buffer, and it takes the second one so the roles of the two never shift
  // between the overloads.
  NodeData b_data;
  if (resolve_node_data(b.vid, m_ctx.m_vector_buf_2, b_data))
    return true;

  return distance(a, b_data, out);
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
  if (store->fetch(mtr, node.nid, /*for_update=*/false, entry, num_valid,
                   get_err_buffer(), get_err_buffer_len(),
                   NodeField::Neighbours)) {
    out.clear();
    return true;
  }

  out.resize(num_valid);
  mtr_ctx.commit();
  return false;
}

bool IndexGraph::resolve_node(LevelId level, NID nid, Node &out) {
  assert(nid.is_valid());
  assert(!nid.is_incoming());

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();
  NeighbourEntry entry;
  size_t num_valid;
  if (store->fetch(mtr, nid, /*for_update=*/false, entry, num_valid,
                   get_err_buffer(), get_err_buffer_len(), NodeField::Owner)) {
    out = Node{};
    return true;
  }

  out = Node{nid, entry.owner};
  mtr_ctx.commit();
  return false;
}

bool IndexGraph::incoming_neighbours(const Node &node, LevelId level,
                                     std::vector<Node> &out) {
  out.clear();
  assert(debug_check_level(node, level));

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);
  const uint32_t max_n = store->max_neighbours();
  const uint32_t capacity = store->overflow_capacity();
  assert(capacity <= m_ctx.m_incoming_buf_1.size());

  // Node's own list first. Read-only, so IncomingFilter::All compacts it to
  // the occupied slots alone -- no on-disk slot has to be tracked here, only
  // which of those links are incoming ones.
  NeighbourEntry entry;
  entry.neighbours = m_ctx.m_node_buf_1.span(max_n);
  size_t num_valid;
  {
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    if (store->fetch(mtr, node.nid, /*for_update=*/false, entry, num_valid,
                     get_err_buffer(), get_err_buffer_len(),
                     NodeField::Neighbours | NodeField::Overflow,
                     IncomingFilter::All))
      return true;
  }

  for (const Node &link : entry.neighbours) {
    if (!link.nid.is_incoming())
      continue;
    out.push_back(Node{link.nid.clear_incoming(), link.vid});
  }

  // Then the overflow chain, which holds incoming links exclusively, one
  // chain entry per mtr (as in drop_overflow_nodes()).
  for (NID cur = entry.overflow; cur.is_valid();) {
    OverflowEntry overflow_entry;
    overflow_entry.incoming = m_ctx.m_incoming_buf_1.span(capacity);
    size_t num_valid;
    {
      MtrCtx mtr_ctx;
      auto mtr = mtr_ctx.start();
      if (store->fetch(mtr, cur, /*for_update=*/false, overflow_entry,
                       num_valid, get_err_buffer(), get_err_buffer_len(),
                       OverflowFieldAll))
        return true;
    }

    for (const NID nid : overflow_entry.incoming) {
      Node resolved;
      if (resolve_node(level, nid.clear_incoming(), resolved))
        return true;
      out.push_back(resolved);
    }
    cur = overflow_entry.overflow;
  }
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
  return m_store.set_entry_point(mtr, node, level, get_err_buffer(),
                                 get_err_buffer_len());
}

bool IndexGraph::ensure_levels(LevelId level) {
  return m_store.ensure_levels(level, get_err_buffer(), get_err_buffer_len()) ==
         nullptr;
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
                            num_valid, get_err_buffer(), get_err_buffer_len(),
                            NodeField::LowerLevel))
      return true;
    // The node one level down doesn't exist yet -- this call is the one
    // that creates it, below.
    assert(!parent_check.lower_level.is_valid());
  }

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);

  // Per the bidirectional-link protocol (see graph.h): this node's edges to
  // neighbours are recorded as incoming links first, since none of them are
  // yet reciprocated -- link_neighbours() lands the neighbour's own directed
  // link back to this node, at which point the flag is cleared and the edge
  // becomes a confirmed outgoing link.
  for (Node &n : neighbours)
    n.nid = n.nid.set_incoming();

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
                    get_err_buffer(), get_err_buffer_len()))
    return true;

  out.nid = new_nid;
  out.vid = entry.owner;

  // Update parent's LowerLevel NID, now that this node's NID is known.
  if (parent) {
    NeighbourEntry parent_entry;
    parent_entry.lower_level = new_nid;
    if (parent_store->update(mtr, parent->nid, parent_entry,
                             NodeField::LowerLevel, m_ctx.m_update_slots,
                             m_ctx.m_neighbour_buf, m_ctx.m_chunk_ids,
                             get_err_buffer(), get_err_buffer_len()))
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
    if (store->fetch(mtr, node.nid, /*for_update=*/false, entry, num_valid,
                     get_err_buffer(), get_err_buffer_len(),
                     NodeField::Overflow))
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
    if (store->fetch(mtr, cur, /*for_update=*/false, entry, num_valid,
                     get_err_buffer(), get_err_buffer_len(),
                     OverflowField::Overflow))
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
                      m_ctx.m_chunk_ids, get_err_buffer(),
                      get_err_buffer_len()))
      return true;
  } else {
    OverflowEntry prev_entry;
    prev_entry.overflow = NID{};
    if (store->update(mtr, link.prev, prev_entry, OverflowField::Overflow,
                      m_ctx.m_update_slots, m_ctx.m_overflow_buf,
                      m_ctx.m_chunk_ids, get_err_buffer(),
                      get_err_buffer_len()))
      return true;
  }

  if (store->remove(mtr, StoreKind::Overflow, link.nid, m_trx_ref,
                    get_err_buffer(), get_err_buffer_len()))
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
                             m_ctx.m_neighbour_buf, m_ctx.m_chunk_ids,
                             get_err_buffer(), get_err_buffer_len()))
      return true;
  }

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);

  if (store->remove(mtr, StoreKind::Neighbour, node.nid, m_trx_ref,
                    get_err_buffer(), get_err_buffer_len()))
    return true;

  mtr_ctx.commit();
  return false;
}

bool IndexGraph::mark_delete(const Node &node, LevelId level,
                             bool delete_mark) {
  assert(debug_check_level(node, level));

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();
  return store->mark_delete(mtr, node.nid, m_trx_ref, delete_mark,
                            get_err_buffer(), get_err_buffer_len());
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
  if (store->fetch(mtr, node.nid, /*for_update=*/false, entry, num_valid,
                   get_err_buffer(), get_err_buffer_len(),
                   NodeField::LowerLevel)) {
    out = Node{};
    return true;
  }

  assert(entry.lower_level.is_valid());
  out = Node{entry.lower_level, node.vid};
  assert(debug_check_level(out, level.lower()));
  mtr_ctx.commit();
  return false;
}

bool IndexGraph::add_overflow_incoming(MtrCtx::Ref mtr, LevelId level,
                                       const Node &neighbour, NID overflow_head,
                                       const Node &node) {
  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);
  const uint32_t capacity = store->overflow_capacity();
  assert(capacity <= m_ctx.m_incoming_buf_1.size());

  // Overflow entries hold only unreciprocated incoming connections, so the
  // stored NID is flagged incoming, mirroring create_node()'s placeholder
  // links -- a future promotion back into the primary neighbour list clears
  // it the same way link_neighbours() does at the end of its own run.
  const NID incoming_nid = node.nid.set_incoming();

  // parent/parent_kind track the record whose Overflow field must be
  // updated to link in a new entry, if the walk below finds every existing
  // entry full.
  NID parent = neighbour.nid;
  StoreKind parent_kind = StoreKind::Neighbour;

  // Walk the chain, one overflow record at a time: each hop opens its own
  // short-lived mtr, latching (and, before advancing, releasing) exactly
  // one overflow page, while mtr keeps neighbour's own page latched for the
  // whole call -- per the lock hierarchy in graph.h, Overflow Storage Locks
  // are acquired only after the Neighbour Storage Lock, and only one is
  // ever held at a time.
  for (NID cur = overflow_head; cur.is_valid();) {
    MtrCtx sub_ctx;
    auto sub_mtr = sub_ctx.start();

    OverflowEntry entry;
    entry.incoming = m_ctx.m_incoming_buf_1.span(capacity);
    size_t num_valid;
    if (store->fetch(sub_mtr, cur, /*for_update=*/true, entry, num_valid,
                     get_err_buffer(), get_err_buffer_len(), OverflowFieldAll))
      return true;

    auto free_slot = std::find_if(entry.incoming.begin(), entry.incoming.end(),
                                  [](const NID &n) { return !n.is_valid(); });
    if (free_slot != entry.incoming.end()) {
      // Room in this entry: land node's incoming NID there.
      auto slot = static_cast<uint16_t>(free_slot - entry.incoming.begin());
      std::array<NID, 1> added{incoming_nid};
      OverflowEntry update_entry;
      update_entry.incoming = added;
      m_ctx.m_update_slots[0] = SlotIndex{slot};
      return store->update(sub_mtr, cur, update_entry, OverflowField::Incoming,
                           m_ctx.m_update_slots, m_ctx.m_overflow_buf,
                           m_ctx.m_chunk_ids, get_err_buffer(),
                           get_err_buffer_len());
    }

    // Full: move on to the next link, remembering this entry as the
    // current tail in case the whole chain turns out to be full.
    parent = cur;
    parent_kind = StoreKind::Overflow;
    cur = entry.overflow;
    sub_ctx.commit();
  }

  // Every existing overflow entry (if any) is full: create a new one
  // holding node's incoming NID, inserted (and committed) before it is
  // linked in below so the chain can never point at a not-yet-existing
  // entry.
  NID new_nid;
  {
    MtrCtx sub_ctx;
    auto sub_mtr = sub_ctx.start();
    std::array<NID, 1> added{incoming_nid};
    OverflowEntry new_entry;
    new_entry.incoming = added;
    if (store->insert(sub_mtr, new_entry, m_trx_ref, m_ctx.m_overflow_buf,
                      new_nid, get_err_buffer(), get_err_buffer_len()))
      return true;
  }

  // Link the new entry in as parent's successor. parent is either
  // neighbour's own record -- still covered by the caller's mtr -- or the
  // last (full) OverflowEntry in the chain, latched afresh here.
  if (parent_kind == StoreKind::Neighbour) {
    NeighbourEntry link_entry;
    link_entry.overflow = new_nid;
    return store->update(mtr, parent, link_entry, NodeField::Overflow,
                         m_ctx.m_update_slots, m_ctx.m_neighbour_buf,
                         m_ctx.m_chunk_ids, get_err_buffer(),
                         get_err_buffer_len());
  }

  MtrCtx sub_ctx;
  auto sub_mtr = sub_ctx.start();
  OverflowEntry link_entry;
  link_entry.overflow = new_nid;
  return store->update(sub_mtr, parent, link_entry, OverflowField::Overflow,
                       m_ctx.m_update_slots, m_ctx.m_overflow_buf,
                       m_ctx.m_chunk_ids, get_err_buffer(),
                       get_err_buffer_len());
}

bool IndexGraph::link_neighbours(const Node &node, LevelId level,
                                 std::vector<Node> &out) {
  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);

  // Level Operation Lock: S. Two concurrently inserted nodes cannot choose
  // one another as neighbours, so no exclusive serialization is needed at
  // the level (see the lock hierarchy in graph.h).
  std::shared_lock<std::shared_mutex> op_lock(store->operation_mutex());

  return link_neighbours_locked(node, level, out);
}

bool IndexGraph::link_neighbours_locked(const Node &node, LevelId level,
                                        std::vector<Node> &out) {
  out.clear();

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);
  const uint32_t max_n = store->max_neighbours();

  NeighbourEntry entry;
  entry.neighbours = m_ctx.m_node_buf_1.span(max_n);
  size_t num_valid;
  {
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    if (store->fetch(mtr, node.nid, /*for_update=*/true, entry, num_valid,
                     get_err_buffer(), get_err_buffer_len(),
                     NodeField::Neighbours, IncomingFilter::All))
      return true;
  }
  std::span<Node> node_slots = m_ctx.m_node_buf_1.span(max_n);

  // Incoming links walked so far, compacted to the front of node_slots with
  // their flag cleared and paired index-wise with m_link_slots -- the
  // ascending on-disk slot each came from. Compacting in place is safe: the
  // write index never runs ahead of the slot being read.
  size_t num_incoming = 0;

  for (uint32_t slot = 0; slot < max_n; ++slot) {
    const Node current = node_slots[slot];
    // Skip empty slots and links already recorded as outgoing: those are
    // either reciprocated already or had their reciprocal edge deliberately
    // withheld, so linking them again would add node to a neighbour twice.
    if (!current.nid.is_valid() || !current.nid.is_incoming())
      continue;

    const Node neighbour{current.nid.clear_incoming(), current.vid};
    node_slots[num_incoming] = neighbour;
    m_ctx.m_link_slots[num_incoming] = SlotIndex{static_cast<uint16_t>(slot)};
    ++num_incoming;

    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();

    // Raw slot layout (for_update=true, slots=nullptr): every on-disk slot
    // up to max_n, valid or not, in place -- needed to find a free slot to
    // land node in. IncomingFilter::All makes num_valid an exact occupied
    // count.
    NeighbourEntry neighbour_entry;
    neighbour_entry.neighbours = m_ctx.m_node_buf_2.span(max_n);
    size_t neighbour_num_valid;

    if (store->fetch(
            mtr, neighbour.nid, /*for_update=*/true, neighbour_entry,
            neighbour_num_valid, get_err_buffer(), get_err_buffer_len(),
            NodeField::Neighbours | NodeField::Overflow, IncomingFilter::All)) {
      // The neighbour's record may be gone: an incoming link may name a node
      // that has since been removed, and reading a freed record is exactly
      // what fails here. A genuine read failure is indistinguishable from it
      // today, so both are swallowed -- there's no neighbour left to
      // reciprocate into. node's own outgoing edge is still confirmed below,
      // left dangling until the neighbour-validation TODO in
      // GraphOperations::remove() repairs it.
      // TODO(villagesql-indexing): propagate a real failure once
      // ColumnStore::fetch() distinguishes it from a free record, which it
      // already detects.
      assert(false);
      continue;
    }

    std::span<Node> neighbour_slots = m_ctx.m_node_buf_2.span(max_n);
    auto free_slot = neighbour_slots.end();
    if (neighbour_num_valid < max_n)
      free_slot = std::find_if(neighbour_slots.begin(), neighbour_slots.end(),
                               [](const Node &n) { return !n.nid.is_valid(); });

    if (free_slot != neighbour_slots.end()) {
      // Room for a direct reciprocal edge: land node there (Bidirectional
      // links, step 2 in graph.h).
      auto free_index =
          static_cast<uint16_t>(free_slot - neighbour_slots.begin());
      std::array<Node, 1> added{node};
      NeighbourEntry update_entry;
      update_entry.neighbours = added;
      m_ctx.m_update_slots[0] = SlotIndex{free_index};
      if (store->update(mtr, neighbour.nid, update_entry, NodeField::Neighbours,
                        m_ctx.m_update_slots, m_ctx.m_neighbour_buf,
                        m_ctx.m_chunk_ids, get_err_buffer(),
                        get_err_buffer_len()))
        return true;
    } else {
      // No room: Add an incoming connection link.
      out.push_back(neighbour);
      if (add_overflow_incoming(mtr, level, neighbour, neighbour_entry.overflow,
                                node))
        return true;
    }
    mtr_ctx.commit();
  }

  // Nothing was flagged incoming: node's list is already fully reciprocated
  // and there is no flag left to clear.
  if (num_incoming == 0)
    return false;

  // Bidirectional links, step 3 in graph.h: every incoming link walked above
  // has now either landed a reciprocal edge directly or been recorded in the
  // neighbour's overflow chain, so confirm them all as real outgoing edges,
  // each written back to the slot m_link_slots recorded for it.
  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();
  NeighbourEntry update_entry;
  update_entry.neighbours = m_ctx.m_node_buf_1.span(num_incoming);
  return store->update(mtr, node.nid, update_entry, NodeField::Neighbours,
                       m_ctx.m_link_slots, m_ctx.m_neighbour_buf,
                       m_ctx.m_chunk_ids, get_err_buffer(),
                       get_err_buffer_len());
}

bool IndexGraph::drop_neighbour_link(LevelId level, const Node &node,
                                     NID neighbour_nid, bool &keeps_link) {
  keeps_link = false;

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);
  const uint32_t max_n = store->max_neighbours();

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();

  // Raw slot layout (for_update=true, slots=nullptr) with IncomingFilter::All,
  // as in unlink_neighbour(): a hit's index is the slot to clear, and the flag
  // on it is exactly what tells the two cases below apart.
  NeighbourEntry entry;
  entry.neighbours = m_ctx.m_node_buf_2.span(max_n);
  size_t num_valid;
  if (store->fetch(mtr, neighbour_nid, /*for_update=*/true, entry, num_valid,
                   get_err_buffer(), get_err_buffer_len(),
                   NodeField::Neighbours | NodeField::Overflow,
                   IncomingFilter::All)) {
    // The neighbour's record is gone -- swallowed exactly as in
    // unlink_neighbour(), and for the same reason: an incoming link may name a
    // node that has since been freed, and there is then nothing recorded to
    // remove.
    return false;
  }

  std::span<Node> neighbour_slots = m_ctx.m_node_buf_2.span(max_n);
  LinkSlot link{};
  for (uint32_t slot = 0; slot < max_n; ++slot) {
    const NID nid = neighbour_slots[slot].nid;
    if (!nid.is_valid() || nid.clear_incoming() != node.nid)
      continue;
    if (!nid.is_incoming()) {
      // The neighbour has an outgoing link of its own back to node: only
      // node's half of that edge is being dropped, so the neighbour's record
      // stays exactly as it is.
      keeps_link = true;
      return false;
    }
    link = LinkSlot{neighbour_nid, StoreKind::Neighbour,
                    SlotIndex{static_cast<uint16_t>(slot)}};
    break;
  }

  // Nothing in the neighbour's primary list names node: the record may be one
  // it had no room for there, kept in its overflow chain instead (see
  // add_overflow_incoming()).
  if (!link.slot.is_valid() &&
      find_overflow_link(level, entry.overflow, node.nid, link))
    return true;

  // Nothing names node anywhere: node's link was a placeholder the neighbour
  // never reciprocated (Bidirectional links, step 1 in graph.h), or one an
  // earlier, interrupted run already removed.
  if (!link.slot.is_valid())
    return false;

  if (clear_link(mtr, level, link))
    return true;

  mtr_ctx.commit();
  return false;
}

bool IndexGraph::replace_neighbours(const Node &node, LevelId level,
                                    const std::vector<Node> &neighbours,
                                    std::vector<Node> &out) {
  out.clear();
  assert(!node.nid.is_incoming());
  assert(debug_check_level(node, level));

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);
  const uint32_t max_n = store->max_neighbours();
  assert(neighbours.size() <= max_n);

  // The steps below match node's slots against the updated list by scanning it,
  // so their cost is on the order of max_n * neighbours.size() -- both bounded
  // by Mmax, a handful of cache lines' worth of NIDs at the default M. Set
  // against the page fetch step B issues per dropped link, indexing either side
  // is not worth what it would cost to hold: another preallocated scratch
  // buffer in GraphContext, or an allocation local to the call.

  // Level Operation Lock: X. This adjusts established connections -- node's own
  // and, on the far side, those of every neighbour it drops -- so it has to be
  // serialized against the other operations that may adjust the same nodes (see
  // the lock hierarchy in graph.h). Holding it across the whole operation is
  // also what lets the reciprocal linking at the end run as
  // link_neighbours_locked(), without reacquiring the lock it already holds.
  std::unique_lock<std::shared_mutex> op_lock(store->operation_mutex());

  // Node's own record of its links, read in raw slot layout and with
  // IncomingFilter::All: an incoming-flagged slot is as much a part of what
  // this replaces as an outgoing one, and every slot has to stay in place for
  // the updates below to write back to the one it came from.
  NeighbourEntry entry;
  entry.neighbours = m_ctx.m_node_buf_1.span(max_n);
  size_t num_valid;
  {
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    if (store->fetch(mtr, node.nid, /*for_update=*/true, entry, num_valid,
                     get_err_buffer(), get_err_buffer_len(),
                     NodeField::Neighbours | NodeField::Overflow,
                     IncomingFilter::All))
      return true;
  }
  std::span<Node> slots = m_ctx.m_node_buf_1.span(max_n);

  // Every link already chained off node is reachable from this head: what step
  // B below adds is appended to the chain's tail, or becomes its head only when
  // there was no chain at all -- and either way names a dropped neighbour,
  // never one the updated list names.
  const NID overflow_head = entry.overflow;

  // A link the updated list names too survives untouched; the rest are the ones
  // being dropped.
  auto in_updated = [&neighbours](NID nid) {
    const NID key = nid.clear_incoming();
    return std::any_of(neighbours.begin(), neighbours.end(),
                       [key](const Node &n) { return n.nid == key; });
  };

  // Staging area for the slot-aligned updates below, paired index-wise with
  // m_update_slots. m_node_buf_1 holds node's own list for the whole call, so
  // the second Node buffer stands in here exactly as it does for the
  // per-neighbour work in link_neighbours() -- which is also what step B's
  // drop_neighbour_link() uses it for, so each set staged here is written out
  // (step A) or built afresh (step C) rather than kept live across a step.
  std::span<Node> staged = m_ctx.m_node_buf_2.span(max_n);

  // A. Demote every dropped link to an incoming one, in a single update, before
  // any of them is touched on the far side: node stops traversing to them right
  // away while still recording which ones are left to visit -- the same
  // ordering unlink_neighbours() relies on. One that is already incoming needs
  // no demoting.
  size_t num_staged = 0;
  for (uint32_t slot = 0; slot < max_n; ++slot) {
    const Node current = slots[slot];
    if (!current.nid.is_valid() || current.nid.is_incoming())
      continue;
    if (in_updated(current.nid))
      continue;
    staged[num_staged] = Node{current.nid.set_incoming(), current.vid};
    m_ctx.m_update_slots[num_staged] = SlotIndex{static_cast<uint16_t>(slot)};
    ++num_staged;
  }
  if (num_staged > 0) {
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    NeighbourEntry update_entry;
    update_entry.neighbours = m_ctx.m_node_buf_2.span(num_staged);
    if (store->update(mtr, node.nid, update_entry, NodeField::Neighbours,
                      m_ctx.m_update_slots, m_ctx.m_neighbour_buf,
                      m_ctx.m_chunk_ids, get_err_buffer(),
                      get_err_buffer_len()))
      return true;
  }

  // B. Remove each dropped link from the far side. A neighbour that has an
  // outgoing link of its own back to node keeps it -- only node's half of that
  // edge is being dropped -- so node has to go on recording that neighbour, as
  // an incoming link in its overflow chain, which frees the primary slot for
  // the updated list. Recording it there before the slot is freed (step D) is
  // what keeps such a link recorded on node's side throughout: the reverse
  // order could leave the neighbour pointing at a node that no longer knows
  // about it.
  for (uint32_t slot = 0; slot < max_n; ++slot) {
    const Node current = slots[slot];
    if (!current.nid.is_valid() || in_updated(current.nid))
      continue;

    const Node dropped{current.nid.clear_incoming(), current.vid};
    bool keeps_link = false;
    if (drop_neighbour_link(level, node, dropped.nid, keeps_link))
      return true;
    if (!keeps_link)
      continue;

    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    // Re-read for the chain head each time round: a previous iteration's
    // add_overflow_incoming() may have been the one to create the chain.
    NeighbourEntry head_entry;
    size_t num_valid;
    if (store->fetch(mtr, node.nid, /*for_update=*/true, head_entry, num_valid,
                     get_err_buffer(), get_err_buffer_len(),
                     NodeField::Overflow))
      return true;
    if (add_overflow_incoming(mtr, level, node, head_entry.overflow, dropped))
      return true;
    mtr_ctx.commit();
  }

  // C. Stage the links the updated list adds. One node already records in its
  // overflow chain is landing back in the primary list rather than arriving
  // fresh, and goes in as a confirmed outgoing link: a chain record is written
  // while the edge it belongs to is being landed (see add_overflow_incoming()),
  // so its source is holding a slot for node either way, and linking it again
  // would only give that source a second one. Everything else goes in
  // incoming-flagged, for step F to reciprocate.
  //
  // Their chain slots are collected as they're found and freed in step E, once
  // node's primary list is the record: find_overflow_link() has already located
  // each one exactly, so there is no reason to walk the chain again looking for
  // them.
  num_staged = 0;
  std::vector<LinkSlot> promoted;
  for (const Node &target : neighbours) {
    assert(target.nid.is_valid() && !target.nid.is_incoming());
    assert(target.nid != node.nid);

    bool present = false;
    for (uint32_t slot = 0; slot < max_n && !present; ++slot)
      present = slots[slot].nid.is_valid() &&
                slots[slot].nid.clear_incoming() == target.nid;
    if (present)
      continue;

    LinkSlot chained{};
    if (find_overflow_link(level, overflow_head, target.nid, chained))
      return true;
    if (chained.slot.is_valid()) {
      staged[num_staged] = target;
      promoted.push_back(chained);
    } else {
      staged[num_staged] = Node{target.nid.set_incoming(), target.vid};
    }
    ++num_staged;
  }

  // D. Write node's new list in a single slot-aligned update, walking the slots
  // in the ascending order update() requires: a kept link is left exactly as it
  // is -- incoming flag included, since the flag records that the neighbour is
  // not known to link back yet and only the linking step can settle that, which
  // step F leaves it to do -- and each staged link takes the next slot a
  // dropped link vacated or one that was free already, with whatever dropped
  // slots are left over cleared.
  size_t num_updates = 0;
  size_t next_staged = 0;
  for (uint32_t slot = 0; slot < max_n; ++slot) {
    const Node current = slots[slot];
    if (current.nid.is_valid() && in_updated(current.nid))
      continue;

    // A dropped link's slot, or one that was free already: the next staged link
    // takes it, and a dropped one left over is cleared.
    Node value{};
    if (next_staged < num_staged)
      value = staged[next_staged++];
    else if (!current.nid.is_valid())
      continue;
    // Compacting the writes into node's own list buffer is safe: the write
    // index never runs ahead of the slot being read.
    slots[num_updates] = value;
    m_ctx.m_update_slots[num_updates] = SlotIndex{static_cast<uint16_t>(slot)};
    ++num_updates;
  }
  // There is always room: the updated list is capped at max_n, and every slot
  // it does not name is either free or one this call just freed.
  assert(next_staged == num_staged);

  if (num_updates > 0) {
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    NeighbourEntry update_entry;
    update_entry.neighbours = m_ctx.m_node_buf_1.span(num_updates);
    if (store->update(mtr, node.nid, update_entry, NodeField::Neighbours,
                      m_ctx.m_update_slots, m_ctx.m_neighbour_buf,
                      m_ctx.m_chunk_ids, get_err_buffer(),
                      get_err_buffer_len()))
      return true;
  }

  // E. Free the chain slots of the links promoted in step C, now that node's
  // primary list is the one recording them. Doing it in this order can only
  // ever leave the same link recorded twice, which costs a wasted visit in
  // unlink_neighbours() and nothing else; the reverse could leave a link node
  // does not record at all.
  if (!promoted.empty()) {
    // Every one of them is an overflow record, which clear_link() latches in a
    // short-lived mtr of its own -- mtr is the caller's latch its other branch
    // needs, and there is none to hold here.
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    for (const LinkSlot &link : promoted) {
      assert(link.kind == StoreKind::Overflow);
      if (clear_link(mtr, level, link))
        return true;
    }
    mtr_ctx.commit();
  }

  // F. Reciprocate the links staged as incoming in step C, along with any kept
  // link still waiting for its own reciprocal edge -- settling those is exactly
  // what link_neighbours() is for. Nothing else in node's list is flagged
  // incoming by now: every link the updated list does not name has been dropped
  // or moved to the overflow chain.
  return link_neighbours_locked(node, level, out);
}

bool IndexGraph::clear_link(MtrCtx::Ref mtr, LevelId level,
                            const LinkSlot &link) {
  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);
  assert(link.slot.is_valid());

  m_ctx.m_update_slots[0] = link.slot;

  if (link.kind == StoreKind::Neighbour) {
    // The caller's mtr already holds link.record's page latch.
    std::array<Node, 1> cleared{};
    NeighbourEntry entry;
    entry.neighbours = cleared;
    return store->update(mtr, link.record, entry, NodeField::Neighbours,
                         m_ctx.m_update_slots, m_ctx.m_neighbour_buf,
                         m_ctx.m_chunk_ids, get_err_buffer(),
                         get_err_buffer_len());
  }

  // An overflow record is latched in its own short-lived mtr, opened while
  // the caller's mtr holds the neighbour-record latches it belongs after.
  MtrCtx sub_ctx;
  auto sub_mtr = sub_ctx.start();
  std::array<NID, 1> cleared{};
  OverflowEntry entry;
  entry.incoming = cleared;
  return store->update(sub_mtr, link.record, entry, OverflowField::Incoming,
                       m_ctx.m_update_slots, m_ctx.m_overflow_buf,
                       m_ctx.m_chunk_ids, get_err_buffer(),
                       get_err_buffer_len());
}

bool IndexGraph::find_overflow_link(LevelId level, NID head, NID target,
                                    LinkSlot &out) {
  out = LinkSlot{};

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);
  const uint32_t capacity = store->overflow_capacity();
  assert(capacity <= m_ctx.m_incoming_buf_2.size());

  for (NID cur = head; cur.is_valid();) {
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();

    // Raw slot layout (for_update=true): every incoming slot stays in place,
    // so a hit's index is the on-disk slot clear_link() must write to.
    OverflowEntry entry;
    entry.incoming = m_ctx.m_incoming_buf_2.span(capacity);
    size_t num_valid;
    if (store->fetch(mtr, cur, /*for_update=*/true, entry, num_valid,
                     get_err_buffer(), get_err_buffer_len(), OverflowFieldAll))
      return true;

    for (uint32_t slot = 0; slot < capacity; ++slot) {
      // Chained incoming links are always flagged incoming (see
      // add_overflow_incoming()), so compare with the flag masked off.
      if (entry.incoming[slot].clear_incoming() != target)
        continue;
      out = LinkSlot{cur, StoreKind::Overflow,
                     SlotIndex{static_cast<uint16_t>(slot)}};
      return false;
    }
    cur = entry.overflow;
  }
  return false;
}

bool IndexGraph::unlink_neighbour(LevelId level, const Node &node,
                                  NID neighbour_nid, Node &out_neighbour,
                                  bool &orphan) {
  orphan = false;
  // Left unset unless a link is actually severed below -- how the caller tells
  // that case from having found nothing to sever.
  out_neighbour = Node{};

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);
  const uint32_t max_n = store->max_neighbours();

  MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();

  // Only the neighbour is latched: node's own list is adjusted separately by
  // unlink_neighbours(), whose Level Operation Lock covers both halves.
  //
  // Raw slot layout (for_update=true, slots=nullptr) with IncomingFilter::All:
  // every on-disk slot in place, so a hit's index is the slot to clear, and an
  // incoming-flagged link is neither skipped nor mistaken for an outgoing one.
  NeighbourEntry entry;
  entry.neighbours = m_ctx.m_node_buf_2.span(max_n);
  size_t num_valid;
  if (store->fetch(mtr, neighbour_nid, /*for_update=*/true, entry, num_valid,
                   get_err_buffer(), get_err_buffer_len(),
                   NodeField::Owner | NodeField::Neighbours |
                       NodeField::Overflow,
                   IncomingFilter::All)) {
    // The neighbour's record is gone: an incoming link may name a node that
    // has since been freed, and reading a freed record is exactly what fails
    // here. A genuine read failure is indistinguishable from it today, so both
    // are swallowed -- either way there is no link left to sever.
    // TODO(villagesql-indexing): propagate a real failure once
    // ColumnStore::fetch() distinguishes it from a free record, which it
    // already detects.
    return false;
  }

  // The neighbour's record of the link, plus the outgoing links it is left
  // with once that record is gone -- none of them makes it an orphan at this
  // level.
  std::span<Node> neighbour_slots = m_ctx.m_node_buf_2.span(max_n);
  LinkSlot link{};
  uint32_t outgoing = 0;
  for (uint32_t slot = 0; slot < max_n; ++slot) {
    const NID nid = neighbour_slots[slot].nid;
    if (!nid.is_valid())
      continue;
    if (nid.clear_incoming() == node.nid) {
      link = LinkSlot{neighbour_nid, StoreKind::Neighbour,
                      SlotIndex{static_cast<uint16_t>(slot)}};
      continue;
    }
    if (!nid.is_incoming())
      ++outgoing;
  }

  // Nothing in the neighbour's primary list names node: the link is one the
  // neighbour had no room for there, recorded as an incoming link in its
  // overflow chain instead (see add_overflow_incoming()).
  if (!link.slot.is_valid() &&
      find_overflow_link(level, entry.overflow, node.nid, link))
    return true;

  // Nothing names node anywhere: node's own link was a placeholder the
  // neighbour never reciprocated (Bidirectional links, step 1 in graph.h), or
  // an earlier, interrupted run of this operation already removed it. Nothing
  // was severed, so nothing is reported about the neighbour either -- only a
  // node that really did link back to node can be called an orphan of it.
  if (!link.slot.is_valid())
    return false;

  if (clear_link(mtr, level, link))
    return true;

  mtr_ctx.commit();
  // The overflow chain carries no vid, so the neighbour's own owner field is
  // the one place either of unlink_neighbours()' walks can resolve it from.
  out_neighbour = Node{neighbour_nid, entry.owner};
  orphan = outgoing == 0;
  return false;
}

bool IndexGraph::unlink_neighbours(const Node &node, LevelId level,
                                   std::vector<Node> &out) {
  out.clear();
  assert(!node.nid.is_incoming());
  assert(debug_check_level(node, level));

  LevelStore *store = m_store.get_level(level);
  assert(store != nullptr);
  const uint32_t max_n = store->max_neighbours();
  const uint32_t capacity = store->overflow_capacity();
  assert(capacity <= m_ctx.m_incoming_buf_1.size());

  // Level Operation Lock: X. Severing established connections adjusts nodes
  // that other operations may be adjusting too, so it has to be serialized
  // against them at this level. Holding it across the whole operation is also
  // what makes the one-latch-at-a-time walk below safe: no concurrent
  // operation can establish or retain a link between node and one of its
  // neighbours while the two sides are taken apart separately (see the lock
  // hierarchy in graph.h).
  std::unique_lock<std::shared_mutex> op_lock(store->operation_mutex());

  // Node's own record of its links: the primary neighbour list -- read in raw
  // slot layout and with IncomingFilter::All, since an incoming-flagged slot
  // records a link to sever just the same -- plus the head of the overflow
  // chain holding the incoming links that had no room in that list.
  NeighbourEntry entry;
  entry.neighbours = m_ctx.m_node_buf_1.span(max_n);
  size_t num_valid;
  {
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    if (store->fetch(mtr, node.nid, /*for_update=*/true, entry, num_valid,
                     get_err_buffer(), get_err_buffer_len(),
                     NodeField::Neighbours | NodeField::Overflow,
                     IncomingFilter::All))
      return true;
  }

  // Compact node's links to the front of node_slots, flagged incoming and
  // paired index-wise with m_link_slots -- the ascending on-disk slot each came
  // from -- as link_neighbours() pairs the links it walks. Compacting in place
  // is safe: the write index never runs ahead of the slot being read.
  std::span<Node> node_slots = m_ctx.m_node_buf_1.span(max_n);
  size_t num_links = 0;
  for (uint32_t slot = 0; slot < max_n; ++slot) {
    const Node current = node_slots[slot];
    if (!current.nid.is_valid())
      continue;
    node_slots[num_links] = Node{current.nid.set_incoming(), current.vid};
    m_ctx.m_link_slots[num_links] = SlotIndex{static_cast<uint16_t>(slot)};
    ++num_links;
  }

  // Demote every one of them to an incoming link in a single update: node
  // keeps the record of which neighbours are left to visit, but no longer has
  // an outgoing link of its own, so it drops out of every traversal
  // (neighbours() excludes incoming links) before any neighbour is touched.
  if (num_links > 0) {
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    NeighbourEntry update_entry;
    update_entry.neighbours = m_ctx.m_node_buf_1.span(num_links);
    if (store->update(mtr, node.nid, update_entry, NodeField::Neighbours,
                      m_ctx.m_link_slots, m_ctx.m_neighbour_buf,
                      m_ctx.m_chunk_ids, get_err_buffer(),
                      get_err_buffer_len()))
      return true;
  }

  // Follow each link and remove the neighbour's own link back to node,
  // collecting the slots to free -- every link's but an orphaned neighbour's,
  // whose slot stays behind as the graph's record that it still needs
  // reconnecting. Both compactions are safe for the same reason as above: the
  // write index trails the index being read.
  size_t num_freed = 0;
  for (size_t i = 0; i < num_links; ++i) {
    Node neighbour;
    bool orphan = false;
    if (unlink_neighbour(level, node, node_slots[i].nid.clear_incoming(),
                         neighbour, orphan))
      return true;
    if (orphan)
      continue;

    m_ctx.m_link_slots[num_freed] = m_ctx.m_link_slots[i];
    node_slots[num_freed] = Node{};
    ++num_freed;
    // Only a severed link yields a neighbour to report; nothing is known about
    // one that turned out to have no link back to node. Bounded by a node's own
    // degree limit besides: out is only a pool of still-connected nodes to
    // search from, and the overflow chain below can otherwise push it
    // arbitrarily far past it.
    if (neighbour.nid.is_valid() && out.size() < max_n)
      out.push_back(neighbour);
  }

  // Free those slots, now that every neighbour has dropped its side. Doing it
  // last is what makes an interrupted run restartable: whatever is still
  // recorded is simply walked again, and a neighbour already done just has no
  // link left to find.
  if (num_freed > 0) {
    MtrCtx mtr_ctx;
    auto mtr = mtr_ctx.start();
    NeighbourEntry update_entry;
    update_entry.neighbours = m_ctx.m_node_buf_1.span(num_freed);
    if (store->update(mtr, node.nid, update_entry, NodeField::Neighbours,
                      m_ctx.m_link_slots, m_ctx.m_neighbour_buf,
                      m_ctx.m_chunk_ids, get_err_buffer(),
                      get_err_buffer_len()))
      return true;
  }

  // The overflow chain holds the same kind of link, one indirection away:
  // every valid incoming NID is a node whose link to node had no slot in the
  // primary list above, severed exactly the same way, with the chain entry's
  // own slots freed in place of node's. The entries themselves are left for
  // drop_node() to unlink and free.
  for (NID cur = entry.overflow; cur.is_valid();) {
    OverflowEntry overflow_entry;
    overflow_entry.incoming = m_ctx.m_incoming_buf_1.span(capacity);
    size_t num_valid;
    {
      MtrCtx mtr_ctx;
      auto mtr = mtr_ctx.start();
      // Raw slot layout, as for node's primary list above.
      if (store->fetch(mtr, cur, /*for_update=*/true, overflow_entry, num_valid,
                       get_err_buffer(), get_err_buffer_len(),
                       OverflowFieldAll))
        return true;
    }

    // Chained links are incoming ones to begin with, so there is nothing to
    // demote here: compact them straight into the same slot pairing the
    // primary list used above.
    std::span<NID> incoming = m_ctx.m_incoming_buf_1.span(capacity);
    size_t num_incoming = 0;
    for (uint32_t slot = 0; slot < capacity; ++slot) {
      if (!incoming[slot].is_valid())
        continue;
      incoming[num_incoming] = incoming[slot];
      m_ctx.m_link_slots[num_incoming] = SlotIndex{static_cast<uint16_t>(slot)};
      ++num_incoming;
    }

    num_freed = 0;
    for (size_t i = 0; i < num_incoming; ++i) {
      Node neighbour;
      bool orphan = false;
      if (unlink_neighbour(level, node, incoming[i].clear_incoming(), neighbour,
                           orphan))
        return true;
      if (orphan)
        continue;

      m_ctx.m_link_slots[num_freed] = m_ctx.m_link_slots[i];
      incoming[num_freed] = NID{};
      ++num_freed;
      if (neighbour.nid.is_valid() && out.size() < max_n)
        out.push_back(neighbour);
    }

    if (num_freed > 0) {
      MtrCtx mtr_ctx;
      auto mtr = mtr_ctx.start();
      OverflowEntry update_entry;
      update_entry.incoming = m_ctx.m_incoming_buf_1.span(num_freed);
      if (store->update(mtr, cur, update_entry, OverflowField::Incoming,
                        m_ctx.m_link_slots, m_ctx.m_overflow_buf,
                        m_ctx.m_chunk_ids, get_err_buffer(),
                        get_err_buffer_len()))
        return true;
    }
    cur = overflow_entry.overflow;
  }
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
