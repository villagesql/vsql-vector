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

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_GRAPH_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_GRAPH_H

#include "../../native_vector.h"
#include "hnsw.h"
#include "storage.h"
#include <cassert>
#include <span>
#include <stack>

// Lock hierarchy
//
//   Graph Lock
//       -> Level Lock
//           -> Level Operation Lock
//               -> Neighbour Storage Lock (InnoDB MTR data-page latch)
//                   -> Overflow Storage Lock (InnoDB MTR data-page latch)
//
// Locks must be acquired from left to right. A lock at a higher level in
// this hierarchy must not be acquired while holding a lock at a lower
// level.
//
//   Level Locks:
//     - Acquire top-down by level.
//
//   Level Operation Locks:
//     - At most one may be held at a time.
//
//   Neighbour Storage Locks across levels:
//     - Acquire top-down by level.
//     - At most two may be held simultaneously.
//
//   Neighbour Storage Locks within one level:
//     - We don't acquire multiple storage locks simultaneously. It would
//       require ordering rules based on underlying pages, if needed.
//     - Instead, hold Level Operation Lock while adjusting links between nodes.
//
//   Overflow Storage Locks:
//     - Acquired only after all corresponding Neighbour Storage Locks.
//
// Operation-specific ordering:
//
//   Insert: linking neighbours
//     - Level Operation Lock: S.
//       Two concurrently inserted nodes cannot choose one another as
//       neighbours. Therefore, concurrent inserts do not require exclusive
//       serialization at the level.
//
//     - Neighbour Storage Locks:
//       Acquire the storage locks needed to update the new node and its
//       selected neighbours while holding the Level Operation Lock.
//
//   Insert/Delete: adjusting an existing node's neighbours
//     - Level Operation Lock: X.
//       This modifies an established connection and must be serialized
//       with other operations that may adjust the same existing node.
//
//     - Neighbour Storage Locks:
//       Acquire the storage locks needed for the affected nodes while
//       holding the Level Operation Lock. The operation lock prevents
//       concurrent graph operations from modifying the same relationships.
//
// Storage locks are logical node locks backed by InnoDB MTR page latches.
// Multiple nodes may reside on the same data page, so two logical storage
// locks may correspond to the same physical page latch.
//
// In particular, an MTR permits repeated X-latching of the same page, but
// does not permit repeated S-latching of the same page. Callers must
// therefore account for page identity when acquiring multiple storage
// locks, even when the corresponding nodes are distinct.
//
// Currently, the link operations do not require multiple shared storage
// latches: an operation holds at most one S|X node storage latch at a
// time.

// Bidirectional links
//
// A node's outgoing link is a directed connection from the current node to
// another node. The target node must exist, but it need not have a link back
// to the current node.
//
// A node's incoming link represents the fact that another node may have a
// connection to the current node. The source node must have existed when the
// incoming link was created, but may subsequently be deleted and exist as a
// free node. The target node must belong to the storage level segment for
// the current level.
//
// A bidirectional connection between N1 and N2 can therefore be established
// without simultaneously latching both nodes:
//
//   1. N1:  Create an incoming link for N2.
//   2. N2:  Create a directed link to N1.
//   3. N1:  Replace the incoming link for N2 with the outgoing link to N2.
//
// Each step is an independent atomic operation and leaves the graph in a
// valid state. In practice, steps 2 and 3 are performed while holding the
// Level Operation Lock.
//
// An incoming link must never be removed without holding the Level Operation
// Lock and the storage latches required to modify both the source and target
// records. The Level Operation Lock prevents a concurrent graph operation
// from establishing or retaining the corresponding outgoing link while the
// incoming link is being removed.

namespace svector::hnsw {

// Reusable scratch buffers shared by graph operations.
// Sized once from the index configuration so hot paths never allocate.
struct GraphContext {
  // max_update_slots sizes the slot list a single update() reads, one entry
  // per record slot. max_update_chunks sizes the chunk id list that same
  // update() emits, which is larger: a neighbour slot costs two chunks (nid
  // and vid) and the scalar fields add a few more.
  GraphContext(size_t neighbour_buf_size, size_t overflow_buf_size,
               size_t vector_buf_size, size_t max_update_slots,
               size_t max_update_chunks, std::span<char> error)
      : m_neighbour_buf(neighbour_buf_size), m_overflow_buf(overflow_buf_size),
        m_vector_buf_1(vector_buf_size), m_vector_buf_2(vector_buf_size),
        m_decoded_buf_1(vector_buf_size), m_decoded_buf_2(vector_buf_size),
        m_update_slots(max_update_slots), m_link_slots(max_update_slots),
        m_chunk_ids(max_update_chunks), m_node_buf_1(max_update_slots),
        m_node_buf_2(max_update_slots), m_incoming_buf_1(max_update_slots),
        m_incoming_buf_2(max_update_slots), m_error(error) {}

  ScratchBytes m_neighbour_buf;
  ScratchBytes m_overflow_buf;
  ScratchBytes m_vector_buf_1;
  ScratchBytes m_vector_buf_2;
  // Decoded native::Data for each distance() operand. Sized to vector_buf_size
  // (the raw max, which is >= the decoded native length), so the leaf distance
  // decodes into these reused buffers instead of allocating per call. Two
  // buffers so both operands are live at once for m_dist_fn.
  ScratchBytes m_decoded_buf_1;
  ScratchBytes m_decoded_buf_2;

  // Source pointer last decoded into m_decoded_buf_1. In a search-layer
  // traversal the first distance() operand (the query/insert vector) is fixed
  // across every candidate, so decoding it once and reusing it skips the
  // per-candidate re-decode. Cleared (nullptr) means the buffer holds nothing
  // reusable. Keyed on the raw source pointer: identical pointer => identical
  // already-decoded bytes.
  const unsigned char *m_decoded_buf_1_src = nullptr;

  ScratchSlots m_update_slots;
  // On-disk slot indices of the incoming-flagged neighbour links
  // link_neighbours() is reciprocating, collected as it walks them and
  // consumed by the flag-clearing update at the end of that walk. A second
  // slot array is needed because m_update_slots is claimed by the
  // per-neighbour update performed inside the same walk.
  ScratchSlots m_link_slots;
  ScratchChunkIds m_chunk_ids;

  // Two reusable Node-array buffers (sized to Mmax0, the largest possible
  // neighbour list) for graph operations that need to hold two decoded
  // neighbour lists live at once -- e.g. link_neighbours() keeps node's own
  // list in one while scanning each neighbour's slot layout into the other.
  ScratchNodes m_node_buf_1;
  ScratchNodes m_node_buf_2;

  // Decoded OverflowEntry::incoming slots (for_update layout: every slot up
  // to overflow_capacity(), valid or not) -- sized to max_update_slots like
  // the buffers above since overflow_capacity() never exceeds it. Two of
  // them for the same reason as the Node buffers above: unlink_neighbours()
  // walks node's own overflow chain in one while the per-neighbour unlink it
  // performs at each step searches a neighbour's chain in the other.
  ScratchNIDs m_incoming_buf_1;
  ScratchNIDs m_incoming_buf_2;

  // Non-owning error buffer for the single API call this GraphContext (and
  // its owning IndexGraph) was constructed for -- IndexGraph is constructed
  // fresh per top-level call by the API entry point that already has its
  // own err/err_len from the caller, rather than shared/reused across calls.
  std::span<char> m_error;
};

class IndexGraph {
public:
  using LevelId = LevelStore::LevelId;

  struct NodeData {
    IndexScanKey::KeyPartData data;
  };

  using Node = hnsw::Node;

  using DistanceType = double;
  enum class LockMode { Shared, Exclusive };

  // RAII guard for the whole-graph lock, which protects graph-wide
  // metadata.
  class LockGraph {
  public:
    LockGraph(IndexGraph &graph, LockMode mode);
    ~LockGraph();

    LockGraph(const LockGraph &) = delete;
    LockGraph &operator=(const LockGraph &) = delete;

    LockGraph(LockGraph &&) = delete;
    LockGraph &operator=(LockGraph &&) = delete;

    // Upgrades the lock from S to X by releasing the shared lock and
    // acquiring the exclusive lock. The operation is not atomic, so
    // there is a window during which no lock is held. Callers must
    // already hold the lock in S mode.
    void upgrade() {
      assert(m_mode == LockMode::Shared);
      relock(LockMode::Exclusive);
    }

    // Downgrades the lock from X to S by releasing the exclusive lock
    // and acquiring the shared lock. The operation is not atomic, so
    // there is a window during which no lock is held. Callers must
    // already hold the lock in X mode.
    void downgrade() {
      assert(m_mode == LockMode::Exclusive);
      relock(LockMode::Shared);
    }

    // Returns the mode in which the graph is currently locked.
    LockMode lock_mode() const { return m_mode; }

  private:
    void relock(LockMode mode);

    IndexGraph &m_graph;
    LockMode m_mode;
  };

  // RAII lock-coupling guard for descending through levels one at a time.
  // Guarantees at least one level stays locked at all times from the
  // first descend() call until the guard is destroyed -- callers must
  // never be left holding zero level locks mid-traversal, since that gap
  // is exactly what the destructive, multi-layer delete this synchronizes
  // against would need to observe to corrupt a concurrent insert/search.
  //
  // mode and policy are fixed for the object's lifetime: every level is
  // locked in the same mode, and every transition follows the same
  // policy. Purely a local, scope-bound guard: neither copyable nor
  // movable, since nothing needs to hand one off.
  class LockLevels {
  public:
    // Keep: every locked level is retained until the guard is destroyed.
    // Release: descending to a new level locks it first, then releases
    // the previous level (hand-over-hand / lock coupling) -- at most the
    // current and about-to-be-released level are ever held at once.
    enum class DescendPolicy { Keep, Release };

    LockLevels(IndexGraph &graph, LockMode mode, LevelId start,
               DescendPolicy policy);
    ~LockLevels();

    LockLevels(const LockLevels &) = delete;
    LockLevels &operator=(const LockLevels &) = delete;

    LockLevels(LockLevels &&) = delete;
    LockLevels &operator=(LockLevels &&) = delete;

    // Descends to the next lower level, if any.
    // @returns The current level after the descent. If already at the lowest
    //          level, no action is taken and the current level is returned.
    LevelId descend();

    void update_policy(DescendPolicy policy);
    LevelId level() const;

  private:
    IndexGraph &m_graph;
    LockMode m_mode;
    DescendPolicy m_policy;
    std::stack<LevelId> m_stack;
  };

  // Constructed fresh by the API entry point handling a single top-level
  // call (e.g. the DML insert() hook), which already has its own index,
  // trx_ref and err/err_len to inject -- not shared or reused across calls.
  IndexGraph(IndexStore &store, const Index &index, Segment::TrxRef trx_ref,
             size_t vector_buf_size, std::span<char> err);

  bool distance(const Node &a, const Node &b, DistanceType &out);

  bool distance(const NodeData &a, const Node &b, DistanceType &out);

  // Resolve a graph node's decoded vector via the store's build-scoped cache,
  // returning a stable pointer (valid for the store's lifetime). Lets a caller
  // that compares one fixed node against many others -- e.g. the Algorithm-4
  // dominance check -- resolve the fixed operand's decoded vector ONCE before
  // the loop and pass it to distance(const native::Data *, const Node &),
  // hoisting the fixed operand's cache lookup out of the inner loop (only the
  // varying operand is looked up per comparison). Returns true on error.
  bool resolve_cached_vector(const Node &node, const native::Data **out);

  // distance() with the first operand already resolved to its decoded vector
  // (see resolve_cached_vector). The second operand is a graph node resolved
  // via the cache. Returns true on error.
  bool distance(const native::Data *a, const Node &b, DistanceType &out);

  // level is node's own level. Callers always already know it (it's how
  // they located node in the first place), so it's taken directly instead
  // of rediscovering it with a locate() call.
  bool neighbours(const Node &node, LevelId level, std::vector<Node> &out);

  // The counterpart to neighbours(): every node still recorded as an incoming
  // link to node, which is what neighbours() drops -- the incoming-flagged
  // slots of node's own list plus every link in its overflow chain, where the
  // incoming links that had no room in that list live. A chained link carries
  // a NID alone, so its vid is read from the owner field of its own record.
  //
  // After unlink_neighbours() these are exactly the neighbours it left
  // orphaned, which is what GraphOperations::remove() reconnects. level is
  // node's own level.
  bool incoming_neighbours(const Node &node, LevelId level,
                           std::vector<Node> &out);

  bool visible(const Node &node, bool &out);

  // Returns the randomly-generated level for the element about to be
  // inserted (Algorithm 1, line 4):
  LevelId get_insert_level();

  // Fetches the graph's current entry points and their level (Algorithm 1,
  // lines 2-3). out is left empty if the graph has no entry point yet
  // (i.e. this insert is the very first), in which case out_level is left
  // unspecified.
  bool get_entry_point(std::vector<Node> &out, LevelId &out_level);

  // Registers nodes as the graph's new entry points at level (Algorithm 1,
  // line 19).
  bool set_entry_point(const std::vector<Node> &nodes, LevelId level);

  // Materializes every level up to and including level. Called by
  // GraphOperations::insert() before it takes any level lock, since a level
  // that does not exist yet cannot be locked. A no-op once the levels
  // exist, so only the insert that raises the graph's height pays for it.
  // The caller must hold the graph lock in X mode whenever levels are
  // actually created (see IndexStore::ensure_levels).
  bool ensure_levels(LevelId level);

  bool create_node(const std::optional<Node> &parent, LevelId level,
                   const NodeData &data, std::vector<Node> &neighbours,
                   Node &out);

  // level is node's own level; when parent is present, it is implicitly at
  // level.value + 1 (mirrors create_node()'s parent/level relationship).
  bool drop_node(const std::optional<Node> &parent, LevelId level,
                 const Node &node);

  // Sets or clears the delete mark on node's own record, leaving every link
  // to and from it in place: a delete-marked node stays part of the graph and
  // keeps serving as a stepping stone for traversals, it just stops being a
  // valid result (see GraphVisiblePolicy). Only this level's record is
  // touched, so a caller marking the whole vector must call this once per
  // level -- GraphOperations::mark_delete() does. level is node's own level.
  bool mark_delete(const Node &node, LevelId level, bool delete_mark);

  // level is node's own level; level must have a lower level (i.e.
  // level.has_lower_level()) -- callers always already know they're
  // descending, never called at level 0. Follows node's LowerLevel NID to the
  // same vector's counterpart node one level down; out's vid is node's own,
  // since a node's vid is invariant across every level it appears at (set
  // identically by every create_node() call for that vector).
  bool get_next_level_node(const Node &node, LevelId level, Node &out);

  // Adds the reciprocal edges back to node for those of node's neighbours
  // whose link is still incoming-flagged, i.e. not yet reciprocated
  // (Bidirectional links, steps 2-3 above), and clears the flag on each once
  // its edge has landed. If doing so would push a neighbour's degree past
  // Mmax for node's level, the edge is withheld and that neighbour is
  // appended to out instead, leaving
  // GraphOperations::insert() to reselect and replace its full neighbour set
  // (Algorithm 1, lines 14-15). level is node's own level.
  bool link_neighbours(const Node &node, LevelId level, std::vector<Node> &out);

  // Replaces the full set of node's neighbours at node's level with
  // neighbours, discarding any existing edges not present in the new list.
  //
  // A link the updated list names too is left exactly where and as it is,
  // incoming flag included: the flag records that the neighbour is not known to
  // link back yet, which only the linking step can settle. The rest of the
  // updated list lands in the slots the dropped links vacate (or in the free
  // ones), and the whole list is then reciprocated as link_neighbours() does,
  // under the same lock -- out reports the neighbours whose own list had no
  // room for the edge, exactly as it does there.
  //
  // A dropped link is severed on the far side as well, unless the neighbour has
  // an outgoing link of its own back to node: only node's half of that edge is
  // being dropped, so node goes on recording the neighbour -- as an incoming
  // link in its overflow chain, where such a record lives once it is out of the
  // primary list. Conversely, a link the updated list names that node already
  // records in that chain is promoted back into the primary list as a confirmed
  // outgoing link rather than added afresh: a chain record is written while the
  // edge it belongs to is being landed, so its source holds a slot for node
  // already and linking it again would only give it a second one.
  //
  // level is node's own level.
  bool replace_neighbours(const Node &node, LevelId level,
                          const std::vector<Node> &neighbours,
                          std::vector<Node> &out);

  // Severs every edge between node and its neighbours at node's level, one
  // node -- and one storage latch -- at a time: node's own links are first all
  // demoted to incoming links, taking node out of every traversal while still
  // recording which neighbours are left to visit, then each neighbour's link
  // back to node is removed. The X-mode Level Operation Lock held throughout
  // is what keeps the two halves consistent (see the lock hierarchy above).
  //
  // Every edge is severed, those of orphaned neighbours -- the ones left with
  // no outgoing link of their own -- included. What tells them apart
  // afterwards is node's own list: their slots are the only ones left behind
  // there (as incoming links), every other link having been freed. A caller
  // that reconnects them can call this again to have those slots freed too,
  // now that they are no longer orphaned.
  //
  // Only a neighbour that really did link back to node is ever called an
  // orphan of it. A link nothing answers to -- a placeholder its source never
  // reciprocated, or one naming a node since freed -- has nothing to sever, so
  // its slot is simply freed and it is left out of both counts.
  //
  // out is set to the neighbours severed from node that keep an outgoing link
  // of their own, capped at Mmax for level: out serves only as a pool of
  // still-connected nodes to search from, so there is no use for more of them
  // than a node's own degree limit. level is node's own level.
  bool unlink_neighbours(const Node &node, LevelId level,
                         std::vector<Node> &out);

  // Configured construction parameters.
  uint32_t M() const;
  uint32_t ef_construction() const;

  // Maximum neighbour degree permitted at level (Mmax, or Mmax0 = 2*M at
  // level 0, per the HNSW paper).
  uint32_t Mmax(LevelId level) const;

private:
  // Lock primitives backing LockGraph/LockLevels.
  void lock_graph(LockMode mode);
  void unlock_graph(LockMode mode);
  void lock_level(LevelId level, LockMode mode);
  void unlock_level(LevelId level, LockMode mode);

  // Convenience accessors for m_ctx.m_error, this call's non-owning error
  // buffer.
  char *get_err_buffer() const { return m_ctx.m_error.data(); }
  uint32_t get_err_buffer_len() const {
    return static_cast<uint32_t>(m_ctx.m_error.size());
  }

#ifndef NDEBUG
  // Debug-only sanity check shared by create_node()/drop_node(): verifies
  // node's NID actually resolves (via IndexStore::locate) to level's store,
  // catching a caller passing the wrong parent/level pairing that would
  // otherwise silently touch the wrong level's storage.
  bool debug_check_level(const Node &node, LevelId level) const;
#endif // NDEBUG

  // Native distance function pointer: at index open, resolve_distance_fn() maps
  // the bound helper's name to the matching native::dist_*, so the leaf distance
  // below is a direct native call instead of a per-call VDF profile-helper
  // dispatch.
  using NativeDistFn = double (*)(const native::Data *, const native::Data *);
  NativeDistFn resolve_distance_fn();

  // The single distance computation both public distance() overloads end at,
  // once each of their operands has been resolved to the vector data it names.
  bool distance(const NodeData &a, const NodeData &b, DistanceType &out);

  // Resolves a node's vid -- which is the server's stable column reference for
  // the vector the node indexes -- back to that vector's data, which the server
  // fills into buf. Each operand of a distance() call gets its own buffer
  // (m_vector_buf_1/m_vector_buf_2), so the first stays valid while the second
  // is resolved.
  bool resolve_node_data(VID vid, ScratchBytes &buf, NodeData &out);

  // One link in a node's overflow chain: the overflow entry to drop (nid),
  // and the record whose Overflow field currently points to it (prev, of
  // kind prev_kind) -- either the owning node's own NeighbourEntry
  // (StoreKind::Neighbour) or an earlier OverflowEntry
  // (StoreKind::Overflow) in the same chain. prev_kind travels with prev so
  // drop_overflow_node() knows which record layout -- and which
  // LevelStore::update() overload -- to use when clearing its Overflow
  // field.
  struct OverflowLink {
    NID nid;
    NID prev;
    StoreKind prev_kind;
  };

  // Drops every overflow entry chained off node at level (via its
  // Overflow field and each subsequent OverflowEntry::overflow), leaving
  // node itself untouched aside from clearing its Overflow field. Called
  // at the start of drop_node() so a node's incoming overflow chain never
  // outlives it.
  bool drop_overflow_nodes(LevelId level, const Node &node);

  // Physically removes the single overflow entry link.nid identifies and
  // clears the Overflow field of the record link.prev/link.prev_kind
  // names. link.nid is always that record's current (and, by
  // construction of drop_overflow_nodes()'s tail-to-head unwind, last
  // remaining) chain successor, so clearing to NID{} rather than splicing
  // in a further successor is always correct.
  bool drop_overflow_node(LevelId level, const OverflowLink &link);

  // link_neighbours() proper, minus the Level Operation Lock: the caller must
  // already hold it, in S mode (link_neighbours()) or X mode (the operations
  // that adjust an existing node's neighbours, e.g. replace_neighbours(),
  // which links its newly added -- and therefore incoming-flagged -- edges
  // under the X lock it already holds).
  bool link_neighbours_locked(const Node &node, LevelId level,
                              std::vector<Node> &out);

  // Records node as an incoming connection in neighbour's overflow chain,
  // for when link_neighbours() finds neighbour's primary neighbour list
  // already full (Algorithm 1, lines 14-15 land the edge here instead).
  // mtr is the caller's already-open mtr, so this joins the same Neighbour
  // Storage Lock ordering the caller established. overflow_head is
  // neighbour's NeighbourEntry::overflow field -- the caller already has it
  // in hand from the same for_update fetch that read neighbour's primary
  // list, on the very page mtr holds exclusively-latched, so it is passed
  // in rather than re-fetched here. level is neighbour's (and node's)
  // level.
  bool add_overflow_incoming(MtrCtx::Ref mtr, LevelId level,
                             const Node &neighbour, NID overflow_head,
                             const Node &node);

  // One endpoint's record of a single edge: the slot, within the record
  // record/kind names, holding the other endpoint's NID -- either a slot in a
  // node's own primary neighbour list (StoreKind::Neighbour) or an incoming
  // slot in one of the OverflowEntry records chained off it
  // (StoreKind::Overflow). kind travels with record so clear_link() knows
  // which record layout -- and which LevelStore::update() overload -- to use.
  struct LinkSlot {
    NID record{};
    StoreKind kind = StoreKind::Neighbour;
    SlotIndex slot{};
  };

  // Clears the single edge record link names, leaving the rest of that record
  // untouched. A Neighbour link is written under mtr, which already holds the
  // record's page latch; an Overflow link gets its own short-lived mtr, per
  // the Overflow-after-Neighbour ordering in the lock hierarchy above.
  bool clear_link(MtrCtx::Ref mtr, LevelId level, const LinkSlot &link);

  // Searches the overflow chain starting at head for the incoming slot
  // holding target, one chain entry per short-lived mtr (as in
  // add_overflow_incoming()). out's slot is left invalid if no entry in the
  // chain names target.
  bool find_overflow_link(LevelId level, NID head, NID target, LinkSlot &out);

  // Resolves nid to a full Node, reading its vid from the owner field of its
  // own record -- for the links recorded as a bare NID in an overflow chain.
  bool resolve_node(LevelId level, NID nid, Node &out);

  // Removes the neighbour's own link back to node, wherever the neighbour
  // records it -- its primary neighbour list, or its overflow chain for a link
  // it had no room for there. Only the neighbour's record is latched; node's
  // own list is adjusted by unlink_neighbours() once every neighbour is done.
  //
  // orphan reports whether the neighbour is left with no outgoing link of its
  // own, and is only ever set for a neighbour that really did have a link back
  // to node. out_neighbour is that neighbour resolved to a full Node, its vid
  // read from the owner field of its own record (a link recorded in an overflow
  // chain carries a NID alone).
  //
  // Nothing having named node is not an error: node's link may be a
  // placeholder incoming link its source never reciprocated (Bidirectional
  // links, step 1 above), one an earlier interrupted run already removed, or
  // one naming a node since freed, which an incoming link may legitimately do
  // (as noted above) and which the read of the neighbour's record fails on.
  // There is then nothing to sever and nothing observed about the neighbour, so
  // out_neighbour is left unset -- which is how the caller tells this case from
  // a severed link -- and orphan stays false.
  bool unlink_neighbour(LevelId level, const Node &node, NID neighbour_nid,
                        Node &out_neighbour, bool &orphan);

  // Removes the neighbour's own record of node -- wherever it keeps it, its
  // primary neighbour list or its overflow chain -- for an edge
  // replace_neighbours() is dropping. Only the neighbour's record is latched;
  // node's own list is adjusted by replace_neighbours() once every dropped link
  // is done.
  //
  // keeps_link reports that the neighbour has an outgoing link of its own back
  // to node, in which case nothing is removed at all: only node's half of that
  // edge is being dropped, and the neighbour's link -- along with node's
  // obligation to go on recording it as an incoming one -- outlives it.
  //
  // Nothing having named node is not an error, for the reasons
  // unlink_neighbour() documents; there is then nothing to remove and
  // keeps_link stays false.
  bool drop_neighbour_link(LevelId level, const Node &node, NID neighbour_nid,
                           bool &keeps_link);

  IndexStore &m_store;
  const Index &m_index;
  Segment::TrxRef m_trx_ref;
  GraphContext m_ctx;
  // Resolved once at construction from the bound helper's name; the leaf
  // distance() calls it directly.
  NativeDistFn m_dist_fn = nullptr;
};

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_GRAPH_H
