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

#include "index.h"
#include <cassert>
#include <stack>

namespace svector::hnsw {

class IndexGraph {
public:
  using LevelId = LevelStore::LevelId;

  struct Node {
    using KeyType = IndexScanKey::KeyPartRef;
    using Data = IndexScanKey::KeyPartData;

    KeyType key() const { return ref; }

    void set_data(const Node &other) { materialized = other.materialized; }

    LevelId level;
    KeyType ref;
    Data materialized;
  };

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

  explicit IndexGraph(IndexStore &store) : m_store(store) {}

  bool distance(const Node &a, const Node &b, DistanceType &out);

  bool neighbours(const Node &node, std::vector<Node> &out);

  bool visible(const Node &node, bool &out);

  void set_level(LevelId level);

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

  bool create_node(const std::optional<Node> &parent, LevelId level,
                   Node &node);

  bool drop_node(const std::optional<Node> &parent, const Node &node);

  bool get_child(const Node &parent, Node &out);

  bool link_neighbours(const Node &node, const std::vector<Node> &neighbours);

  // Adds the reciprocal edges from node's neighbours back to node. If doing
  // so would push a neighbour's degree past Mmax for node's level, the edge
  // is withheld and that neighbour is appended to out instead, leaving
  // GraphOperations::insert() to reselect and replace its full neighbour set
  // (Algorithm 1, lines 14-15).
  bool link_neighbours_back(const Node &node, std::vector<Node> &out);

  // Replaces the full set of node's neighbours at node's level with
  // neighbours, discarding any existing edges not present in the new list.
  bool replace_neighbours(const Node &node,
                          const std::vector<Node> &neighbours);

  // No: unlink only the neighbours that have another edge of their own and
  // so won't be left orphaned by losing this one; out is set to the
  // neighbours actually unlinked. Yes: unlink the rest (the ones that do
  // become orphaned); out is set to those.
  enum class UnlinkOrphans { No, Yes };
  bool unlink_neighbours(const Node &node, UnlinkOrphans orphans,
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

  IndexStore &m_store;
};

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_GRAPH_H
