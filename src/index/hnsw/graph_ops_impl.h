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

// Out-of-line template definition for Inserter, declared in graph_ops.h.
// Split out into this header (rather than folded into graph_ops.h
// directly) so a caller can explicitly instantiate Inserter for its own
// Graph without also pulling in graph.h -- see graph_ops.cc for the
// IndexGraph instantiation.

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_GRAPH_OPS_IMPL_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_GRAPH_OPS_IMPL_H

#include <algorithm>
#include <vector>

#include "graph_ops.h"
#include "layer_ops.h"
#include "visibility_policy.h"

namespace svector::hnsw {

// Algorithm 1, INSERT.
template <typename Graph>
bool GraphOperations<Graph>::insert(const NodeData &new_node_data) {
  using LockMode = typename Graph::LockMode;
  using LockGraph = typename Graph::LockGraph;
  using LockLevels = typename Graph::LockLevels;
  using DescendPolicy = typename Graph::LockLevels::DescendPolicy;

  using LayerOps = LayerOperations<Graph, AlwaysVisiblePolicy>;
  using ExtendCandidates = typename LayerOps::ExtendCandidates;
  using KeepPrunedConnections = typename LayerOps::KeepPrunedConnections;

  // Lock graph in Shared Mode.
  LockGraph graph_lock(m_graph, LockMode::Shared);

  // Lines 2-3: ep <- entry point for hnsw, L <- level of ep.
  std::vector<Node> candidates;
  LevelId entry_level{0};
  if (m_graph.get_entry_point(candidates, entry_level)) {
    return true;
  }

  // Line 4: l <- new element's level.
  const LevelId insert_level = m_graph.get_insert_level();
  bool new_entry_point = false;

  if (candidates.empty() || insert_level > entry_level) {
    // A new entry point may be required, so acquire the graph in X mode.
    graph_lock.upgrade();

    // Upgrade is not atomic, so another thread may have modified the
    // entry point while no lock was held. Re-read the current state.
    if (m_graph.get_entry_point(candidates, entry_level)) {
      return true;
    }
    new_entry_point = (candidates.empty() || insert_level > entry_level);
    // Intentionally keep the X lock even if a new entry point is no longer
    // required. Avoiding an X → S downgrade prevents repeated lock mode
    // transitions under contention, and this path is expected to be rare.
  }

  auto level = std::max(entry_level, insert_level);
  LockLevels levels(m_graph, LockMode::Shared, level, DescendPolicy::Release);
  LayerOps layer(m_graph, new_node_data);

  // Traverse the levels until we reach the insertion level.
  // Lines 5-7: greedily descend from L down to l+1 with ef=1, narrowing to
  // the single nearest element found at each layer.
  while (level > insert_level) {
    // Search current layer to zoom into nearest candidates.
    if (layer.search_layer(candidates, GREEDY_DESCENT_EF)) {
      return true;
    }
    // Prepare for the next layer by replacing the current candidates
    // with their next-level counterparts.
    assert(level.has_lower_level());
    if (advance_to_next_level(candidates)) {
      return true;
    }
    // Acquire level lock for next layer.
    level = levels.descend();
  }

  assert(level == insert_level);
  levels.update_policy(DescendPolicy::Keep);

  std::stack<std::pair<Node, LevelId>> inserted_nodes;
  std::optional<Node> parent;
  Node top_node{};

  // Lines 8-16: from min(L, l) down to 0, gather efConstruction candidates
  // and connect q to its selected neighbours at each layer.
  for (;;) {
    // Line 17 (folded in early): search_layer() replaces candidates with W,
    // which then also serves as ep for the next (lower) layer's search.
    if (layer.search_layer(candidates, m_graph.ef_construction())) {
      return true;
    }
    std::vector<Node> neighbours;
    // Use the standard HNSW Algorithm 4 settings:
    //   - extend_candidates = false
    //   - keep_pruned_connections = true
    // TODO(villagesql-indexing): Consider making these options configurable.
    if (layer.select_neighbours_heuristic(
            candidates, m_graph.M(), ExtendCandidates::No,
            KeepPrunedConnections::Yes, neighbours)) {
      return true;
    }
    // create_node() also creates the outgoing links to neighbours.
    // Reciprocal links are added after reaching layer 0 by traversing the
    // inserted nodes bottom-up. This preserves the invariant that any node
    // reachable at level N has already been fully inserted at all lower
    // levels.
    Node node{};
    if (m_graph.create_node(parent, level, new_node_data, neighbours, node)) {
      return true;
    }

    if (!parent) {
      top_node = node;
    }

    parent = node;
    inserted_nodes.push({node, level});

    if (level.has_lower_level()) {
      if (level <= entry_level) {
        // candidates is this level's real search result -- promote it to
        // its counterparts one level down before searching there next.
        // Above entry_level nothing real has been searched yet, so there's
        // nothing to promote.
        if (advance_to_next_level(candidates)) {
          return true;
        }
      }
      level = levels.descend();
    } else {
      break;
    }
  }

  // Complete the insertion by adding the reciprocal links bottom-up. Shared
  // locks are retained across all levels to prevent concurrent deletion of
  // the inserted nodes while still allowing concurrent searches and inserts.
  while (!inserted_nodes.empty()) {
    auto [node, node_level] = inserted_nodes.top();
    inserted_nodes.pop();

    // Neighbours whose degree would exceed Mmax once linked back to node;
    // link_neighbours() withholds those edges rather than adding them
    // outright, leaving shrink_neighbours() below to decide their final
    // neighbour set (Algorithm 1, lines 14-15).
    std::vector<Node> overflowed;
    if (m_graph.link_neighbours(node, overflowed)) {
      return true;
    }
    if (shrink_neighbours(layer, node, node_level, overflowed)) {
      return true;
    }
  }

  // Lines 18-19: Make q the new entry point if the graph is empty or if q's
  // level exceeds the previous top level.
  if (new_entry_point) {
    assert(graph_lock.lock_mode() == LockMode::Exclusive);
    return m_graph.set_entry_point({top_node}, insert_level);
  }
  return false;
}

// Two-phase delete: first sever target_node's edges at every level (X lock,
// hand-over-hand descent) -- repairing each neighbour that would otherwise
// be left orphaned at that same level along the way -- then remove the
// per-level records bottom-up (S lock, held across all levels).
template <typename Graph>
bool GraphOperations<Graph>::remove(const Node &target_node,
                                    LevelId target_level) {
  using LockMode = typename Graph::LockMode;
  using LockGraph = typename Graph::LockGraph;
  using LockLevels = typename Graph::LockLevels;
  using DescendPolicy = typename Graph::LockLevels::DescendPolicy;
  using UnlinkOrphans = typename Graph::UnlinkOrphans;

  using LayerOps = LayerOperations<Graph, AlwaysVisiblePolicy>;
  using ExtendCandidates = typename LayerOps::ExtendCandidates;
  using KeepPrunedConnections = typename LayerOps::KeepPrunedConnections;

  // Lock graph in Shared Mode.
  LockGraph graph_lock(m_graph, LockMode::Shared);

  // target_node may be the graph's entry point; check the same way
  // insert() checks whether a new one is needed. Only target_node being
  // the *sole* entry point matters here: if others remain, they're still
  // valid and target_node simply drops out of contention without needing
  // a replacement.
  std::vector<Node> entry_points;
  LevelId entry_level{0};
  if (m_graph.get_entry_point(entry_points, entry_level)) {
    return true;
  }
  auto is_sole_entry_point = [&] {
    return target_level == entry_level && entry_points.size() == 1 &&
           entry_points.front().key() == target_node.key();
  };
  bool new_entry_point = false;

  if (is_sole_entry_point()) {
    // Removing the entry point may require selecting a new one, so
    // acquire the graph in X mode.
    graph_lock.upgrade();

    // Upgrade is not atomic, so another thread may have modified the
    // entry point while no lock was held. Re-read the current state.
    if (m_graph.get_entry_point(entry_points, entry_level)) {
      return true;
    }
    new_entry_point = is_sole_entry_point();
    // Intentionally keep the X lock even if a new entry point is no
    // longer required -- as in insert(), avoiding an X → S downgrade
    // prevents repeated lock mode transitions under contention, and this
    // path is expected to be rare.
  }

  std::stack<Node> nodes;

  // If target_node turns out to be the entry point, its replacement is
  // the first survivor found at the topmost level that still has one --
  // the levels below are visited top-down, so the first non-empty
  // 'unlinked' set below is exactly that. Left unset if nothing survives
  // target_node's removal at any level, meaning target_node was the
  // graph's only element.
  std::optional<Node> replacement;
  LevelId replacement_level{0};

  // Reused across every level and every orphaned neighbour below --
  // reset() rebinds the query without reallocating scratch buffers.
  LayerOps layer(m_graph, target_node);

  // Phase 1: descend from target_node's top level down to level 0 in X
  // mode (lock-coupled), severing target_node's edges at every level.
  // Once this completes target_node is unreachable from any concurrent
  // traversal -- nothing else in the graph points to it any more -- so
  // it's safe to drop every level lock before phase 2 re-acquires them.
  {
    LevelId level = target_level;
    LockLevels levels(m_graph, LockMode::Exclusive, target_level,
                      DescendPolicy::Release);
    Node current = target_node;
    for (;;) {
      // First, sever only the edges that won't orphan their other
      // endpoint, keeping the rest intact -- they're good entry points
      // for repairing the ones that would be orphaned.
      std::vector<Node> unlinked;
      if (m_graph.unlink_neighbours(current, UnlinkOrphans::No, unlinked)) {
        return true;
      }

      // Whatever is still linked to current would be orphaned once its
      // last edge (to current) is cut. Repair each of them at this same
      // level -- being orphaned is a level-specific phenomenon, so we
      // never descend further to do it -- using the survivors above as
      // search_layer's entry points.
      std::vector<Node> orphaned;
      if (m_graph.neighbours(current, orphaned)) {
        return true;
      }
      for (const Node &orphan : orphaned) {
        if (unlinked.empty()) {
          // current was the sole connection for everything left at this
          // level, so there's nothing to seed a search from yet. Seed
          // with this orphan itself so later orphans in this loop have
          // something to attach to (and, via link_neighbours below, a way
          // to end up connected to them in turn).
          unlinked.push_back(orphan);
          continue;
        }
        std::vector<Node> candidates(unlinked);
        layer.reset(orphan);
        if (layer.search_layer(candidates, m_graph.ef_construction())) {
          return true;
        }
        std::vector<Node> new_neighbours;
        if (layer.select_neighbours_heuristic(
                candidates, m_graph.M(), ExtendCandidates::No,
                KeepPrunedConnections::Yes, new_neighbours)) {
          return true;
        }
        if (m_graph.replace_neighbours(orphan, new_neighbours)) {
          return true;
        }
        std::vector<Node> overflowed;
        if (m_graph.link_neighbours(orphan, overflowed)) {
          return true;
        }
        if (shrink_neighbours(layer, orphan, level, overflowed)) {
          return true;
        }
        // Grows the entry point pool for subsequent orphans as this
        // level's repaired core expands.
        unlinked.push_back(orphan);
      }

      if (!replacement && !unlinked.empty()) {
        replacement = unlinked.front();
        replacement_level = level;
      }

      // Now that the orphaned ones are repaired elsewhere, sever the
      // remaining edges too.
      std::vector<Node> unused;
      if (m_graph.unlink_neighbours(current, UnlinkOrphans::Yes, unused)) {
        return true;
      }

      nodes.push(current);
      if (!level.has_lower_level()) {
        break;
      }
      Node next{};
      if (m_graph.get_next_level(current, next)) {
        return true;
      }
      current = next;
      level = levels.descend();
    }
  }

  // Phase 2: re-descend from the top level down to level 0, this time in
  // Shared mode and holding every level (Keep). LockLevels only ever
  // descends, so re-acquiring top-down is the only way to end up holding
  // every level needed to unwind the stack bottom-up.
  LockLevels levels(m_graph, LockMode::Shared, target_level,
                    DescendPolicy::Keep);
  while (levels.level().has_lower_level()) {
    levels.descend();
  }

  // Unwind bottom-up (level 0 first), dropping each level's node and
  // passing its parent -- the node one level up, gathered in phase 1 -- so
  // the parent's next-level pointer can be fixed up too.
  while (!nodes.empty()) {
    Node current = nodes.top();
    nodes.pop();
    std::optional<Node> parent;
    if (!nodes.empty()) {
      parent = nodes.top();
    }
    if (m_graph.drop_node(parent, current)) {
      return true;
    }
  }

  if (new_entry_point) {
    assert(graph_lock.lock_mode() == LockMode::Exclusive);
    if (replacement) {
      return m_graph.set_entry_point({*replacement}, replacement_level);
    }
    // target_node was the graph's only element; there's nothing left to
    // point to.
    return m_graph.set_entry_point({}, LevelId{0});
  }

  return false;
}

// Algorithm 5, K-NN-SEARCH.
template <typename Graph>
bool GraphOperations<Graph>::search_knn(const NodeData &query_node_data,
                                        uint32_t k, uint32_t ef_search,
                                        std::vector<Node> &nearest_nodes) {
  using LockMode = typename Graph::LockMode;
  using LockGraph = typename Graph::LockGraph;
  using LockLevels = typename Graph::LockLevels;
  using DescendPolicy = typename Graph::LockLevels::DescendPolicy;

  // Upper layers are pure navigation: a deleted-but-still-linked node is a
  // perfectly good stepping stone towards the bottom layer, so visibility
  // is irrelevant there. Only the bottom layer's search actually produces
  // the result set, so that's the only one filtered for visibility.
  using UpperLayerOps = LayerOperations<Graph, AlwaysVisiblePolicy>;
  using BottomLayerOps = LayerOperations<Graph, GraphVisiblePolicy>;

  assert(k <= ef_search);

  // Lock graph in Shared Mode.
  LockGraph graph_lock(m_graph, LockMode::Shared);

  // Lines 2-3: ep <- entry point for hnsw, L <- level of ep.
  std::vector<Node> candidates;
  LevelId entry_level{0};
  if (m_graph.get_entry_point(candidates, entry_level)) {
    return true;
  }
  if (candidates.empty()) {
    nearest_nodes.clear();
    return false;
  }

  LockLevels levels(m_graph, LockMode::Shared, entry_level,
                    DescendPolicy::Release);
  UpperLayerOps upper_layer(m_graph, query_node_data);

  // Lines 4-6: greedily descend from L down to 1 with ef=1, narrowing to
  // the single nearest element found at each layer.
  for (auto level = entry_level; level.has_lower_level();
       level = levels.descend()) {
    if (upper_layer.search_layer(candidates, GREEDY_DESCENT_EF)) {
      return true;
    }
    // Prepare for the next layer by replacing the current candidates
    // with their next-level counterparts.
    assert(level.has_lower_level());
    if (advance_to_next_level(candidates)) {
      return true;
    }
  }

  // Line 7: search the bottom layer with ef=ef_search. This is the only
  // layer whose result is returned, so it's the only one filtered for
  // visibility.
  BottomLayerOps bottom_layer(m_graph, query_node_data);
  if (bottom_layer.search_layer(candidates, ef_search)) {
    return true;
  }

  // Line 8: return K nearest elements from candidates. search_layer()
  // leaves candidates ordered by increasing distance, so the K nearest are
  // simply its first K elements.
  if (candidates.size() > k) {
    candidates.resize(k);
  }
  nearest_nodes = std::move(candidates);
  return false;
}

template <typename Graph>
bool GraphOperations<Graph>::advance_to_next_level(
    std::vector<Node> &candidates) {
  for (Node &n : candidates) {
    Node next{};
    if (m_graph.get_next_level(n, next)) {
      return true;
    }
    n = next;
  }
  return false;
}

// Algorithm 1, lines 14-15: shrink connections of neighbours whose degree
// would otherwise exceed Mmax. link_neighbours() withholds the edge to
// linked_node for exactly these nodes rather than adding it outright, so
// linked_node is re-added here as a plain candidate alongside each node's
// existing connections and the heuristic decides whether it survives.
template <typename Graph>
bool GraphOperations<Graph>::shrink_neighbours(
    LayerOps &layer, const Node &linked_node, LevelId level,
    const std::vector<Node> &overflowed) {
  using ExtendCandidates = typename LayerOps::ExtendCandidates;
  using KeepPrunedConnections = typename LayerOps::KeepPrunedConnections;

  for (const Node &neighbour : overflowed) {
    std::vector<Node> connections;
    if (m_graph.neighbours(neighbour, connections)) {
      return true;
    }
    connections.push_back(linked_node);

    layer.reset(neighbour);
    std::vector<Node> shrunk;
    if (layer.select_neighbours_heuristic(connections, m_graph.Mmax(level),
                                          ExtendCandidates::No,
                                          KeepPrunedConnections::No, shrunk)) {
      return true;
    }
    if (m_graph.replace_neighbours(neighbour, shrunk)) {
      return true;
    }
  }
  return false;
}

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_GRAPH_OPS_IMPL_H
