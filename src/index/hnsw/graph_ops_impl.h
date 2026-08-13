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
#include <cassert>
#include <vector>

#include "graph_ops.h"
#include "layer_ops.h"
#include "visibility_policy.h"

namespace svector::hnsw {

// Algorithm 1, INSERT.
template <typename Graph>
bool GraphOperations<Graph>::insert(const NodeData &new_node_data,
                                    Node &out_node) {
  using LockMode = typename Graph::LockMode;
  using LockGraph = typename Graph::LockGraph;
  using LockLevels = typename Graph::LockLevels;
  using DescendPolicy = typename Graph::LockLevels::DescendPolicy;

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

  // Materialize the levels this insert will touch before locking any of
  // them: when insert_level > entry_level the top levels do not exist yet,
  // and a level cannot be locked until it does. Mutating level storage
  // requires the graph lock in X mode, which is exactly the case that
  // reached the upgrade above -- when level <= entry_level every level
  // already exists and this is a no-op.
  if (m_graph.ensure_levels(level)) {
    return true;
  }

  LockLevels levels(m_graph, LockMode::Shared, level, DescendPolicy::Release);
  LayerOps layer(m_graph, new_node_data);

  // Traverse the levels until we reach the insertion level.
  // Lines 5-7: greedily descend from L down to l+1 with ef=1, narrowing to
  // the single nearest element found at each layer.
  while (level > insert_level) {
    // Search current layer to zoom into nearest candidates.
    if (layer.search(candidates, level, GREEDY_DESCENT_EF)) {
      return true;
    }
    layer.consume_all(candidates);
    // Prepare for the next layer by replacing the current candidates
    // with their next-level counterparts.
    assert(level.has_lower_level());
    if (advance_to_next_level(candidates, level)) {
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
  // and connect q to its selected neighbours at each layer. Levels above L
  // (only possible when l > L, i.e. q is becoming the new tallest entry
  // point) have nothing else in them yet -- nothing to search or link
  // against -- so q is created there with no neighbours instead; candidates
  // stays exactly what get_entry_point() returned until level actually
  // reaches L, at which point it's finally valid to search with.
  for (;;) {
    std::vector<Node> neighbours;
    if (level > entry_level) {
      // Brand-new top level: no search, no neighbours.
    } else {
      // Line 17 (folded in early): search() populates W, which then also
      // serves as ep for the next (lower) layer's search -- consume_heuristic()
      // below hands it back via candidate_pool, straight into candidates,
      // without re-evaluating any of its distances.
      if (layer.search(candidates, level, m_graph.ef_construction())) {
        return true;
      }
      if (consume_heuristic(layer, m_graph.M(), neighbours, &candidates)) {
        return true;
      }
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
        // candidates is this level's real search result -- translate it to
        // its counterparts one level down before searching there next.
        if (advance_to_next_level(candidates, level)) {
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
    if (m_graph.link_neighbours(node, node_level, overflowed)) {
      return true;
    }
    if (shrink_neighbours(layer, node, node_level, overflowed)) {
      return true;
    }
  }
  out_node = top_node;

  // Lines 18-19: Make q the new entry point if the graph is empty or if q's
  // level exceeds the previous top level.
  if (new_entry_point) {
    assert(graph_lock.lock_mode() == LockMode::Exclusive);
    return m_graph.set_entry_point({top_node}, insert_level);
  }
  return false;
}

// A vector is one node per level, each with its own delete mark, so marking
// the vector means walking that whole chain: descend from target_node's top
// level down to level 0, marking every node along the way. Nothing structural
// changes -- the marked nodes keep every edge they have and stay navigable --
// so this only needs each level to hold still while it is visited, which the
// Shared locks give it: the destructive multi-level delete it has to be kept
// out of takes them exclusively (remove()'s phase 1).
//
// An error partway down is reported as-is, leaving the levels already visited
// marked. Unwinding them here would buy nothing: a crash at that same point
// leaves exactly the same partially marked chain with nothing left running to
// undo it, so rollback has to cope with one regardless -- which makes it the
// one place that has to get this right, and an in-line unwind redundant.
template <typename Graph>
bool GraphOperations<Graph>::mark_delete(const Node &target_node,
                                         LevelId target_level,
                                         bool delete_mark) {
  using LockMode = typename Graph::LockMode;
  using LockGraph = typename Graph::LockGraph;
  using LockLevels = typename Graph::LockLevels;
  using DescendPolicy = typename Graph::LockLevels::DescendPolicy;

  // Nothing here touches the graph-wide metadata this lock protects: it is
  // held only because the level locks below it are never acquired without it
  // (see the lock hierarchy in graph.h).
  LockGraph graph_lock(m_graph, LockMode::Shared);
  LockLevels levels(m_graph, LockMode::Shared, target_level,
                    DescendPolicy::Release);

  Node current = target_node;
  for (LevelId level = target_level;;) {
    if (m_graph.mark_delete(current, level, delete_mark)) {
      return true;
    }
    if (!level.has_lower_level()) {
      return false;
    }
    Node next{};
    if (m_graph.get_next_level_node(current, level, next)) {
      return true;
    }
    current = next;
    level = levels.descend();
  }
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

  std::stack<std::pair<Node, LevelId>> nodes;

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
      // Sever every edge at this level. The neighbours that keep an edge of
      // their own come back in unlinked -- they're good entry points for
      // repairing the ones that don't.
      std::vector<Node> unlinked;
      if (m_graph.unlink_neighbours(current, level, unlinked)) {
        return true;
      }

      // The rest were orphaned by losing their edge to current, and are the
      // links current is still recorded as the target of. Repair each of them
      // at this same level -- being orphaned is a level-specific phenomenon,
      // so we never descend further to do it -- using the survivors above as
      // search_layer's entry points.
      std::vector<Node> orphaned;
      if (m_graph.incoming_neighbours(current, level, orphaned)) {
        return true;
      }
      // TODO(villagesql-indexing): Validate/repair stale incoming links on
      // all surviving neighbours, not just those orphaned by this removal.
      // Such links may be left behind by a failure while replacing neighbours.
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
        if (layer.search(candidates, level, m_graph.ef_construction())) {
          return true;
        }
        std::vector<Node> new_neighbours;
        if (consume_heuristic(layer, m_graph.M(), new_neighbours)) {
          return true;
        }
        // The search seeds its result set with every entry point that isn't
        // the query node itself, construction treats every node as visible,
        // and the nearest result can never be dominated -- so a non-empty
        // candidates always yields at least one neighbour. unlinked can't
        // consist of orphan alone either: it is appended to after the repair,
        // never before, and what unlink_neighbours() put there is disjoint
        // from the orphans by construction.
        assert(!new_neighbours.empty());
        // replace_neighbours() adds the reciprocal edges itself, under the same
        // lock, and reports the neighbours whose own list had no room for one.
        std::vector<Node> overflowed;
        if (m_graph.replace_neighbours(orphan, level, new_neighbours,
                                       overflowed)) {
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

      // The repaired orphans stay recorded as incoming links on current,
      // which nothing reads any more: no node points at current at this
      // level, and drop_node() below frees the record they live in.
      nodes.push({current, level});
      if (!level.has_lower_level()) {
        break;
      }
      Node next{};
      if (m_graph.get_next_level_node(current, level, next)) {
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
    auto [current, current_level] = nodes.top();
    nodes.pop();
    std::optional<Node> parent;
    if (!nodes.empty()) {
      parent = nodes.top().first;
    }
    if (m_graph.drop_node(parent, current_level, current)) {
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
    if (upper_layer.search(candidates, level, GREEDY_DESCENT_EF)) {
      return true;
    }
    upper_layer.consume_all(candidates);
    // Prepare for the next layer by replacing the current candidates
    // with their next-level counterparts.
    assert(level.has_lower_level());
    if (advance_to_next_level(candidates, level)) {
      return true;
    }
  }

  // Line 7: search the bottom layer with ef=ef_search. This is the only
  // layer whose result is returned, so it's the only one filtered for
  // visibility.
  BottomLayerOps bottom_layer(m_graph, query_node_data);
  if (bottom_layer.search(candidates, LevelId{0}, ef_search)) {
    return true;
  }

  // Line 8: return K nearest elements from candidates. consume_simple()
  // truncates the (already ascending-by-distance) search result to the K
  // nearest directly.
  bottom_layer.consume_simple(k, candidates);
  nearest_nodes = std::move(candidates);
  return false;
}

template <typename Graph>
bool GraphOperations<Graph>::advance_to_next_level(
    std::vector<Node> &candidates, LevelId level) {
  for (Node &n : candidates) {
    Node next{};
    if (m_graph.get_next_level_node(n, level, next)) {
      return true;
    }
    n = next;
  }
  return false;
}

template <typename Graph>
bool GraphOperations<Graph>::consume_heuristic(
    LayerOps &layer, uint32_t M, std::vector<Node> &out,
    std::vector<Node> *candidate_pool, ExtendCandidates extend_candidates,
    KeepPrunedConnections keep_pruned_connections) {
  return layer.consume_heuristic(M, extend_candidates, keep_pruned_connections,
                                 out, candidate_pool);
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
  // Whatever replace_neighbours() has to withhold in turn is left withheld:
  // the shrink stops at the neighbours linked_node itself overflowed, and does
  // not go on replacing the neighbours of those. The graph stays consistent
  // either way -- a withheld edge is recorded as an incoming link on the node
  // that had no room for it -- and this keeps one insert from cascading into an
  // unbounded rewrite of the layer, which the heuristic's extendCandidates can
  // otherwise feed by selecting nodes that were not connections to begin with.
  std::vector<Node> withheld;
  for (const Node &neighbour : overflowed) {
    std::vector<Node> connections;
    if (m_graph.neighbours(neighbour, level, connections)) {
      return true;
    }
    connections.push_back(linked_node);

    layer.reset(neighbour);
    // No preceding graph traversal is needed here -- connections is
    // already the exact candidate set to select Mmax(level) neighbours
    // from -- so seed() (Step 1 without the expansion loop) stands in for
    // search().
    if (layer.seed(connections, level)) {
      return true;
    }
    std::vector<Node> shrunk;
    if (consume_heuristic(layer, m_graph.Mmax(level), shrunk)) {
      return true;
    }
    if (m_graph.replace_neighbours(neighbour, level, shrunk, withheld)) {
      return true;
    }
  }
  return false;
}

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_GRAPH_OPS_IMPL_H
