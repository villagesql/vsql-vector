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

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_GRAPH_OPS_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_GRAPH_OPS_H

#include <vector>

#include "layer_ops.h"
#include "visibility_policy.h"

namespace svector::hnsw {

// Requirements:
//
// In addition to the Graph requirements documented in layer_ops.h (Node,
// DistanceType, distance(), neighbours()), Graph must provide:
//   using LevelId = ...;
//
//   // Returns the level the new element should be inserted at (Algorithm
//   // 1, line 4).
//   LevelId get_insert_level();
//
//   // Sets 'out'/'out_level' to the graph's current entry points and
//   // their level; 'out' is left empty if the graph has no entry point
//   // yet.
//   bool get_entry_point(std::vector<Node>& out, LevelId& out_level);
//
//   // Registers 'nodes' as the graph's new entry points at 'level'.
//   bool set_entry_point(const std::vector<Node>& nodes, LevelId level);
//
//   // Materializes every level up to and including 'level', so that the
//   // level can be locked and its nodes created. A no-op for levels that
//   // already exist. Called before any level is locked, since locking a
//   // level that does not exist yet is not possible.
//   bool ensure_levels(LevelId level);
//
//   // Configured construction parameters (Algorithm 1's M, efConstruction).
//   uint32_t M() const;
//   uint32_t ef_construction() const;
//
//   // Maximum neighbour degree permitted at 'level' (Mmax / Mmax0).
//   uint32_t Mmax(LevelId level) const;
//
//   // Severs every edge between 'node' and its neighbours at 'level' (node's
//   // own level). 'out' is set to the neighbours whose link slots were freed
//   // -- the ones not left orphaned (with no outgoing link of their own) by
//   // losing this edge -- capped at Mmax(level); an orphaned neighbour keeps
//   // its slot in node's list, which is what records that it still needs
//   // reconnecting.
//   bool unlink_neighbours(const Node& node, LevelId level,
//                          std::vector<Node>& out);
//
//   // Sets 'out' to every node still recorded as an incoming link to 'node'
//   // at 'level' -- the links neighbours() drops. After unlink_neighbours()
//   // these are exactly the neighbours it left orphaned.
//   bool incoming_neighbours(const Node& node, LevelId level,
//                            std::vector<Node>& out);
template <typename Graph> class GraphOperations {
public:
  using Node = typename Graph::Node;
  using NodeData = typename Graph::NodeData;
  using LevelId = typename Graph::LevelId;
  using LayerOps = LayerOperations<Graph, AlwaysVisiblePolicy>;
  using ExtendCandidates = typename LayerOps::ExtendCandidates;
  using KeepPrunedConnections = typename LayerOps::KeepPrunedConnections;

  // Algorithm 4 heuristic parameters used by all GraphOperations calls to
  // consume_heuristic(): insert()'s own-neighbour selection, remove()'s
  // repair-neighbour selection, and shrink_neighbours()'s Mmax reselection.
  //
  // Per Section 3 of the HNSW paper, extendCandidates, when enabled, extends
  // the candidate set with neighbours of the candidates. It defaults to
  // false, and we keep that default.
  //
  // keepPrunedConnections retains pruned candidates to maintain a fixed
  // number of connections per element. We disable it, allowing the heuristic
  // to select the most diverse neighbours without retaining pruned candidates.
  //
  // Keep these centralized as named constants rather than literals at each
  // call site, so there is a single place to change the policy if needed.
  // TODO(villagesql-indexing): Consider making SHOULD_EXTEND_CANDIDATES and
  // SHOULD_KEEP_PRUNED_CONNECTIONS configurable.
  static constexpr ExtendCandidates SHOULD_EXTEND_CANDIDATES =
      ExtendCandidates::No;
  static constexpr KeepPrunedConnections SHOULD_KEEP_PRUNED_CONNECTIONS =
      KeepPrunedConnections::No;

  GraphOperations(Graph &graph) : m_graph(graph) {};

  bool insert(const NodeData &new_node_data);

  bool remove(const Node &target_node, LevelId target_level);

  bool search_knn(const NodeData &query_node_data, uint32_t k,
                  uint32_t ef_search, std::vector<Node> &nearest_nodes);

private:
  // Algorithm 1, lines 5-7: ef is fixed at 1 for the greedy descent.
  static constexpr uint32_t GREEDY_DESCENT_EF = 1;

  // Replaces each candidate with its counterpart at the next lower level.
  // level is the level every candidate currently lives at.
  bool advance_to_next_level(std::vector<Node> &candidates, LevelId level);

  // Wraps LayerOps::consume_heuristic(), defaulting extend_candidates and
  // keep_pruned_connections to SHOULD_EXTEND_CANDIDATES and
  // SHOULD_KEEP_PRUNED_CONNECTIONS so call sites only need to supply what
  // varies -- M, the output, and (optionally) candidate_pool -- while still
  // allowing either policy to be overridden explicitly.
  bool consume_heuristic(
      LayerOps &layer, uint32_t M, std::vector<Node> &out,
      std::vector<Node> *candidate_pool = nullptr,
      ExtendCandidates extend_candidates = SHOULD_EXTEND_CANDIDATES,
      KeepPrunedConnections keep_pruned_connections =
          SHOULD_KEEP_PRUNED_CONNECTIONS);

  // Reselects up to Mmax(level) neighbours for each node in 'overflowed',
  // after 'linked_node' was withheld from being linked back to it by
  // link_neighbours() (Algorithm 1, lines 14-15). level is the level shared
  // by linked_node and every node in overflowed.
  bool shrink_neighbours(LayerOps &layer, const Node &linked_node,
                         LevelId level, const std::vector<Node> &overflowed);

  Graph &m_graph;
};

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_GRAPH_OPS_H
