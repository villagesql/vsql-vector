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
//   // Configured construction parameters (Algorithm 1's M, efConstruction).
//   uint32_t M() const;
//   uint32_t ef_construction() const;
//
//   // Maximum neighbour degree permitted at 'level' (Mmax / Mmax0).
//   uint32_t Mmax(LevelId level) const;
//
//   // Severs the edges between 'node' and its neighbours at 'level' (node's
//   // own level). No: unlinks only the neighbours that have another edge of
//   // their own and so won't be left orphaned by losing this one; 'out' is
//   // set to the neighbours actually unlinked. Yes: unlinks the rest -- the
//   // ones that do become orphaned; 'out' is set to those.
//   enum class UnlinkOrphans { No, Yes };
//   bool unlink_neighbours(const Node& node, LevelId level,
//                          UnlinkOrphans orphans, std::vector<Node>& out);
template <typename Graph> class GraphOperations {
public:
  using Node = typename Graph::Node;
  using NodeData = typename Graph::NodeData;
  using LevelId = typename Graph::LevelId;

  GraphOperations(Graph &graph) : m_graph(graph) {};

  bool insert(const NodeData &new_node_data);

  bool remove(const Node &target_node, LevelId target_level);

  bool search_knn(const NodeData &query_node_data, uint32_t k,
                  uint32_t ef_search, std::vector<Node> &nearest_nodes);

private:
  using LayerOps = LayerOperations<Graph, AlwaysVisiblePolicy>;

  // Algorithm 1, lines 5-7: ef is fixed at 1 for the greedy descent.
  static constexpr uint32_t GREEDY_DESCENT_EF = 1;

  // Replaces each candidate with its counterpart at the next lower level.
  // level is the level every candidate currently lives at.
  bool advance_to_next_level(std::vector<Node> &candidates, LevelId level);

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
