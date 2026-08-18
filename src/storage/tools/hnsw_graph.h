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

#ifndef VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_HNSW_GRAPH_H_
#define VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_HNSW_GRAPH_H_

#include <cstdint>
#include <ostream>

#include "hnsw_layout.h"
#include "page_reader.h"

namespace svector {
namespace tool {

// Selects how render_hnsw_graph() prints each level's local neighbour graph.
enum class GraphStyle {
  // A spanning tree from the level's local entry node, using tree-drawing
  // connectors (see render_hnsw_graph()'s comment below).
  Tree,
  // One line per reachable node, in BFS order: "<label>(<vid>) -> "
  // "[<neighbour_ref>(<neighbour_vid>), ...] degree=<n>" (only the line's
  // own node gets the bullet, its neighbours are bare "page:slot" refs),
  // with no nesting
  // or connectors -- easier to read once the level has many edges per node.
  List,
};

// Walks and prints the HNSW graph reachable from index_meta's entry point,
// one level at a time -- without needing the caller to supply a root page
// number per level. This works because a NID's value embeds the
// Column::Ref (data page + slot) of its own record (see hnsw_decode_ref),
// and every level's NeighbourEntry column_size is determined by M alone
// (via hnsw_neighbour_column_size), which is itself recovered from level
// 0's column_size. So each node visited is fetched by reading its data page
// directly, then followed level-by-level via its lower_level_nid link, the
// same descent GraphOperations::search_knn makes at query time.
//
// index_meta must be the level-0 primary NeighbourEntry store's decoded
// metadata (see HnswIndexMetadata::is_overflow(); level0_column_size is that
// same store's root page column_size) -- the only store whose metadata
// carries a real entry point. Prints an explanatory error to out instead of
// a graph if that does not hold, or if there is no entry point (empty
// graph).
//
// Per level, walks that level's local neighbour graph starting at the
// level's local entry node (the same vertex's record at that level) and
// prints it per style. A neighbour already visited within the same level's
// walk is printed as a leaf annotated "..." (Tree) or simply not re-visited
// (List, where every node's own line already lists it as a neighbour)
// rather than re-expanded, since HNSW levels are not trees (nodes have
// multiple incoming edges). Only outgoing edges are drawn -- a neighbour
// slot flagged "incoming" (Id<NIDTag>::is_incoming(), mirrored as
// HNSW_NID_INCOMING_BIT) records someone else's edge into this node, not an
// edge out of it, and a node's own overflow chain is incoming-only for the
// same reason (see LevelStore::insert()/link_neighbours() in
// ../../index/hnsw/graph_ops_impl.h) -- so neither contributes to the
// walk.
//
// max_nodes bounds the total number of distinct records fetched and
// expanded across the whole walk (every level combined); once exhausted,
// remaining nodes are printed as unexpanded "(truncated)" leaves (Tree) or
// left out of the walk (List) and the walk stops. verbose additionally
// prints each node's owner VID; ignored for GraphStyle::List, which always
// includes it as part of the format.
void render_hnsw_graph(PageReader &reader, const HnswIndexMetadata &index_meta,
                       uint16_t level0_column_size, uint32_t max_nodes,
                       bool verbose, GraphStyle style, std::ostream &out);

} // namespace tool
} // namespace svector

#endif // VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_HNSW_GRAPH_H_
