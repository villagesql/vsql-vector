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

#ifndef VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_HNSW_DOT_H_
#define VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_HNSW_DOT_H_

#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svector {
namespace tool {

// Graphviz DOT export of the HNSW graph. This file holds the pure part: an
// in-memory model of each level's adjacency, a breadth-first collector that
// fills it through a caller-supplied record fetcher, and the DOT writer. It
// has no dependency on PageReader or the SDK, so it is unit-testable with a
// fake fetcher; hnsw_graph.cc supplies the real one (reusing the same record
// decoding as the ASCII renderers).

// One decoded NeighbourEntry, reduced to what the graph needs.
struct HnswNodeRecord {
  uint64_t owner_vid = 0;
  // Delete-marked: the record is still allocated (and so still traversed).
  bool is_deleted = false;
  // Transaction id (decimal) that last wrote the record.
  uint64_t trx_ref = 0;
  // Lower-level NID ref of the same vertex; 0 when absent.
  uint64_t lower_level_nid = 0;
  // Outgoing slots only, as (nid ref, vid): incoming-flagged and empty slots
  // are already dropped by the fetcher.
  std::vector<std::pair<uint64_t, uint64_t>> neighbours;
};

// Fetches nid_ref's record at level. On failure returns false with error set
// to why (page unreadable, slot not live, ...).
using HnswRecordFetcher = std::function<bool(
    uint64_t nid_ref, uint8_t level, HnswNodeRecord &rec, std::string &error)>;

enum class HnswNodeState {
  Expanded,   // Record fetched; its outgoing edges are all in the model.
  Truncated,  // Seen as a neighbour but the node budget ran out first.
  Unresolved, // Referenced, but the record could not be read as a live node.
  Unreachable // Live record the entry-point walk never reached.
};

struct HnswGraphNode {
  uint64_t nid = 0; // This level's record identity (page:slot).
  uint64_t vid = 0; // The vertex's identity across levels; valid if has_vid.
  bool has_vid = false;
  HnswNodeState state = HnswNodeState::Expanded;
  bool is_deleted = false;
  uint64_t trx_ref = 0;  // Valid once the record was read (has_vid nodes).
  bool is_entry = false; // The level's local entry node.
  std::string error;     // Why state == Unresolved.
};

struct HnswGraphEdge {
  uint64_t from = 0;
  uint64_t to = 0;
};

struct HnswLevelGraph {
  uint8_t level = 0;
  std::vector<HnswGraphNode> nodes;
  std::vector<HnswGraphEdge> edges; // Distinct outgoing edges, one per entry.
  std::unordered_map<uint64_t, size_t> node_index; // nid -> nodes[] position
  uint32_t duplicate_edges = 0; // Repeats of an edge within one adjacency list.
};

struct HnswGraphModel {
  uint32_t M = 0;
  uint8_t entry_level = 0;
  uint32_t max_nodes = 0;
  std::vector<HnswLevelGraph> levels; // Top level first.
  // True if the walk stopped short of the full reachable graph: either some
  // node was left unexpanded or a lower level was never walked.
  bool truncated = false;
  // Free-text notes shown in the diagram (why a walk stopped, why the
  // unreachable analysis was skipped, ...).
  std::vector<std::string> notes;
  // Level-0 live-record count; set only when the unreachable analysis ran.
  bool has_allocated_count = false;
  uint32_t allocated_level0 = 0;
  uint32_t unreachable_total = 0; // May exceed the number actually drawn.
};

// Walks entry_level down to level 0 from entry_ref, one breadth-first walk
// per level starting at that level's local entry node (the lower_level_nid of
// the previous level's entry), sharing one node budget of max_nodes across
// all levels. Only edges actually stored in a fetched record are recorded;
// nothing is inferred.
HnswGraphModel collect_hnsw_graph(uint64_t entry_ref, uint8_t entry_level,
                                  uint32_t M, uint32_t max_nodes,
                                  const HnswRecordFetcher &fetch);

// Adds every record in live_level0 that the walk did not reach to the level-0
// graph as an Unreachable node with its outgoing edges, up to max_nodes of
// them. live_level0 must be the complete set of live level-0 records, and the
// walk must not have been truncated (otherwise unvisited nodes cannot be told
// from unreachable ones); returns false, adding a note instead, if the model
// is truncated. A neighbour of such a node that is not in live_level0 becomes
// an Unresolved node.
bool add_unreachable_level0(
    HnswGraphModel &model,
    const std::vector<std::pair<uint64_t, HnswNodeRecord>> &live_level0);

// Writes model as one DOT digraph with one cluster per level. level_filter
// >= 0 emits only that level. Reciprocal edge pairs are one bold double-
// headed edge; one-way edges are thin single-headed; edges to a node whose
// adjacency is unknown are dashed (reciprocity cannot be judged).
void write_hnsw_dot(const HnswGraphModel &model, int level_filter,
                    std::ostream &out);

// Escapes s for use inside a double-quoted DOT string.
std::string dot_escape(const std::string &s);

} // namespace tool
} // namespace svector

#endif // VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_HNSW_DOT_H_
