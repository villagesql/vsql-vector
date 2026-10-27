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

#include "hnsw_dot.h"

#include <set>
#include <unordered_set>

#include "hnsw_layout.h"

namespace svector {
namespace tool {

namespace {

using EdgeKey = std::pair<uint64_t, uint64_t>;

std::string ref_text(uint64_t ref_value) {
  HnswColumnRef ref = hnsw_decode_ref(ref_value);
  return std::to_string(ref.page_ref) + ":" + std::to_string(ref.slot_index);
}

// Returns nid's position in g.nodes, adding an (as yet unexpanded) node with
// the given neighbour-supplied vid if it is new.
size_t ensure_node(HnswLevelGraph &g, uint64_t nid, uint64_t vid, bool has_vid,
                   bool &added) {
  auto it = g.node_index.find(nid);
  if (it != g.node_index.end()) {
    added = false;
    HnswGraphNode &node = g.nodes[it->second];
    if (!node.has_vid && has_vid) {
      node.vid = vid;
      node.has_vid = true;
    }
    return it->second;
  }
  added = true;
  HnswGraphNode node;
  node.nid = nid;
  node.vid = vid;
  node.has_vid = has_vid;
  g.nodes.push_back(node);
  g.node_index.emplace(nid, g.nodes.size() - 1);
  return g.nodes.size() - 1;
}

// Records from -> to once per level, counting repeats within the walk.
void add_edge(HnswLevelGraph &g, std::set<EdgeKey> &seen, uint64_t from,
              uint64_t to) {
  if (!seen.insert({from, to}).second) {
    ++g.duplicate_edges;
    return;
  }
  g.edges.push_back({from, to});
}

// Breadth-first walk of one level from entry_ref. Mirrors the ASCII list
// renderer's walk: a node costs one unit of budget when fetched, and once the
// budget is gone the remaining discovered nodes stay Truncated. Reports the
// entry node's lower-level link through lower_ref/has_lower.
void walk_level(HnswGraphModel &model, HnswLevelGraph &g, uint64_t entry_ref,
                uint32_t &budget, const HnswRecordFetcher &fetch,
                uint64_t &lower_ref, bool &has_lower) {
  std::set<EdgeKey> seen_edges;
  bool added;
  size_t entry = ensure_node(g, entry_ref, 0, false, added);
  g.nodes[entry].is_entry = true;

  // g.nodes doubles as the BFS queue: nodes are appended in discovery order.
  for (size_t qi = 0; qi < g.nodes.size(); ++qi) {
    uint64_t nid = g.nodes[qi].nid;

    if (budget == 0) {
      g.nodes[qi].state = HnswNodeState::Truncated;
      model.truncated = true;
      continue;
    }
    --budget;

    HnswNodeRecord rec;
    std::string error;
    if (!fetch(nid, g.level, rec, error)) {
      g.nodes[qi].state = HnswNodeState::Unresolved;
      g.nodes[qi].error = error;
      continue;
    }

    g.nodes[qi].state = HnswNodeState::Expanded;
    g.nodes[qi].vid = rec.owner_vid;
    g.nodes[qi].has_vid = true;
    g.nodes[qi].is_deleted = rec.is_deleted;

    if (qi == 0 && g.level > 0 && rec.lower_level_nid != 0) {
      lower_ref = rec.lower_level_nid & HNSW_NID_REF_MASK;
      has_lower = true;
    }

    for (const auto &neighbour : rec.neighbours) {
      uint64_t nref = neighbour.first & HNSW_NID_REF_MASK;
      ensure_node(g, nref, neighbour.second, true, added);
      add_edge(g, seen_edges, nid, nref);
    }
  }
}

size_t count_state(const HnswLevelGraph &g, HnswNodeState state) {
  size_t n = 0;
  for (const auto &node : g.nodes)
    n += node.state == state;
  return n;
}

std::string node_id(uint8_t level, uint64_t nid) {
  HnswColumnRef ref = hnsw_decode_ref(nid);
  return "L" + std::to_string((int)level) + "_" + std::to_string(ref.page_ref) +
         "_" + std::to_string(ref.slot_index);
}

// A line of text for a graph/cluster label, left-justified (\l).
std::string label_line(const std::string &text) {
  return dot_escape(text) + "\\l";
}

void write_node(const HnswLevelGraph &g, const HnswGraphNode &node,
                std::ostream &out) {
  std::string label = dot_escape(ref_text(node.nid));
  if (node.has_vid)
    label += "\\n(" + dot_escape(ref_text(node.vid)) + ")";
  std::string attrs;
  switch (node.state) {
  case HnswNodeState::Expanded:
    break;
  case HnswNodeState::Truncated:
    label += "\\ntruncated";
    attrs = " style=dashed color=\"#777777\" fontcolor=\"#555555\"";
    break;
  case HnswNodeState::Unresolved:
    label += "\\nunresolved";
    attrs = " style=dashed color=\"#c00000\" fontcolor=\"#c00000\" tooltip=\"" +
            dot_escape(node.error) + "\"";
    break;
  case HnswNodeState::Unreachable:
    label += "\\nunreachable";
    attrs = " style=filled fillcolor=\"#ffd8a8\"";
    break;
  }
  if (node.is_deleted) {
    label += "\\ndeleted";
    if (node.state == HnswNodeState::Expanded)
      attrs = " style=filled fillcolor=\"#f2b8b5\"";
  }
  if (node.is_entry)
    attrs += " peripheries=2";
  out << "    \"" << node_id(g.level, node.nid) << "\" [label=\"" << label
      << "\"" << attrs << "];\n";
}

void write_level(const HnswLevelGraph &g, std::ostream &out) {
  std::string label = "Level " + std::to_string((int)g.level) + ": " +
                      std::to_string(count_state(g, HnswNodeState::Expanded)) +
                      " reachable";
  size_t n;
  if ((n = count_state(g, HnswNodeState::Unreachable)) > 0)
    label += ", " + std::to_string(n) + " unreachable";
  if ((n = count_state(g, HnswNodeState::Unresolved)) > 0)
    label += ", " + std::to_string(n) + " unresolved";
  if ((n = count_state(g, HnswNodeState::Truncated)) > 0)
    label += ", " + std::to_string(n) + " truncated";
  if (g.duplicate_edges > 0)
    label += ", " + std::to_string(g.duplicate_edges) + " duplicate edge(s)";

  out << "  subgraph \"cluster_L" << (int)g.level << "\" {\n";
  out << "    label=\"" << dot_escape(label) << "\"; labeljust=l;\n";
  for (const auto &node : g.nodes)
    write_node(g, node, out);

  std::set<EdgeKey> edge_set;
  for (const auto &e : g.edges)
    edge_set.insert({e.from, e.to});

  std::set<EdgeKey> drawn;
  for (const auto &e : g.edges) {
    std::string from = node_id(g.level, e.from), to = node_id(g.level, e.to);
    bool reciprocal = e.from != e.to && edge_set.count({e.to, e.from});
    if (reciprocal) {
      // Emit the pair once, in whichever direction was discovered first.
      if (drawn.insert({e.from, e.to}).second && !drawn.count({e.to, e.from}))
        out << "    \"" << from << "\" -> \"" << to
            << "\" [dir=both penwidth=2 color=\"#1f5fbf\"];\n";
      continue;
    }
    const HnswGraphNode &target = g.nodes[g.node_index.at(e.to)];
    bool known = target.state == HnswNodeState::Expanded ||
                 target.state == HnswNodeState::Unreachable;
    if (known)
      out << "    \"" << from << "\" -> \"" << to
          << "\" [color=\"#d95f02\"];\n";
    else
      out << "    \"" << from << "\" -> \"" << to
          << "\" [style=dashed color=\"#777777\" "
             "tooltip=\"reciprocity unknown: target not expanded\"];\n";
  }
  out << "  }\n";
}

} // namespace

std::string dot_escape(const std::string &s) {
  std::string r;
  for (char c : s) {
    switch (c) {
    case '"':
      r += "\\\"";
      break;
    case '\\':
      r += "\\\\";
      break;
    case '\n':
      r += "\\n";
      break;
    case '\r':
      break;
    default:
      r += c;
    }
  }
  return r;
}

HnswGraphModel collect_hnsw_graph(uint64_t entry_ref, uint8_t entry_level,
                                  uint32_t M, uint32_t max_nodes,
                                  const HnswRecordFetcher &fetch) {
  HnswGraphModel model;
  model.M = M;
  model.entry_level = entry_level;
  model.max_nodes = max_nodes;

  uint32_t budget = max_nodes;
  uint8_t level = entry_level;
  uint64_t ref = entry_ref & HNSW_NID_REF_MASK;
  for (;;) {
    HnswLevelGraph g;
    g.level = level;
    uint64_t lower_ref = 0;
    bool has_lower = false;
    walk_level(model, g, ref, budget, fetch, lower_ref, has_lower);
    model.levels.push_back(std::move(g));

    if (level == 0)
      break;
    if (budget == 0) {
      model.truncated = true;
      model.notes.push_back("Levels " + std::to_string((int)level - 1) +
                            " to 0 were not walked: node budget exhausted");
      break;
    }
    if (!has_lower) {
      model.notes.push_back(
          "Level " + std::to_string((int)level) +
          " entry node has no lower-level link; lower levels not walked");
      break;
    }
    ref = lower_ref;
    --level;
  }
  return model;
}

bool add_unreachable_level0(
    HnswGraphModel &model,
    const std::vector<std::pair<uint64_t, HnswNodeRecord>> &live_level0) {
  const char *skipped = "Unreachable-node analysis skipped: ";
  if (model.levels.empty() || model.levels.back().level != 0) {
    model.notes.push_back(std::string(skipped) + "level 0 was not walked");
    return false;
  }
  if (model.truncated) {
    model.notes.push_back(
        std::string(skipped) +
        "the walk was truncated, so unvisited nodes cannot be told from "
        "unreachable ones; rerun with a higher --max-nodes");
    return false;
  }

  HnswLevelGraph &g = model.levels.back();
  std::unordered_set<uint64_t> live;
  for (const auto &entry : live_level0)
    live.insert(entry.first);

  // Pass 1: add the unreached live records as nodes (capped).
  std::vector<const std::pair<uint64_t, HnswNodeRecord> *> added_records;
  uint32_t total = 0;
  for (const auto &entry : live_level0) {
    if (g.node_index.count(entry.first))
      continue;
    ++total;
    if (added_records.size() >= model.max_nodes)
      continue;
    bool added;
    size_t i = ensure_node(g, entry.first, entry.second.owner_vid, true, added);
    g.nodes[i].state = HnswNodeState::Unreachable;
    g.nodes[i].is_deleted = entry.second.is_deleted;
    added_records.push_back(&entry);
  }
  model.has_allocated_count = true;
  model.allocated_level0 = static_cast<uint32_t>(live_level0.size());
  model.unreachable_total = total;
  if (total > added_records.size()) {
    model.truncated = true;
    model.notes.push_back("Only " + std::to_string(added_records.size()) +
                          " of " + std::to_string(total) +
                          " unreachable nodes are drawn (--max-nodes)");
  }

  // Pass 2: their outgoing edges. A target that is live but was not drawn is
  // Truncated; one that is not a live record at all is Unresolved.
  std::set<EdgeKey> seen_edges;
  for (const auto *entry : added_records) {
    for (const auto &neighbour : entry->second.neighbours) {
      uint64_t nref = neighbour.first & HNSW_NID_REF_MASK;
      bool added;
      size_t i = ensure_node(g, nref, neighbour.second, true, added);
      if (added) {
        if (live.count(nref)) {
          g.nodes[i].state = HnswNodeState::Truncated;
        } else {
          g.nodes[i].state = HnswNodeState::Unresolved;
          g.nodes[i].error = "not a live level-0 record";
        }
      }
      add_edge(g, seen_edges, entry->first, nref);
    }
  }
  return true;
}

void write_hnsw_dot(const HnswGraphModel &model, int level_filter,
                    std::ostream &out) {
  std::string label;
  label +=
      label_line("HNSW graph: M=" + std::to_string(model.M) + ", entry level " +
                 std::to_string((int)model.entry_level) + ", up to " +
                 std::to_string(model.max_nodes) + " nodes");
  label += label_line("Node label: NID page:slot, then (VID page:slot); "
                      "double ring = level entry node");
  label += label_line("Edge: blue bold double arrow = reciprocal; orange "
                      "arrow = one-way; dashed grey = reciprocity unknown");
  if (level_filter >= 0)
    label +=
        label_line("Showing level " + std::to_string(level_filter) + " only");
  if (model.has_allocated_count)
    label +=
        label_line("Level 0: " + std::to_string(model.allocated_level0) +
                   " live records, " + std::to_string(model.unreachable_total) +
                   " not reachable from the entry point");
  if (model.truncated)
    label += label_line("PARTIAL GRAPH: truncated; rerun with a higher "
                        "--max-nodes");
  for (const auto &note : model.notes)
    label += label_line(note);

  out << "digraph hnsw {\n";
  out << "  graph [rankdir=TB labelloc=t fontname=\"Helvetica\" label=\""
      << label << "\"];\n";
  out << "  node [shape=circle fontname=\"Helvetica\" fontsize=10];\n";
  out << "  edge [arrowsize=0.7];\n";
  for (const auto &g : model.levels) {
    if (level_filter >= 0 && g.level != level_filter)
      continue;
    write_level(g, out);
  }
  out << "}\n";
}

} // namespace tool
} // namespace svector
