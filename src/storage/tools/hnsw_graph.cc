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

#include "hnsw_graph.h"

#include <sstream>
#include <unordered_set>
#include <vector>

#include "../root_page.h"
#include "data_page_parser.h"
#include "hnsw_dot.h"

namespace svector {
namespace tool {

namespace {

// Mirrors LevelStore::max_neighbours (see ../../index/hnsw/storage.h):
// level 0 gets twice the configured degree, every level above it gets M.
uint32_t level_max_neighbours(uint8_t level, uint32_t M) {
  return level == 0 ? 2 * M : M;
}

// Compact "page:slot" form of a NID/VID's Column::Ref, as opposed to
// format_hnsw_ref()'s verbose "Page #<p>, Slot #<s>" -- graph output prints
// one of these per node (and, in verbose mode, a second for its VID), so
// terseness matters more here than in a single-record dump.
std::string ref_label(uint64_t ref_value) {
  HnswColumnRef ref = hnsw_decode_ref(ref_value);
  return std::to_string(ref.page_ref) + ":" + std::to_string(ref.slot_index);
}

std::string node_label(uint64_t nid_ref) {
  return "\xE2\x80\xA2" + ref_label(nid_ref); // U+2022 BULLET
}

// Fetches nid_ref's NeighbourEntry record at level. Returns false (with a
// message already printed at line_prefix + label) if the page or slot
// cannot be read as a live Neighbour record.
bool fetch_record(PageReader &reader, uint64_t nid_ref, uint8_t level,
                  uint32_t M, const std::string &line_prefix,
                  const std::string &label, std::ostream &out,
                  DataPageParser::RecordStatus &rec) {
  HnswColumnRef ref = hnsw_decode_ref(nid_ref);
  bool has_lower = level > 0;
  uint16_t column_size =
      hnsw_neighbour_column_size(level_max_neighbours(level, M), has_lower);

  auto page_data = reader.read_page(ref.page_ref);
  if (!page_data) {
    out << line_prefix << label << "  (error reading page " << ref.page_ref
        << ": " << reader.get_error() << ")\n";
    return false;
  }

  DataPageParser::DataPageInfo info;
  std::string error;
  if (!DataPageParser::parse(*page_data, column_size, info, error,
                             HnswRecordKind::Neighbour, has_lower)) {
    out << line_prefix << label << "  (error decoding page " << ref.page_ref
        << ": " << error << ")\n";
    return false;
  }
  if (ref.slot_index >= info.records.size() ||
      info.records[ref.slot_index].is_free) {
    out << line_prefix << label << "  (slot " << ref.slot_index << " on page "
        << ref.page_ref << " is not a live record)\n";
    return false;
  }

  rec = info.records[ref.slot_index];
  return true;
}

// Recursively prints nid_ref and its subtree using tree-drawing connectors.
// prefix is the indentation already emitted for this node's own line;
// is_root suppresses the connector (the level's local entry point gets a
// bare line, like `tree`'s root). visited/budget/total_shown are threaded
// through the whole level's walk. When this call is the level's root and it
// has a lower-level link, out_lower_ref/out_has_lower report it back to the
// caller so the next level's walk knows where to start.
void print_node(PageReader &reader, uint8_t level, uint32_t M, bool verbose,
                uint64_t nid_ref, const std::string &prefix, bool is_root,
                bool is_last, std::unordered_set<uint64_t> &visited,
                uint32_t &budget, uint32_t &total_shown, std::ostream &out,
                uint64_t &out_lower_ref, bool &out_has_lower) {
  std::string label = node_label(nid_ref);
  std::string line_prefix =
      is_root ? prefix
              : prefix + (is_last ? "\xE2\x94\x94\xE2\x94\x80 "
                                  : "\xE2\x94\x9C\xE2\x94\x80 ");
  // U+2514 U+2500 ("+- ") for the last child, U+251C U+2500 ("|- ") otherwise.

  if (visited.count(nid_ref)) {
    out << line_prefix << label << "  ...\n";
    return;
  }

  if (budget == 0) {
    out << line_prefix << label << "  (truncated)\n";
    return;
  }

  visited.insert(nid_ref);
  --budget;
  ++total_shown;

  DataPageParser::RecordStatus rec;
  if (!fetch_record(reader, nid_ref, level, M, line_prefix, label, out, rec))
    return;

  if (verbose)
    out << line_prefix << label << "(" << ref_label(rec.owner_vid) << ")\n";
  else
    out << line_prefix << label << "\n";

  if (level > 0 && rec.lower_level_nid != 0) {
    out_lower_ref = rec.lower_level_nid & HNSW_NID_REF_MASK;
    out_has_lower = true;
  }

  // Only forward (non-incoming) edges are drawn -- see hnsw_graph.h.
  std::vector<uint64_t> children;
  for (const auto &neighbour : rec.neighbours) {
    uint64_t nid = neighbour.first;
    if (nid == 0 || (nid & HNSW_NID_INCOMING_BIT))
      continue;
    children.push_back(nid & HNSW_NID_REF_MASK);
  }

  std::string child_prefix =
      is_root ? prefix : prefix + (is_last ? "   " : "\xE2\x94\x82  ");
  // "   " under a last child, "|  " (U+2502) otherwise.

  for (size_t i = 0; i < children.size(); ++i) {
    uint64_t dummy_ref = 0;
    bool dummy_has = false;
    print_node(reader, level, M, verbose, children[i], child_prefix,
               /*is_root=*/false, /*is_last=*/i + 1 == children.size(), visited,
               budget, total_shown, out, dummy_ref, dummy_has);
  }
}

// Walks the level's local neighbour graph breadth-first from entry_ref,
// printing one line per node: "<label>(<vid>) -> [<neighbour_ref>"
// "(<neighbour_vid>), ...] degree=<n>" (only the line's own node is
// bulleted; its neighbours print as bare "page:slot" refs). Each neighbour
// pair comes straight from the record's own on-disk Node{nid,vid} slot, so
// no extra page reads are needed to label them. Reports entry_ref's own
// lower-level link via out_lower_ref/out_has_lower, same as print_node().
void print_level_list(PageReader &reader, uint8_t level, uint32_t M,
                      uint64_t entry_ref, uint32_t &budget,
                      uint32_t &total_shown, std::ostream &out,
                      uint64_t &out_lower_ref, bool &out_has_lower) {
  std::unordered_set<uint64_t> visited;
  std::vector<uint64_t> queue{entry_ref};
  visited.insert(entry_ref);

  for (size_t qi = 0; qi < queue.size(); ++qi) {
    uint64_t nid_ref = queue[qi];
    std::string label = node_label(nid_ref);

    if (budget == 0) {
      out << label << "  (truncated)\n";
      continue;
    }
    --budget;
    ++total_shown;

    DataPageParser::RecordStatus rec;
    if (!fetch_record(reader, nid_ref, level, M, "", label, out, rec))
      continue;

    if (qi == 0 && level > 0 && rec.lower_level_nid != 0) {
      out_lower_ref = rec.lower_level_nid & HNSW_NID_REF_MASK;
      out_has_lower = true;
    }

    out << label << "(" << ref_label(rec.owner_vid) << ") -> [";
    uint32_t degree = 0;
    for (const auto &neighbour : rec.neighbours) {
      uint64_t nid = neighbour.first;
      if (nid == 0 || (nid & HNSW_NID_INCOMING_BIT))
        continue;
      uint64_t nref = nid & HNSW_NID_REF_MASK;
      if (degree > 0)
        out << ", ";
      out << ref_label(nref) << "(" << ref_label(neighbour.second) << ")";
      ++degree;
      if (visited.insert(nref).second)
        queue.push_back(nref);
    }
    out << "] degree=" << degree << "\n";
  }
}

} // namespace

namespace {

enum class Preflight { Ok, Empty, Error };

// Checks that index_meta can seed a graph walk and infers M from the level-0
// column size, shared by every graph renderer. Anything but Ok leaves the
// reason (newline-terminated) in msg.
Preflight graph_preflight(const HnswIndexMetadata &index_meta,
                          uint16_t level0_column_size, uint32_t &M,
                          std::ostream &msg) {
  if (index_meta.is_overflow() || index_meta.level != 0) {
    msg << "--graph requires the level-0 primary NeighbourEntry store's "
           "root page (its metadata carries the graph's entry point); this "
           "root page is "
        << (index_meta.is_overflow() ? "an overflow store" : "level ")
        << (index_meta.is_overflow() ? "" : std::to_string(index_meta.level))
        << ".\n";
    return Preflight::Error;
  }

  if (index_meta.entry_points.empty() || index_meta.entry_points[0] == 0) {
    msg << "Graph is empty (no entry point set).\n";
    return Preflight::Empty;
  }

  uint32_t max_n0 = hnsw_max_neighbours(level0_column_size,
                                        /*has_lower_level=*/false);
  if (max_n0 == 0 || max_n0 % 2 != 0) {
    msg << "Could not infer M from level-0 column size (" << level0_column_size
        << " bytes); refusing to guess a possibly wrong record layout for "
           "other levels.\n";
    return Preflight::Error;
  }
  M = max_n0 / 2;
  return Preflight::Ok;
}

// Outgoing neighbour slots of a decoded record as (nid ref, vid): empty and
// incoming-flagged slots are other nodes' edges, so they are dropped.
std::vector<std::pair<uint64_t, uint64_t>>
outgoing_neighbours(const DataPageParser::RecordStatus &rec) {
  std::vector<std::pair<uint64_t, uint64_t>> out;
  for (const auto &neighbour : rec.neighbours) {
    if (neighbour.first == 0 || (neighbour.first & HNSW_NID_INCOMING_BIT))
      continue;
    out.push_back({neighbour.first & HNSW_NID_REF_MASK, neighbour.second});
  }
  return out;
}

// Strips the "  (...)\n" wrapping fetch_record() puts around its message.
std::string unwrap_message(std::string msg) {
  size_t b = msg.find_first_not_of(' ');
  size_t e = msg.find_last_not_of(" \n");
  if (b == std::string::npos || e == std::string::npos)
    return "";
  msg = msg.substr(b, e - b + 1);
  if (msg.size() >= 2 && msg.front() == '(' && msg.back() == ')')
    msg = msg.substr(1, msg.size() - 2);
  return msg;
}

} // namespace

void render_hnsw_graph(PageReader &reader, const HnswIndexMetadata &index_meta,
                       uint16_t level0_column_size, uint32_t max_nodes,
                       bool verbose, GraphStyle style, std::ostream &out) {
  uint32_t M = 0;
  std::ostringstream msg;
  if (graph_preflight(index_meta, level0_column_size, M, msg) !=
      Preflight::Ok) {
    out << msg.str();
    return;
  }

  uint8_t level = index_meta.entry_level;
  uint64_t entry_ref = index_meta.entry_points[0] & HNSW_NID_REF_MASK;
  uint32_t budget = max_nodes;
  uint32_t total_shown = 0;

  out << "HNSW Graph (M=" << M << ", entry level "
      << (int)index_meta.entry_level << ", showing up to " << max_nodes
      << " nodes)\n\n";

  for (;;) {
    out << "Level " << (int)level
        << (level == index_meta.entry_level ? " (entry)" : "") << "\n";

    uint64_t lower_ref = 0;
    bool has_lower = false;
    if (style == GraphStyle::Tree) {
      std::unordered_set<uint64_t> visited;
      print_node(reader, level, M, verbose, entry_ref, "", /*is_root=*/true,
                 /*is_last=*/true, visited, budget, total_shown, out, lower_ref,
                 has_lower);
    } else {
      print_level_list(reader, level, M, entry_ref, budget, total_shown, out,
                       lower_ref, has_lower);
    }
    out << "\n";

    if (level == 0)
      break;
    if (budget == 0) {
      out << "... graph truncated after " << max_nodes
          << " nodes; rerun with a higher --max-nodes to see more\n";
      break;
    }
    if (!has_lower) {
      out << "(entry node has no lower-level link; stopping)\n";
      break;
    }

    entry_ref = lower_ref;
    --level;
  }
}

bool render_hnsw_graph_dot(PageReader &reader,
                           const HnswIndexMetadata &index_meta,
                           const HnswRootInfo &root, uint32_t max_nodes,
                           int level_filter, bool find_unreachable,
                           std::ostream &out, std::ostream &err) {
  uint32_t M = 0;
  std::ostringstream msg;
  Preflight pre = graph_preflight(index_meta, root.column_size, M, msg);
  if (pre == Preflight::Error) {
    err << msg.str();
    return false;
  }
  if (pre == Preflight::Empty) {
    out << "digraph hnsw {\n  label=\"Graph is empty (no entry point "
           "set)\";\n}\n";
    return true;
  }

  HnswRecordFetcher fetch = [&](uint64_t nid_ref, uint8_t level,
                                HnswNodeRecord &node, std::string &error) {
    DataPageParser::RecordStatus rec;
    std::ostringstream fetch_msg;
    if (!fetch_record(reader, nid_ref, level, M, "", "", fetch_msg, rec)) {
      error = unwrap_message(fetch_msg.str());
      return false;
    }
    node.owner_vid = rec.owner_vid;
    node.is_deleted = rec.is_deleted;
    node.lower_level_nid = rec.lower_level_nid;
    node.neighbours = outgoing_neighbours(rec);
    return true;
  };

  HnswGraphModel model =
      collect_hnsw_graph(index_meta.entry_points[0] & HNSW_NID_REF_MASK,
                         index_meta.entry_level, M, max_nodes, fetch);

  if (find_unreachable) {
    // Enumerate every live level-0 record independently of the entry-point
    // walk, by following the store's own all-pages list from its root page.
    std::vector<std::pair<uint64_t, HnswNodeRecord>> live;
    std::unordered_set<uint32_t> seen_pages;
    std::string scan_error;
    uint32_t page_num = root.all_slot_head;
    while (page_num != RootPage::NULL_FREE_PAGE_REF &&
           seen_pages.insert(page_num).second) {
      auto page_data = reader.read_page(page_num);
      DataPageParser::DataPageInfo info;
      std::string error;
      if (!page_data) {
        scan_error = "error reading page " + std::to_string(page_num) + ": " +
                     reader.get_error();
        break;
      }
      if (!DataPageParser::parse(*page_data, root.column_size, info, error,
                                 HnswRecordKind::Neighbour,
                                 /*has_lower_level=*/false)) {
        scan_error =
            "error decoding page " + std::to_string(page_num) + ": " + error;
        break;
      }
      for (size_t slot = 0; slot < info.records.size(); ++slot) {
        const auto &rec = info.records[slot];
        if (rec.is_free)
          continue;
        HnswNodeRecord node;
        node.owner_vid = rec.owner_vid;
        node.is_deleted = rec.is_deleted;
        node.neighbours = outgoing_neighbours(rec);
        live.emplace_back(
            hnsw_encode_ref(page_num, static_cast<uint16_t>(slot)),
            std::move(node));
      }
      page_num = info.fil_page_next;
    }
    if (!scan_error.empty())
      model.notes.push_back("Unreachable-node analysis skipped: " + scan_error);
    else
      add_unreachable_level0(model, live);
  }

  write_hnsw_dot(model, level_filter, out);
  return true;
}

} // namespace tool
} // namespace svector
