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

// Standalone, assert()-based unit test for the HNSW DOT export
// (src/storage/tools/hnsw_dot.h). Drives collect_hnsw_graph() with a fake
// record fetcher instead of an .ibd file, so it has no dependency on the
// VillageSQL SDK.

#include <cassert>
#include <cstdio>
#include <map>
#include <sstream>
#include <string>

#include "../src/storage/tools/hnsw_dot.h"
#include "../src/storage/tools/hnsw_layout.h"

using namespace svector::tool;

namespace {

// A NID/VID "page:slot" as a ref value.
uint64_t R(uint32_t page, uint16_t slot) { return hnsw_encode_ref(page, slot); }

// Fake store: (level, nid) -> record. A missing key is an unresolved node.
struct FakeStore {
  std::map<std::pair<int, uint64_t>, HnswNodeRecord> recs;
  int fetches = 0;

  void add(int level, uint64_t nid, uint64_t vid,
           std::vector<std::pair<uint64_t, uint64_t>> neighbours,
           uint64_t lower = 0, bool deleted = false) {
    HnswNodeRecord rec;
    rec.owner_vid = vid;
    rec.lower_level_nid = lower;
    rec.is_deleted = deleted;
    rec.neighbours = std::move(neighbours);
    recs[{level, nid}] = rec;
  }

  HnswRecordFetcher fetcher() {
    return [this](uint64_t nid, uint8_t level, HnswNodeRecord &rec,
                  std::string &error) {
      ++fetches;
      auto it = recs.find({level, nid});
      if (it == recs.end()) {
        error = "slot is not a live \"record\"\\\n";
        return false;
      }
      rec = it->second;
      return true;
    };
  }
};

std::string dot(const HnswGraphModel &model, int level = -1) {
  std::ostringstream out;
  write_hnsw_dot(model, level, out);
  return out.str();
}

bool has(const std::string &s, const std::string &needle) {
  return s.find(needle) != std::string::npos;
}

size_t count(const std::string &s, const std::string &needle) {
  size_t n = 0;
  for (size_t p = s.find(needle); p != std::string::npos;
       p = s.find(needle, p + needle.size()))
    ++n;
  return n;
}

// Structural validity: braces balance outside strings, every string closes,
// and the text is one digraph.
void assert_valid_dot(const std::string &s) {
  assert(s.rfind("digraph hnsw {\n", 0) == 0);
  int depth = 0;
  bool in_string = false;
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (in_string) {
      if (c == '\\')
        ++i;
      else if (c == '"')
        in_string = false;
      else
        assert(c != '\n'); // Raw newlines must be escaped inside strings.
    } else if (c == '"') {
      in_string = true;
    } else if (c == '{') {
      ++depth;
    } else if (c == '}') {
      --depth;
      assert(depth >= 0);
    }
  }
  assert(!in_string && depth == 0);
}

const HnswGraphNode *find_node(const HnswLevelGraph &g, uint64_t nid) {
  auto it = g.node_index.find(nid);
  return it == g.node_index.end() ? nullptr : &g.nodes[it->second];
}

// A<->B reciprocal, B->C one-way, C->A closing a cycle, and D reached from
// two nodes (A and C), with A listing D twice.
void test_single_level_edges() {
  FakeStore s;
  uint64_t A = R(12, 5), B = R(12, 4), C = R(12, 3), D = R(12, 2);
  s.add(0, A, R(11, 5), {{B, R(11, 4)}, {D, R(11, 2)}, {D, R(11, 2)}});
  s.add(0, B, R(11, 4), {{A, R(11, 5)}, {C, R(11, 3)}});
  s.add(0, C, R(11, 3), {{A, R(11, 5)}, {D, R(11, 2)}});
  s.add(0, D, R(11, 2), {});

  HnswGraphModel m = collect_hnsw_graph(A, 0, 16, 50, s.fetcher());
  assert(!m.truncated);
  assert(m.levels.size() == 1);
  const HnswLevelGraph &g = m.levels[0];
  assert(g.nodes.size() == 4);    // D is one node, not two.
  assert(g.edges.size() == 6);    // A->B A->D B->A B->C C->A C->D
  assert(g.duplicate_edges == 1); // The second A->D.
  assert(find_node(g, A)->is_entry && !find_node(g, B)->is_entry);
  assert(find_node(g, D)->vid == R(11, 2) && find_node(g, D)->has_vid);
  assert(s.fetches == 4); // Each node fetched exactly once.

  std::string out = dot(m);
  assert_valid_dot(out);
  assert(has(out, "label=\"12:5\\n(11:5)\\ntrx 0\" peripheries=2"));
  assert(count(out, "dir=both") == 1); // A<->B drawn once, double-headed.
  assert(has(out, "\"L0_12_5\" -> \"L0_12_4\" [dir=both"));
  assert(!has(out, "\"L0_12_4\" -> \"L0_12_5\""));
  assert(has(out, "\"L0_12_4\" -> \"L0_12_3\" [color=\"#d95f02\"]"));
  assert(has(out, "\"L0_12_3\" -> \"L0_12_5\" [color=\"#d95f02\"]"));
  assert(has(out, "1 duplicate edge(s)"));
}

// Self-loop is one-way, not "reciprocal with itself".
void test_self_loop_is_one_way() {
  FakeStore s;
  uint64_t A = R(1, 0);
  s.add(0, A, R(2, 0), {{A, R(2, 0)}});
  std::string out = dot(collect_hnsw_graph(A, 0, 16, 50, s.fetcher()));
  assert(!has(out, "dir=both"));
  assert(has(out, "\"L0_1_0\" -> \"L0_1_0\" [color=\"#d95f02\"]"));
}

// Incoming-flagged and empty slots are the fetcher's to drop; a NID that
// still carries the flag must resolve to the same node as the plain ref.
void test_incoming_bit_masked() {
  FakeStore s;
  uint64_t A = R(1, 0), B = R(1, 1);
  s.add(0, A, R(2, 0), {{B | HNSW_NID_INCOMING_BIT, R(2, 1)}});
  s.add(0, B, R(2, 1), {});
  HnswGraphModel m = collect_hnsw_graph(A, 0, 16, 50, s.fetcher());
  assert(m.levels[0].nodes.size() == 2);
  assert(find_node(m.levels[0], B)->state == HnswNodeState::Expanded);
}

// Levels stay separate: same vertex has a different NID per level, edges are
// never drawn across levels, and the level filter drops the others.
void test_multiple_levels() {
  FakeStore s;
  uint64_t E1 = R(17, 1), X1 = R(17, 2);
  uint64_t E0 = R(12, 5), X0 = R(12, 6);
  s.add(1, E1, R(11, 5), {{X1, R(11, 6)}}, /*lower=*/E0);
  s.add(1, X1, R(11, 6), {{E1, R(11, 5)}}, X0);
  s.add(0, E0, R(11, 5), {{X0, R(11, 6)}});
  s.add(0, X0, R(11, 6), {});

  HnswGraphModel m = collect_hnsw_graph(E1, 1, 16, 50, s.fetcher());
  assert(m.levels.size() == 2 && m.levels[0].level == 1 &&
         m.levels[1].level == 0);
  assert(m.levels[0].edges.size() == 2 && m.levels[1].edges.size() == 1);
  // Same VID, different per-level nodes.
  assert(find_node(m.levels[0], E1)->vid == find_node(m.levels[1], E0)->vid);

  std::string all = dot(m);
  assert_valid_dot(all);
  assert(has(all, "cluster_L1") && has(all, "cluster_L0"));
  assert(has(all, "\"L1_17_1\" -> \"L1_17_2\" [dir=both"));
  assert(has(all, "\"L0_12_5\" -> \"L0_12_6\" [color"));
  assert(!has(all, "\"L1_17_1\" -> \"L0_"));

  std::string only0 = dot(m, 0);
  assert_valid_dot(only0);
  assert(has(only0, "cluster_L0") && !has(only0, "cluster_L1"));
  assert(has(only0, "Showing level 0 only"));
}

// Entry node with no lower link: lower levels are not walked, and says so.
void test_missing_lower_link_noted() {
  FakeStore s;
  uint64_t E1 = R(17, 1);
  s.add(1, E1, R(11, 5), {});
  HnswGraphModel m = collect_hnsw_graph(E1, 1, 16, 50, s.fetcher());
  assert(m.levels.size() == 1);
  assert(m.notes.size() == 1 && has(m.notes[0], "no lower-level link"));
  assert(has(dot(m), "no lower-level link"));
}

// A neighbour whose record cannot be read stays in the graph as an
// unresolved node, with the reason in its tooltip, escaped.
void test_unresolved_neighbour_and_escaping() {
  FakeStore s;
  uint64_t A = R(12, 5), GONE = R(12, 9);
  s.add(0, A, R(11, 5), {{GONE, R(11, 9)}});
  HnswGraphModel m = collect_hnsw_graph(A, 0, 16, 50, s.fetcher());
  const HnswGraphNode *n = find_node(m.levels[0], GONE);
  assert(n && n->state == HnswNodeState::Unresolved && n->has_vid);
  assert(!m.truncated);

  std::string out = dot(m);
  assert_valid_dot(out);
  assert(has(out, "label=\"12:9\\n(11:9)\\nunresolved\""));
  // The fetcher's error text contains a quote, a backslash and a newline.
  assert(has(out, "tooltip=\"slot is not a live \\\"record\\\"\\\\\\n\""));
  assert(has(out, "\"L0_12_5\" -> \"L0_12_9\" [style=dashed"));
}

// With max_nodes smaller than the graph: frontier nodes stay Truncated, the
// diagram says it is partial, and nothing beyond the budget is fetched.
void test_truncation() {
  FakeStore s;
  uint64_t n[5];
  for (int i = 0; i < 5; ++i)
    n[i] = R(12, i);
  for (int i = 0; i < 4; ++i)
    s.add(0, n[i], R(11, i), {{n[i + 1], R(11, i + 1)}});
  s.add(0, n[4], R(11, 4), {});

  HnswGraphModel m = collect_hnsw_graph(n[0], 0, 16, 2, s.fetcher());
  assert(m.truncated);
  assert(s.fetches == 2);
  assert(find_node(m.levels[0], n[1])->state == HnswNodeState::Expanded);
  assert(find_node(m.levels[0], n[2])->state == HnswNodeState::Truncated);
  assert(!find_node(m.levels[0], n[3]));

  std::string out = dot(m);
  assert_valid_dot(out);
  assert(has(out, "PARTIAL GRAPH"));
  assert(has(out, "1 truncated"));
  assert(has(out, "\"L0_12_1\" -> \"L0_12_2\" [style=dashed"));

  // Exactly enough budget is not truncation.
  FakeStore s2 = s;
  assert(!collect_hnsw_graph(n[0], 0, 16, 5, s2.fetcher()).truncated);
}

// Budget exhausted on an upper level: lower levels are never walked, which
// counts as truncation even though every node that was seen got expanded.
void test_truncation_skips_lower_levels() {
  FakeStore s;
  uint64_t E1 = R(17, 1), E0 = R(12, 5);
  s.add(1, E1, R(11, 5), {}, E0);
  s.add(0, E0, R(11, 5), {});
  HnswGraphModel m = collect_hnsw_graph(E1, 1, 16, 1, s.fetcher());
  assert(m.truncated && m.levels.size() == 1);
  assert(has(dot(m), "were not walked"));
}

// Unreachable analysis: live records the walk never reached are drawn with
// their own edges; a dangling edge from one becomes unresolved.
void test_unreachable_nodes() {
  FakeStore s;
  uint64_t A = R(12, 5), B = R(12, 4), O1 = R(12, 3), O2 = R(12, 2);
  uint64_t DEAD = R(12, 8);
  s.add(0, A, R(11, 5), {{B, R(11, 4)}});
  s.add(0, B, R(11, 4), {{A, R(11, 5)}});
  s.add(0, O1, R(11, 3), {{O2, R(11, 2)}, {A, R(11, 5)}, {DEAD, R(11, 8)}});
  s.add(0, O2, R(11, 2), {{O1, R(11, 3)}}, 0, /*deleted=*/true);

  HnswGraphModel m = collect_hnsw_graph(A, 0, 16, 50, s.fetcher());
  std::vector<std::pair<uint64_t, HnswNodeRecord>> live;
  for (const auto &kv : s.recs)
    live.emplace_back(kv.first.second, kv.second);
  assert(add_unreachable_level0(m, live));
  assert(m.allocated_level0 == 4 && m.unreachable_total == 2);

  const HnswLevelGraph &g = m.levels[0];
  assert(find_node(g, O1)->state == HnswNodeState::Unreachable);
  assert(find_node(g, O2)->is_deleted);
  assert(find_node(g, A)->state == HnswNodeState::Expanded);
  assert(find_node(g, DEAD)->state == HnswNodeState::Unresolved);

  std::string out = dot(m);
  assert_valid_dot(out);
  assert(has(out, "2 not reachable from the entry point"));
  assert(has(out, "label=\"12:3\\n(11:3)\\ntrx 0\\nunreachable\""));
  assert(has(out, "\\ndeleted"));
  assert(has(out, "\"L0_12_3\" -> \"L0_12_5\" [color=\"#d95f02\"]"));
  assert(count(out, "dir=both") == 2); // A<->B and O1<->O2, each drawn once.
}

// The analysis must refuse, not guess, when the walk was cut short.
void test_unreachable_refused_when_truncated() {
  FakeStore s;
  uint64_t A = R(12, 5), B = R(12, 4);
  s.add(0, A, R(11, 5), {{B, R(11, 4)}});
  s.add(0, B, R(11, 4), {});
  HnswGraphModel m = collect_hnsw_graph(A, 0, 16, 1, s.fetcher());
  std::vector<std::pair<uint64_t, HnswNodeRecord>> live;
  for (const auto &kv : s.recs)
    live.emplace_back(kv.first.second, kv.second);
  assert(!add_unreachable_level0(m, live));
  assert(!m.has_allocated_count);
  assert(find_node(m.levels[0], B)->state == HnswNodeState::Truncated);
  assert(has(dot(m), "Unreachable-node analysis skipped"));
}

// More unreachable nodes than max_nodes: capped, counted, flagged partial.
void test_unreachable_capped() {
  FakeStore s;
  uint64_t A = R(12, 0);
  s.add(0, A, R(11, 0), {});
  for (int i = 1; i <= 4; ++i)
    s.add(0, R(12, i), R(11, i), {});
  HnswGraphModel m = collect_hnsw_graph(A, 0, 16, 3, s.fetcher());
  std::vector<std::pair<uint64_t, HnswNodeRecord>> live;
  for (const auto &kv : s.recs)
    live.emplace_back(kv.first.second, kv.second);
  assert(add_unreachable_level0(m, live));
  assert(m.unreachable_total == 4 && m.levels[0].nodes.size() == 1 + 3);
  assert(m.truncated);
  assert(has(dot(m), "Only 3 of 4 unreachable nodes are drawn"));
}

void test_dot_escape() {
  assert(dot_escape("plain") == "plain");
  assert(dot_escape("a\"b") == "a\\\"b");
  assert(dot_escape("a\\b") == "a\\\\b");
  assert(dot_escape("a\nb\r") == "a\\nb");
}

} // namespace

int main() {
  test_single_level_edges();
  test_self_loop_is_one_way();
  test_incoming_bit_masked();
  test_multiple_levels();
  test_missing_lower_link_noted();
  test_unresolved_neighbour_and_escaping();
  test_truncation();
  test_truncation_skips_lower_levels();
  test_unreachable_nodes();
  test_unreachable_refused_when_truncated();
  test_unreachable_capped();
  test_dot_escape();

  std::printf("All hnsw_dot tests passed.\n");
  return 0;
}
