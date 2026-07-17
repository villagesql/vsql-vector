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

// Standalone, assert()-based unit test for LayerOperations<Graph>. Has no
// dependency on the VillageSQL SDK: LayerOperations is a pure template over a
// user-supplied Graph, so it's exercised here against a small mock graph.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// Nodes are integers 0..n-1 on a line; neighbours(v) = {v-1, v+1} (clamped),
// distance(a, b) = |a - b|. Follows the codebase convention: bool-returning
// operations return true on error, false on success.
struct LineGraph {
  struct Node {
    using KeyType = int;
    int value;
    KeyType key() const { return value; }
  };
  using DistanceType = float;

  int n = 0;
  bool fail_distance = false;
  bool fail_neighbours = false;
  bool fail_is_visible = false;
  std::vector<int> invisible;

  bool distance(const Node &a, const Node &b, DistanceType &out) {
    if (fail_distance) {
      return true;
    }
    out = std::abs(a.value - b.value);
    return false;
  }

  bool neighbours(const Node &node, std::vector<Node> &out) {
    if (fail_neighbours) {
      return true;
    }
    out.clear();
    if (node.value - 1 >= 0) {
      out.push_back(Node{node.value - 1});
    }
    if (node.value + 1 < n) {
      out.push_back(Node{node.value + 1});
    }
    return false;
  }

  bool is_visible(const Node &node, bool &out) {
    if (fail_is_visible) {
      return true;
    }
    out = std::find(invisible.begin(), invisible.end(), node.value) ==
          invisible.end();
    return false;
  }
};

// Consults LineGraph::is_visible(), as a real query-time caller would.
struct VisibilityPolicy {
  template <typename Graph>
  bool is_visible(Graph &graph, const typename Graph::Node &node,
                  bool &out) const {
    return graph.is_visible(node, out);
  }
};

// Treats every node as visible without consulting the graph, as an
// index-construction caller would.
struct AlwaysVisiblePolicy {
  template <typename Graph>
  bool is_visible(Graph &, const typename Graph::Node &, bool &out) const {
    out = true;
    return false;
  }
};

} // namespace

// LayerOperations's member functions are defined in hnsw_search.cc, not the
// header, so they must be explicitly instantiated for LineGraph somewhere
// that includes the definitions -- there's no separately-built object file
// to link against for an arbitrary test-only Graph type.
#include "../src/index/hnsw_search.cc"

namespace svector::hnsw {
template class LayerOperations<LineGraph, VisibilityPolicy>;
template class LayerOperations<LineGraph, AlwaysVisiblePolicy>;
} // namespace svector::hnsw

namespace {

std::vector<int> values(const std::vector<LineGraph::Node> &nodes) {
  std::vector<int> out;
  out.reserve(nodes.size());
  for (const auto &node : nodes) {
    out.push_back(node.value);
  }
  return out;
}

void test_basic_knn() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::Node query{15};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  std::vector<LineGraph::Node> entry_points{LineGraph::Node{0}};
  std::vector<LineGraph::Node> result;
  assert(!layer.search_layer(entry_points, 5, result));
  assert(result.size() == 5);

  // Nearest 5 nodes to 15 among 0..19 are {13,14,15,16,17}.
  std::vector<int> got = values(result);
  std::vector<int> sorted_got = got;
  std::sort(sorted_got.begin(), sorted_got.end());
  assert((sorted_got == std::vector<int>{13, 14, 15, 16, 17}));

  // search_layer() must order the result nearest-first.
  float prev = -1;
  for (int v : got) {
    float d = std::abs(v - query.value);
    assert(d >= prev);
    prev = d;
  }
}

void test_reuse_after_search_layer() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::Node query{15};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  std::vector<LineGraph::Node> first_entry{LineGraph::Node{0}};
  std::vector<LineGraph::Node> first_result;
  assert(!layer.search_layer(first_entry, 5, first_result));
  assert(first_result.size() == 5);

  // search_layer() resets internally, so it can be called again with the
  // same query without an explicit reset() call.
  std::vector<LineGraph::Node> second_entry{LineGraph::Node{19}};
  std::vector<LineGraph::Node> second_result;
  assert(!layer.search_layer(second_entry, 3, second_result));
  assert(second_result.size() == 3);

  std::vector<int> sorted_got = values(second_result);
  std::sort(sorted_got.begin(), sorted_got.end());
  assert((sorted_got == std::vector<int>{14, 15, 16}));
}

void test_ef_bounds_result_size() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::Node query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  std::vector<LineGraph::Node> entry_points{LineGraph::Node{10}};
  std::vector<LineGraph::Node> result;
  assert(!layer.search_layer(entry_points, 1, result));
  assert(result.size() == 1);
  assert(result[0].value == 10);
}

void test_distance_failure_propagates() {
  LineGraph graph;
  graph.n = 20;
  graph.fail_distance = true;
  LineGraph::Node query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  std::vector<LineGraph::Node> entry_points{LineGraph::Node{10}};
  std::vector<LineGraph::Node> result;
  assert(layer.search_layer(entry_points, 5, result));
}

void test_neighbours_failure_propagates() {
  LineGraph graph;
  graph.n = 20;
  graph.fail_neighbours = true;
  LineGraph::Node query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  // With ef=1 and a single entry point equal to the query (distance 0),
  // Algorithm 2's break condition (line 7) never triggers, forcing an
  // expand() call that hits the failing neighbours().
  std::vector<LineGraph::Node> entry_points{LineGraph::Node{10}};
  std::vector<LineGraph::Node> result;
  assert(layer.search_layer(entry_points, 1, result));
}

void test_is_visible_failure_propagates() {
  LineGraph graph;
  graph.n = 20;
  graph.fail_is_visible = true;
  LineGraph::Node query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  std::vector<LineGraph::Node> entry_points{LineGraph::Node{10}};
  std::vector<LineGraph::Node> result;
  assert(layer.search_layer(entry_points, 1, result));
}

void test_invisible_nodes_excluded_from_result() {
  LineGraph graph;
  graph.n = 20;
  // 15 is the nearest node to the query but is invisible, so the search
  // must traverse through it to reach the surrounding visible nodes
  // instead of returning it.
  graph.invisible = {15};
  LineGraph::Node query{15};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  // Entry point is off to one side (0) so the search sweeps in from a
  // single direction, avoiding same-distance ties in the result set at
  // the ef cutoff (unlike starting from the query itself).
  std::vector<LineGraph::Node> entry_points{LineGraph::Node{0}};
  std::vector<LineGraph::Node> result;
  assert(!layer.search_layer(entry_points, 6, result));

  // Without 15, the nearest 6 visible nodes to 15 are {12,13,14,16,17,18}
  // (distances 3,2,1,1,2,3); 15 itself (distance 0) is excluded despite
  // being the closest possible node.
  std::vector<int> got = values(result);
  std::vector<int> sorted_got = got;
  std::sort(sorted_got.begin(), sorted_got.end());
  assert((sorted_got == std::vector<int>{12, 13, 14, 16, 17, 18}));

  float prev = -1;
  for (int v : got) {
    float d = std::abs(v - query.value);
    assert(d >= prev);
    prev = d;
  }
}

void test_invisible_entry_point_does_not_seed_result() {
  LineGraph graph;
  graph.n = 20;
  graph.invisible = {0};
  LineGraph::Node query{0};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  // The sole entry point is invisible and is also the query itself, so
  // with ef=1 the search must expand past it to find a visible result
  // instead of returning early with an empty/invisible result.
  std::vector<LineGraph::Node> entry_points{LineGraph::Node{0}};
  std::vector<LineGraph::Node> result;
  assert(!layer.search_layer(entry_points, 1, result));
  assert((values(result) == std::vector<int>{1}));
}

void test_always_visible_policy_ignores_invisibility() {
  LineGraph graph;
  graph.n = 20;
  // Would exclude 0 from the result and fail the search under
  // VisibilityPolicy, but AlwaysVisiblePolicy never consults either.
  graph.invisible = {0};
  graph.fail_is_visible = true;
  LineGraph::Node query{0};
  svector::hnsw::LayerOperations<LineGraph, AlwaysVisiblePolicy> layer(
      graph, query, AlwaysVisiblePolicy());

  std::vector<LineGraph::Node> entry_points{LineGraph::Node{0}};
  std::vector<LineGraph::Node> result;
  assert(!layer.search_layer(entry_points, 1, result));
  assert((values(result) == std::vector<int>{0}));
}

void test_select_neighbours_simple() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::Node query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  std::vector<LineGraph::Node> candidates{
      LineGraph::Node{0},  LineGraph::Node{7},  LineGraph::Node{9},
      LineGraph::Node{12}, LineGraph::Node{19}, LineGraph::Node{10}};

  std::vector<LineGraph::Node> result;
  assert(!layer.select_neighbours_simple(candidates, 3, result));
  assert(result.size() == 3);

  // Nearest 3 candidates to 10 are {10,9,12} (distances 0,1,2), ordered by
  // increasing distance.
  std::vector<int> got = values(result);
  assert((got == std::vector<int>{10, 9, 12}));

  // M larger than the candidate set returns every candidate, still ordered
  // by increasing distance.
  std::vector<LineGraph::Node> all;
  assert(!layer.select_neighbours_simple(candidates, 10, all));
  std::vector<int> got_all = values(all);
  assert((got_all == std::vector<int>{10, 9, 12, 7, 19, 0}));
}

void test_select_neighbours_simple_distance_failure_propagates() {
  LineGraph graph;
  graph.n = 20;
  graph.fail_distance = true;
  LineGraph::Node query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  std::vector<LineGraph::Node> candidates{LineGraph::Node{5}};
  std::vector<LineGraph::Node> result;
  assert(layer.select_neighbours_simple(candidates, 1, result));
}

void test_select_neighbours_heuristic_prefers_diverse_candidates() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::Node query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  // Distances to 10: 5->5, 9->1, 12->2, 16->6. Nearest-first: 9,12,5,16.
  // 5 is dominated by 9 (dist(5,9)=4 < dist(5,10)=5) and 16 is dominated by
  // 12 (dist(16,12)=4 < dist(16,10)=6), so only {9,12} pass the heuristic
  // even though M=3 asks for more -- unlike select_neighbours_simple,
  // which would also return the third-nearest candidate.
  std::vector<LineGraph::Node> candidates{
      LineGraph::Node{5}, LineGraph::Node{9}, LineGraph::Node{12},
      LineGraph::Node{16}};

  std::vector<LineGraph::Node> result;
  assert(!layer.select_neighbours_heuristic(candidates, 3,
                                            /*extend_candidates=*/false,
                                            /*keep_pruned_connections=*/false,
                                            result));
  assert((values(result) == std::vector<int>{9, 12}));
}

void test_select_neighbours_heuristic_keeps_pruned_connections() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::Node query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  // Same candidates as above; with keep_pruned_connections set, the
  // remaining slot is backfilled from the discarded candidates, nearest
  // to the query first: 5 (dist 5) before 16 (dist 6).
  std::vector<LineGraph::Node> candidates{
      LineGraph::Node{5}, LineGraph::Node{9}, LineGraph::Node{12},
      LineGraph::Node{16}};

  std::vector<LineGraph::Node> result;
  assert(!layer.select_neighbours_heuristic(candidates, 3,
                                            /*extend_candidates=*/false,
                                            /*keep_pruned_connections=*/true,
                                            result));
  assert((values(result) == std::vector<int>{9, 12, 5}));
}

void test_select_neighbours_heuristic_extend_candidates() {
  LineGraph graph;
  graph.n = 30;
  LineGraph::Node query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  // Candidates {6,17} extend to {6,17,5,7,16,18} (neighbours of 6 and 17).
  // Nearest-first by distance to 10: 7(3),6(4),5(5),16(6),17(7),18(8).
  // 6 and 5 are dominated by 7 (dist 1 and 2, both < their distance to the
  // query); 17 and 18 are dominated by 16 (dist 1 and 2). So extending the
  // candidate set replaces the originally-nearest {6,17} with {7,16}.
  std::vector<LineGraph::Node> candidates{LineGraph::Node{6},
                                          LineGraph::Node{17}};

  std::vector<LineGraph::Node> without_extend;
  assert(!layer.select_neighbours_heuristic(candidates, 3,
                                            /*extend_candidates=*/false,
                                            /*keep_pruned_connections=*/false,
                                            without_extend));
  assert((values(without_extend) == std::vector<int>{6, 17}));

  std::vector<LineGraph::Node> with_extend;
  assert(!layer.select_neighbours_heuristic(candidates, 3,
                                            /*extend_candidates=*/true,
                                            /*keep_pruned_connections=*/false,
                                            with_extend));
  assert((values(with_extend) == std::vector<int>{7, 16}));

  // With keep_pruned_connections, the remaining slot backfills from the
  // discarded candidates nearest to the query: 6 (dist 4) before 5, 17, 18.
  std::vector<LineGraph::Node> with_extend_and_keep;
  assert(!layer.select_neighbours_heuristic(candidates, 3,
                                            /*extend_candidates=*/true,
                                            /*keep_pruned_connections=*/true,
                                            with_extend_and_keep));
  assert((values(with_extend_and_keep) == std::vector<int>{7, 16, 6}));
}

void test_select_neighbours_heuristic_m_zero() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::Node query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  std::vector<LineGraph::Node> candidates{LineGraph::Node{9},
                                          LineGraph::Node{11}};
  std::vector<LineGraph::Node> result;
  assert(!layer.select_neighbours_heuristic(candidates, 0,
                                            /*extend_candidates=*/false,
                                            /*keep_pruned_connections=*/false,
                                            result));
  assert(result.empty());
}

void test_select_neighbours_heuristic_distance_failure_propagates() {
  LineGraph graph;
  graph.n = 20;
  graph.fail_distance = true;
  LineGraph::Node query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  std::vector<LineGraph::Node> candidates{LineGraph::Node{5},
                                          LineGraph::Node{9}};
  std::vector<LineGraph::Node> result{LineGraph::Node{-1}};
  assert(layer.select_neighbours_heuristic(candidates, 2,
                                           /*extend_candidates=*/false,
                                           /*keep_pruned_connections=*/false,
                                           result));
  // Failure must leave out unchanged.
  assert((values(result) == std::vector<int>{-1}));
}

void test_select_neighbours_heuristic_neighbours_failure_propagates() {
  LineGraph graph;
  graph.n = 20;
  graph.fail_neighbours = true;
  LineGraph::Node query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, query, VisibilityPolicy());

  std::vector<LineGraph::Node> candidates{LineGraph::Node{5},
                                          LineGraph::Node{9}};
  std::vector<LineGraph::Node> result{LineGraph::Node{-1}};
  assert(layer.select_neighbours_heuristic(candidates, 2,
                                           /*extend_candidates=*/true,
                                           /*keep_pruned_connections=*/false,
                                           result));
  assert((values(result) == std::vector<int>{-1}));
}

} // namespace

int main() {
  test_basic_knn();
  test_reuse_after_search_layer();
  test_ef_bounds_result_size();
  test_distance_failure_propagates();
  test_neighbours_failure_propagates();
  test_is_visible_failure_propagates();
  test_invisible_nodes_excluded_from_result();
  test_invisible_entry_point_does_not_seed_result();
  test_always_visible_policy_ignores_invisibility();
  test_select_neighbours_simple();
  test_select_neighbours_simple_distance_failure_propagates();
  test_select_neighbours_heuristic_prefers_diverse_candidates();
  test_select_neighbours_heuristic_keeps_pruned_connections();
  test_select_neighbours_heuristic_extend_candidates();
  test_select_neighbours_heuristic_m_zero();
  test_select_neighbours_heuristic_distance_failure_propagates();
  test_select_neighbours_heuristic_neighbours_failure_propagates();

  std::printf("All hnsw_search tests passed.\n");
  return 0;
}
