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
  struct NodeData {
    int value;
    // Distinguishes the query vector (bound once via m_query) from a node's
    // vector resolved as a fixed operand (resolve_fixed_operand). Only the
    // former counts as a query-to-candidate distance; is_dominated() compares a
    // resolved node against others, which is not a query distance.
    bool is_query = true;
  };
  using DistanceType = float;
  // LineGraph is a single flat graph with no notion of levels; LevelId
  // exists only to satisfy the Graph interface and carries no state.
  struct LevelId {
    bool operator==(const LevelId &) const = default;
  };

  int n = 0;
  bool fail_distance = false;
  bool fail_neighbours = false;
  bool fail_is_visible = false;
  std::vector<int> invisible;
  // Node values for which is_visible() fails, distinct from fail_is_visible
  // (which fails unconditionally on every call regardless of which node is
  // being checked).
  std::vector<int> fail_is_visible_for;
  // If non-negative, distance() succeeds for its first fail_distance_after
  // calls and fails from the next call onward, letting a test target a
  // specific call site (e.g. is_dominated()'s) rather than the first
  // distance() call made overall.
  int fail_distance_after = -1;
  int distance_calls = 0;
  // Counts only the query-to-node overload below, distinct from
  // distance_calls (which counts both overloads) -- lets a test check
  // specifically whether a query-to-candidate distance was recomputed
  // rather than reused from a preceding search()/seed() call.
  int query_distance_calls = 0;

  bool distance(const Node &a, const Node &b, DistanceType &out) {
    ++distance_calls;
    if (fail_distance) {
      return true;
    }
    if (fail_distance_after >= 0 && distance_calls > fail_distance_after) {
      return true;
    }
    out = std::abs(a.value - b.value);
    return false;
  }

  bool distance(const NodeData &a, const Node &b, DistanceType &out) {
    ++distance_calls;
    if (a.is_query) ++query_distance_calls;
    if (fail_distance) {
      return true;
    }
    if (fail_distance_after >= 0 && distance_calls > fail_distance_after) {
      return true;
    }
    out = std::abs(a.value - b.value);
    return false;
  }

  // Resolve a node to its NodeData for use as a fixed distance operand (see
  // IndexGraph::resolve_fixed_operand). For LineGraph a node's data is just its
  // value; resolving is not a distance call, so no counter changes.
  bool resolve_fixed_operand(const Node &node, NodeData &out) {
    out = NodeData{node.value, /*is_query=*/false};
    return false;
  }

  bool neighbours(const Node &node, LevelId /*level*/, std::vector<Node> &out) {
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
    if (std::find(fail_is_visible_for.begin(), fail_is_visible_for.end(),
                  node.value) != fail_is_visible_for.end()) {
      return true;
    }
    out = std::find(invisible.begin(), invisible.end(), node.value) ==
          invisible.end();
    return false;
  }
};

// Consults LineGraph::is_visible(), as a real query-time caller would.
template <typename Graph> struct VisibilityPolicy {
  using Node = typename Graph::Node;
  static bool is_visible(Graph &graph, const Node &node, bool &out) {
    return graph.is_visible(node, out);
  }
};

// Treats every node as visible without consulting the graph, as an
// index-construction caller would.
template <typename Graph> struct AlwaysVisiblePolicy {
  using Node = typename Graph::Node;
  static bool is_visible(Graph &, const Node &, bool &out) {
    out = true;
    return false;
  }
};

} // namespace

// LayerOperations's member functions are defined in layer_ops_impl.h, not
// layer_ops.h, so they must be explicitly instantiated for LineGraph
// somewhere that includes the definitions -- there's no separately-built
// object file to link against for an arbitrary test-only Graph type.
#include "../src/index/hnsw/layer_ops_impl.h"

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
  LineGraph::NodeData query{15};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  std::vector<LineGraph::Node> candidates{LineGraph::Node{0}};
  assert(!layer.search(candidates, LineGraph::LevelId{}, 5));
  layer.consume_all(candidates);
  assert(candidates.size() == 5);

  // Nearest 5 nodes to 15 among 0..19 are {13,14,15,16,17}.
  std::vector<int> got = values(candidates);
  std::vector<int> sorted_got = got;
  std::sort(sorted_got.begin(), sorted_got.end());
  assert((sorted_got == std::vector<int>{13, 14, 15, 16, 17}));

  // consume_all() must order the result nearest-first.
  float prev = -1;
  for (int v : got) {
    float d = std::abs(v - query.value);
    assert(d >= prev);
    prev = d;
  }
}

void test_reuse_after_search() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::NodeData query{15};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  std::vector<LineGraph::Node> first_candidates{LineGraph::Node{0}};
  assert(!layer.search(first_candidates, LineGraph::LevelId{}, 5));
  layer.consume_all(first_candidates);
  assert(first_candidates.size() == 5);

  // search() resets internally, so it can be called again with the same
  // query without an explicit reset() call.
  std::vector<LineGraph::Node> second_candidates{LineGraph::Node{19}};
  assert(!layer.search(second_candidates, LineGraph::LevelId{}, 3));
  layer.consume_all(second_candidates);
  assert(second_candidates.size() == 3);

  std::vector<int> sorted_got = values(second_candidates);
  std::sort(sorted_got.begin(), sorted_got.end());
  assert((sorted_got == std::vector<int>{14, 15, 16}));
}

void test_search_then_consume_heuristic_reuses_search_distances() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  std::vector<LineGraph::Node> candidates{LineGraph::Node{0}};
  assert(!layer.search(candidates, LineGraph::LevelId{}, 5));
  int query_calls_after_search = graph.query_distance_calls;
  assert(query_calls_after_search > 0);

  std::vector<LineGraph::Node> result;
  assert(!layer.consume_heuristic(3, decltype(layer)::ExtendCandidates::No,
                                  decltype(layer)::KeepPrunedConnections::No,
                                  result));

  // consume_heuristic() with ExtendCandidates::No must not re-evaluate the
  // query-to-candidate distance for any node search() already scored --
  // that's the whole point of splitting search() from the consume_*()
  // step: the result's distances are handed forward instead of being
  // recomputed.
  assert(graph.query_distance_calls == query_calls_after_search);
}

void test_reset_rebinds_query() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::NodeData initial_query{5};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(
      graph, initial_query);

  // Rebind to a different query node via the public reset(query) overload,
  // before search()'s own internal no-arg reset() (which must preserve
  // whatever query is currently bound) ever runs.
  LineGraph::NodeData new_query{15};
  layer.reset(new_query);

  std::vector<LineGraph::Node> candidates{LineGraph::Node{0}};
  assert(!layer.search(candidates, LineGraph::LevelId{}, 5));
  layer.consume_all(candidates);

  // Nearest 5 nodes to 15 (the rebound query), not to 5 (the constructor's
  // query), are {13,14,15,16,17}.
  std::vector<int> sorted_got = values(candidates);
  std::sort(sorted_got.begin(), sorted_got.end());
  assert((sorted_got == std::vector<int>{13, 14, 15, 16, 17}));

  // The rebind must stick across further reuse too: search()'s internal
  // no-arg reset() call must not revert to the constructor's query.
  std::vector<LineGraph::Node> second_candidates{LineGraph::Node{19}};
  assert(!layer.search(second_candidates, LineGraph::LevelId{}, 3));
  layer.consume_all(second_candidates);
  std::vector<int> sorted_second = values(second_candidates);
  std::sort(sorted_second.begin(), sorted_second.end());
  assert((sorted_second == std::vector<int>{14, 15, 16}));
}

void test_ef_bounds_result_size() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  std::vector<LineGraph::Node> candidates{LineGraph::Node{10}};
  uint32_t ef = 1;
  assert(!layer.search(candidates, LineGraph::LevelId{}, ef));
  layer.consume_all(candidates);
  assert(candidates.size() == ef);
  assert(candidates[0].value == 10);
}

void test_search_dedups_entry_points() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  // 10 appears twice among the entry points; seed()'s m_visited set must
  // ensure it's only added to the candidate/result sets once. If it were
  // added twice, it would occupy two of the five result slots instead of
  // one, changing which of its neighbours make the cut.
  std::vector<LineGraph::Node> candidates{LineGraph::Node{10},
                                          LineGraph::Node{10}};
  assert(!layer.search(candidates, LineGraph::LevelId{}, 5));
  layer.consume_all(candidates);

  std::vector<int> sorted_got = values(candidates);
  std::sort(sorted_got.begin(), sorted_got.end());
  assert((sorted_got == std::vector<int>{8, 9, 10, 11, 12}));
}

void test_distance_failure_propagates() {
  LineGraph graph;
  graph.n = 20;
  graph.fail_distance = true;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  std::vector<LineGraph::Node> candidates{LineGraph::Node{10}};
  assert(layer.search(candidates, LineGraph::LevelId{}, 5));
}

void test_neighbours_failure_propagates() {
  LineGraph graph;
  graph.n = 20;
  graph.fail_neighbours = true;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  // With ef=1 and a single entry point equal to the query (distance 0),
  // Algorithm 2's break condition (line 7) never triggers, forcing an
  // expand() call that hits the failing neighbours().
  std::vector<LineGraph::Node> candidates{LineGraph::Node{10}};
  assert(layer.search(candidates, LineGraph::LevelId{}, 1));
}

void test_is_visible_failure_propagates() {
  LineGraph graph;
  graph.n = 20;
  graph.fail_is_visible = true;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  std::vector<LineGraph::Node> candidates{LineGraph::Node{10}};
  assert(layer.search(candidates, LineGraph::LevelId{}, 1));
}

void test_expand_is_visible_failure_propagates() {
  LineGraph graph;
  graph.n = 20;
  // Distinct from test_is_visible_failure_propagates(): the entry point
  // (0) passes is_visible() inside seed(), so the failure can only occur
  // later, when expand() calls is_visible() for 15 -- a node discovered
  // mid-search, at expand()'s own call site rather than seed()'s. This is
  // the same graph shape as test_invisible_nodes_excluded_from_result(),
  // which already establishes that the search reaches and checks node 15
  // via expand() before it can be excluded from the result.
  graph.fail_is_visible_for = {15};
  LineGraph::NodeData query{15};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  std::vector<LineGraph::Node> candidates{LineGraph::Node{0}};
  assert(layer.search(candidates, LineGraph::LevelId{}, 6));
}

void test_invisible_nodes_excluded_from_result() {
  LineGraph graph;
  graph.n = 20;
  // 15 is the nearest node to the query but is invisible, so the search
  // must traverse through it to reach the surrounding visible nodes
  // instead of returning it.
  graph.invisible = {15};
  LineGraph::NodeData query{15};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  // Entry point is off to one side (0) so the search sweeps in from a
  // single direction, avoiding same-distance ties in the result set at
  // the ef cutoff (unlike starting from the query itself).
  std::vector<LineGraph::Node> candidates{LineGraph::Node{0}};
  assert(!layer.search(candidates, LineGraph::LevelId{}, 6));
  layer.consume_all(candidates);

  // Without 15, the nearest 6 visible nodes to 15 are {12,13,14,16,17,18}
  // (distances 3,2,1,1,2,3); 15 itself (distance 0) is excluded despite
  // being the closest possible node.
  std::vector<int> got = values(candidates);
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
  LineGraph::NodeData query{0};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  // The sole entry point is invisible and is also the query itself, so
  // with ef=1 the search must expand past it to find a visible result
  // instead of returning early with an empty/invisible result.
  std::vector<LineGraph::Node> candidates{LineGraph::Node{0}};
  assert(!layer.search(candidates, LineGraph::LevelId{}, 1));
  layer.consume_all(candidates);
  assert((values(candidates) == std::vector<int>{1}));
}

void test_always_visible_policy_ignores_invisibility() {
  LineGraph graph;
  graph.n = 20;
  // Would exclude 0 from the result and fail the search under
  // VisibilityPolicy, but AlwaysVisiblePolicy never consults either.
  graph.invisible = {0};
  graph.fail_is_visible = true;
  LineGraph::NodeData query{0};
  svector::hnsw::LayerOperations<LineGraph, AlwaysVisiblePolicy> layer(graph,
                                                                       query);

  std::vector<LineGraph::Node> candidates{LineGraph::Node{0}};
  assert(!layer.search(candidates, LineGraph::LevelId{}, 1));
  layer.consume_all(candidates);
  assert((values(candidates) == std::vector<int>{0}));
}

void test_consume_simple() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  std::vector<LineGraph::Node> candidates{
      LineGraph::Node{0},  LineGraph::Node{7},  LineGraph::Node{9},
      LineGraph::Node{12}, LineGraph::Node{19}, LineGraph::Node{10}};

  // No preceding graph traversal is needed to score this fixed candidate
  // set, so seed() (Step 1 without the expansion loop) stands in for
  // search().
  assert(!layer.seed(candidates, LineGraph::LevelId{}));
  std::vector<LineGraph::Node> result;
  layer.consume_simple(3, result);
  assert(result.size() == 3);

  // Nearest 3 candidates to 10 are {10,9,12} (distances 0,1,2), ordered by
  // increasing distance.
  std::vector<int> got = values(result);
  assert((got == std::vector<int>{10, 9, 12}));

  // M larger than the candidate set returns every candidate, still ordered
  // by increasing distance.
  assert(!layer.seed(candidates, LineGraph::LevelId{}));
  std::vector<LineGraph::Node> all;
  layer.consume_simple(10, all);
  std::vector<int> got_all = values(all);
  assert((got_all == std::vector<int>{10, 9, 12, 7, 19, 0}));
}

void test_seed_distance_failure_propagates() {
  LineGraph graph;
  graph.n = 20;
  graph.fail_distance = true;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  std::vector<LineGraph::Node> candidates{LineGraph::Node{5}};
  assert(layer.seed(candidates, LineGraph::LevelId{}));
}

void test_consume_heuristic_prefers_diverse_candidates() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  // Distances to 10: 5->5, 9->1, 12->2, 16->6. Nearest-first: 9,12,5,16.
  // 5 is dominated by 9 (dist(5,9)=4 < dist(5,10)=5) and 16 is dominated by
  // 12 (dist(16,12)=4 < dist(16,10)=6), so only {9,12} pass the heuristic
  // even though M=3 asks for more -- unlike consume_simple(), which would
  // also return the third-nearest candidate.
  std::vector<LineGraph::Node> candidates{
      LineGraph::Node{5}, LineGraph::Node{9}, LineGraph::Node{12},
      LineGraph::Node{16}};
  assert(!layer.seed(candidates, LineGraph::LevelId{}));

  std::vector<LineGraph::Node> result;
  assert(!layer.consume_heuristic(3, decltype(layer)::ExtendCandidates::No,
                                  decltype(layer)::KeepPrunedConnections::No,
                                  result));
  assert((values(result) == std::vector<int>{9, 12}));
}

void test_consume_heuristic_keeps_pruned_connections() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  // Same candidates as above; with keep_pruned_connections set, the
  // remaining slot is backfilled from the discarded candidates, nearest
  // to the query first: 5 (dist 5) before 16 (dist 6).
  std::vector<LineGraph::Node> candidates{
      LineGraph::Node{5}, LineGraph::Node{9}, LineGraph::Node{12},
      LineGraph::Node{16}};
  assert(!layer.seed(candidates, LineGraph::LevelId{}));

  std::vector<LineGraph::Node> result;
  assert(!layer.consume_heuristic(3, decltype(layer)::ExtendCandidates::No,
                                  decltype(layer)::KeepPrunedConnections::Yes,
                                  result));
  assert((values(result) == std::vector<int>{9, 12, 5}));
}

void test_consume_heuristic_extend_candidates() {
  LineGraph graph;
  graph.n = 30;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  // Candidates {6,17} extend to {6,17,5,7,16,18} (neighbours of 6 and 17).
  // Nearest-first by distance to 10: 7(3),6(4),5(5),16(6),17(7),18(8).
  // 6 and 5 are dominated by 7 (dist 1 and 2, both < their distance to the
  // query); 17 and 18 are dominated by 16 (dist 1 and 2). So extending the
  // candidate set replaces the originally-nearest {6,17} with {7,16}.
  // Each variant below needs its own seed(): consume_heuristic() consumes
  // the Consume state exactly once, so a fresh seed() is required before
  // each independent consume_heuristic() call on the same candidates.
  std::vector<LineGraph::Node> candidates{LineGraph::Node{6},
                                          LineGraph::Node{17}};

  assert(!layer.seed(candidates, LineGraph::LevelId{}));
  std::vector<LineGraph::Node> without_extend;
  assert(!layer.consume_heuristic(3, decltype(layer)::ExtendCandidates::No,
                                  decltype(layer)::KeepPrunedConnections::No,
                                  without_extend));
  assert((values(without_extend) == std::vector<int>{6, 17}));

  assert(!layer.seed(candidates, LineGraph::LevelId{}));
  std::vector<LineGraph::Node> with_extend;
  assert(!layer.consume_heuristic(3, decltype(layer)::ExtendCandidates::Yes,
                                  decltype(layer)::KeepPrunedConnections::No,
                                  with_extend));
  assert((values(with_extend) == std::vector<int>{7, 16}));

  // With keep_pruned_connections, the remaining slot backfills from the
  // discarded candidates nearest to the query: 6 (dist 4) before 5, 17, 18.
  assert(!layer.seed(candidates, LineGraph::LevelId{}));
  std::vector<LineGraph::Node> with_extend_and_keep;
  assert(!layer.consume_heuristic(3, decltype(layer)::ExtendCandidates::Yes,
                                  decltype(layer)::KeepPrunedConnections::Yes,
                                  with_extend_and_keep));
  assert((values(with_extend_and_keep) == std::vector<int>{7, 16, 6}));
}

void test_consume_heuristic_extend_candidates_dedups_overlap() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  // 8 is a neighbour of both 7 and 9, so extending each candidate must add
  // it to the working set only once, via the shared visited set -- not once
  // per candidate that reaches it. Distances to
  // 10: 7->3, 9->1, 6->4, 8->2, 10->0 (10 is also pulled in, as a
  // neighbour of 9).
  std::vector<LineGraph::Node> candidates{LineGraph::Node{7},
                                          LineGraph::Node{9}};
  assert(!layer.seed(candidates, LineGraph::LevelId{}));

  std::vector<LineGraph::Node> result;
  assert(!layer.consume_heuristic(4, decltype(layer)::ExtendCandidates::Yes,
                                  decltype(layer)::KeepPrunedConnections::Yes,
                                  result));

  // Nearest-first: 10,9,8,7,6. 10 and 9 pass the heuristic; 8, 7 and 6 are
  // each dominated (by 9 or 10) and backfilled nearest-first since
  // keep_pruned_connections is set. Had 8 been added twice instead of
  // deduped, it would occupy both backfill slots here in place of 7.
  assert((values(result) == std::vector<int>{10, 9, 8, 7}));
}

void test_consume_heuristic_m_zero() {
  LineGraph graph;
  graph.n = 20;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  std::vector<LineGraph::Node> candidates{LineGraph::Node{9},
                                          LineGraph::Node{11}};
  assert(!layer.seed(candidates, LineGraph::LevelId{}));
  std::vector<LineGraph::Node> result;
  assert(!layer.consume_heuristic(0, decltype(layer)::ExtendCandidates::No,
                                  decltype(layer)::KeepPrunedConnections::No,
                                  result));
  assert(result.empty());
}

void test_consume_heuristic_is_dominated_distance_failure_propagates() {
  LineGraph graph;
  graph.n = 20;
  // Let the three distance() calls inside seed()'s evaluate_distances()
  // succeed, and fail from the fourth call onward -- the first call made
  // from within is_dominated() itself. This is the only way to reach
  // is_dominated()'s own distance() failure branch: seed() always
  // evaluates every candidate's distance up front, so a graph that fails
  // unconditionally (fail_distance) would always fail there first, before
  // is_dominated() ever runs.
  graph.fail_distance_after = 3;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  // Nearest-first: 9(1),12(2),5(5). 9 is admitted without consulting
  // is_dominated() (the result set is still empty). Checking 12 against 9
  // is the first is_dominated() distance() call -- the fourth overall.
  std::vector<LineGraph::Node> candidates{
      LineGraph::Node{5}, LineGraph::Node{9}, LineGraph::Node{12}};
  assert(!layer.seed(candidates, LineGraph::LevelId{}));

  std::vector<LineGraph::Node> result{LineGraph::Node{-1}};
  assert(layer.consume_heuristic(3, decltype(layer)::ExtendCandidates::No,
                                 decltype(layer)::KeepPrunedConnections::No,
                                 result));
  // Failure must leave out unchanged.
  assert((values(result) == std::vector<int>{-1}));
}

void test_consume_heuristic_extend_candidates_neighbours_failure_propagates() {
  LineGraph graph;
  graph.n = 20;
  graph.fail_neighbours = true;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  // fail_neighbours doesn't affect seed() (which never calls
  // neighbours()), so the failure can only be reached inside
  // consume_heuristic()'s own extension step, when ExtendCandidates::Yes
  // makes it fetch each candidate's neighbours.
  std::vector<LineGraph::Node> candidates{LineGraph::Node{5},
                                          LineGraph::Node{9}};
  assert(!layer.seed(candidates, LineGraph::LevelId{}));

  std::vector<LineGraph::Node> result{LineGraph::Node{-1}};
  assert(layer.consume_heuristic(2, decltype(layer)::ExtendCandidates::Yes,
                                 decltype(layer)::KeepPrunedConnections::No,
                                 result));
  assert((values(result) == std::vector<int>{-1}));
}

void test_consume_heuristic_extend_candidates_distance_failure_propagates() {
  LineGraph graph;
  graph.n = 30;
  // seed()'s own evaluate_distances() call (for candidates {6,17}) makes
  // the first two distance() calls; let those succeed and fail from the
  // third call onward -- the first call evaluating a newly-extended
  // neighbour's distance, a branch only reachable once extend_candidates
  // adds nodes beyond the seeded set.
  graph.fail_distance_after = 2;
  LineGraph::NodeData query{10};
  svector::hnsw::LayerOperations<LineGraph, VisibilityPolicy> layer(graph,
                                                                    query);

  std::vector<LineGraph::Node> candidates{LineGraph::Node{6},
                                          LineGraph::Node{17}};
  assert(!layer.seed(candidates, LineGraph::LevelId{}));

  std::vector<LineGraph::Node> result{LineGraph::Node{-1}};
  assert(layer.consume_heuristic(3, decltype(layer)::ExtendCandidates::Yes,
                                 decltype(layer)::KeepPrunedConnections::No,
                                 result));
  assert((values(result) == std::vector<int>{-1}));
}

} // namespace

int main() {
  test_basic_knn();
  test_reuse_after_search();
  test_search_then_consume_heuristic_reuses_search_distances();
  test_reset_rebinds_query();
  test_ef_bounds_result_size();
  test_search_dedups_entry_points();
  test_distance_failure_propagates();
  test_neighbours_failure_propagates();
  test_is_visible_failure_propagates();
  test_expand_is_visible_failure_propagates();
  test_invisible_nodes_excluded_from_result();
  test_invisible_entry_point_does_not_seed_result();
  test_always_visible_policy_ignores_invisibility();
  test_consume_simple();
  test_seed_distance_failure_propagates();
  test_consume_heuristic_prefers_diverse_candidates();
  test_consume_heuristic_keeps_pruned_connections();
  test_consume_heuristic_extend_candidates();
  test_consume_heuristic_extend_candidates_dedups_overlap();
  test_consume_heuristic_m_zero();
  test_consume_heuristic_is_dominated_distance_failure_propagates();
  test_consume_heuristic_extend_candidates_neighbours_failure_propagates();
  test_consume_heuristic_extend_candidates_distance_failure_propagates();

  std::printf("All layer_ops tests passed.\n");
  return 0;
}
