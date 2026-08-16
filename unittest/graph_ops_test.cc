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

// Standalone, assert()-based unit test for GraphOperations<Graph>. Has no
// dependency on the VillageSQL SDK: like LayerOperations, GraphOperations is
// a pure template over a user-supplied Graph, so it's exercised here against
// a small mock graph (see unittest/layer_ops_test.cc for the sibling test).
//
// Unlike LayerOperations's Graph -- a read-only distance/neighbour oracle --
// GraphOperations's Graph is a stateful, multi-level, lock-protected
// structure: nodes are created/dropped per level, edges are linked/unlinked,
// and the entry point is read-modify-written under a two-mode (Shared/
// Exclusive) lock. MockGraph below models all of that with plain in-memory
// containers: levels are small integers, nodes are (level, id) pairs
// sharing a per-id "position" for distance (same line-graph convention as
// LineGraph in layer_ops_test.cc), and locking is single-threaded
// bookkeeping only -- real concurrency safety of the locking protocol is
// out of scope for an assert()-based single-threaded test; only the
// sequence of mode/level transitions GraphOperations itself relies on for
// correctness is exercised.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <compare>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Mirrors svector::hnsw::LevelStore::LevelId (src/index/hnsw/storage.h) closely
// enough to exercise GraphOperations's level-descent logic without a
// dependency on it.
struct MockLevelId {
  uint8_t value = 0;

  constexpr MockLevelId() = default;
  constexpr explicit MockLevelId(uint8_t v) : value(v) {}
  constexpr bool has_lower_level() const { return value > 0; }
  constexpr MockLevelId lower() const {
    return value > 0 ? MockLevelId(static_cast<uint8_t>(value - 1))
                     : MockLevelId(0);
  }
  constexpr bool operator==(const MockLevelId &) const = default;
  constexpr auto operator<=>(const MockLevelId &) const = default;
};

// A small stateful multi-level graph: elements sit on a line indexed by id
// (distance(a, b) = |position(a) - position(b)|), and accumulate per-level
// adjacency as GraphOperations links them in. An id present at level L is
// assumed -- by construction, mirroring the real invariant -- to also be
// present at every level below L.
struct MockGraph {
  struct Node {
    using KeyType = int;
    MockLevelId level;
    int id = -1;
    KeyType key() const { return id; }
  };
  // Stand-in for the real NodeData's materialized vector payload. id is a
  // mock-only convenience letting tests pick a new element's id up front,
  // unlike the real system where create_node() assigns it.
  struct NodeData {
    int id = -1;
    int data = 0;
  };
  using LevelId = MockLevelId;
  using DistanceType = float;
  enum class LockMode { Shared, Exclusive };
  enum class UnlinkOrphans { No, Yes };

  // ---- RAII lock guards. Single-threaded bookkeeping only: these tests
  // don't exercise real mutual exclusion, only the sequence of mode/level
  // transitions GraphOperations itself relies on for correctness (e.g.
  // upgrading S->X only when a new entry point turns out to be needed).
  class LockGraph {
  public:
    LockGraph(MockGraph &graph, LockMode mode) : m_graph(graph), m_mode(mode) {}
    ~LockGraph() = default;
    LockGraph(const LockGraph &) = delete;
    LockGraph &operator=(const LockGraph &) = delete;
    LockGraph(LockGraph &&) = delete;
    LockGraph &operator=(LockGraph &&) = delete;

    void upgrade() {
      if (m_mode == LockMode::Exclusive) {
        return;
      }
      m_mode = LockMode::Exclusive;
      ++m_graph.upgrade_calls;
    }
    LockMode lock_mode() const { return m_mode; }

  private:
    MockGraph &m_graph;
    LockMode m_mode;
  };

  class LockLevels {
  public:
    enum class DescendPolicy { Keep, Release };

    LockLevels(MockGraph &graph, LockMode mode, LevelId start,
               DescendPolicy policy)
        : m_mode(mode), m_policy(policy) {
      // Mirror the real IndexGraph::lock_level(): a level store must exist to
      // be locked. GraphOperations::insert() constructs LockLevels(start =
      // max(entry_level, insert_level)) before create_node()/ensure_levels()
      // materializes a new top level, so `start` may exceed the created levels.
      // The real code applies a create-on-demand stopgap in lock_level()
      // (ensure_levels() before locking); model that here by raising
      // created_max_level, so this mock reflects the fixed behavior rather than
      // aborting. See MockGraph::created_max_level and the TEMPORARY note in
      // src/index/hnsw/graph.cc lock_level().
      if (start.value > graph.created_max_level) {
        graph.created_max_level = start.value;
      }
      m_stack.push(start);
    }
    ~LockLevels() = default;
    LockLevels(const LockLevels &) = delete;
    LockLevels &operator=(const LockLevels &) = delete;
    LockLevels(LockLevels &&) = delete;
    LockLevels &operator=(LockLevels &&) = delete;

    LevelId descend() {
      LevelId current = m_stack.top();
      if (!current.has_lower_level()) {
        return current;
      }
      LevelId next = current.lower();
      if (m_policy == DescendPolicy::Release) {
        m_stack.pop();
      }
      m_stack.push(next);
      return next;
    }

    void update_policy(DescendPolicy policy) { m_policy = policy; }
    LevelId level() const { return m_stack.top(); }

  private:
    LockMode m_mode;
    DescendPolicy m_policy;
    std::stack<LevelId> m_stack;
  };

  // ---- Spatial model (for distance()).
  std::unordered_map<int, int> position;

  // ---- Per-level graph state: presence and adjacency, keyed by level value.
  std::map<uint8_t, std::set<int>> present;
  std::map<uint8_t, std::unordered_map<int, std::vector<int>>> adjacency;

  // ---- Highest level whose store has been CREATED. Level 0 always exists;
  // higher levels come into existence only when create_node() materializes
  // them (the real system's ensure_levels()). LockLevels asserts it never
  // locks above this. This is what surfaces the "lock a level before it is
  // created" ordering bug that the real IndexGraph hits but a permissive mock
  // hides.
  uint8_t created_max_level = 0;

  // ---- Stand-in for external storage: create_node() persists each id's
  // data here, independent of any in-memory Node the caller happens to be
  // holding -- Node itself carries no data, mirroring the real system,
  // so tests read this map directly to check data survived a round trip.
  std::unordered_map<int, int> node_data;

  // ---- Soft-deletion, for visible().
  std::set<int> deleted;

  // ---- Entry point(s). Real HNSW only ever has one, but the interface
  // takes a vector (see is_sole_entry_point() in remove()), so tests can
  // populate more than one to exercise that branch.
  std::vector<Node> entry_points;
  LevelId entry_level{0};

  // ---- Construction parameters. m_Mmax defaults far above m_M so existing
  // fixtures -- built before Mmax-shrinking existed -- can't incidentally
  // trigger it; tests exercising the shrink path set it explicitly.
  uint32_t m_M = 3;
  uint32_t m_ef_construction = 10;
  uint32_t m_Mmax = 1000;

  // ---- get_insert_level() is randomised in a real graph; tests drive it
  // deterministically by queueing the levels successive insert() calls
  // should use (default: level 0 if the queue is empty).
  std::deque<uint8_t> insert_levels;

  // ---- get_entry_point() can be told to report empty on its first call
  // regardless of actual state, to simulate a concurrent insert completing
  // during insert()/remove()'s S->X upgrade window (see
  // test_insert_upgrade_recheck_suppresses_unneeded_entry_point_update).
  bool get_entry_point_first_call_empty = false;
  int get_entry_point_calls = 0;

  // ---- Failure injection, one flag per Graph method GraphOperations calls.
  bool fail_distance = false;
  bool fail_neighbours = false;
  bool fail_visible = false;
  bool fail_get_entry_point = false;
  bool fail_set_entry_point = false;
  bool fail_create_node = false;
  bool fail_drop_node = false;
  bool fail_get_next_level = false;
  bool fail_link_neighbours = false;
  bool fail_replace_neighbours = false;
  bool fail_unlink_neighbours = false;

  int upgrade_calls = 0;
  std::vector<std::string> call_log;

  uint32_t M() const { return m_M; }
  uint32_t ef_construction() const { return m_ef_construction; }
  uint32_t Mmax(LevelId /*level*/) const { return m_Mmax; }

  bool distance(const Node &a, const Node &b, DistanceType &out) {
    if (fail_distance) {
      return true;
    }
    out = std::abs(position.at(a.id) - position.at(b.id));
    return false;
  }

  bool distance(const NodeData &a, const Node &b, DistanceType &out) {
    if (fail_distance) {
      return true;
    }
    out = std::abs(position.at(a.id) - position.at(b.id));
    return false;
  }

  bool neighbours(const Node &node, LevelId level, std::vector<Node> &out) {
    assert(level == node.level);
    if (fail_neighbours) {
      return true;
    }
    out.clear();
    auto level_it = adjacency.find(node.level.value);
    if (level_it != adjacency.end()) {
      auto id_it = level_it->second.find(node.id);
      if (id_it != level_it->second.end()) {
        for (int nb : id_it->second) {
          out.push_back(Node{node.level, nb});
        }
      }
    }
    return false;
  }

  bool visible(const Node &node, bool &out) {
    if (fail_visible) {
      return true;
    }
    out = deleted.find(node.id) == deleted.end();
    return false;
  }

  LevelId get_insert_level() {
    if (insert_levels.empty()) {
      return LevelId(0);
    }
    uint8_t v = insert_levels.front();
    insert_levels.pop_front();
    return LevelId(v);
  }

  bool get_entry_point(std::vector<Node> &out, LevelId &out_level) {
    if (fail_get_entry_point) {
      return true;
    }
    if (get_entry_point_first_call_empty && get_entry_point_calls == 0) {
      ++get_entry_point_calls;
      out.clear();
      out_level = LevelId(0);
      return false;
    }
    ++get_entry_point_calls;
    out = entry_points;
    out_level = entry_level;
    return false;
  }

  bool set_entry_point(const std::vector<Node> &nodes, LevelId level) {
    if (fail_set_entry_point) {
      return true;
    }
    entry_points = nodes;
    entry_level = level;
    call_log.push_back("set_entry_point:" +
                       (nodes.empty() ? std::string("none")
                                      : std::to_string(nodes.front().id)) +
                       "@" + std::to_string(level.value));
    return false;
  }

  bool create_node(const std::optional<Node> &parent, LevelId level,
                   const NodeData &data, const std::vector<Node> &neighbours,
                   Node &out) {
    if (fail_create_node) {
      return true;
    }
    // When parent is unset, this is the topmost level for the element being
    // inserted, and its id comes from data; otherwise it's the same element
    // one level down, so its id is derived from parent.
    int id = parent ? parent->id : data.id;
    // The real create_node() calls ensure_levels() on its topmost (parentless)
    // call, materializing every level up to `level`. Mirror that so LockLevels'
    // created-level check reflects reality.
    if (!parent && level.value > created_max_level) {
      created_max_level = level.value;
    }
    present[level.value].insert(id);
    node_data[id] = data.data;
    out = Node{level, id};
    auto &adj = adjacency[level.value][id];
    for (const Node &nb : neighbours) {
      if (std::find(adj.begin(), adj.end(), nb.id) == adj.end()) {
        adj.push_back(nb.id);
      }
    }
    call_log.push_back("create:" + std::to_string(id) + "@" +
                       std::to_string(level.value));
    if (!neighbours.empty()) {
      call_log.push_back("link:" + std::to_string(id));
    }
    return false;
  }

  bool drop_node(const std::optional<Node> &parent, LevelId level,
                 const Node &node) {
    assert(level == node.level);
    if (fail_drop_node) {
      return true;
    }
    (void)parent;
    present[node.level.value].erase(node.id);
    adjacency[node.level.value].erase(node.id);
    call_log.push_back("drop:" + std::to_string(node.id) + "@" +
                       std::to_string(node.level.value));
    return false;
  }

  bool get_next_level_node(const Node &node, LevelId level, Node &out) {
    assert(level == node.level);
    if (fail_get_next_level) {
      return true;
    }
    out = Node{level.lower(), node.id};
    return false;
  }

  bool link_neighbours(const Node &node, LevelId level,
                       std::vector<Node> &out) {
    assert(level == node.level);
    if (fail_link_neighbours) {
      return true;
    }
    out.clear();
    // Snapshot first: node's own list isn't touched below, but this keeps
    // the iteration independent of any aliasing between node's and a
    // neighbour's adjacency vector.
    std::vector<int> forward = adjacency[node.level.value][node.id];
    for (int nb_id : forward) {
      auto &back_adj = adjacency[node.level.value][nb_id];
      if (std::find(back_adj.begin(), back_adj.end(), node.id) !=
          back_adj.end()) {
        continue;
      }
      if (back_adj.size() >= m_Mmax) {
        // Adding the edge would push nb_id over Mmax: withhold it and let
        // the caller reselect nb_id's full neighbour set instead.
        out.push_back(Node{node.level, nb_id});
        continue;
      }
      back_adj.push_back(node.id);
    }
    call_log.push_back("link_back:" + std::to_string(node.id));
    return false;
  }

  bool replace_neighbours(const Node &node, LevelId level,
                          const std::vector<Node> &new_neighbours) {
    assert(level == node.level);
    if (fail_replace_neighbours) {
      return true;
    }
    auto &adj = adjacency[node.level.value][node.id];
    adj.clear();
    for (const Node &nb : new_neighbours) {
      adj.push_back(nb.id);
    }
    call_log.push_back("replace:" + std::to_string(node.id));
    return false;
  }

  bool unlink_neighbours(const Node &node, LevelId level, UnlinkOrphans orphans,
                         std::vector<Node> &out) {
    assert(level == node.level);
    if (fail_unlink_neighbours) {
      return true;
    }
    out.clear();
    std::vector<int> to_remove;
    if (orphans == UnlinkOrphans::No) {
      // Sever exactly the neighbours that have another edge of their own.
      for (int nb_id : adjacency[node.level.value][node.id]) {
        auto &nb_adj = adjacency[node.level.value][nb_id];
        bool has_other_edge = nb_adj.size() > 1 ||
                              (nb_adj.size() == 1 && nb_adj.front() != node.id);
        if (has_other_edge) {
          to_remove.push_back(nb_id);
        }
      }
    } else {
      // Whatever's still linked at this point is exactly what the prior No
      // call left behind on purpose: the ones that would have been
      // orphaned by cutting them too.
      to_remove = adjacency[node.level.value][node.id];
    }
    for (int nb_id : to_remove) {
      auto &node_adj = adjacency[node.level.value][node.id];
      node_adj.erase(std::remove(node_adj.begin(), node_adj.end(), nb_id),
                     node_adj.end());
      auto &nb_adj = adjacency[node.level.value][nb_id];
      nb_adj.erase(std::remove(nb_adj.begin(), nb_adj.end(), node.id),
                   nb_adj.end());
      out.push_back(Node{node.level, nb_id});
    }
    call_log.push_back(
        (orphans == UnlinkOrphans::No ? "unlink_no:" : "unlink_yes:") +
        std::to_string(node.id));
    return false;
  }
};

} // namespace

// GraphOperations's member functions are defined in graph_ops_impl.h, not
// graph_ops.h, so they must be explicitly instantiated for MockGraph
// somewhere that includes the definitions -- there's no separately-built
// object file to link against for an arbitrary test-only Graph type (see
// graph_ops.cc for the IndexGraph instantiation). GraphOperations in turn
// uses LayerOperations, whose own out-of-line definitions live in
// layer_ops_impl.h -- graph_ops.h only forward-declares them, so that must
// be pulled in explicitly here too.
#include "../src/index/hnsw/graph_ops_impl.h"
#include "../src/index/hnsw/layer_ops_impl.h"

namespace svector::hnsw {
template class GraphOperations<MockGraph>;
} // namespace svector::hnsw

namespace {

using Node = MockGraph::Node;
using NodeData = MockGraph::NodeData;
using LevelId = MockGraph::LevelId;

std::vector<int> values(const std::vector<Node> &nodes) {
  std::vector<int> out;
  out.reserve(nodes.size());
  for (const auto &n : nodes) {
    out.push_back(n.id);
  }
  return out;
}

// ---- Fixture helpers ----

void add_element(MockGraph &g, int id, int pos) { g.position[id] = pos; }

void add_presence(MockGraph &g, int id, uint8_t level) {
  g.present[level].insert(id);
}

void add_edge(MockGraph &g, uint8_t level, int a, int b) {
  auto &aa = g.adjacency[level][a];
  if (std::find(aa.begin(), aa.end(), b) == aa.end()) {
    aa.push_back(b);
  }
  auto &bb = g.adjacency[level][b];
  if (std::find(bb.begin(), bb.end(), a) == bb.end()) {
    bb.push_back(a);
  }
}

// Builds a level-0 line: ids 0..n-1 at position == id, neighbours(v) =
// {v-1, v+1} (clamped) -- the same shape LineGraph used in layer_ops_test.cc.
void build_line_level0(MockGraph &g, int n) {
  for (int id = 0; id < n; ++id) {
    add_element(g, id, id);
    add_presence(g, id, 0);
  }
  for (int id = 0; id + 1 < n; ++id) {
    add_edge(g, 0, id, id + 1);
  }
}

bool do_insert(MockGraph &g, int id, int pos, int data = 0) {
  add_element(g, id, pos);
  svector::hnsw::GraphOperations<MockGraph> ops(g);
  return ops.insert(NodeData{id, data});
}

// ============================== search_knn ==============================

void test_search_knn_empty_graph() {
  MockGraph g;
  svector::hnsw::GraphOperations<MockGraph> ops(g);

  add_element(g, 0, 0); // query point only, not part of the graph
  std::vector<Node> result{Node{LevelId(0), -1}};
  assert(!ops.search_knn(NodeData{0}, 3, 5, result));
  assert(result.empty());
}

void test_search_knn_single_level() {
  MockGraph g;
  build_line_level0(g, 20);
  g.entry_points = {Node{LevelId(0), 0}};
  g.entry_level = LevelId(0);
  svector::hnsw::GraphOperations<MockGraph> ops(g);

  std::vector<Node> result;
  assert(!ops.search_knn(NodeData{15}, 5, 5, result));

  // Same shape as layer_ops_test.cc's test_basic_knn: nearest 5 to 15
  // among 0..19, starting the search from entry point 0.
  std::vector<int> sorted_got = values(result);
  std::sort(sorted_got.begin(), sorted_got.end());
  assert((sorted_got == std::vector<int>{13, 14, 15, 16, 17}));
}

void test_search_knn_k_truncates_ef_search_results() {
  MockGraph g;
  build_line_level0(g, 20);
  g.entry_points = {Node{LevelId(0), 0}};
  g.entry_level = LevelId(0);
  svector::hnsw::GraphOperations<MockGraph> ops(g);

  // k=1 avoids the 3-way tie at distance 1 (14, 16, 17) among the nearest
  // 5 -- 15 itself (distance 0) is the unique nearest, so truncating to
  // k=1 has one unambiguous right answer regardless of search_layer's
  // internal tie-break order.
  std::vector<Node> result;
  assert(!ops.search_knn(NodeData{15}, 1, 5, result));
  assert((values(result) == std::vector<int>{15}));
}

void test_search_knn_multi_level_descent() {
  MockGraph g;
  build_line_level0(g, 20);
  // A sparse level-1 chain over a subset of the same ids, connecting them
  // in a line just like level 0 does.
  for (int id : {0, 5, 10, 15}) {
    add_presence(g, id, 1);
  }
  add_edge(g, 1, 0, 5);
  add_edge(g, 1, 5, 10);
  add_edge(g, 1, 10, 15);
  g.entry_points = {Node{LevelId(1), 15}};
  g.entry_level = LevelId(1);
  svector::hnsw::GraphOperations<MockGraph> ops(g);

  // k=1 for the same unambiguous-nearest reason as the truncation test:
  // 18 is the unique nearest node to the query, reachable from the
  // level-1 entry point 15 via a handful of level-0 hops.
  std::vector<Node> result;
  assert(!ops.search_knn(NodeData{18}, 1, 4, result));
  assert((values(result) == std::vector<int>{18}));
}

void test_search_knn_visibility_filters_only_bottom_layer() {
  MockGraph g;
  build_line_level0(g, 20);
  for (int id : {0, 5, 10, 15}) {
    add_presence(g, id, 1);
  }
  add_edge(g, 1, 0, 5);
  add_edge(g, 1, 5, 10);
  add_edge(g, 1, 10, 15);
  // The level-1 entry point starts on the opposite side of the query from
  // the deleted nodes, so the upper-layer greedy descent must pass through
  // 15 (deleted) to reach the query's neighbourhood at all.
  g.entry_points = {Node{LevelId(1), 0}};
  g.entry_level = LevelId(1);
  // 15 is the query itself (would otherwise be the unique nearest); 16 is
  // its only distance-1 rival. Deleting both leaves 14 as the unique
  // nearest *visible* node.
  g.deleted = {15, 16};
  svector::hnsw::GraphOperations<MockGraph> ops(g);

  std::vector<Node> result;
  assert(!ops.search_knn(NodeData{15}, 1, 6, result));
  assert((values(result) == std::vector<int>{14}));
}

void test_search_knn_distance_failure_propagates() {
  MockGraph g;
  build_line_level0(g, 20);
  g.entry_points = {Node{LevelId(0), 0}};
  g.entry_level = LevelId(0);
  g.fail_distance = true;
  svector::hnsw::GraphOperations<MockGraph> ops(g);

  std::vector<Node> result;
  assert(ops.search_knn(NodeData{15}, 5, 5, result));
}

void test_search_knn_get_entry_point_failure_propagates() {
  MockGraph g;
  g.fail_get_entry_point = true;
  svector::hnsw::GraphOperations<MockGraph> ops(g);

  std::vector<Node> result;
  assert(ops.search_knn(NodeData{0}, 1, 1, result));
}

void test_search_knn_get_next_level_failure_propagates() {
  MockGraph g;
  build_line_level0(g, 20);
  for (int id : {0, 5, 10, 15}) {
    add_presence(g, id, 1);
  }
  add_edge(g, 1, 0, 5);
  add_edge(g, 1, 5, 10);
  add_edge(g, 1, 10, 15);
  g.entry_points = {Node{LevelId(1), 15}};
  g.entry_level = LevelId(1);
  g.fail_get_next_level = true;
  svector::hnsw::GraphOperations<MockGraph> ops(g);

  std::vector<Node> result;
  assert(ops.search_knn(NodeData{18}, 1, 4, result));
}

// ================================ insert =================================

void test_insert_bootstraps_entry_point_on_empty_graph() {
  MockGraph g;
  assert(!do_insert(g, 42, 10));

  assert(g.entry_points.size() == 1);
  assert(g.entry_points.front().id == 42);
  assert(g.entry_level == LevelId(0));
  assert(g.present[0].count(42) == 1);
  assert(g.adjacency[0][42].empty());
  // candidates was empty (first-ever element), so a new entry point was
  // unavoidable: the S->X upgrade must have happened.
  assert(g.upgrade_calls == 1);
}

void test_insert_links_nearest_neighbour_without_unnecessary_upgrade() {
  MockGraph g;
  assert(!do_insert(g, 0, 0));
  // Isolate the second insert's upgrade behaviour from the first's (which
  // unavoidably upgrades, since the graph starts empty).
  g.upgrade_calls = 0;
  assert(!do_insert(g, 1, 5));

  // 1's only possible neighbour is 0; both the forward link (from
  // create_node) and its reciprocal (from link_neighbours) must be present.
  assert((g.adjacency[0][1] == std::vector<int>{0}));
  assert((g.adjacency[0][0] == std::vector<int>{1}));
  // 0 is already the entry point at level 0, and this insert's level (0)
  // doesn't exceed it, so no new entry point -- and thus no upgrade --
  // should have been needed for the second insert.
  assert(g.upgrade_calls == 0);
  assert(g.entry_points.front().id == 0);
}

void test_insert_new_top_level_creates_records_at_every_level() {
  MockGraph g;
  assert(!do_insert(g, 0, 0));
  g.upgrade_calls = 0; // isolate the second insert's upgrade behaviour

  g.insert_levels = {2};
  assert(!do_insert(g, 1, 1));

  // The new element's insert level (2) exceeds the prior entry level (0),
  // so a new entry point was required.
  assert(g.upgrade_calls == 1);
  assert(g.entry_level == LevelId(2));
  assert(g.present[2].count(1) == 1);
  assert(g.present[1].count(1) == 1);
  assert(g.present[0].count(1) == 1);
  assert(g.entry_points.front().id == 1);
  // The registered entry point must itself carry the true top level (2),
  // not do_insert()'s placeholder level-0 Node argument -- otherwise a
  // later get_entry_point() caller would look up this element's adjacency
  // at the wrong level. See graph_ops_impl.h: insert() captures the Node
  // produced by the topmost (parent-less) create_node() call into a
  // dedicated top_node local specifically so it survives to
  // set_entry_point() with that level intact.
  assert(g.entry_points.front().level == LevelId(2));
}

void test_insert_upgrade_recheck_suppresses_unneeded_entry_point_update() {
  MockGraph g;
  // Simulate a concurrent insert that has *already* bootstrapped the entry
  // point by the time this insert's S->X upgrade completes: the first
  // get_entry_point() call (before the upgrade) reports an empty graph,
  // but the real state -- visible on the second, post-upgrade call -- is
  // already non-empty at a level this insert doesn't exceed.
  add_element(g, 999, 999);
  g.entry_points = {Node{LevelId(0), 999}};
  g.entry_level = LevelId(0);
  g.get_entry_point_first_call_empty = true;

  assert(!do_insert(g, 1, 1));

  // The upgrade still happened (the stale first read looked like an empty
  // graph), but the recheck must have found new_entry_point no longer
  // necessary, per the comment at graph_ops_impl.h:77-80 -- so the entry
  // point recorded by the *other* insert must survive untouched.
  assert(g.upgrade_calls == 1);
  assert(g.entry_points.front().id == 999);
  assert(g.entry_level == LevelId(0));
}

void test_insert_reciprocal_links_after_all_levels_created() {
  MockGraph g;
  g.insert_levels = {1};
  assert(!do_insert(g, 0, 0));
  // Isolate the second insert's own call sequence from the first's.
  g.call_log.clear();

  g.insert_levels = {1};
  assert(!do_insert(g, 1, 1));

  // Every "create"/"link" entry (top-down record creation and forward
  // linking) must precede every "link_back" entry (bottom-up reciprocal
  // linking) -- see the comment at graph_ops_impl.h:137-140: any node
  // reachable at level N must already be fully inserted at all lower
  // levels before back-links are added.
  size_t last_forward = 0, first_back = g.call_log.size();
  for (size_t i = 0; i < g.call_log.size(); ++i) {
    const std::string &entry = g.call_log[i];
    if (entry.rfind("create:", 0) == 0 || entry.rfind("link:", 0) == 0) {
      last_forward = i;
    } else if (entry.rfind("link_back:", 0) == 0) {
      first_back = std::min(first_back, i);
    }
  }
  assert(first_back > last_forward);
}

void test_insert_get_entry_point_failure_propagates() {
  MockGraph g;
  g.fail_get_entry_point = true;
  assert(do_insert(g, 0, 0));
}

void test_insert_create_node_failure_propagates() {
  MockGraph g;
  g.fail_create_node = true;
  assert(do_insert(g, 0, 0));
}

void test_insert_link_neighbours_failure_propagates() {
  MockGraph g;
  assert(!do_insert(g, 0, 0));
  g.fail_link_neighbours = true;
  assert(do_insert(g, 1, 1));
}

void test_insert_shrinks_overflowed_neighbours_past_mmax() {
  MockGraph g;
  assert(!do_insert(g, 0, 0));
  assert(!do_insert(g, 1, 1));
  // Cap every node at a single neighbour, so inserting a third element
  // forces both 0 and 1 to choose between their existing edge and the new
  // one -- neither has a free slot to simply keep both.
  g.m_Mmax = 1;
  g.call_log.clear();

  assert(!do_insert(g, 2, 3));

  using GraphOps = svector::hnsw::GraphOperations<MockGraph>;

  // 2's own neighbour selection: candidates are 1 (distance 2) and 0
  // (distance 3). 1 is nearest and always kept. 0 is dominated -- closer to
  // 1 (distance 1) than to the query -- so whether it survives depends on
  // GraphOps::KEEP_PRUNED_CONNECTIONS: Yes backfills it in anyway, No
  // discards it outright. Branching on the same constant GraphOperations
  // itself uses means this test keeps working whichever way that default is
  // set, without needing hand-editing every time it's revisited.
  //
  // 2's own outgoing edges are unaffected either way: shrinking only touches
  // the *neighbour's* list (graph_ops_impl.h's shrink_neighbours()), never
  // the newly inserted node's own list.
  if (GraphOps::KEEP_PRUNED_CONNECTIONS ==
      GraphOps::KeepPrunedConnections::Yes) {
    assert((g.adjacency[0][2] == std::vector<int>{1, 0}));
  } else {
    assert((g.adjacency[0][2] == std::vector<int>{1}));
  }

  // 1's reciprocal link back to 2 would push it over Mmax=1, so
  // shrink_neighbours() reselects from 1's existing connection (0) plus the
  // tentative one (2): 0 is nearer to 1 (distance 1) than 2 is (distance 2),
  // so 0 wins regardless of keep_pruned_connections -- 1 ends up back on its
  // original edge.
  assert((g.adjacency[0][1] == std::vector<int>{0}));

  // 0 only gets a reciprocal-link/shrink attempt at all if 2 selected it as
  // a neighbour in the first place, which is exactly the branch above.
  // Either way 0 keeps its original edge to 1: with KEEP_PRUNED_CONNECTIONS
  // == Yes, 0's own shrink (candidates {1, 2}, Mmax=1) keeps nearer 1 over
  // 2; with No, 0 is never a candidate for anything and is simply
  // untouched.
  assert((g.adjacency[0][0] == std::vector<int>{1}));

  // Exactly as many neighbours as 2 ended up linking to (1, or 1 and 0) go
  // through the shrink-and-replace path.
  int replace_count = 0;
  for (const std::string &entry : g.call_log) {
    if (entry.rfind("replace:", 0) == 0) {
      ++replace_count;
    }
  }
  int expected_replace_count =
      GraphOps::KEEP_PRUNED_CONNECTIONS == GraphOps::KeepPrunedConnections::Yes
          ? 2
          : 1;
  assert(replace_count == expected_replace_count);
}

void test_insert_replace_neighbours_failure_propagates() {
  MockGraph g;
  assert(!do_insert(g, 0, 0));
  assert(!do_insert(g, 1, 1));
  g.m_Mmax = 1;
  g.fail_replace_neighbours = true;
  assert(do_insert(g, 2, 3));
}

void test_insert_set_entry_point_failure_propagates() {
  MockGraph g;
  g.fail_set_entry_point = true;
  // First-ever insert always needs a new entry point.
  assert(do_insert(g, 0, 0));
}

void test_insert_get_next_level_failure_propagates() {
  MockGraph g;
  g.insert_levels = {1};
  assert(!do_insert(g, 0, 0));

  // This insert's level (0) is below the current entry level (1), so
  // greedy descent runs and calls advance_to_next_level() -> get_next_level().
  g.fail_get_next_level = true;
  assert(do_insert(g, 1, 1));
}

// insert()/search_knn() round trip: the graph_ops.h::create_node() contract
// says each level's record is seeded from new_node's identity/data, not
// invented separately -- this checks that data actually survives a real
// insert and comes back out through search_knn(), not just through the
// in-process Node objects insert() happens to pass around internally.
void test_insert_then_search_returns_inserted_data() {
  MockGraph g;
  // data is deliberately distinguishable from id/position, so a bug that
  // returns the wrong node, or the right node with stale/default data,
  // is caught independently of id.
  for (int id = 0; id < 10; ++id) {
    assert(!do_insert(g, id, id, id * 100));
  }
  svector::hnsw::GraphOperations<MockGraph> ops(g);

  std::vector<Node> result;
  assert(!ops.search_knn(NodeData{5}, 3, 5, result));

  std::sort(result.begin(), result.end(),
            [](const Node &a, const Node &b) { return a.id < b.id; });
  assert((values(result) == std::vector<int>{4, 5, 6}));
  for (const Node &n : result) {
    assert(g.node_data.at(n.id) == n.id * 100);
  }
}

// ================================ remove =================================

void test_remove_severs_edges_without_orphaning_safe_neighbours() {
  MockGraph g;
  // 0-1-2-3-4 line, plus leaf 5 hanging off of 2 (5's only edge).
  for (int id = 0; id <= 4; ++id) {
    add_element(g, id, id);
    add_presence(g, id, 0);
  }
  add_edge(g, 0, 0, 1);
  add_edge(g, 0, 1, 2);
  add_edge(g, 0, 2, 3);
  add_edge(g, 0, 3, 4);
  add_element(g, 5, 1); // co-located with 1, so it's the unique nearest
                        // survivor once 2 (position 2) is gone.
  add_presence(g, 5, 0);
  add_edge(g, 0, 2, 5);
  g.entry_points = {Node{LevelId(0), 0}};
  g.entry_level = LevelId(0);
  g.m_M = 1;
  svector::hnsw::GraphOperations<MockGraph> ops(g);

  assert(!ops.remove(Node{LevelId(0), 2}, LevelId(0)));

  // 2 is fully gone.
  assert(g.present[0].count(2) == 0);
  assert(g.adjacency[0].count(2) == 0 || g.adjacency[0][2].empty());
  // 3 had another edge (to 4), so it was severed from 2 without needing
  // repair -- 2 simply disappears from its adjacency.
  assert((g.adjacency[0][3] == std::vector<int>{4}));
  // 5's only edge was to 2, so it must have been repaired: linked to its
  // nearest surviving neighbour (1). 1 keeps its original edge to 0 and
  // gains the reciprocal of that repair.
  assert((g.adjacency[0][5] == std::vector<int>{1}));
  assert((g.adjacency[0][1] == std::vector<int>{0, 5}));
  // 2 wasn't the entry point, so no upgrade was needed.
  assert(g.upgrade_calls == 0);
  assert(g.entry_points.front().id == 0);
}

void test_remove_sole_entry_point_selects_survivor_as_replacement() {
  MockGraph g;
  add_element(g, 0, 0);
  add_presence(g, 0, 0);
  add_element(g, 1, 1);
  add_presence(g, 1, 0);
  add_edge(g, 0, 0, 1);
  g.entry_points = {Node{LevelId(0), 0}};
  g.entry_level = LevelId(0);
  svector::hnsw::GraphOperations<MockGraph> ops(g);

  assert(!ops.remove(Node{LevelId(0), 0}, LevelId(0)));

  // 1 was 0's only neighbour and had no other edge, so it's the orphan
  // that "seeds" the survivor pool (graph_ops_impl.h:266-274) rather than
  // going through a real repair search -- and it becomes the sole entry
  // point's replacement.
  assert(g.present[0].count(0) == 0);
  assert(g.adjacency[0][1].empty());
  assert(g.upgrade_calls == 1);
  assert(g.entry_points.size() == 1);
  assert(g.entry_points.front().id == 1);
  assert(g.entry_level == LevelId(0));
}

void test_remove_only_element_clears_entry_point() {
  MockGraph g;
  add_element(g, 0, 0);
  add_presence(g, 0, 0);
  g.entry_points = {Node{LevelId(0), 0}};
  g.entry_level = LevelId(0);
  svector::hnsw::GraphOperations<MockGraph> ops(g);

  assert(!ops.remove(Node{LevelId(0), 0}, LevelId(0)));

  // Nothing survives node's removal at any level, so there's no
  // replacement (graph_ops_impl.h:346-352): the entry point is cleared.
  assert(g.entry_points.empty());
  assert(g.entry_level == LevelId(0));
  assert(g.upgrade_calls == 1);
}

void test_remove_non_sole_entry_point_needs_no_upgrade() {
  MockGraph g;
  add_element(g, 0, 0);
  add_presence(g, 0, 0);
  add_element(g, 1, 1);
  add_presence(g, 1, 0);
  add_edge(g, 0, 0, 1);
  // Two entry points at the same level: removing one of them still leaves
  // the other valid, so no replacement should be sought.
  g.entry_points = {Node{LevelId(0), 0}, Node{LevelId(0), 1}};
  g.entry_level = LevelId(0);
  svector::hnsw::GraphOperations<MockGraph> ops(g);

  assert(!ops.remove(Node{LevelId(0), 0}, LevelId(0)));
  assert(g.upgrade_calls == 0);
}

void test_remove_get_entry_point_failure_propagates() {
  MockGraph g;
  g.fail_get_entry_point = true;
  svector::hnsw::GraphOperations<MockGraph> ops(g);
  assert(ops.remove(Node{LevelId(0), 0}, LevelId(0)));
}

void test_remove_neighbours_failure_propagates() {
  MockGraph g;
  add_element(g, 0, 0);
  add_presence(g, 0, 0);
  g.entry_points = {Node{LevelId(0), 0}};
  g.entry_level = LevelId(0);
  g.fail_neighbours = true;
  svector::hnsw::GraphOperations<MockGraph> ops(g);
  assert(ops.remove(Node{LevelId(0), 0}, LevelId(0)));
}

void test_remove_unlink_neighbours_failure_propagates() {
  MockGraph g;
  add_element(g, 0, 0);
  add_presence(g, 0, 0);
  add_element(g, 1, 1);
  add_presence(g, 1, 0);
  add_edge(g, 0, 0, 1);
  g.entry_points = {Node{LevelId(0), 5}}; // not node 0, keeps this focused
  g.entry_level = LevelId(0);
  g.fail_unlink_neighbours = true;
  svector::hnsw::GraphOperations<MockGraph> ops(g);
  assert(ops.remove(Node{LevelId(0), 0}, LevelId(0)));
}

void test_remove_drop_node_failure_propagates() {
  MockGraph g;
  add_element(g, 0, 0);
  add_presence(g, 0, 0);
  g.entry_points = {Node{LevelId(0), 5}};
  g.entry_level = LevelId(0);
  g.fail_drop_node = true;
  svector::hnsw::GraphOperations<MockGraph> ops(g);
  assert(ops.remove(Node{LevelId(0), 0}, LevelId(0)));
}

void test_remove_set_entry_point_failure_propagates() {
  MockGraph g;
  add_element(g, 0, 0);
  add_presence(g, 0, 0);
  g.entry_points = {Node{LevelId(0), 0}};
  g.entry_level = LevelId(0);
  g.fail_set_entry_point = true;
  svector::hnsw::GraphOperations<MockGraph> ops(g);
  // Sole entry point, no survivors -> new_entry_point path -> set_entry_point.
  assert(ops.remove(Node{LevelId(0), 0}, LevelId(0)));
}

void test_remove_get_next_level_failure_propagates() {
  MockGraph g;
  // Node 7 exists at levels 1 and 0, so removing it descends through its
  // own level chain via get_next_level() (graph_ops_impl.h:313-317).
  add_element(g, 7, 7);
  add_presence(g, 7, 1);
  add_presence(g, 7, 0);
  g.entry_points = {Node{LevelId(0), 99}};
  g.entry_level = LevelId(0);
  g.fail_get_next_level = true;
  svector::hnsw::GraphOperations<MockGraph> ops(g);
  assert(ops.remove(Node{LevelId(1), 7}, LevelId(1)));
}

} // namespace

int main() {
  test_search_knn_empty_graph();
  test_search_knn_single_level();
  test_search_knn_k_truncates_ef_search_results();
  test_search_knn_multi_level_descent();
  test_search_knn_visibility_filters_only_bottom_layer();
  test_search_knn_distance_failure_propagates();
  test_search_knn_get_entry_point_failure_propagates();
  test_search_knn_get_next_level_failure_propagates();

  test_insert_bootstraps_entry_point_on_empty_graph();
  test_insert_links_nearest_neighbour_without_unnecessary_upgrade();
  test_insert_new_top_level_creates_records_at_every_level();
  test_insert_upgrade_recheck_suppresses_unneeded_entry_point_update();
  test_insert_reciprocal_links_after_all_levels_created();
  test_insert_get_entry_point_failure_propagates();
  test_insert_create_node_failure_propagates();
  test_insert_link_neighbours_failure_propagates();
  test_insert_shrinks_overflowed_neighbours_past_mmax();
  test_insert_replace_neighbours_failure_propagates();
  test_insert_set_entry_point_failure_propagates();
  test_insert_get_next_level_failure_propagates();
  test_insert_then_search_returns_inserted_data();

  test_remove_severs_edges_without_orphaning_safe_neighbours();
  test_remove_sole_entry_point_selects_survivor_as_replacement();
  test_remove_only_element_clears_entry_point();
  test_remove_non_sole_entry_point_needs_no_upgrade();
  test_remove_get_entry_point_failure_propagates();
  test_remove_neighbours_failure_propagates();
  test_remove_unlink_neighbours_failure_propagates();
  test_remove_drop_node_failure_propagates();
  test_remove_set_entry_point_failure_propagates();
  test_remove_get_next_level_failure_propagates();

  std::printf("All graph_ops tests passed.\n");
  return 0;
}
