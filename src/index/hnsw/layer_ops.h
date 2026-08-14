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

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_LAYER_OPS_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_LAYER_OPS_H

#include <algorithm>
#include <cstdint>
#include <queue>
#include <unordered_set>
#include <variant>
#include <vector>

namespace svector::hnsw {

namespace detail {

// std::priority_queue with clear() added: empties the queue while keeping
// the underlying container's capacity, unlike replacing the queue with a
// freshly-constructed one, which drops the capacity and forces the next
// search to reallocate.
template <typename... Args>
class PriorityQueue : public std::priority_queue<Args...> {
public:
  void clear() { c.clear(); }

private:
  using std::priority_queue<Args...>::c;
};

} // namespace detail

// Requirements:
//
// Graph must provide:
//   using Node = ...;
//   using NodeData = ...;
//   using DistanceType = ...;  // DefaultConstructible and LessThanComparable
//
//   // Computes the distance between two nodes and stores the result in 'out'.
//   bool distance(const Node&, const Node&, DistanceType& out);
//
//   // Computes the distance between the query and a node and stores the
//   // result in 'out'.
//   bool distance(const NodeData&, const Node&, DistanceType& out);
//
//   // Replaces the contents of 'out' with the neighbours of the node.
//   bool neighbours(const Node&, std::vector<Node>& out);
//
// Graph::Node must provide:
//   using KeyType = ...;  // Hashable and equality comparable.
//   KeyType key() const;
//
// Policy<Graph> must provide:
//   using Node = typename Graph::Node;
//   // Sets 'out' to whether the node is eligible for inclusion in the
//   // result set. Nodes that are not eligible are still traversed so the
//   // search can reach eligible nodes beyond them, but they never appear
//   // in search_layer()'s output or count against ef. The bool return
//   // value follows the same convention as Graph's methods above: it
//   // signals failure of the visibility check itself, not the node's
//   // visibility -- that result is carried in 'out'.
//   static bool is_visible(Graph&, const Node&, bool& out);
template <typename Graph, template <typename> class Policy>
class LayerOperations {
public:
  using PolicyType = Policy<Graph>;
  using Node = typename Graph::Node;
  using NodeData = typename Graph::NodeData;
  enum class ExtendCandidates { No, Yes };
  enum class KeepPrunedConnections { No, Yes };

  // Creates a search context for a single query node. The query may be a
  // NodeData not yet present in the graph (e.g. the vector being inserted
  // or searched for) or an existing Node already stored in the graph. The
  // object may be reused to perform multiple searches for the same query
  // by calling search()/seed() repeatedly.
  LayerOperations(Graph &graph, const NodeData &query);
  LayerOperations(Graph &graph, const Node &query);

  // LayerOperations follows a two-step protocol. Step 1 -- search() or
  // seed() -- gathers a result set and puts the object in the Consume
  // state; step 2 -- exactly one of consume_all(), consume_simple() or
  // consume_heuristic() -- extracts it and puts the object back in the
  // Init state.

  // Step 1: executes the search from the entry points (Algorithm 2,
  // SEARCH-LAYER). At most ef visible candidates are retained during the
  // search; invisible candidates are still traversed but never retained,
  // so they don't count against ef or appear in the result. Returns true
  // if a graph operation fails.
  bool search(const std::vector<Node> &entry_points, uint32_t ef);

  // Step 1, traversal-free variant: evaluates the distance from the query
  // to each element of candidates directly (Algorithm 2, lines 1-3,
  // without the expansion loop), for callers that already know the exact
  // set to select from and don't need search()'s neighbourhood traversal
  // (e.g. shrink_neighbours() re-selecting among a node's existing
  // neighbours plus one new one). Returns true if a graph operation
  // fails.
  bool seed(const std::vector<Node> &candidates);

  // Step 2, variant A: extracts every node found by the preceding
  // search()/seed() call, ordered by increasing distance from the query
  // node (nearest first).
  void consume_all(std::vector<Node> &out);

  // Step 2, variant B: extracts the M elements of the preceding
  // search()/seed() call's result nearest to the query node, ordered by
  // increasing distance (Algorithm 3, SELECT-NEIGHBORS-SIMPLE).
  void consume_simple(uint32_t M, std::vector<Node> &out);

  // Step 2, variant C: selects up to M elements of the preceding
  // search()/seed() call's result via the heuristic (Algorithm 4,
  // SELECT-NEIGHBORS-HEURISTIC): elements are taken nearest-first
  // (optionally extended with their neighbours), and admitted only if
  // they are closer to the query node than to every element already
  // selected -- this favours diversity over picking the M nearest
  // overall. If keep_pruned_connections is set, remaining slots are
  // backfilled from the elements that were rejected by that check.
  // Returns true if a graph operation fails, in which case out is left
  // unchanged. If candidate_pool is non-null, it's filled with the full
  // preceding search()/seed() result (i.e. what consume_all() would have
  // produced), regardless of extend_candidates -- for callers that, like
  // Algorithm 1's "ep <- W" step, need that alongside the narrowed
  // result and would otherwise have to re-seed() it at the cost of
  // recomputing its distances.
  bool consume_heuristic(uint32_t M, ExtendCandidates extend_candidates,
                         KeepPrunedConnections keep_pruned_connections,
                         std::vector<Node> &out,
                         std::vector<Node> *candidate_pool = nullptr);

  // Resets the search state, allowing the object to be reused for another
  // search with the current query node kept. Puts the object back in the
  // Init state.
  void reset();

  // Resets the search state and rebinds the object to a new query node.
  void reset(const NodeData &query);
  void reset(const Node &query);

private:
  // Init: no result is pending consumption, either because no search()/
  // seed() call has been made yet or because the last one's result was
  // already consumed. Consume: search()/seed() has populated a result
  // pending exactly one consume_*() call.
  enum class State { Init, Consume };

  using Distance = typename Graph::DistanceType;

  struct Candidate {
    Node node;
    Distance distance{};

    Candidate(Node n) : node(std::move(n)) {}

    bool operator<(const Candidate &rhs) const {
      return distance < rhs.distance;
    }

    struct MinComparator {
      bool operator()(const Candidate &lhs, const Candidate &rhs) const {
        return rhs < lhs;
      }
    };

    struct MaxComparator {
      bool operator()(const Candidate &lhs, const Candidate &rhs) const {
        return lhs < rhs;
      }
    };
  };

  using MinQueue = detail::PriorityQueue<Candidate, std::vector<Candidate>,
                                         typename Candidate::MinComparator>;

  using MaxQueue = detail::PriorityQueue<Candidate, std::vector<Candidate>,
                                         typename Candidate::MaxComparator>;

  // Moves the search result into out, consuming it in the process.
  // The output vector is replaced with the search result, ordered by
  // increasing distance from the query node (nearest first). Leaves
  // m_candidates and m_visited as-is; the next search()/seed() call is
  // responsible for discarding them via reset().
  void consume_result(std::vector<Node> &out);

  // Fills in the distance for each candidate from index begin onward.
  // Elements before begin are assumed to already carry a valid distance
  // (e.g. copied over from a prior search()/seed() result) and are left
  // untouched. Returns true if a graph operation fails.
  bool evaluate_distances(std::vector<Candidate> &candidates, size_t begin = 0);

  // Seeds m_visited, m_candidates and m_results from the entry points
  // (Algorithm 2, lines 1-3). Returns true if a graph operation fails.
  // The public search() and seed() both wrap this, after their own reset().
  bool seed_impl(const std::vector<Node> &entry_points);

  // Expands node's neighbourhood into m_candidates and m_results
  // (Algorithm 2, lines 9-17). Returns true if a graph operation fails.
  bool expand(const Node &node, uint32_t ef);

  // Extends m_expand_buf, holding W, with each of its candidates'
  // neighbours (Algorithm 4, lines 3-7), evaluating distances only for the
  // newly-added ones -- W's own are already known from the preceding
  // search()/seed() call. Only called for ExtendCandidates::Yes. Returns
  // true if a graph operation fails.
  bool extend_with_neighbours();

  // Returns true if a graph operation fails. On success, sets dominated
  // to whether e is closer to some element of result than to the query
  // node -- i.e. whether some element of result already "covers" e
  // (Algorithm 4, line 11, negated).
  bool is_dominated(const Candidate &e, const std::vector<Node> &result,
                    bool &dominated);

  Graph &m_graph;
  std::variant<Node, NodeData> m_query;
  State m_state = State::Init;

  std::unordered_set<typename Node::KeyType> m_visited;
  MinQueue m_candidates;
  MaxQueue m_results;

  // Scratch buffers reused across calls, so that repeated operations
  // don't reallocate their backing storage. Cleared by reset(); methods
  // that repurpose them for a second role within a single call (e.g.
  // consume_heuristic()) clear them again as needed.
  std::vector<Node> m_neighbour_buf;
  std::vector<Candidate> m_expand_buf;
};

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_LAYER_OPS_H
