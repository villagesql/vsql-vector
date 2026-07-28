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
//   using DistanceType = ...;  // DefaultConstructible and LessThanComparable
//
//   // Computes the distance between two nodes and stores the result in 'out'.
//   bool distance(const Node&, const Node&, DistanceType& out);
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
  enum class ExtendCandidates { No, Yes };
  enum class KeepPrunedConnections { No, Yes };

  // Creates a search context for a single query node.
  // The object may be reused to perform multiple searches for the same
  // query by calling search_layer() repeatedly.
  LayerOperations(Graph &graph, const Node &query);

  // Executes the search from the entry points in candidates (Algorithm 2,
  // SEARCH-LAYER), discarding any state left over from a previous call
  // first. At most ef visible candidates are retained during the search;
  // invisible candidates are still traversed but never retained, so they
  // don't count against ef or appear in the result. Returns true if a
  // graph operation fails, in which case candidates is left unchanged. On
  // success, candidates is replaced with the search result, ordered by
  // increasing distance from the query node (nearest first).
  bool search_layer(std::vector<Node> &candidates, uint32_t ef);

  // Returns the M elements of candidates nearest to the query node,
  // ordered by increasing distance (Algorithm 3,
  // SELECT-NEIGHBORS-SIMPLE). Discards any state left over from a
  // previous call first. Returns true if a graph operation fails.
  bool select_neighbours_simple(const std::vector<Node> &candidates, uint32_t M,
                                std::vector<Node> &out);

  // Returns up to M elements of candidates selected by the heuristic
  // (Algorithm 4, SELECT-NEIGHBORS-HEURISTIC): elements are taken
  // nearest-first from candidates (optionally extended with their
  // neighbours), and admitted only if they are closer to the query node
  // than to every element already selected -- this favours diversity over
  // picking the M nearest overall. If keep_pruned_connections is set,
  // remaining slots are backfilled from the elements that were rejected
  // by that check. Discards any state left over from a previous call
  // first. Returns true if a graph operation fails, in which case out is
  // left unchanged.
  bool
  select_neighbours_heuristic(const std::vector<Node> &candidates, uint32_t M,
                              ExtendCandidates extend_candidates,
                              KeepPrunedConnections keep_pruned_connections,
                              std::vector<Node> &out);

  // Resets the search state, allowing the object to be reused for
  // another search. If query is non-null, the object is rebound to that
  // query node; otherwise the current query node is kept.
  void reset(const Node *query = nullptr);

private:
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
  // m_candidates and m_visited as-is; the next public call is
  // responsible for discarding them via reset().
  void consume_result(std::vector<Node> &out);

  // Fills in the distance for each candidate. Returns true if a graph
  // operation fails.
  bool evaluate_distances(std::vector<Candidate> &candidates);

  // Seeds m_visited, m_candidates and m_results from the entry points
  // (Algorithm 2, lines 1-3). Returns true if a graph operation fails.
  bool seed(const std::vector<Node> &entry_points);

  // Expands node's neighbourhood into m_candidates and m_results
  // (Algorithm 2, lines 9-17). Returns true if a graph operation fails.
  bool expand(const Node &node, uint32_t ef);

  // Fills m_expand_buf with the deduplicated candidates, extended with
  // their neighbours if extend_candidates is set, and evaluates each
  // element's distance to the query node (Algorithm 4, lines 2-8).
  // Returns true if a graph operation fails.
  bool gather_candidates(const std::vector<Node> &candidates,
                         ExtendCandidates extend_candidates);

  // Returns true if a graph operation fails. On success, sets dominated
  // to whether e is closer to some element of result than to the query
  // node -- i.e. whether some element of result already "covers" e
  // (Algorithm 4, line 11, negated).
  bool is_dominated(const Candidate &e, const std::vector<Node> &result,
                    bool &dominated);

  Graph &m_graph;
  Node m_query;

  std::unordered_set<typename Node::KeyType> m_visited;
  MinQueue m_candidates;
  MaxQueue m_results;

  // Scratch buffers reused across calls, so that repeated operations
  // don't reallocate their backing storage. Cleared by reset(); methods
  // that repurpose them for a second role within a single call (e.g.
  // select_neighbours_heuristic()) clear them again as needed.
  std::vector<Node> m_neighbour_buf;
  std::vector<Candidate> m_expand_buf;
};

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_LAYER_OPS_H
