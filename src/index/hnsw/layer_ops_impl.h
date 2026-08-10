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

// Out-of-line template definitions for LayerOperations, declared in
// layer_ops.h. Split out into this header (rather than folded into
// layer_ops.h directly) so a caller can explicitly instantiate
// LayerOperations for its own Graph/Policy without also linking against a
// production Graph type -- see layer_ops.cc for the IndexGraph
// instantiation, and unittest/layer_ops_test.cc for the standalone-mock
// instantiation.

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_LAYER_OPS_IMPL_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_LAYER_OPS_IMPL_H

#include "layer_ops.h"

namespace svector::hnsw {

template <typename Graph, template <typename> class Policy>
LayerOperations<Graph, Policy>::LayerOperations(Graph &graph, const Node &query)
    : m_graph(graph), m_query(query) {}

template <typename Graph, template <typename> class Policy>
bool LayerOperations<Graph, Policy>::search_layer(std::vector<Node> &candidates,
                                                  uint32_t ef) {
  reset();

  if (seed(candidates)) {
    return true;
  }

  // Algorithm 2, lines 4-8: repeatedly expand the nearest unexpanded
  // candidate until the closest remaining candidate is further than the
  // furthest element already found.
  while (!m_candidates.empty()) {
    Candidate c = m_candidates.top();
    m_candidates.pop();
    // Algorithm 2, line 7: distance(c, q) > distance(f, q). m_results can
    // be empty here if every candidate seen so far was invisible; in that
    // case there's no result yet to compare against, so keep expanding.
    if (!m_results.empty() && m_results.top() < c) {
      break;
    }
    if (expand(c.node, ef)) {
      return true;
    }
  }

  consume_result(candidates);
  return false;
}

template <typename Graph, template <typename> class Policy>
void LayerOperations<Graph, Policy>::consume_result(std::vector<Node> &out) {
  out.clear();
  if (!m_results.empty()) {
    // m_results is a max-heap (largest distance first). Fill out
    // back-to-front as we pop, so the final order is ascending by distance
    // without a separate reverse pass.
    out.resize(m_results.size());
    for (auto i = out.size(); i-- > 0;) {
      out[i] = m_results.top().node;
      m_results.pop();
    }
  }
}

template <typename Graph, template <typename> class Policy>
void LayerOperations<Graph, Policy>::reset(const Node *query) {
  if (query != nullptr) {
    m_query = *query;
  }

  m_visited.clear();
  m_candidates.clear();
  m_results.clear();

  m_expand_buf.clear();
  m_neighbour_buf.clear();
}

template <typename Graph, template <typename> class Policy>
bool LayerOperations<Graph, Policy>::select_neighbours_simple(
    const std::vector<Node> &candidates, uint32_t M, std::vector<Node> &out) {
  reset();

  // Algorithm 3, SELECT-NEIGHBORS-SIMPLE: return the M elements of
  // candidates nearest to the query node.
  m_expand_buf.reserve(candidates.size());
  for (const Node &node : candidates) {
    m_expand_buf.emplace_back(node);
  }

  if (evaluate_distances(m_expand_buf)) {
    return true;
  }

  const size_t count = std::min<size_t>(M, m_expand_buf.size());
  std::partial_sort(m_expand_buf.begin(), m_expand_buf.begin() + count,
                    m_expand_buf.end());

  out.clear();
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    out.push_back(std::move(m_expand_buf[i].node));
  }
  return false;
}

template <typename Graph, template <typename> class Policy>
bool LayerOperations<Graph, Policy>::select_neighbours_heuristic(
    const std::vector<Node> &candidates, uint32_t M,
    ExtendCandidates extend_candidates,
    KeepPrunedConnections keep_pruned_connections, std::vector<Node> &out) {
  reset();

  // Algorithm 4, lines 2-8: W <- C, optionally extended with neighbours.
  if (gather_candidates(candidates, extend_candidates)) {
    return true;
  }
  for (const Candidate &candidate : m_expand_buf) {
    m_candidates.push(candidate);
  }

  // R is accumulated into m_neighbour_buf rather than out directly, so
  // that out is left unchanged if a later graph operation fails.
  m_neighbour_buf.clear();

  // Wd reuses m_expand_buf now that its contents have been copied into
  // the m_candidates heap above. Discarded candidates land in it in
  // ascending-distance order for free, since they're rejected in the
  // same nearest-first order they're popped off W.
  m_expand_buf.clear();

  // Algorithm 4, lines 9-14.
  while (!m_candidates.empty() && m_neighbour_buf.size() < M) {
    Candidate e = m_candidates.top();
    m_candidates.pop();

    bool dominated = false;
    if (is_dominated(e, m_neighbour_buf, dominated)) {
      return true;
    }

    if (dominated) {
      m_expand_buf.push_back(std::move(e));
    } else {
      m_neighbour_buf.push_back(std::move(e.node));
    }
  }

  // Algorithm 4, lines 15-17: backfill from the discarded candidates,
  // nearest-first.
  if (keep_pruned_connections == KeepPrunedConnections::Yes) {
    for (size_t i = 0; i < m_expand_buf.size() && m_neighbour_buf.size() < M;
         ++i) {
      m_neighbour_buf.push_back(std::move(m_expand_buf[i].node));
    }
  }

  out.clear();
  out.reserve(m_neighbour_buf.size());
  for (Node &node : m_neighbour_buf) {
    out.push_back(std::move(node));
  }
  return false;
}

template <typename Graph, template <typename> class Policy>
bool LayerOperations<Graph, Policy>::gather_candidates(
    const std::vector<Node> &candidates, ExtendCandidates extend_candidates) {
  // Algorithm 4, line 2: W <- C.
  for (const Node &node : candidates) {
    if (m_visited.insert(node.key()).second) {
      m_expand_buf.emplace_back(node);
    }
  }

  // Algorithm 4, lines 3-7: extend W with each candidate's neighbours.
  if (extend_candidates == ExtendCandidates::Yes) {
    for (const Node &node : candidates) {
      if (m_graph.neighbours(node, m_neighbour_buf)) {
        return true;
      }
      for (const Node &neighbour : m_neighbour_buf) {
        if (m_visited.insert(neighbour.key()).second) {
          m_expand_buf.emplace_back(neighbour);
        }
      }
    }
  }

  return evaluate_distances(m_expand_buf);
}

template <typename Graph, template <typename> class Policy>
bool LayerOperations<Graph, Policy>::is_dominated(
    const Candidate &e, const std::vector<Node> &result, bool &dominated) {
  // Algorithm 4, line 11 (negated): does some element of result already
  // cover e, i.e. is it closer to e than the query node is?
  for (const Node &r : result) {
    Distance dist_er{};
    if (m_graph.distance(e.node, r, dist_er)) {
      return true;
    }
    if (dist_er < e.distance) {
      dominated = true;
      return false;
    }
  }
  dominated = false;
  return false;
}

template <typename Graph, template <typename> class Policy>
bool LayerOperations<Graph, Policy>::evaluate_distances(
    std::vector<Candidate> &candidates) {
  // Distance computations are independent and could be parallelized
  // if Graph::distance() is thread-safe.
  for (Candidate &candidate : candidates) {
    if (m_graph.distance(m_query, candidate.node, candidate.distance)) {
      return true;
    }
  }
  return false;
}

template <typename Graph, template <typename> class Policy>
bool LayerOperations<Graph, Policy>::seed(
    const std::vector<Node> &entry_points) {
  // Algorithm 2, lines 1-3: v = C = W = ep.
  for (const Node &node : entry_points) {
    if (m_visited.insert(node.key()).second) {
      m_expand_buf.emplace_back(node);
    }
  }

  if (evaluate_distances(m_expand_buf)) {
    return true;
  }

  for (const Candidate &candidate : m_expand_buf) {
    m_candidates.push(candidate);

    // Invisible entry points are still traversed from, but must not seed
    // the result set.
    bool visible = false;
    if (PolicyType::is_visible(m_graph, candidate.node, visible)) {
      return true;
    }
    if (visible) {
      m_results.push(candidate);
    }
  }
  return false;
}

template <typename Graph, template <typename> class Policy>
bool LayerOperations<Graph, Policy>::expand(const Node &node, uint32_t ef) {
  if (m_graph.neighbours(node, m_neighbour_buf)) {
    return true;
  }

  // Algorithm 2, lines 10-11: v = v U e, for each unvisited neighbour.
  m_expand_buf.clear();
  for (const Node &neighbour : m_neighbour_buf) {
    if (m_visited.insert(neighbour.key()).second) {
      m_expand_buf.emplace_back(neighbour);
    }
  }

  if (evaluate_distances(m_expand_buf)) {
    return true;
  }

  for (const Candidate &candidate : m_expand_buf) {
    // Algorithm 2, line 13: distance(e, q) < distance(f, q) or |W| < ef.
    if (m_results.size() < ef || candidate < m_results.top()) {
      m_candidates.push(candidate);

      // Invisible candidates are still traversed so the search can reach
      // visible nodes beyond them, but must not enter the result set or
      // count against ef.
      bool visible = false;
      if (Policy<Graph>::is_visible(m_graph, candidate.node, visible)) {
        return true;
      }
      if (visible) {
        m_results.push(candidate);
        if (m_results.size() > ef) {
          m_results.pop();
        }
      }
    }
  }
  return false;
}

} // namespace svector::hnsw

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_LAYER_OPS_IMPL_H
