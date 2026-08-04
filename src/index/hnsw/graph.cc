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

#include "graph.h"

#include <cmath>
#include <random>

namespace svector::hnsw {

bool IndexGraph::distance(const Node & /*a*/, const Node & /*b*/,
                          DistanceType &out) {
  out = 0.0;
  return false;
}

bool IndexGraph::distance(const NodeData & /*a*/, const Node & /*b*/,
                          DistanceType &out) {
  out = 0.0;
  return false;
}

bool IndexGraph::neighbours(const Node & /*node*/, std::vector<Node> &out) {
  out.clear();
  return false;
}

bool IndexGraph::visible(const Node & /*node*/, bool &out) {
  out = true;
  return false;
}

IndexGraph::LevelId IndexGraph::get_insert_level() {
  thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());
  // uniform_real_distribution's range is half-open [0, 1); flip it to
  // (0, 1] so unif() is never exactly 0, which would make -ln(unif)
  // diverge below.
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double unif = 1.0 - dist(gen);

  // Algorithm 1, line 4:
  // l <- floor(-ln(unif(0,1)) * mL)
  // where mL = 1 / ln(M)
  double level = std::floor(-std::log(unif) * m_store.level_norm_factor());

  double max_level = static_cast<double>(IndexStore::max_level().value);
  level = std::min(level, max_level);
  return LevelId{static_cast<uint8_t>(level)};
}

bool IndexGraph::get_entry_point(std::vector<Node> &out,
                                 LevelId & /*out_level*/) {
  out.clear();
  return false;
}

bool IndexGraph::set_entry_point(const std::vector<Node> & /*nodes*/,
                                 LevelId /*level*/) {
  return false;
}

bool IndexGraph::create_node(const std::optional<Node> & /*parent*/,
                             LevelId /*level*/, const NodeData & /*data*/,
                             const std::vector<Node> & /*neighbours*/,
                             Node & /*out*/) {
  return false;
}

bool IndexGraph::drop_node(const std::optional<Node> & /*parent*/,
                           const Node & /*node*/) {
  return false;
}

bool IndexGraph::get_next_level(const Node & /*node*/, Node &out) {
  out = Node{};
  return false;
}

bool IndexGraph::link_neighbours(const Node & /*node*/,
                                 std::vector<Node> &out) {
  out.clear();
  return false;
}

bool IndexGraph::replace_neighbours(const Node & /*node*/,
                                    const std::vector<Node> & /*neighbours*/) {
  return false;
}

bool IndexGraph::unlink_neighbours(const Node & /*node*/,
                                   UnlinkOrphans /*orphans*/,
                                   std::vector<Node> &out) {
  out.clear();
  return false;
}

uint32_t IndexGraph::M() const { return m_store.num_neighbours(); }
uint32_t IndexGraph::ef_construction() const {
  return m_store.ef_construction();
}
uint32_t IndexGraph::Mmax(LevelId level) const {
  auto *store = m_store.level(level);
  assert(store != nullptr);
  return store->max_neighbours();
}

// Lock primitives backing LockGraph/LockLevels below.
void IndexGraph::lock_graph(LockMode mode) {
  if (mode == LockMode::Shared)
    m_store.mutex().lock_shared();
  else
    m_store.mutex().lock();
}

void IndexGraph::unlock_graph(LockMode mode) {
  if (mode == LockMode::Shared)
    m_store.mutex().unlock_shared();
  else
    m_store.mutex().unlock();
}

void IndexGraph::lock_level(LevelId level, LockMode mode) {
  auto *store = m_store.level(level);
  assert(store != nullptr);
  if (mode == LockMode::Shared)
    store->mutex().lock_shared();
  else
    store->mutex().lock();
}

void IndexGraph::unlock_level(LevelId level, LockMode mode) {
  auto *store = m_store.level(level);
  assert(store != nullptr);
  if (mode == LockMode::Shared)
    store->mutex().unlock_shared();
  else
    store->mutex().unlock();
}

IndexGraph::LockGraph::LockGraph(IndexGraph &graph, LockMode mode)
    : m_graph(graph), m_mode(mode) {
  m_graph.lock_graph(m_mode);
}

void IndexGraph::LockGraph::relock(LockMode mode) {
  if (m_mode == mode) {
    return;
  }
  m_graph.unlock_graph(m_mode);
  m_graph.lock_graph(mode);
  m_mode = mode;
}

IndexGraph::LockGraph::~LockGraph() { m_graph.unlock_graph(m_mode); }

IndexGraph::LockLevels::LockLevels(IndexGraph &graph, LockMode mode,
                                   LevelId start, DescendPolicy policy)
    : m_graph(graph), m_mode(mode), m_policy(policy) {
  m_graph.lock_level(start, m_mode);
  m_stack.push(start);
}

IndexGraph::LockLevels::~LockLevels() {
  // Release level locks in LIFO order.
  while (!m_stack.empty()) {
    m_graph.unlock_level(m_stack.top(), m_mode);
    m_stack.pop();
  }
}

IndexGraph::LevelId IndexGraph::LockLevels::descend() {
  auto upper_level = level();
  if (!upper_level.has_lower_level()) {
    return upper_level;
  }
  auto lower_level = upper_level.lower();
  // Lock the lower level before releasing the current one (if any), so
  // this guard never momentarily holds zero level locks.
  m_graph.lock_level(lower_level, m_mode);

  if (m_policy == DescendPolicy::Release) {
    m_graph.unlock_level(upper_level, m_mode);
    m_stack.pop();
  }
  m_stack.push(lower_level);
  return lower_level;
}

IndexGraph::LevelId IndexGraph::LockLevels::level() const {
  assert(!m_stack.empty());
  return m_stack.top();
}

void IndexGraph::LockLevels::update_policy(DescendPolicy policy) {
  m_policy = policy;
}

} // namespace svector::hnsw
