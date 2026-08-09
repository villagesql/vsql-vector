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

#ifndef VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_HNSW_H
#define VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_HNSW_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <vector>

#include <villagesql/preview/storage_api.h>

namespace svector::hnsw {

using vsql::preview_storage::Column;

// Fixed-size reusable scratch storage.
// Allocated once during construction and reused to avoid allocations in
// performance-critical graph operations.
template <typename T> struct ScratchArray {
  explicit ScratchArray(size_t size) : m_data(size) {}

  void clear() { fill(T{}); }

  void fill(const T &value) { std::fill(m_data.begin(), m_data.end(), value); }

  T &operator[](size_t i) { return m_data[i]; }
  const T &operator[](size_t i) const { return m_data[i]; }

  T *data() { return m_data.data(); }
  const T *data() const { return m_data.data(); }

  std::span<T> span() { return m_data; }
  std::span<const T> span() const { return m_data; }
  std::span<T> span(size_t n) { return span().first(n); }
  std::span<const T> span(size_t n) const { return span().first(n); }

  size_t size() const { return m_data.size(); }
  bool empty() const { return m_data.empty(); }

  auto begin() { return m_data.begin(); }
  auto end() { return m_data.end(); }

  auto begin() const { return m_data.begin(); }
  auto end() const { return m_data.end(); }

private:
  std::vector<T> m_data;
};

struct SlotIndex {
  static constexpr uint16_t INVALID = std::numeric_limits<uint16_t>::max();

  uint16_t value = INVALID;

  constexpr SlotIndex(uint16_t v) : value(v) {}
  constexpr SlotIndex() = default;

  constexpr bool is_valid() const { return value != INVALID; }
};

using ScratchBytes = ScratchArray<std::byte>;
using ScratchSlots = ScratchArray<SlotIndex>;
using ScratchChunkIds = ScratchArray<uint16_t>;

template <typename Tag> struct Id {
  static constexpr uint64_t INVALID = 0;
  static constexpr size_t STORAGE_SIZE = 6;

  uint64_t value{INVALID};

  constexpr Id() = default;
  constexpr explicit Id(uint64_t v) : value(v) {}

  constexpr bool is_valid() const { return value != INVALID; }

  constexpr auto operator<=>(const Id &) const = default;
};

struct NIDTag {};
struct VIDTag {};

// NID's value is the Column::Ref of the NeighbourEntry/OverflowEntry record
// it names. Column Storage guarantees the high 18 bits of a Column::Ref are
// always zero (see DataPage::MAX_SLOT_INDEX), so bit 46 -- the lowest of
// those guaranteed-free bits -- is repurposed here as the "incoming" flag.
template <> struct Id<NIDTag> {
  static constexpr uint64_t INVALID = 0;
  static constexpr size_t STORAGE_SIZE = 6;

  static constexpr uint64_t INCOMING_BIT = uint64_t{1} << 46;
  static constexpr uint64_t COLUMN_REF_MASK = INCOMING_BIT - 1;

  uint64_t value{INVALID};

  constexpr Id() = default;
  constexpr explicit Id(uint64_t v) : value(v) {}

  constexpr bool is_valid() const { return value != INVALID; }
  constexpr explicit operator bool() const { return is_valid(); }

  constexpr auto operator<=>(const Id &) const = default;

  // The Column::Ref this NID's record lives at, with the incoming flag
  // masked off.
  constexpr Column::Ref column_ref() const {
    return static_cast<Column::Ref>(value & COLUMN_REF_MASK);
  }

  constexpr bool is_incoming() const { return (value & INCOMING_BIT) != 0; }
  constexpr Id set_incoming() const { return Id(value | INCOMING_BIT); }
  constexpr Id clear_incoming() const { return Id(value & ~INCOMING_BIT); }
};

// Node ID
using NID = Id<NIDTag>;

// Vector data ID
using VID = Id<VIDTag>;

// A node with its ID and vector data ID
struct Node {
  NID nid{};
  VID vid{};

  using KeyType = NID;
  KeyType key() const { return nid; }

  static constexpr size_t STORAGE_SIZE = NID::STORAGE_SIZE + VID::STORAGE_SIZE;
};

// Which of a level's two stores a node (or NID) belongs to.
enum class StoreKind { Neighbour, Overflow };

// A Neighbour entry in HNSW index.
struct NeighbourEntry {
  // Vector owning this neighbour entry.
  VID owner{};
  // Reference to the same vector's neighbour entry in the next lower level.
  // Invalid for level 0.
  NID lower_level{};
  std::span<Node> neighbours{};
  // Reference to the first overflow entry, if any.
  NID overflow{};

  static constexpr size_t storage_size(size_t max_neighbours,
                                       bool has_lower_level) {
    return VID::STORAGE_SIZE + (has_lower_level ? NID::STORAGE_SIZE : 0) +
           max_neighbours * Node::STORAGE_SIZE + NID::STORAGE_SIZE;
  }
};

// Selects which fields of a NeighbourEntry a partial update writes or a
// partial fetch unmarshals.
enum class NodeField : uint32_t {
  Owner = 1 << 0,
  LowerLevel = 1 << 1,
  Neighbours = 1 << 2,
  Overflow = 1 << 3,
};

constexpr NodeField operator|(NodeField a, NodeField b) {
  return static_cast<NodeField>(static_cast<uint32_t>(a) |
                                static_cast<uint32_t>(b));
}

constexpr bool has(NodeField mask, NodeField field) {
  return (static_cast<uint32_t>(mask) & static_cast<uint32_t>(field)) != 0;
}

// Every NeighbourEntry field -- the default fetch mask for a full read.
constexpr NodeField FieldAll = NodeField::Owner | NodeField::LowerLevel |
                               NodeField::Neighbours | NodeField::Overflow;

// Whether LevelStore::fetch()'s NeighbourEntry overload includes or drops
// neighbours whose NID has the incoming flag set (Id<NIDTag>::is_incoming())
// when decoding NodeField::Neighbours.
enum class IncomingFilter { All, ExcludeIncoming };

// Overflow entry to hold extra incoming connections to a Node. Overflow
// entries are chained.
struct OverflowEntry {
  std::span<NID> incoming{};
  // Reference to the next overflow entry, if any.
  NID overflow{};

  static constexpr size_t storage_size(size_t capacity) {
    return capacity * NID::STORAGE_SIZE + NID::STORAGE_SIZE;
  }
};

// Selects which fields of an OverflowEntry a partial update writes.
enum class OverflowField : uint32_t {
  Incoming = 1 << 0,
  Overflow = 1 << 1,
};

constexpr OverflowField operator|(OverflowField a, OverflowField b) {
  return static_cast<OverflowField>(static_cast<uint32_t>(a) |
                                    static_cast<uint32_t>(b));
}

constexpr bool has(OverflowField mask, OverflowField field) {
  return (static_cast<uint32_t>(mask) & static_cast<uint32_t>(field)) != 0;
}

// Every OverflowEntry field -- the default fetch mask for a full read.
constexpr OverflowField OverflowFieldAll =
    OverflowField::Incoming | OverflowField::Overflow;

} // namespace svector::hnsw

template <typename Tag> struct std::hash<svector::hnsw::Id<Tag>> {
  size_t operator()(const svector::hnsw::Id<Tag> &id) const noexcept {
    return std::hash<uint64_t>{}(id.value);
  }
};

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_HNSW_H
