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

#include <cstdint>
#include <functional>
#include <span>

namespace svector::hnsw {

template <typename Tag> struct Id {
  static constexpr uint64_t INVALID = 0;
  static constexpr size_t STORAGE_SIZE = 6;

  uint64_t value{INVALID};

  constexpr Id() = default;
  constexpr explicit Id(uint64_t v) : value(v) {}

  constexpr bool is_valid() const { return value != INVALID; }
  constexpr explicit operator bool() const { return is_valid(); }

  constexpr auto operator<=>(const Id &) const = default;
};

struct NIDTag {};
struct VIDTag {};

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

// Overflow entry to hold extra incoming connections to a Node. Overflow
// entries are chained.
struct OverflowEntry {
  std::span<NID> incoming{};
  // Reference to the next overflow entry, if any.
  NID overflow{};
};

} // namespace svector::hnsw

template <typename Tag> struct std::hash<svector::hnsw::Id<Tag>> {
  size_t operator()(const svector::hnsw::Id<Tag> &id) const noexcept {
    return std::hash<uint64_t>{}(id.value);
  }
};

#endif // VILLAGESQL_VSQL_VECTOR_SRC_INDEX_HNSW_HNSW_H
