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

#ifndef VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_DISTANCE_REGISTRY_H
#define VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_DISTANCE_REGISTRY_H

#include <cstring>

#include "native_vector.h"

// The single owner of the distance metric NAME <-> KERNEL correspondence.
//
// One name string per metric, and one map from name to the matching kernel in
// native_vector.h. Both sides of the extension bind against these symbols:
//   - SVECTOR function/index registration (vector.cc) names its helper
//     functions with kDist*, and
//   - index code resolves a bound helper's name back to its kernel via
//     dist_for_name().
// Keeping both on these symbols means a rename can never silently mismatch the
// two. This lives in its own file (not native_vector.h) so the kernels stay
// pure math and the name policy has one obvious home.
namespace svector::native {

using DistFn = double (*)(const Data *, const Data *);

inline constexpr const char kDistL1[] = "l1_distance";
inline constexpr const char kDistL2Squared[] = "l2_squared_distance";
inline constexpr const char kDistCosine[] = "cosine_distance";
inline constexpr const char kDistInnerProduct[] = "inner_product";

// Map a metric name to its kernel; nullptr if the name is unknown.
inline DistFn dist_for_name(const char *name) {
  if (name == nullptr) return nullptr;
  if (std::strcmp(name, kDistL2Squared) == 0) return &dist_squared_l2;
  if (std::strcmp(name, kDistL1) == 0) return &dist_l1;
  if (std::strcmp(name, kDistCosine) == 0) return &dist_cosine;
  if (std::strcmp(name, kDistInnerProduct) == 0) return &dist_inner_product;
  return nullptr;
}

}  // namespace svector::native

#endif  // VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_DISTANCE_REGISTRY_H
