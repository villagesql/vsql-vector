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

#include "quantize.h"

#include <algorithm>
#include <cmath>

namespace svector::quant {

void quantize(const float *src, uint32_t dim, uint32_t padded_dim, QData *out) {
  float max_abs = 0.0f;
  for (uint32_t i = 0; i < dim; i++)
    max_abs = std::max(max_abs, std::abs(src[i]));

  const float scale = max_abs > 0.0f ? max_abs / 32767.0f : 1.0f;
  out->dim = dim;
  out->scale = scale;

  int64_t dot = 0;
  for (uint32_t i = 0; i < dim; i++) {
    int32_t q = static_cast<int32_t>(std::lround(src[i] / scale));
    if (q > 32767) q = 32767;
    if (q < -32768) q = -32768;
    out->dims[i] = static_cast<int16_t>(q);
    dot += static_cast<int32_t>(out->dims[i]) *
           static_cast<int32_t>(out->dims[i]);
  }
  for (uint32_t i = dim; i < padded_dim; i++) out->dims[i] = 0;
  out->abs2 = 0.5f * scale * scale * static_cast<float>(dot);
}

}  // namespace svector::quant
