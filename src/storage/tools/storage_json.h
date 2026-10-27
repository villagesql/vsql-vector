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

#ifndef VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_STORAGE_JSON_H_
#define VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_STORAGE_JSON_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "data_page_parser.h"
#include "hnsw_layout.h"
#include "root_page_parser.h"

namespace svector {
namespace tool {

// JSON export of one storage page, for the graph viewer's storage view. The
// output is a single object on out and nothing else. Page references that
// are the "no page" value (RootPage::NULL_FREE_PAGE_REF) are written as null,
// and NIDs/VIDs as "page:slot" strings (null when 0, i.e. unset).

// Root page. index_meta is the decoded HNSW StorageMeta, or null when the
// store is a plain SVECTOR column (or the metadata did not decode).
// other_root_names[i] is the store name of info.other_root_page_refs[i] ("" if
// unknown); it may be shorter than the refs, and missing names are "".
void write_root_page_json(uint32_t page_num,
                          const RootPageParser::RootPageInfo &info,
                          const HnswIndexMetadata *index_meta,
                          const std::vector<std::string> &other_root_names,
                          std::ostream &out);

// Data page. How each record's fields are written follows info.index_kind.
void write_data_page_json(uint32_t page_num,
                          const DataPageParser::DataPageInfo &info,
                          std::ostream &out);

// Escapes s for use inside a double-quoted JSON string.
std::string json_escape(const std::string &s);

} // namespace tool
} // namespace svector

#endif // VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_STORAGE_JSON_H_
