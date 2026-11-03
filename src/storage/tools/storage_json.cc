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

#include "storage_json.h"

#include <cstdio>

#include "../root_page.h"

namespace svector {
namespace tool {

namespace {

void write_page_ref(std::ostream &out, uint32_t page) {
  if (page == RootPage::NULL_FREE_PAGE_REF)
    out << "null";
  else
    out << page;
}

void write_ref(std::ostream &out, uint64_t ref_value) {
  if (ref_value == 0) {
    out << "null";
    return;
  }
  HnswColumnRef ref = hnsw_decode_ref(ref_value);
  out << "\"" << ref.page_ref << ":" << ref.slot_index << "\"";
}

} // namespace

std::string json_escape(const std::string &s) {
  std::string r;
  for (unsigned char c : s) {
    switch (c) {
    case '"':
      r += "\\\"";
      break;
    case '\\':
      r += "\\\\";
      break;
    case '\n':
      r += "\\n";
      break;
    case '\t':
      r += "\\t";
      break;
    default:
      if (c < 0x20 || c >= 0x7f) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
        r += buf;
      } else {
        r += static_cast<char>(c);
      }
    }
  }
  return r;
}

void write_root_page_json(uint32_t page_num,
                          const RootPageParser::RootPageInfo &info,
                          const HnswIndexMetadata *index_meta,
                          const std::vector<std::string> &other_root_names,
                          std::ostream &out) {
  out << "{\"kind\":\"root\",\"page\":" << page_num
      << ",\"version\":" << static_cast<int>(info.version)
      << ",\"column_size\":" << info.column_size << ",\"all_head\":";
  write_page_ref(out, info.all_slot_head);
  out << ",\"all_tail\":";
  write_page_ref(out, info.all_slot_tail);
  out << ",\"total_data_pages\":" << info.total_data_pages
      << ",\"total_free_pages\":" << info.total_free_pages
      << ",\"free_slot_array_max\":" << info.free_slot_array_max_size
      << ",\"other_roots\":[";
  for (size_t i = 0; i < info.other_root_page_refs.size(); ++i) {
    if (i > 0)
      out << ",";
    out << "{\"page\":";
    write_page_ref(out, info.other_root_page_refs[i]);
    out << ",\"name\":\""
        << json_escape(i < other_root_names.size() ? other_root_names[i] : "")
        << "\"}";
  }
  out << "],\"free_pages\":[";
  for (size_t i = 0; i < info.free_slots.size(); ++i) {
    if (i > 0)
      out << ",";
    write_page_ref(out, info.free_slots[i]);
  }
  out << "],\"index\":";
  if (index_meta == nullptr) {
    out << "null";
  } else {
    out << "{\"name\":\"" << json_escape(index_meta->name)
        << "\",\"level\":" << static_cast<int>(index_meta->level)
        << ",\"overflow\":" << (index_meta->is_overflow() ? "true" : "false")
        << "}";
  }
  out << "}\n";
}

void write_data_page_json(uint32_t page_num,
                          const DataPageParser::DataPageInfo &info,
                          std::ostream &out) {
  const char *kind = info.index_kind == HnswRecordKind::Neighbour  ? "neighbour"
                     : info.index_kind == HnswRecordKind::Overflow ? "overflow"
                                                                   : "vector";
  out << "{\"kind\":\"data\",\"page\":" << page_num
      << ",\"root\":" << info.root_page_ref << ",\"prev\":";
  write_page_ref(out, info.fil_page_prev);
  out << ",\"next\":";
  write_page_ref(out, info.fil_page_next);
  out << ",\"free_prev\":";
  write_page_ref(out, info.prev_free_page);
  out << ",\"free_next\":";
  write_page_ref(out, info.next_free_page);
  out << ",\"free_slot_number\":";
  if (info.free_slot_number == DataPage::INVALID_SLOT)
    out << "null";
  else
    out << info.free_slot_number;
  out << ",\"version\":" << static_cast<int>(info.version)
      << ",\"max\":" << info.max_num_recs << ",\"free\":" << info.num_free_recs
      << ",\"allocated\":" << info.num_allocated()
      << ",\"deleted\":" << info.num_deleted() << ",\"record_kind\":\"" << kind
      << "\",\"has_lower_level\":" << (info.has_lower_level ? "true" : "false")
      << ",\"records\":[";
  for (size_t i = 0; i < info.records.size(); ++i) {
    const auto &rec = info.records[i];
    if (i > 0)
      out << ",";
    // s: F(ree), D(eleted) or A(llocated).
    out << "{\"i\":" << i << ",\"s\":\""
        << (rec.is_free ? 'F' : (rec.is_deleted ? 'D' : 'A')) << "\"";
    if (!rec.is_free) {
      out << ",\"trx\":" << rec.trx_ref << ",\"dtrx\":" << rec.delete_trx_ref;
      if (info.index_kind == HnswRecordKind::Neighbour) {
        out << ",\"vid\":";
        write_ref(out, rec.owner_vid);
        if (info.has_lower_level) {
          out << ",\"lower\":";
          write_ref(out, rec.lower_level_nid);
        }
        out << ",\"neighbours\":[";
        bool first = true;
        for (const auto &neighbour : rec.neighbours) {
          if (neighbour.first == 0)
            continue;
          if (!first)
            out << ",";
          first = false;
          out << "{\"nid\":";
          write_ref(out, neighbour.first & HNSW_NID_REF_MASK);
          out << ",\"vid\":";
          write_ref(out, neighbour.second);
          out << ",\"incoming\":"
              << ((neighbour.first & HNSW_NID_INCOMING_BIT) ? "true" : "false")
              << "}";
        }
        out << "],\"overflow\":";
        write_ref(out, rec.overflow_nid);
      } else if (info.index_kind == HnswRecordKind::Overflow) {
        out << ",\"incoming\":[";
        bool first = true;
        for (uint64_t nid : rec.incoming) {
          if (nid == 0)
            continue;
          if (!first)
            out << ",";
          first = false;
          write_ref(out, nid);
        }
        out << "],\"overflow\":";
        write_ref(out, rec.overflow_nid);
      }
    }
    out << "}";
  }
  out << "]}\n";
}

} // namespace tool
} // namespace svector
