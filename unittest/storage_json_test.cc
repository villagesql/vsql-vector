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

// Standalone, assert()-based unit test for the JSON page export
// (src/storage/tools/storage_json.h). Builds parsed page structs by hand
// instead of reading an .ibd file.

#include <cassert>
#include <cstdio>
#include <sstream>
#include <string>

#include "../src/storage/tools/hnsw_layout.h"
#include "../src/storage/tools/storage_json.h"

using namespace svector::tool;

namespace {

uint64_t R(uint32_t page, uint16_t slot) { return hnsw_encode_ref(page, slot); }

bool has(const std::string &s, const std::string &needle) {
  return s.find(needle) != std::string::npos;
}

DataPageParser::RecordStatus rec(bool free, bool deleted) {
  DataPageParser::RecordStatus r{};
  r.is_free = free;
  r.is_deleted = deleted;
  return r;
}

DataPageParser::DataPageInfo neighbour_page() {
  DataPageParser::DataPageInfo info{};
  info.fil_page_prev = 0xFFFFFFFF;
  info.fil_page_next = 9;
  info.free_slot_number = 0xFFFF;
  info.root_page_ref = 4;
  info.prev_free_page = 0xFFFFFFFF;
  info.next_free_page = 0xFFFFFFFF;
  info.max_num_recs = 3;
  info.num_free_recs = 1;
  info.index_kind = HnswRecordKind::Neighbour;

  auto a = rec(false, false);
  a.trx_ref = 7;
  a.owner_vid = R(11, 5);
  a.neighbours = {{R(12, 1), R(11, 6)},
                  {0, 0}, // empty slot: not written
                  {R(12, 2) | HNSW_NID_INCOMING_BIT, R(11, 7)}};
  info.records = {a, rec(false, true), rec(true, false)};
  info.records[1].owner_vid = R(11, 6);
  return info;
}

void test_data_page() {
  std::ostringstream out;
  write_data_page_json(5, neighbour_page(), out);
  std::string j = out.str();
  assert(has(j, "\"kind\":\"data\",\"page\":5,\"root\":4,\"prev\":null,"
                "\"next\":9"));
  assert(has(j, "\"free_slot_number\":null"));
  assert(has(j, "\"record_kind\":\"neighbour\""));
  assert(has(j, "\"allocated\":2,\"deleted\":1"));
  // States per slot, in slot order: allocated, delete-marked, free.
  assert(j.find("\"s\":\"A\"") < j.find("\"s\":\"D\""));
  assert(j.find("\"s\":\"D\"") < j.find("\"s\":\"F\""));
  assert(has(j, "\"vid\":\"11:5\""));
  assert(has(j, "{\"nid\":\"12:1\",\"vid\":\"11:6\",\"incoming\":false}"));
  // The incoming flag is reported, not left in the NID; empty slot dropped.
  assert(has(j, "{\"nid\":\"12:2\",\"vid\":\"11:7\",\"incoming\":true}"));
  assert(!has(j, "\"nid\":null"));
  assert(has(j, "\"overflow\":null"));
  assert(j.back() == '\n');
}

void test_free_record_has_no_fields() {
  std::ostringstream out;
  write_data_page_json(5, neighbour_page(), out);
  assert(has(out.str(), "{\"i\":2,\"s\":\"F\"}"));
}

void test_vector_page_has_no_vid() {
  DataPageParser::DataPageInfo info = neighbour_page();
  info.index_kind = HnswRecordKind::None;
  std::ostringstream out;
  write_data_page_json(5, info, out);
  assert(has(out.str(), "\"record_kind\":\"vector\""));
  assert(!has(out.str(), "\"vid\""));
}

void test_overflow_page() {
  DataPageParser::DataPageInfo info = neighbour_page();
  info.index_kind = HnswRecordKind::Overflow;
  info.records[0].incoming = {R(3, 1), 0, R(3, 2)};
  info.records[0].overflow_nid = R(8, 0);
  std::ostringstream out;
  write_data_page_json(5, info, out);
  assert(has(out.str(), "\"incoming\":[\"3:1\",\"3:2\"],\"overflow\":\"8:0\""));
}

void test_root_page() {
  RootPageParser::RootPageInfo info{};
  info.version = 1;
  info.column_size = 128;
  info.all_slot_head = 5;
  info.all_slot_tail = 0xFFFFFFFF;
  info.total_data_pages = 2;
  info.total_free_pages = 1;
  info.other_root_page_refs = {7, 9};
  info.free_slots = {5};
  HnswIndexMetadata meta;
  meta.name = "HNSW-L0";
  std::ostringstream out;
  write_root_page_json(4, info, &meta, {"HNSW-L0-OV", ""}, out);
  std::string j = out.str();
  assert(has(j, "\"kind\":\"root\",\"page\":4"));
  assert(has(j, "\"all_head\":5,\"all_tail\":null"));
  assert(has(j, "\"other_roots\":[{\"page\":7,\"name\":\"HNSW-L0-OV\"},"
                "{\"page\":9,\"name\":\"\"}]"));
  assert(has(j, "\"free_pages\":[5]"));
  assert(has(j, "\"index\":{\"name\":\"HNSW-L0\",\"level\":0,"
                "\"overflow\":false}"));

  std::ostringstream plain;
  write_root_page_json(4, info, nullptr, {}, plain);
  assert(has(plain.str(), "\"index\":null"));
}

void test_json_escape() {
  assert(json_escape("plain") == "plain");
  assert(json_escape("a\"b\\c\n") == "a\\\"b\\\\c\\n");
  assert(json_escape(std::string("\x01", 1)) == "\\u0001");
}

} // namespace

int main() {
  test_data_page();
  test_free_record_has_no_fields();
  test_vector_page_has_no_vid();
  test_overflow_page();
  test_root_page();
  test_json_escape();

  std::printf("All storage_json tests passed.\n");
  return 0;
}
