// Copyright (c) 2026 VillageSQL Contributors
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License, version 2.0,
// as published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License, version 2.0, for more details.

// Standalone, assert()-based unit test for the SVECTOR column storage layer
// (svector::MultiColumnStore / ColumnStore), driven WITHOUT InnoDB or the
// server. The storage primitives (Page/Segment/MtrCtx) in
// <villagesql/preview/storage_api.h> call through a vtable
// (vsql::preview_storage::detail::g_abi) that the server normally populates
// with InnoDB-backed functions. Here we install an in-memory fake backend --
// a map of page-number -> fixed-size page buffer -- so the real storage code
// can run in a plain unit-test process.
//
// PURPOSE: reproduce BUG 7 -- the multi-segment level-store page-allocation
// bug. The HNSW IndexStore lays out per-level stores across 2 segments
// (Primary = level 0, Secondary = levels >= 1). Segments are created ONCE, on
// the primary (level-0) root page (MultiColumnStore::create ->
// Segment::create). A level >= 1 store's root page is allocated later by
// init_root_page() and formatted with num_segments = 0. But ColumnStore::insert
// reads its segment header from the store's OWN root page --
// Segment::get_header(root_page, 0) (src/storage/storage.cc) -- which for a
// level >= 1 store has num_segments == 0, so Segment::get_header aborts on
// `assert(seg_no < num_segments)` (storage_api.h). The fix (owned by the
// storage layer) is for a non-primary store's insert to source the segment
// from the PRIMARY root page at its assigned segment index (Secondary), the
// way init_root_page() already does. Until then, the test below documents the
// crash: an insert into a level >= 1 store aborts.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <villagesql/preview/storage_api.h>

#include "../src/storage/storage.h"

namespace {

using vsql::preview_storage::Column;
using vsql::preview_storage::Segment;
using vsql::preview_storage::Space;

// -------------------------------------------------------------------------
// In-memory fake storage backend (stand-in for InnoDB's g_abi vtable).
// Single-threaded; latches are bookkeeping no-ops. Pages are fixed-size
// buffers kept in a std::map (node pointers are stable, so a page's data
// pointer stays valid for the life of the store).
// -------------------------------------------------------------------------
class FakeStorage {
public:
  static constexpr uint32_t kPageSize = 16384;

  // std::map: element addresses are stable across insertions, so the
  // unsigned char* we hand back from get_data stays valid.
  std::map<uint32_t, std::vector<unsigned char>> pages;
  uint32_t next_page = 1;
  // A single fake mtr token (non-null). Its identity is irrelevant to the
  // fake; the real code only needs a non-null ref back from mtr_start.
  int mtr_token = 0;

  static FakeStorage &instance() {
    static FakeStorage s;
    return s;
  }

  std::vector<unsigned char> &page(uint32_t num) {
    auto it = pages.find(num);
    assert(it != pages.end());
    return it->second;
  }

  uint32_t allocate() {
    uint32_t num = next_page++;
    pages.emplace(num, std::vector<unsigned char>(kPageSize, 0));
    return num;
  }
};

// Big-endian integer write, matching Page::mach_read_from_* (storage_api.h).
void be_write(unsigned char *p, uint64_t value, int bytes) {
  for (int i = bytes - 1; i >= 0; --i) {
    p[i] = static_cast<unsigned char>(value & 0xFF);
    value >>= 8;
  }
}

// ---- g_abi function implementations ----

vef_storage_mtr_ref_t
fake_mtr_start(void * /*buffer*/, uint32_t /*buffer_size*/,
               uint32_t * /*required_size*/, uint32_t * /*required_alignment*/,
               char * /*error_msg*/, uint32_t /*error_msg_len*/) {
  return &FakeStorage::instance().mtr_token;
}

void fake_mtr_commit(vef_storage_mtr_ref_t /*ref*/) {}

int fake_segment_create(vef_storage_space_ref_t /*space_ref*/,
                        uint8_t num_segments, vef_storage_trx_ref_t /*trx_ref*/,
                        vef_storage_page_num_t *root_page_num_p,
                        char * /*error_msg*/, uint32_t /*error_msg_len*/) {
  auto &fs = FakeStorage::instance();
  uint32_t root = fs.allocate();
  // Segment headers live at the start of the root page's DATA area; the byte
  // at Page::HEADER_SIZE is the segment count (read by
  // Page::read_num_segments).
  fs.page(root)[vsql::preview_storage::Page::HEADER_SIZE] = num_segments;
  *root_page_num_p = root;
  return VEF_STORAGE_SUCCESS;
}

int fake_segment_drop(vef_storage_space_ref_t, vef_storage_trx_ref_t,
                      vef_storage_page_num_t, char *, uint32_t) {
  return VEF_STORAGE_SUCCESS;
}

int fake_page_load(vef_storage_block_ref_t *block_p, uint64_t *position_p,
                   unsigned char **data_p, uint32_t *data_size_p,
                   vef_storage_space_ref_t /*space_ref*/,
                   vef_storage_page_num_t page_num,
                   vef_storage_latch_t /*latch_mode*/,
                   vef_storage_mtr_ref_t /*mtr_ref*/, char * /*error_msg*/,
                   uint32_t /*error_msg_len*/) {
  auto &fs = FakeStorage::instance();
  auto it = fs.pages.find(page_num);
  if (it == fs.pages.end())
    return VEF_STORAGE_ERROR_PAGE_LOAD;
  unsigned char *data = it->second.data();
  *block_p = data;        // block ref == page data pointer
  *position_p = page_num; // stored position (non-INVALID -> release works)
  *data_p = data;
  *data_size_p = FakeStorage::kPageSize;
  return VEF_STORAGE_SUCCESS;
}

int fake_page_allocate_and_load(vef_storage_block_ref_t *block_p,
                                vef_storage_page_num_t *page_num_p,
                                unsigned char **data_p, uint32_t *data_size_p,
                                unsigned char * /*segment_header*/,
                                vef_storage_mtr_ref_t /*mtr_ref*/,
                                char * /*error_msg*/,
                                uint32_t /*error_msg_len*/) {
  auto &fs = FakeStorage::instance();
  uint32_t num = fs.allocate();
  unsigned char *data = fs.page(num).data();
  *block_p = data;
  *page_num_p = num;
  *data_p = data;
  *data_size_p = FakeStorage::kPageSize;
  return VEF_STORAGE_SUCCESS;
}

int fake_page_latch(vef_storage_block_ref_t, uint64_t, vef_storage_latch_t,
                    vef_storage_mtr_ref_t, char *, uint32_t) {
  return VEF_STORAGE_SUCCESS;
}

int fake_page_release(vef_storage_block_ref_t, uint64_t, vef_storage_mtr_ref_t,
                      char *, uint32_t) {
  return VEF_STORAGE_SUCCESS;
}

uint32_t fake_page_get_size(vef_storage_space_ref_t /*space_ref*/) {
  return FakeStorage::kPageSize;
}

void fake_page_write_integer(vef_storage_block_ref_t block,
                             vef_storage_page_offset_t offset, uint64_t value,
                             vef_storage_integer_bytes_t bytes,
                             vef_storage_mtr_ref_t /*mtr_ref*/) {
  auto *p = static_cast<unsigned char *>(block) + offset;
  switch (bytes) {
  case VEF_STORAGE_PAGE_INT_1BYTE:
    be_write(p, value, 1);
    break;
  case VEF_STORAGE_PAGE_INT_2BYTES:
    be_write(p, value, 2);
    break;
  case VEF_STORAGE_PAGE_INT_4BYTES:
    be_write(p, value, 4);
    break;
  case VEF_STORAGE_PAGE_INT_8BYTES:
    be_write(p, value, 8);
    break;
  default:
    assert(false);
  }
}

void fake_page_write_string(vef_storage_block_ref_t block,
                            vef_storage_page_offset_t offset,
                            const unsigned char *str, uint32_t len,
                            vef_storage_mtr_ref_t /*mtr_ref*/) {
  std::memcpy(static_cast<unsigned char *>(block) + offset, str, len);
}

const vef_preview_storage_t kFakeAbi = {
    /*version=*/VEF_STORAGE_SE_INTF_VERSION,
    fake_mtr_start,
    fake_mtr_commit,
    fake_segment_create,
    fake_segment_drop,
    fake_page_load,
    fake_page_allocate_and_load,
    fake_page_latch,
    fake_page_release,
    fake_page_get_size,
    fake_page_write_integer,
    fake_page_write_string,
};

void install_fake_abi() { vsql::preview_storage::detail::g_abi = &kFakeAbi; }

// -------------------------------------------------------------------------

// Reproduces BUG 7. Mirrors the HNSW IndexStore's two-segment layout:
// Primary (segment 0, level-0 store) and Secondary (segment 1, level >= 1
// stores). A first insert into the level-1 store takes the pessimistic path
// (no free data page yet) and calls Segment::get_header(own_root_page, 0),
// which aborts because that store's root page has num_segments == 0.
//
// Expected on the CURRENT (buggy) code: abort with
//   "Assertion failed: (seg_no < num_segments) ... storage_api.h"
// When the storage layer is fixed to source the segment from the primary root
// page at the store's segment index, this insert succeeds and the final
// assert(!failed) holds.
void test_insert_into_level1_store_hits_segment_bug() {
  install_fake_abi();

  char err[512] = {};
  constexpr uint16_t kColLen = 16; // arbitrary fixed column width

  svector::MultiColumnStore mcs;
  std::vector<svector::Storage_spec> specs;
  specs.push_back({kColLen, "L0"}); // primary   (segment 0)
  specs.push_back({kColLen, "L1"}); // secondary (segment 1)

  // 2 segments: Primary + Secondary, exactly like IndexStore::create for
  // level-0 vs level>=1.
  bool failed = mcs.create(/*space_ref=*/1, /*trx_ref=*/1, specs,
                           /*num_segments=*/2, err, sizeof(err));
  assert(!failed && "MultiColumnStore::create should succeed");

  // Materialize the level-1 store's root page from the Secondary segment
  // (seg_idx = 1), root_idx = 1 -- exactly what IndexStore does for a new
  // higher level.
  failed = mcs.init_root_page(/*seg_idx=*/1, /*root_idx=*/1, err, sizeof(err));
  assert(!failed && "init_root_page for the level-1 store should succeed");

  // Insert into the level-1 store. First insert -> no free data page ->
  // pessimistic path -> Segment::get_header(own_root_page, 0). That root page
  // has num_segments == 0, so this aborts on the current (buggy) code.
  std::vector<unsigned char> value(kColLen, 0x7);
  Column::Data col_data{value.data(), static_cast<uint32_t>(value.size())};
  Column::Ref col_ref = Column::EMPTY_REF;

  vsql::preview_storage::MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();
  failed = mcs.m_stores[1].insert(mtr, /*trx_ref=*/1, col_data, col_ref, err,
                                  sizeof(err));
  mtr_ctx.commit();

  // Only reached once the bug is fixed; documents the intended post-fix
  // behavior.
  assert(!failed && "insert into level-1 store should succeed once BUG 7 is "
                    "fixed (segment sourced from primary root page)");
}

// Regression test for the reload path of the segment fix. A store reopened via
// MultiColumnStore::load() (the reopen-table path -- ALTER, restart, second
// connection) must restore each store's
// m_primary_root_page_ref/m_segment_index, exactly as create() does. Otherwise
// the PRIMARY store's m_primary_root_page_ref stays INVALID_REF, insert() takes
// the "not the primary" branch, and loads a segment header from an invalid page
// -> InnoDB aborts (fil0fil.cc invalid page access). This is the single-store
// base-column case (one spec, one segment), which is what the SVECTOR column
// uses; it never touches a level>=1 store, so the original test above did not
// catch it.
void test_insert_after_reload_uses_primary_segment() {
  install_fake_abi();

  char err[512] = {};
  constexpr uint16_t kColLen = 16;

  Column::StorageRef storage_ref;
  {
    svector::MultiColumnStore mcs;
    std::vector<svector::Storage_spec> specs{{kColLen, "SVECTOR"}};
    bool failed = mcs.create(/*space_ref=*/1, /*trx_ref=*/1, specs,
                             /*num_segments=*/1, err, sizeof(err));
    assert(!failed && "create (single base-column store) should succeed");
    // Deliberately do NOT insert here: create() formats only the root page, no
    // data page. That way the FIRST insert after reload takes the pessimistic
    // "allocate a new data page" path -- the one that reads the segment header
    // and hits the bug. (An insert here would allocate the data page, so the
    // post-reload insert would reuse it and skip the segment path, masking the
    // bug -- which is exactly why ALTER/readme_examples, not the vector_insert
    // single-session tests, are what crashed in CI.)
    storage_ref = mcs.m_ref;
  }

  // Reopen from the persisted ref (fake pages survive in the shared std::map),
  // as if the table were closed and reopened.
  svector::MultiColumnStore reopened;
  std::vector<svector::Storage_spec> specs{{0, ""}}; // ColumnStorage::load form
  bool failed = reopened.load(storage_ref, specs, err, sizeof(err));
  assert(!failed && "load (reopen) should succeed");

  // First insert AFTER reload -> allocates the first data page -> reads the
  // segment header. Pre-fix m_primary_root_page_ref == INVALID_REF, so this
  // took the "not the primary" branch and loaded a segment from an invalid
  // page (InnoDB: fil0fil.cc invalid page access; fake harness: load error).
  std::vector<unsigned char> value2(kColLen, 0x2);
  Column::Data col_data2{value2.data(), kColLen};
  Column::Ref col_ref2 = Column::EMPTY_REF;
  vsql::preview_storage::MtrCtx mtr_ctx;
  auto mtr = mtr_ctx.start();
  failed = reopened.m_stores[0].insert(mtr, /*trx_ref=*/1, col_data2, col_ref2,
                                       err, sizeof(err));
  mtr_ctx.commit();
  assert(!failed && "insert after reload should succeed (primary segment "
                    "restored by load())");
}

} // namespace

int main() {
  test_insert_into_level1_store_hits_segment_bug();
  test_insert_after_reload_uses_primary_segment();
  std::printf("All storage tests passed.\n");
  return 0;
}
