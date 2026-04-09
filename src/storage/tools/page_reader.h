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

#ifndef VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_PAGE_READER_H_
#define VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_PAGE_READER_H_

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace svector {
namespace tool {

// Simple page reader for reading raw pages from IBD files
// This is a standalone utility that doesn't depend on mysqld
class PageReader {
 public:
  // Standard InnoDB page size
  static constexpr uint32_t DEFAULT_PAGE_SIZE = 16384;

  // Page header offsets (from InnoDB format)
  static constexpr uint32_t FIL_PAGE_OFFSET = 4;   // Page number
  static constexpr uint32_t FIL_PAGE_TYPE = 24;    // Page type
  static constexpr uint32_t FIL_PAGE_DATA = 38;    // Start of page data
  static constexpr uint32_t FIL_PAGE_END_LSN = 8;  // Trailer offset from end

  // Constructor
  explicit PageReader(const std::string &ibd_file_path);

  // Destructor
  ~PageReader();

  // Open the IBD file
  bool open();

  // Close the IBD file
  void close();

  // Read a page by page number
  // Returns nullptr on error
  std::unique_ptr<std::vector<uint8_t>> read_page(uint32_t page_number);

  // Get page size
  uint32_t get_page_size() const { return page_size_; }

  // Get error message
  const std::string &get_error() const { return error_; }

  // Check if file is open
  bool is_open() const { return file_stream_.is_open(); }

 private:
  std::string ibd_file_path_;
  std::ifstream file_stream_;
  uint32_t page_size_;
  std::string error_;
};

}  // namespace tool
}  // namespace svector

#endif  // VILLAGESQL_EXAMPLES_VSQL_SVECTOR_SRC_STORAGE_TOOLS_PAGE_READER_H_
