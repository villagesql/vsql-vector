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

#include "page_reader.h"

#include <fstream>
#include <iostream>

namespace svector {
namespace tool {

PageReader::PageReader(const std::string &ibd_file_path)
    : ibd_file_path_(ibd_file_path),
      file_stream_(),
      page_size_(DEFAULT_PAGE_SIZE),
      error_("") {}

PageReader::~PageReader() { close(); }

bool PageReader::open() {
  if (is_open()) {
    error_ = "File already open";
    return false;
  }

  file_stream_.open(ibd_file_path_, std::ios::binary | std::ios::in);
  if (!file_stream_.is_open()) {
    error_ = "Failed to open file";
    return false;
  }

  // Get file size to validate
  file_stream_.seekg(0, std::ios::end);
  std::streampos file_size = file_stream_.tellg();
  file_stream_.seekg(0, std::ios::beg);

  if (file_stream_.fail()) {
    error_ = "Failed to get file size";
    file_stream_.close();
    return false;
  }

  if (file_size < static_cast<std::streampos>(page_size_)) {
    error_ = "File too small to contain even one page";
    file_stream_.close();
    return false;
  }

  return true;
}

void PageReader::close() {
  if (file_stream_.is_open()) {
    file_stream_.close();
  }
}

std::unique_ptr<std::vector<uint8_t>> PageReader::read_page(
    uint32_t page_number) {
  if (!is_open()) {
    error_ = "File not open";
    return nullptr;
  }

  // Calculate offset
  std::streampos offset = static_cast<std::streampos>(page_number) *
                          static_cast<std::streampos>(page_size_);

  // Seek to page
  file_stream_.seekg(offset);
  if (file_stream_.fail()) {
    error_ = "Failed to seek to page";
    file_stream_.clear();
    return nullptr;
  }

  // Allocate buffer
  auto buffer = std::make_unique<std::vector<uint8_t>>(page_size_);

  // Read page
  file_stream_.read(reinterpret_cast<char *>(buffer->data()), page_size_);
  if (file_stream_.fail() && !file_stream_.eof()) {
    error_ = "Failed to read page";
    file_stream_.clear();
    return nullptr;
  }

  std::streamsize bytes_read = file_stream_.gcount();
  if (bytes_read != static_cast<std::streamsize>(page_size_)) {
    error_ = "Incomplete page read";
    file_stream_.clear();
    return nullptr;
  }

  return buffer;
}

}  // namespace tool
}  // namespace svector
