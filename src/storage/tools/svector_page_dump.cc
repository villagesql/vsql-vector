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

// SVECTOR Page Dump Tool
//
// Usage: svector_page_dump <ibd_file> <root_page_num> [options]
//
// This tool reads and displays SVECTOR storage pages from an InnoDB tablespace
// (.ibd) file. It can show the root page metadata and optionally traverse and
// display all data pages.
//
// Options:
//   -v, --verbose       Show detailed information
//   -r, --records       Show record data
//   -d, --data-page N   Show specific data page number
//   -a, --all-pages     Show all data pages (traverse from root)
//   -h, --help          Show this help message

#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "data_page_parser.h"
#include "page_reader.h"
#include "root_page_parser.h"

namespace svector {
namespace tool {

class SvectorPageDump {
 public:
  SvectorPageDump()
      : verbose_(false),
        show_records_(false),
        show_all_pages_(false),
        specific_data_page_(0),
        has_specific_data_page_(false),
        record_start_(0),
        record_count_(10) {}

  int run(int argc, char *argv[]) {
    // Show help if called with no arguments
    if (argc < 2) {
      print_help(argv[0]);
      return 0;
    }

    if (!parse_args(argc, argv)) {
      return 1;
    }

    if (show_help_) {
      print_help(argv[0]);
      return 0;
    }

    if (ibd_file_.empty() || root_page_num_ == 0) {
      std::cerr << "Error: IBD file and root page number are required\n\n";
      print_usage(argv[0]);
      return 1;
    }

    // Open IBD file
    PageReader reader(ibd_file_);
    if (!reader.open()) {
      std::cerr << "Error: " << reader.get_error() << "\n";
      return 1;
    }

    // Read and parse root page
    auto root_page_data = reader.read_page(root_page_num_);
    if (!root_page_data) {
      std::cerr << "Error reading root page: " << reader.get_error() << "\n";
      return 1;
    }

    RootPageParser::RootPageInfo root_info;
    std::string error;
    if (!RootPageParser::parse(*root_page_data, root_info, error)) {
      std::cerr << "Error parsing root page: " << error << "\n";
      return 1;
    }

    // Display root page
    std::cout << "IBD File: " << ibd_file_ << "\n";
    std::cout << "Root Page Number: " << root_page_num_ << "\n\n";
    RootPageParser::display(root_info, verbose_);

    // Show specific data page if requested
    if (has_specific_data_page_) {
      std::cout << "\n" << std::string(80, '=') << "\n\n";
      if (!show_data_page(reader, specific_data_page_, root_info.column_size)) {
        return 1;
      }
    }

    // Show all data pages if requested
    if (show_all_pages_) {
      std::cout << "\n" << std::string(80, '=') << "\n";
      std::cout << "Traversing all data pages...\n";
      std::cout << std::string(80, '=') << "\n\n";

      // Start from head of all-pages list
      uint32_t page_num = root_info.all_slot_head;
      int count = 0;
      const int max_pages = 100;  // Safety limit

      while (page_num != RootPage::NULL_FREE_PAGE_REF && count < max_pages) {
        if (!show_data_page(reader, page_num, root_info.column_size)) {
          std::cerr << "Error displaying data page " << page_num << "\n";
          break;
        }

        // Read page to get next link
        auto page_data = reader.read_page(page_num);
        if (!page_data) {
          std::cerr << "Error reading page " << page_num << ": "
                    << reader.get_error() << "\n";
          break;
        }

        DataPageParser::DataPageInfo page_info;
        if (!DataPageParser::parse(*page_data, root_info.column_size, page_info,
                                   error)) {
          std::cerr << "Error parsing page " << page_num << ": " << error
                    << "\n";
          break;
        }

        // Move to next page
        page_num = page_info.next_free_page;
        count++;

        if (page_num != RootPage::NULL_FREE_PAGE_REF) {
          std::cout << "\n" << std::string(80, '-') << "\n\n";
        }
      }

      if (count >= max_pages) {
        std::cout << "\nReached safety limit of " << max_pages << " pages\n";
      }
    }

    return 0;
  }

 private:
  bool parse_args(int argc, char *argv[]) {
    if (argc < 2) {
      return false;
    }

    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];

      if (arg == "-h" || arg == "--help") {
        show_help_ = true;
        return true;
      } else if (arg == "-v" || arg == "--verbose") {
        verbose_ = true;
      } else if (arg == "-r" || arg == "--records") {
        show_records_ = true;
        // Check if next arg is a number (start position)
        if (i + 1 < argc && argv[i + 1][0] != '-') {
          try {
            record_start_ = std::stoul(argv[i + 1]);
            ++i;
            // Check if there's another number (count)
            if (i + 1 < argc && argv[i + 1][0] != '-') {
              try {
                record_count_ = std::stoul(argv[i + 1]);
                ++i;
              } catch (...) {
                // Not a number, continue
              }
            }
          } catch (...) {
            // Not a number, continue with defaults
          }
        }
      } else if (arg == "-a" || arg == "--all-pages") {
        show_all_pages_ = true;
      } else if (arg == "-d" || arg == "--data-page") {
        if (i + 1 >= argc) {
          std::cerr << "Error: " << arg << " requires a page number\n";
          return false;
        }
        specific_data_page_ = std::stoul(argv[++i]);
        has_specific_data_page_ = true;
      } else if (arg[0] == '-') {
        std::cerr << "Error: Unknown option: " << arg << "\n";
        return false;
      } else {
        // Positional arguments
        if (ibd_file_.empty()) {
          ibd_file_ = arg;
        } else if (root_page_num_ == 0) {
          root_page_num_ = std::stoul(arg);
        } else {
          std::cerr << "Error: Too many positional arguments\n";
          return false;
        }
      }
    }

    return true;
  }

  void print_usage(const char *prog_name) {
    std::cerr << "Usage: " << prog_name
              << " <ibd_file> <root_page_num> [options]\n";
    std::cerr << "Try '" << prog_name << " --help' for more information.\n";
  }

  void print_help(const char *prog_name) {
    std::cout << "SVECTOR Page Dump Tool\n\n";
    std::cout << "Usage: " << prog_name
              << " <ibd_file> <root_page_num> [options]\n\n";
    std::cout << "This tool reads and displays SVECTOR storage pages from an "
                 "InnoDB tablespace\n";
    std::cout << "(.ibd) file. It can show the root page metadata and "
                 "optionally traverse and\n";
    std::cout << "display all data pages.\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  <ibd_file>         Path to the .ibd file\n";
    std::cout << "  <root_page_num>    Root page number in the tablespace\n\n";
    std::cout << "Options:\n";
    std::cout << "  -v, --verbose         Show detailed information\n";
    std::cout << "  -r, --records [S] [N] Show record data\n";
    std::cout << "                        S = start position (default: 0)\n";
    std::cout
        << "                        N = number of records (default: 10)\n";
    std::cout << "  -d, --data-page N     Show specific data page number\n";
    std::cout << "  -a, --all-pages       Show all data pages (traverse from "
                 "root)\n";
    std::cout << "  -h, --help            Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  # Show root page only\n";
    std::cout << "  " << prog_name << " table.ibd 4\n\n";
    std::cout << "  # Show root page with verbose output\n";
    std::cout << "  " << prog_name << " table.ibd 4 -v\n\n";
    std::cout << "  # Show specific data page with first 10 records\n";
    std::cout << "  " << prog_name << " table.ibd 4 -d 5 -r\n\n";
    std::cout << "  # Show records starting at position 20, show 5 records\n";
    std::cout << "  " << prog_name << " table.ibd 4 -d 5 -r 20 5\n\n";
    std::cout << "  # Show records starting at position 50\n";
    std::cout << "  " << prog_name << " table.ibd 4 -d 5 -r 50\n\n";
    std::cout << "  # Traverse and show all data pages\n";
    std::cout << "  " << prog_name << " table.ibd 4 -a -v\n\n";
  }

  bool show_data_page(PageReader &reader, uint32_t page_num,
                      uint16_t column_size) {
    auto page_data = reader.read_page(page_num);
    if (!page_data) {
      std::cerr << "Error reading data page: " << reader.get_error() << "\n";
      return false;
    }

    DataPageParser::DataPageInfo page_info;
    std::string error;
    if (!DataPageParser::parse(*page_data, column_size, page_info, error)) {
      std::cerr << "Error parsing data page: " << error << "\n";
      return false;
    }

    std::cout << "Data Page Number: " << page_num << "\n\n";
    DataPageParser::display(page_info, verbose_, show_records_, record_start_,
                            record_count_);
    return true;
  }

  std::string ibd_file_;
  uint32_t root_page_num_ = 0;
  bool verbose_;
  bool show_records_;
  bool show_all_pages_;
  uint32_t specific_data_page_;
  bool has_specific_data_page_;
  uint32_t record_start_;
  uint32_t record_count_;
  bool show_help_ = false;
};

}  // namespace tool
}  // namespace svector

int main(int argc, char *argv[]) {
  svector::tool::SvectorPageDump tool;
  return tool.run(argc, argv);
}
