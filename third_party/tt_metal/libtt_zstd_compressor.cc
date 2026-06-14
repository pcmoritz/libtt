#include <zstd.h>

#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
  int level = 19;
  if (argc > 1) {
    level = std::atoi(argv[1]);
  }

  std::vector<char> input((std::istreambuf_iterator<char>(std::cin)),
                          std::istreambuf_iterator<char>());
  const size_t bound = ZSTD_compressBound(input.size());
  std::vector<char> output(bound);
  const size_t compressed_size =
      ZSTD_compress(output.data(), output.size(), input.data(), input.size(), level);
  if (ZSTD_isError(compressed_size)) {
    std::cerr << ZSTD_getErrorName(compressed_size) << "\n";
    return 1;
  }

  std::cout.write(output.data(), static_cast<std::streamsize>(compressed_size));
  return std::cout ? 0 : 1;
}
