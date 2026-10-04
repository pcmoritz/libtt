#include <zstd.h>

#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>

// Usage: libtt_zstd_compressor [level [window_log]] < input > output
//
// A window log enables long-distance matching with that window. The
// runtime archive is mostly the SFPI toolchain, whose cc1, cc1plus and lto1
// share most of their code and whose driver and binutils executables appear
// twice; a long window finds those repeats. Keep the window log at most 27:
// libarchive extracts the archive with zstd's default decoder window limit,
// ZSTD_WINDOWLOG_LIMIT_DEFAULT, which is 27.
constexpr int kDecoderWindowLogLimit = 27;

int main(int argc, char** argv) {
  int level = 9;
  int window_log = 0;
  if (argc > 1) {
    level = std::atoi(argv[1]);
  }
  if (argc > 2) {
    window_log = std::atoi(argv[2]);
  }
  if (window_log > kDecoderWindowLogLimit) {
    std::cerr << "window log " << window_log << " exceeds the decoder's default limit "
              << kDecoderWindowLogLimit << "\n";
    return 1;
  }

  std::vector<char> input((std::istreambuf_iterator<char>(std::cin)),
                          std::istreambuf_iterator<char>());
  ZSTD_CCtx* context = ZSTD_createCCtx();
  if (context == nullptr) {
    std::cerr << "ZSTD_createCCtx failed\n";
    return 1;
  }
  size_t status = ZSTD_CCtx_setParameter(context, ZSTD_c_compressionLevel, level);
  if (!ZSTD_isError(status) && window_log > 0) {
    status = ZSTD_CCtx_setParameter(context, ZSTD_c_enableLongDistanceMatching, 1);
    if (!ZSTD_isError(status)) {
      status = ZSTD_CCtx_setParameter(context, ZSTD_c_windowLog, window_log);
    }
  }
  std::vector<char> output(ZSTD_compressBound(input.size()));
  if (!ZSTD_isError(status)) {
    status = ZSTD_compress2(context, output.data(), output.size(), input.data(), input.size());
  }
  ZSTD_freeCCtx(context);
  if (ZSTD_isError(status)) {
    std::cerr << ZSTD_getErrorName(status) << "\n";
    return 1;
  }

  std::cout.write(output.data(), static_cast<std::streamsize>(status));
  return std::cout ? 0 : 1;
}
