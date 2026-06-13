#include <cstdio>
#include <cstdlib>
#include <string>

#include "flatbuffers/base.h"
#include "flatbuffers/flatc.h"
#include "idl_gen_cpp.h"

namespace {

const char* g_program_name = nullptr;

void Warn(const flatbuffers::FlatCompiler* flatc, const std::string& warn, bool show_exe_name) {
    (void)flatc;
    if (show_exe_name) {
        std::fprintf(stderr, "%s: ", g_program_name);
    }
    std::fprintf(stderr, "\nwarning:\n  %s\n\n", warn.c_str());
}

void Error(const flatbuffers::FlatCompiler* flatc, const std::string& err, bool usage, bool show_exe_name) {
    if (show_exe_name) {
        std::fprintf(stderr, "%s: ", g_program_name);
    }
    if (usage && flatc) {
        std::fprintf(stderr, "%s\n", flatc->GetShortUsageString(g_program_name).c_str());
    }
    std::fprintf(stderr, "\nerror:\n  %s\n\n", err.c_str());
    std::exit(1);
}

}  // namespace

namespace flatbuffers {

void LogCompilerWarn(const std::string& warn) { Warn(nullptr, warn, true); }

void LogCompilerError(const std::string& err) { Error(nullptr, err, false, true); }

}  // namespace flatbuffers

int main(int argc, const char* argv[]) {
    g_program_name = argv[0];

    flatbuffers::FlatCompiler::InitParams params;
    params.warn_fn = Warn;
    params.error_fn = Error;

    flatbuffers::FlatCompiler flatc(params);
    flatc.RegisterCodeGenerator(
        flatbuffers::FlatCOption{"c", "cpp", "", "Generate C++ headers for tables/structs"},
        flatbuffers::NewCppCodeGenerator());

    const flatbuffers::FlatCOptions& options = flatc.ParseFromCommandLineArguments(argc, argv);
    return flatc.Compile(options);
}
