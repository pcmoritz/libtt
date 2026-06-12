#include <string>

#include "flatbuffers/idl.h"

namespace flatbuffers {

bool GenerateCppGRPC(const Parser&, const std::string&, const std::string&) { return false; }

bool GenerateGoGRPC(const Parser&, const std::string&, const std::string&) { return false; }

bool GenerateJavaGRPC(const Parser&, const std::string&, const std::string&) { return false; }

bool GeneratePythonGRPC(const Parser&, const std::string&, const std::string&) { return false; }

bool GenerateSwiftGRPC(const Parser&, const std::string&, const std::string&) { return false; }

bool GenerateTSGRPC(const Parser&, const std::string&, const std::string&) { return false; }

}  // namespace flatbuffers
