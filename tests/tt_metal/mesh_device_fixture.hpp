// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include "rules_cc/cc/runfiles/runfiles.h"

namespace tt::tt_metal::distributed::test {

// Opens the system mesh for each test, like tt-metal's GenericMeshDeviceFixture.
class GenericMeshDeviceFixture : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(std::nullopt)); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // Absolute path of a kernel source under tests/tt_metal/kernels.
    static std::string kernel_path(const std::string& name) {
        std::string error;
        std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
            rules_cc::cc::runfiles::Runfiles::CreateForTest(&error));
        EXPECT_NE(runfiles, nullptr) << error;
        return runfiles ? runfiles->Rlocation("_main/tests/tt_metal/kernels/" + name) : name;
    }

    // A workload that runs one data movement kernel on core (0, 0) of every device.
    MeshWorkload single_core_workload(const std::string& kernel, const std::vector<uint32_t>& runtime_args = {}) {
        Program program = CreateProgram();
        const CoreCoord core{0, 0};
        const KernelHandle handle = CreateKernel(
            program,
            kernel_path(kernel),
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program, handle, core, runtime_args);
        MeshWorkload workload;
        workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
        return workload;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

}  // namespace tt::tt_metal::distributed::test
