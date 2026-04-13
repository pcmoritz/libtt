use crate::device::Device;
use crate::dispatch::{MathFidelity, Program};
use crate::dram::DType;
use crate::hw::TensixL1;
use crate::log::log;
use std::collections::{BTreeMap, HashMap, hash_map::DefaultHasher};
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

const PCIE_NOC_X: u8 = 19;
const PCIE_NOC_Y: u8 = 24;

const INCLUDE_PATHS: &[&str] = &[
    "tt_metal/hw/inc",
    "tt_metal/hostdevcommon/api",
    "tt_metal/api",
    "tt_metal/include",
    "tt_metal/hw/inc/internal/tt-1xx",
    "tt_metal/hw/inc/internal/tt-1xx/blackhole",
    "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc",
    "tt_metal/hw/ckernels/blackhole/metal/llk_io",
    "tt_metal/hw/ckernels/blackhole/metal/common",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu",
    "tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc",
    "tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib",
    "runtime/sfpi/include",
];

const CFLAGS: &[&str] = &[
    "-std=c++17",
    "-flto=auto",
    "-ffast-math",
    "-fno-exceptions",
    "-fno-use-cxa-atexit",
];

const LFLAGS: &[&str] = &[
    "-Wl,-z,max-page-size=16",
    "-Wl,-z,common-page-size=16",
    "-nostartfiles",
];

const PERF_COUNTER_DEFINES: &[&str] = &["-DPROFILE_PERF_COUNTERS=0x3f"];

const CQ_SOURCES: &[(&str, &str, u8)] = &[
    ("prefetch_brisc", "cq_prefetch.cpp", 0),
    ("dispatch_brisc", "cq_dispatch.cpp", 1),
    ("dispatch_s_ncrisc", "cq_dispatch_subordinate.cpp", 1),
];

const FW_TARGETS: &[FwTarget] = &[
    FwTarget {
        src_name: "brisc.cc",
        target: "brisc",
        target_defs: &[
            "-DCOMPILE_FOR_BRISC",
            "-DPROCESSOR_INDEX=0",
            "-DNOC_INDEX=1",
            "-DNOC_MODE=0",
        ],
        mcpu: &["-mcpu=tt-bh", "-fno-tree-loop-distribute-patterns"],
        opt: "-Os",
        extra_objs: &["noc.o"],
    },
    FwTarget {
        src_name: "ncrisc.cc",
        target: "ncrisc",
        target_defs: &[
            "-DCOMPILE_FOR_NCRISC",
            "-DPROCESSOR_INDEX=1",
            "-DNOC_INDEX=0",
            "-DNOC_MODE=0",
        ],
        mcpu: &["-mcpu=tt-bh", "-fno-tree-loop-distribute-patterns"],
        opt: "-Os",
        extra_objs: &[],
    },
    FwTarget {
        src_name: "trisc.cc",
        target: "trisc0",
        target_defs: &[
            "-DCOMPILE_FOR_TRISC=0",
            "-DPROCESSOR_INDEX=2",
            "-DUCK_CHLKC_UNPACK",
            "-DNAMESPACE=chlkc_unpack",
        ],
        mcpu: &["-mcpu=tt-bh-tensix"],
        opt: "-O3",
        extra_objs: &[],
    },
    FwTarget {
        src_name: "trisc.cc",
        target: "trisc1",
        target_defs: &[
            "-DCOMPILE_FOR_TRISC=1",
            "-DPROCESSOR_INDEX=3",
            "-DUCK_CHLKC_MATH",
            "-DNAMESPACE=chlkc_math",
        ],
        mcpu: &["-mcpu=tt-bh-tensix"],
        opt: "-O3",
        extra_objs: &[],
    },
    FwTarget {
        src_name: "trisc.cc",
        target: "trisc2",
        target_defs: &[
            "-DCOMPILE_FOR_TRISC=2",
            "-DPROCESSOR_INDEX=4",
            "-DUCK_CHLKC_PACK",
            "-DNAMESPACE=chlkc_pack",
        ],
        mcpu: &["-mcpu=tt-bh-tensix"],
        opt: "-O3",
        extra_objs: &[],
    },
];

static FIRMWARE_CACHE: OnceLock<Mutex<HashMap<String, FirmwareCacheEntry>>> = OnceLock::new();
static KERNEL_CACHE: OnceLock<Mutex<HashMap<String, KernelCacheEntry>>> = OnceLock::new();

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PTLoad {
    pub paddr: u32,
    pub data: Vec<u8>,
    pub memsz: u32,
    pub flags: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledKernel {
    pub xip: Vec<u8>,
    pub xip_text_bytes: usize,
    pub disassembly: String,
    pub elf_bytes: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledFirmware {
    pub elf_bytes: Vec<u8>,
    pub segments: Vec<PTLoad>,
    pub scratch_base: u32,
}

impl CompiledFirmware {
    pub fn text_base(&self) -> Option<u32> {
        self.segments.first().map(|segment| segment.paddr)
    }
}

#[derive(Clone)]
struct FirmwareCacheEntry {
    result: HashMap<String, CompiledFirmware>,
}

#[derive(Clone)]
struct KernelCacheEntry {
    result: CompiledKernel,
}

#[derive(Clone, Copy)]
struct FwTarget {
    src_name: &'static str,
    target: &'static str,
    target_defs: &'static [&'static str],
    mcpu: &'static [&'static str],
    opt: &'static str,
    extra_objs: &'static [&'static str],
}

#[derive(Clone, Copy)]
struct SectionHeader {
    sh_type: u32,
    sh_flags: u32,
    sh_addr: u32,
    sh_offset: u32,
    sh_size: u32,
    sh_link: u32,
    sh_info: u32,
    sh_entsize: u32,
}

#[derive(Clone, Copy)]
struct Symbol {
    st_value: u32,
}

pub struct Compiler {
    cc: PathBuf,
    objcopy: PathBuf,
    includes: Vec<String>,
    kernel_defines: Vec<String>,
    firmware: HashMap<String, CompiledFirmware>,
    profile: bool,
}

impl Compiler {
    pub fn new(
        num_dram_banks: usize,
        num_l1_banks: usize,
        prefetch_core: (u8, u8),
        dispatch_core: (u8, u8),
    ) -> io::Result<Self> {
        let cc = sfpi_dir().join("riscv-tt-elf-g++");
        if !cc.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "missing compiler: {}\nDownload toolchain to {}",
                    cc.display(),
                    deps_root().join("sfpi-toolchain").display()
                ),
            ));
        }

        let objcopy = sfpi_dir().join("riscv-tt-elf-objcopy");
        if !objcopy.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("missing objcopy: {}", objcopy.display()),
            ));
        }

        let device_defines =
            device_defines(num_dram_banks, num_l1_banks, prefetch_core, dispatch_core);
        let mut kernel_defines = vec![
            "-DTENSIX_FIRMWARE".to_owned(),
            "-DLOCAL_MEM_EN=0".to_owned(),
            "-DARCH_BLACKHOLE".to_owned(),
            "-DDISPATCH_MESSAGE_ADDR=0xFFB70438".to_owned(),
            "-DKERNEL_BUILD".to_owned(),
        ];
        kernel_defines.extend(device_defines.iter().cloned());

        let profile = profiler_enabled();
        let firmware = compile_firmware(
            num_dram_banks,
            num_l1_banks,
            prefetch_core,
            dispatch_core,
            profile,
            &cc,
        )?;

        Ok(Self {
            cc,
            objcopy,
            includes: includes_with_dot(),
            kernel_defines,
            firmware,
            profile,
        })
    }

    pub fn for_device(device: &Device) -> io::Result<Self> {
        let prefetch = device.prefetch_core.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("device {} is missing prefetch core metadata", device.id),
            )
        })?;
        let dispatch = device.dispatch_core.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("device {} is missing dispatch core metadata", device.id),
            )
        })?;
        if device.active_dram_banks == 0 || device.all_worker_cores.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "device {} is missing Blackhole topology metadata",
                    device.id
                ),
            ));
        }
        Self::new(
            device.active_dram_banks,
            device.all_worker_cores.len(),
            (prefetch.x, prefetch.y),
            (dispatch.x, dispatch.y),
        )
    }

    pub fn disassemble(&self, kernel: &CompiledKernel) -> String {
        if !kernel.disassembly.is_empty() {
            return kernel.disassembly.clone();
        }
        kernel
            .elf_bytes
            .as_deref()
            .map(|elf| self.disassemble_elf(elf))
            .unwrap_or_default()
    }

    pub fn firmware(&self) -> &HashMap<String, CompiledFirmware> {
        &self.firmware
    }

    pub fn compile_dataflow(
        &self,
        src: &str,
        processor: &str,
        noc_index: Option<u8>,
    ) -> io::Result<CompiledKernel> {
        match processor {
            "brisc" => self.compile_dataflow_inner(
                src,
                "brisc",
                noc_index.unwrap_or(1),
                &[],
                &[],
                false,
                true,
            ),
            "ncrisc" => self.compile_dataflow_inner(
                src,
                "ncrisc",
                noc_index.unwrap_or(0),
                &[],
                &[],
                false,
                true,
            ),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("processor must be 'brisc' or 'ncrisc', got: {other}"),
            )),
        }
    }

    pub fn compile_compute(
        &self,
        src: &str,
        program: &Program,
    ) -> io::Result<(CompiledKernel, CompiledKernel, CompiledKernel)> {
        Ok((
            self.compile_trisc(src, 0, program)?,
            self.compile_trisc(src, 1, program)?,
            self.compile_trisc(src, 2, program)?,
        ))
    }

    pub fn compile_cq_kernels(&self) -> io::Result<BTreeMap<String, CompiledKernel>> {
        let cq_dir = repo_root().join("firmware").join("cq");
        let cq_includes = vec![
            cq_dir.display().to_string(),
            cq_dir.join("includes").display().to_string(),
        ];

        let mut kernels = BTreeMap::new();
        for &(name, src_name, noc_index) in CQ_SOURCES {
            let source = fs::read_to_string(cq_dir.join(src_name))?;
            let processor = if name.contains("ncrisc") {
                "ncrisc"
            } else {
                "brisc"
            };
            let kernel = self.compile_dataflow_inner(
                &source,
                processor,
                noc_index,
                &[],
                &cq_includes,
                true,
                false,
            )?;
            kernels.insert(name.to_owned(), kernel);
        }
        Ok(kernels)
    }

    fn compile_dataflow_inner(
        &self,
        src: &str,
        target: &str,
        noc_index: u8,
        extra_defines: &[String],
        extra_includes: &[String],
        xip_relocate: bool,
        profiler: bool,
    ) -> io::Result<CompiledKernel> {
        let mut defines = self.kernel_defines.clone();
        defines.push(format!("-DCOMPILE_FOR_{}", target.to_uppercase()));
        defines.push(format!(
            "-DPROCESSOR_INDEX={}",
            if target == "brisc" { 0 } else { 1 }
        ));
        defines.push(format!("-DNOC_INDEX={noc_index}"));
        defines.push("-DNOC_MODE=0".to_owned());
        defines.extend(extra_defines.iter().cloned());
        if self.profile && profiler {
            append_profile_defines(&mut defines);
        }
        let extra_objs = if target == "brisc" {
            vec![
                deps_root()
                    .join("lib")
                    .join("blackhole")
                    .join("noc.o")
                    .display()
                    .to_string(),
            ]
        } else {
            Vec::new()
        };
        self.build(
            src,
            target,
            &defines,
            &extra_objs,
            "-O2",
            false,
            extra_includes,
            xip_relocate,
            &Program::default(),
        )
    }

    fn compile_trisc(
        &self,
        src: &str,
        trisc_id: usize,
        program: &Program,
    ) -> io::Result<CompiledKernel> {
        let stage = ["unpack", "math", "pack"][trisc_id];
        let mut defines = self.kernel_defines.clone();
        defines.push(format!("-DCOMPILE_FOR_TRISC={trisc_id}"));
        defines.push(format!("-DPROCESSOR_INDEX={}", trisc_id + 2));
        defines.push(format!("-DUCK_CHLKC_{}", stage.to_uppercase()));
        defines.push(format!("-DNAMESPACE=chlkc_{stage}"));
        if self.profile {
            append_profile_defines(&mut defines);
        }
        self.build(
            src,
            &format!("trisc{trisc_id}"),
            &defines,
            &[],
            "-O3",
            true,
            &[],
            false,
            program,
        )
    }

    fn build(
        &self,
        kernel_source: &str,
        target: &str,
        defines: &[String],
        extra_objs: &[String],
        opt: &str,
        trisc: bool,
        extra_includes: &[String],
        xip_relocate: bool,
        program: &Program,
    ) -> io::Result<CompiledKernel> {
        let headers = ckernel_headers(program);
        let mut include_content = Vec::new();
        for dir in extra_includes {
            for file in sorted_files_recursive(Path::new(dir))? {
                include_content.extend(fs::read(file)?);
            }
        }

        let fw = self.firmware.get(target).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("missing compiled firmware for {target}"),
            )
        })?;

        let key = kernel_cache_key(
            kernel_source,
            target,
            defines,
            opt,
            trisc,
            xip_relocate,
            &headers,
            &fw.elf_bytes,
            &include_content,
        );
        if let Some(entry) = kernel_cache()
            .lock()
            .expect("kernel cache poisoned")
            .get(&key)
            .cloned()
        {
            return Ok(entry.result);
        }

        let mut mcpu = if trisc {
            vec![
                "-mcpu=tt-bh-tensix".to_owned(),
                "-mno-tt-tensix-optimize-replay".to_owned(),
            ]
        } else {
            vec![
                "-mcpu=tt-bh".to_owned(),
                "-mno-tt-tensix-optimize-replay".to_owned(),
                "-fno-tree-loop-distribute-patterns".to_owned(),
            ]
        };

        let fw_src = deps_root().join("firmware-src").join(if trisc {
            "trisck.cc".to_owned()
        } else {
            format!("{target}k.cc")
        });

        let mut includes = self.includes.clone();
        includes.extend(extra_includes.iter().map(|dir| format!("-I{dir}")));
        let (effective_opt, cflags, debug_flags) = kernel_build_flags(opt);

        let mut compile_args = Vec::new();
        compile_args.push(effective_opt.clone());
        compile_args.extend(cflags.clone());
        compile_args.extend(debug_flags);
        compile_args.push("-MMD".to_owned());
        compile_args.append(&mut mcpu.clone());
        compile_args.extend(includes);
        compile_args.extend(defines.iter().cloned());

        let fw_link_elf = std::cell::RefCell::new(None::<PathBuf>);
        let elf = compile_and_link(
            &self.cc,
            &fw_src,
            &compile_args,
            |_build| {
                let linker_script = deps_root()
                    .join("toolchain")
                    .join("blackhole")
                    .join(format!("kernel_{target}.ld"));
                let mut args = vec![effective_opt.clone()];
                args.extend(cflags.clone());
                args.extend(LFLAGS.iter().map(|value| (*value).to_owned()));
                args.append(&mut mcpu);
                args.push(format!("-T{}", linker_script.display()));
                args.push("-Wl,--emit-relocs".to_owned());
                args.push(format!(
                    "-Wl,--just-symbols={}",
                    fw_link_elf
                        .borrow()
                        .as_ref()
                        .expect("firmware link path must be prepared")
                        .display()
                ));
                args.push("out.o".to_owned());
                args.extend(extra_objs.iter().cloned());
                args.push(
                    deps_root()
                        .join("lib")
                        .join("blackhole")
                        .join("substitutes.o")
                        .display()
                        .to_string(),
                );
                args
            },
            &format!("tt-{target}-"),
            Some(|build: &Path| {
                *fw_link_elf.borrow_mut() = Some(self.weaken_fw_symbols(build, &fw.elf_bytes)?);
                fs::write(build.join("kernel_includes.hpp"), kernel_source)?;
                for (name, content) in &headers {
                    fs::write(build.join(name), content)?;
                }
                if trisc {
                    fs::write(build.join("defines_generated.h"), "")?;
                    for (stage, macro_name) in [
                        ("unpack", "TRISC_UNPACK"),
                        ("math", "TRISC_MATH"),
                        ("pack", "TRISC_PACK"),
                    ] {
                        let source = format!(
                            "#define {macro_name}\n#include \"defines_generated.h\"\n#include <kernel_includes.hpp>\n"
                        );
                        fs::write(build.join(format!("chlkc_{stage}.cpp")), source)?;
                    }
                }
                Ok(())
            }),
        )?;

        let (xip, xip_text_bytes) = pack_xip_elf(&elf, xip_relocate)?;
        let result = CompiledKernel {
            xip,
            xip_text_bytes,
            disassembly: self.disassemble_elf(&elf),
            elf_bytes: Some(elf),
        };
        kernel_cache()
            .lock()
            .expect("kernel cache poisoned")
            .insert(
                key,
                KernelCacheEntry {
                    result: result.clone(),
                },
            );
        Ok(result)
    }

    fn weaken_fw_symbols(&self, build: &Path, fw: &[u8]) -> io::Result<PathBuf> {
        let out = build.join("fw.elf");
        fs::write(&out, fw)?;
        run_command(
            &self.objcopy,
            &[
                "--localize-symbol=_start".to_owned(),
                "--localize-symbol=main".to_owned(),
                "--localize-symbol=exit".to_owned(),
                "--weaken".to_owned(),
                out.display().to_string(),
            ],
            build,
        )?;
        Ok(out)
    }

    fn disassemble_elf(&self, elf: &[u8]) -> String {
        let objdump = sfpi_dir().join("riscv-tt-elf-objdump");
        if !objdump.is_file() {
            return String::new();
        }

        let build = match unique_temp_dir("tt-disasm-") {
            Ok(dir) => dir,
            Err(_) => return String::new(),
        };
        let path = build.join("out.elf");
        if fs::write(&path, elf).is_err() {
            let _ = fs::remove_dir_all(&build);
            return String::new();
        }

        let output = Command::new(&objdump).arg("-d").arg(&path).output();
        let _ = fs::remove_dir_all(&build);
        let Ok(output) = output else {
            return String::new();
        };
        if !output.status.success() {
            return String::new();
        }
        normalize_objdump_addresses(&String::from_utf8_lossy(&output.stdout))
    }
}

fn compile_firmware(
    num_dram_banks: usize,
    num_l1_banks: usize,
    prefetch_core: (u8, u8),
    dispatch_core: (u8, u8),
    profile: bool,
    cc: &Path,
) -> io::Result<HashMap<String, CompiledFirmware>> {
    let fw_src_dir = repo_root().join("firmware");
    let unique_srcs = {
        let mut names = FW_TARGETS
            .iter()
            .map(|target| target.src_name)
            .collect::<Vec<_>>();
        names.sort_unstable();
        names.dedup();
        names
    };
    let key = firmware_cache_key(
        profile,
        num_dram_banks,
        num_l1_banks,
        prefetch_core,
        dispatch_core,
        &fw_src_dir,
        &unique_srcs,
    )?;
    if let Some(entry) = firmware_cache()
        .lock()
        .expect("firmware cache poisoned")
        .get(&key)
        .cloned()
    {
        return Ok(entry.result);
    }

    let mut common_defines = vec![
        "-DTENSIX_FIRMWARE".to_owned(),
        "-DFW_BUILD".to_owned(),
        "-DARCH_BLACKHOLE".to_owned(),
        "-DLOCAL_MEM_EN=0".to_owned(),
        "-DDISPATCH_MESSAGE_ADDR=0xFFB70438".to_owned(),
    ];
    common_defines.extend(device_defines(
        num_dram_banks,
        num_l1_banks,
        prefetch_core,
        dispatch_core,
    ));
    if profile {
        append_profile_defines(&mut common_defines);
    }

    let lib_dir = deps_root().join("lib").join("blackhole");
    let ld_dir = deps_root().join("toolchain").join("blackhole");

    let mut result = HashMap::new();
    for target in FW_TARGETS {
        let linker_script = ld_dir.join(format!("firmware_{}.ld", target.target));
        let src = fw_src_dir.join(target.src_name);
        let mut compile_args = vec![target.opt.to_owned()];
        compile_args.extend(CFLAGS.iter().map(|value| (*value).to_owned()));
        compile_args.extend(target.mcpu.iter().map(|value| (*value).to_owned()));
        compile_args.push("-mno-tt-tensix-optimize-replay".to_owned());
        compile_args.extend(common_defines.iter().cloned());
        compile_args.extend(target.target_defs.iter().map(|value| (*value).to_owned()));
        if profile && target.target == "trisc1" {
            compile_args.extend(PERF_COUNTER_DEFINES.iter().map(|value| (*value).to_owned()));
        }
        compile_args.extend(includes_without_dot());

        let elf = compile_and_link(
            cc,
            &src,
            &compile_args,
            |_| {
                let mut args = vec![target.opt.to_owned()];
                args.extend(CFLAGS.iter().map(|value| (*value).to_owned()));
                args.extend(LFLAGS.iter().map(|value| (*value).to_owned()));
                args.extend(target.mcpu.iter().map(|value| (*value).to_owned()));
                args.push("-mno-tt-tensix-optimize-replay".to_owned());
                args.push(format!("-T{}", linker_script.display()));
                args.push(lib_dir.join("tmu-crt0.o").display().to_string());
                args.push("out.o".to_owned());
                for extra in target.extra_objs {
                    args.push(lib_dir.join(extra).display().to_string());
                }
                args.push(lib_dir.join("substitutes.o").display().to_string());
                args
            },
            &format!("tt-fw-{}-", target.target),
            None::<fn(&Path) -> io::Result<()>>,
        )?;
        let segments = iter_pt_load(&elf)?;
        result.insert(
            target.target.to_owned(),
            CompiledFirmware {
                elf_bytes: elf,
                segments,
                scratch_base: init_scratch(target.target),
            },
        );
    }

    firmware_cache()
        .lock()
        .expect("firmware cache poisoned")
        .insert(
            key,
            FirmwareCacheEntry {
                result: result.clone(),
            },
        );
    Ok(result)
}

fn includes_without_dot() -> Vec<String> {
    let inc = deps_root().join("include");
    let mut includes = vec![format!("-I{}", inc.display())];
    includes.extend(
        INCLUDE_PATHS
            .iter()
            .map(|path| format!("-I{}", inc.join(path).display())),
    );
    includes
}

fn includes_with_dot() -> Vec<String> {
    let mut includes = vec!["-I.".to_owned()];
    includes.extend(includes_without_dot());
    includes
}

fn kernel_build_flags(opt: &str) -> (String, Vec<String>, Vec<String>) {
    (
        opt.to_owned(),
        CFLAGS.iter().map(|value| (*value).to_owned()).collect(),
        Vec::new(),
    )
}

fn device_defines(
    num_dram_banks: usize,
    num_l1_banks: usize,
    prefetch_core: (u8, u8),
    dispatch_core: (u8, u8),
) -> Vec<String> {
    let mut defs = vec![
        format!("-DNUM_DRAM_BANKS={num_dram_banks}"),
        format!("-DNUM_L1_BANKS={num_l1_banks}"),
        format!("-DPREFETCH_NOC_X={}", prefetch_core.0),
        format!("-DPREFETCH_NOC_Y={}", prefetch_core.1),
        format!("-DDISPATCH_NOC_X={}", dispatch_core.0),
        format!("-DDISPATCH_NOC_Y={}", dispatch_core.1),
        format!("-DPCIE_NOC_X={PCIE_NOC_X}"),
        format!("-DPCIE_NOC_Y={PCIE_NOC_Y}"),
        "-DIS_NOT_POW2_NUM_L1_BANKS=1".to_owned(),
    ];
    if num_dram_banks == 8 {
        defs.push("-DLOG_BASE_2_OF_NUM_DRAM_BANKS=3".to_owned());
    } else {
        defs.push("-DIS_NOT_POW2_NUM_DRAM_BANKS=1".to_owned());
    }
    defs
}

fn ckernel_headers(program: &Program) -> BTreeMap<String, String> {
    let mut formats = vec![DType::Float16B; 32];
    for cb in &program.cbs {
        if cb.index < formats.len() {
            formats[cb.index] = cb.dtype;
        }
    }
    let tile_sizes = formats
        .iter()
        .map(|format| format.tile_size())
        .collect::<Vec<_>>();

    let join = |values: &[DType]| -> String {
        values
            .iter()
            .map(|value| (*value as i32).to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let join_usize = |values: &[usize]| -> String {
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let repeat32 = |value: usize| -> String {
        std::iter::repeat_n(value, 32)
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let cbool = |value: bool| if value { "true" } else { "false" };
    let dst_sync = if program.dst_full_sync {
        "DstSync::SyncFull"
    } else {
        "DstSync::SyncHalf"
    };

    let data_fmt = |prefix: &str, ctype: &str| -> String {
        format!(
            "#pragma once\n#include <cstdint>\nconstexpr {ctype} {prefix}_src_format[32] = {{{}}};\nconstexpr {ctype} {prefix}_dst_format[32] = {{{}}};\n",
            join(&formats),
            join(&formats)
        )
    };
    let tile_dims = |prefix: &str| -> String {
        format!(
            "#pragma once\n#include <cstdint>\nconstexpr uint8_t {prefix}_tile_num_faces[32] = {{{}}};\nconstexpr uint8_t {prefix}_partial_face[32] = {{{}}};\nconstexpr uint8_t {prefix}_tile_face_r_dim[32] = {{{}}};\nconstexpr uint8_t {prefix}_narrow_tile[32] = {{{}}};\nconstexpr uint8_t {prefix}_tile_r_dim[32] = {{{}}};\nconstexpr uint8_t {prefix}_tile_c_dim[32] = {{{}}};\nconstexpr uint16_t {prefix}_tile_size[32] = {{{}}};\n",
            repeat32(4),
            repeat32(0),
            repeat32(16),
            repeat32(0),
            repeat32(32),
            repeat32(32),
            join_usize(&tile_sizes)
        )
    };

    BTreeMap::from([
        (
            "chlkc_unpack_data_format.h".to_owned(),
            data_fmt("unpack", "std::int32_t"),
        ),
        (
            "chlkc_pack_data_format.h".to_owned(),
            data_fmt("pack", "unsigned char"),
        ),
        ("chlkc_unpack_tile_dims.h".to_owned(), tile_dims("unpack")),
        ("chlkc_pack_tile_dims.h".to_owned(), tile_dims("pack")),
        (
            "chlkc_dst_accum_mode.h".to_owned(),
            format!(
                "#pragma once\nconstexpr bool DST_ACCUM_MODE = {};\n",
                cbool(program.dst_accum_mode)
            ),
        ),
        (
            "chlkc_dst_sync_mode.h".to_owned(),
            format!("#pragma once\n#define DST_SYNC_MODE {dst_sync}\n"),
        ),
        (
            "chlkc_math_fidelity.h".to_owned(),
            format!(
                "#pragma once\n#include <cstdint>\nconstexpr std::int32_t MATH_FIDELITY = {};\n",
                program.math_fidelity as i32
            ),
        ),
        (
            "chlkc_math_approx_mode.h".to_owned(),
            format!(
                "#pragma once\nconstexpr bool APPROX = {};\n",
                cbool(program.approx)
            ),
        ),
    ])
}

fn iter_pt_load(elf: &[u8]) -> io::Result<Vec<PTLoad>> {
    let e_phoff = read_u32(elf, 28)? as usize;
    let e_phentsize = read_u16(elf, 42)? as usize;
    let e_phnum = read_u16(elf, 44)? as usize;
    if e_phentsize < 32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad e_phentsize: {e_phentsize}"),
        ));
    }

    let mut segments = Vec::new();
    for index in 0..e_phnum {
        let offset = e_phoff + index * e_phentsize;
        let p_type = read_u32(elf, offset)?;
        if p_type != 1 {
            continue;
        }
        let p_offset = read_u32(elf, offset + 4)? as usize;
        let p_vaddr = read_u32(elf, offset + 8)?;
        let p_paddr = read_u32(elf, offset + 12)?;
        let p_filesz = read_u32(elf, offset + 16)? as usize;
        let p_memsz = read_u32(elf, offset + 20)?;
        let p_flags = read_u32(elf, offset + 24)?;
        let end = p_offset
            .checked_add(p_filesz)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "PT_LOAD overflow"))?;
        if end > elf.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "PT_LOAD range exceeds ELF size",
            ));
        }
        segments.push(PTLoad {
            paddr: if p_paddr != 0 { p_paddr } else { p_vaddr },
            data: elf[p_offset..end].to_vec(),
            memsz: p_memsz,
            flags: p_flags,
        });
    }
    Ok(segments)
}

fn xipify_riscv32_elf(elf: &[u8]) -> Vec<u8> {
    let mut data = elf.to_vec();
    let Ok(e_phoff) = read_u32(&data, 28).map(|v| v as usize) else {
        return elf.to_vec();
    };
    let Ok(e_shoff) = read_u32(&data, 32).map(|v| v as usize) else {
        return elf.to_vec();
    };
    let Ok(e_phentsize) = read_u16(&data, 42).map(|v| v as usize) else {
        return elf.to_vec();
    };
    let Ok(e_phnum) = read_u16(&data, 44).map(|v| v as usize) else {
        return elf.to_vec();
    };
    let Ok(e_shentsize) = read_u16(&data, 46).map(|v| v as usize) else {
        return elf.to_vec();
    };
    let Ok(e_shnum) = read_u16(&data, 48).map(|v| v as usize) else {
        return elf.to_vec();
    };
    if e_phentsize < 32 || e_shentsize < 40 {
        return elf.to_vec();
    }

    let r_riscv_hi20 = 26u32;
    let r_riscv_lo12_i = 27u32;
    let r_riscv_lo12_s = 28u32;

    let mut text_vaddr = None;
    let mut text_memsz = None;
    for index in 0..e_phnum {
        let offset = e_phoff + index * e_phentsize;
        let Ok(p_type) = read_u32(&data, offset) else {
            return elf.to_vec();
        };
        if p_type != 1 {
            continue;
        }
        let Ok(p_vaddr) = read_u32(&data, offset + 8) else {
            return elf.to_vec();
        };
        let Ok(p_memsz) = read_u32(&data, offset + 20) else {
            return elf.to_vec();
        };
        let Ok(p_flags) = read_u32(&data, offset + 24) else {
            return elf.to_vec();
        };
        if (p_flags & 1) != 0 {
            text_vaddr = Some(p_vaddr);
            text_memsz = Some(p_memsz);
            break;
        }
    }

    let (Some(text_vaddr), Some(text_memsz)) = (text_vaddr, text_memsz) else {
        return elf.to_vec();
    };
    let text_end = text_vaddr.saturating_add(text_memsz);
    let is_text = |addr: u32| text_vaddr <= addr && addr <= text_end;

    for rel_sec_idx in 0..e_shnum {
        let Some(rel_sec) = section_header(&data, e_shoff, e_shentsize, rel_sec_idx) else {
            return elf.to_vec();
        };
        if rel_sec.sh_type != 4 || rel_sec.sh_entsize < 12 || rel_sec.sh_size == 0 {
            continue;
        }
        let Some(target_sec) =
            section_header(&data, e_shoff, e_shentsize, rel_sec.sh_info as usize)
        else {
            continue;
        };
        if (target_sec.sh_flags & 0x2) == 0 || target_sec.sh_type == 8 {
            continue;
        }

        let mut hi_by_sym: HashMap<u32, Vec<(u32, i32)>> = HashMap::new();
        let mut lo_relocs = Vec::<(u32, u32, u32)>::new();
        for index in 0..(rel_sec.sh_size / rel_sec.sh_entsize) {
            let offset = rel_sec.sh_offset as usize + index as usize * rel_sec.sh_entsize as usize;
            let Ok(r_offset) = read_u32(&data, offset) else {
                return elf.to_vec();
            };
            let Ok(r_info) = read_u32(&data, offset + 4) else {
                return elf.to_vec();
            };
            let Ok(r_addend) = read_i32(&data, offset + 8) else {
                return elf.to_vec();
            };
            let r_type = r_info & 0xff;
            let r_sym = r_info >> 8;
            if r_type == r_riscv_hi20 {
                let Some(symbol) = symbol(
                    &data,
                    e_shoff,
                    e_shentsize,
                    rel_sec.sh_link as usize,
                    r_sym as usize,
                ) else {
                    continue;
                };
                if is_text(symbol.st_value) {
                    hi_by_sym
                        .entry(r_sym)
                        .or_default()
                        .push((r_offset, r_addend));
                }
            } else if r_type == r_riscv_lo12_i || r_type == r_riscv_lo12_s {
                lo_relocs.push((r_offset, r_sym, r_type));
            }
        }

        for items in hi_by_sym.values_mut() {
            items.sort_by_key(|item| item.0);
        }

        for (lo_offset, lo_sym, lo_type) in lo_relocs {
            let Some(hi_list) = hi_by_sym.get(&lo_sym) else {
                continue;
            };
            let mut hi_offset = hi_list[0].0;
            let mut hi_addend = hi_list[0].1;
            for &(candidate_offset, candidate_addend) in hi_list {
                if candidate_offset < lo_offset {
                    hi_offset = candidate_offset;
                    hi_addend = candidate_addend;
                } else {
                    break;
                }
            }

            let Some(symbol) = symbol(
                &data,
                e_shoff,
                e_shentsize,
                rel_sec.sh_link as usize,
                lo_sym as usize,
            ) else {
                continue;
            };
            let value = (symbol.st_value as i64 + i64::from(hi_addend) - i64::from(hi_offset))
                .rem_euclid(1i64 << 32) as u32;
            let Some(hi_file_offset) = section_file_offset(target_sec, hi_offset) else {
                continue;
            };
            let Ok(hi_insn) = read_u32(&data, hi_file_offset) else {
                continue;
            };
            if (hi_insn & 0x7f) != 0x37 {
                continue;
            }
            let rd = (hi_insn >> 7) & 0x1f;
            let new_hi =
                ((((value.wrapping_add(0x800)) >> 12) & 0x000f_ffff) << 12) | (rd << 7) | 0x17;
            if write_u32(&mut data, hi_file_offset, new_hi).is_err() {
                return elf.to_vec();
            }

            let Some(lo_file_offset) = section_file_offset(target_sec, lo_offset) else {
                continue;
            };
            let Ok(lo_insn) = read_u32(&data, lo_file_offset) else {
                continue;
            };
            let lo12 = value & 0x0fff;
            let new_lo = if lo_type == r_riscv_lo12_i {
                (lo_insn & 0x000f_ffff) | (lo12 << 20)
            } else {
                (lo_insn & !((0x7f << 25) | (0x1f << 7)))
                    | ((lo12 >> 5) << 25)
                    | ((lo12 & 0x1f) << 7)
            };
            if write_u32(&mut data, lo_file_offset, new_lo).is_err() {
                return elf.to_vec();
            }
        }
    }

    data
}

pub fn pack_xip_elf(elf: &[u8], xip_relocate: bool) -> io::Result<(Vec<u8>, usize)> {
    let elf = if xip_relocate {
        xipify_riscv32_elf(elf)
    } else {
        elf.to_vec()
    };
    let segments = iter_pt_load(&elf)?;
    if segments.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no PT_LOAD segments",
        ));
    }

    let mut l1 = segments
        .into_iter()
        .filter(|segment| {
            (segment.memsz != 0 || !segment.data.is_empty()) && segment.paddr < TensixL1::SIZE
        })
        .collect::<Vec<_>>();
    l1.sort_by_key(|segment| segment.paddr);
    if l1.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no L1 PT_LOAD segments",
        ));
    }

    let base = l1[0].paddr;
    let mut out = Vec::new();
    for segment in &l1 {
        let start = (segment.paddr - base) as usize;
        let end = start + usize::max(segment.memsz as usize, segment.data.len());
        if out.len() < end {
            out.resize(end, 0);
        }
        out[start..start + segment.data.len()].copy_from_slice(&segment.data);
    }
    let text = l1
        .iter()
        .find(|segment| (segment.flags & 1) != 0 && !segment.data.is_empty())
        .unwrap_or(&l1[0]);
    Ok((out, text.data.len()))
}

fn run_command(exe: &Path, args: &[String], cwd: &Path) -> io::Result<()> {
    log(format!(
        "compiler exec cwd={} cmd={}",
        cwd.display(),
        render_command(exe, args)
    ));
    let output = Command::new(exe).args(args).current_dir(cwd).output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "{} failed:\n{}",
            exe.file_name().and_then(OsStr::to_str).unwrap_or("command"),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(())
}

fn render_command(exe: &Path, args: &[String]) -> String {
    std::iter::once(exe.display().to_string())
        .chain(args.iter().map(|arg| shell_quote(arg)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(arg: &str) -> String {
    if !arg.is_empty()
        && arg
            .bytes()
            .all(|byte| matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'/' | b'.' | b'_' | b'-' | b'=' | b':' | b',' | b'+'))
    {
        return arg.to_owned();
    }
    format!("'{}'", arg.replace('\'', "'\\''"))
}

fn compile_and_link<P, L>(
    cc: &Path,
    src: &Path,
    compile_args: &[String],
    link_args: L,
    tmp_prefix: &str,
    prepare: Option<P>,
) -> io::Result<Vec<u8>>
where
    P: FnOnce(&Path) -> io::Result<()>,
    L: FnOnce(&Path) -> Vec<String>,
{
    let build = unique_temp_dir(tmp_prefix)?;
    let result = (|| {
        if let Some(prepare) = prepare {
            prepare(&build)?;
        }

        let mut compile = compile_args.to_vec();
        compile.push("-c".to_owned());
        compile.push("-o".to_owned());
        compile.push("out.o".to_owned());
        compile.push(src.display().to_string());
        run_command(cc, &compile, &build)?;

        let mut link = link_args(&build);
        link.push("-o".to_owned());
        link.push("out.elf".to_owned());
        run_command(cc, &link, &build)?;
        fs::read(build.join("out.elf"))
    })();
    let _ = fs::remove_dir_all(&build);
    result
}

fn profiler_enabled() -> bool {
    matches!(env::var("PROFILE").as_deref(), Ok("1"))
}

fn append_profile_defines(defines: &mut Vec<String>) {
    defines.push("-DPROFILE_KERNEL=1".to_owned());
    defines.push(format!(
        "-DPROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC={}",
        TensixL1::PROFILER_HOST_BUFFER_BYTES_PER_RISC
    ));
}

fn repo_root() -> &'static Path {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
        .as_path()
}

fn deps_root() -> PathBuf {
    repo_root().join("tt-metal-deps")
}

fn sfpi_dir() -> PathBuf {
    deps_root().join("sfpi-toolchain").join("bin")
}

fn init_scratch(target: &str) -> u32 {
    let base = TensixL1::KERNEL_CONFIG_BASE;
    match target {
        "brisc" => base,
        "ncrisc" => base + 0x2000,
        "trisc0" => base + 0x4000,
        "trisc1" => base + 0x5000,
        "trisc2" => base + 0x6000,
        _ => base,
    }
}

fn unique_temp_dir(prefix: &str) -> io::Result<PathBuf> {
    let base = env::temp_dir();
    let pid = std::process::id();
    for attempt in 0..1024u32 {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = base.join(format!("{prefix}{pid}-{nanos}-{attempt}"));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "failed to allocate unique temp dir",
    ))
}

fn sorted_files_recursive(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if !root.exists() {
        return Ok(files);
    }
    collect_files_recursive(root, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_files_recursive(root: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    if root.is_file() {
        files.push(root.to_path_buf());
        return Ok(());
    }

    let mut entries = fs::read_dir(root)?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_files_recursive(&path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

fn firmware_cache_key(
    profile: bool,
    num_dram_banks: usize,
    num_l1_banks: usize,
    prefetch_core: (u8, u8),
    dispatch_core: (u8, u8),
    fw_src_dir: &Path,
    unique_srcs: &[&str],
) -> io::Result<String> {
    let mut hasher = DefaultHasher::new();
    "fw-v3".hash(&mut hasher);
    profile.hash(&mut hasher);
    num_dram_banks.hash(&mut hasher);
    num_l1_banks.hash(&mut hasher);
    prefetch_core.hash(&mut hasher);
    dispatch_core.hash(&mut hasher);
    unique_srcs.hash(&mut hasher);
    for src in unique_srcs {
        src.hash(&mut hasher);
        fs::read(fw_src_dir.join(src))?.hash(&mut hasher);
    }
    Ok(format!("{:016x}", hasher.finish()))
}

fn kernel_cache_key(
    kernel_source: &str,
    target: &str,
    defines: &[String],
    opt: &str,
    trisc: bool,
    xip_relocate: bool,
    headers: &BTreeMap<String, String>,
    fw_elf: &[u8],
    include_content: &[u8],
) -> String {
    let mut hasher = DefaultHasher::new();
    "kern-v3".hash(&mut hasher);
    kernel_source.hash(&mut hasher);
    target.hash(&mut hasher);
    defines.hash(&mut hasher);
    opt.hash(&mut hasher);
    trisc.hash(&mut hasher);
    xip_relocate.hash(&mut hasher);
    headers.len().hash(&mut hasher);
    for (name, content) in headers {
        name.hash(&mut hasher);
        content.hash(&mut hasher);
    }
    fw_elf.hash(&mut hasher);
    include_content.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn read_u16(data: &[u8], offset: usize) -> io::Result<u16> {
    let bytes = data
        .get(offset..offset + 2)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "short read"))?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(data: &[u8], offset: usize) -> io::Result<u32> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "short read"))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_i32(data: &[u8], offset: usize) -> io::Result<i32> {
    Ok(read_u32(data, offset)? as i32)
}

fn write_u32(data: &mut [u8], offset: usize, value: u32) -> io::Result<()> {
    let bytes = data
        .get_mut(offset..offset + 4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "short write"))?;
    bytes.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn section_header(
    data: &[u8],
    e_shoff: usize,
    e_shentsize: usize,
    index: usize,
) -> Option<SectionHeader> {
    let offset = e_shoff.checked_add(index.checked_mul(e_shentsize)?)?;
    Some(SectionHeader {
        sh_type: read_u32(data, offset + 4).ok()?,
        sh_flags: read_u32(data, offset + 8).ok()?,
        sh_addr: read_u32(data, offset + 12).ok()?,
        sh_offset: read_u32(data, offset + 16).ok()?,
        sh_size: read_u32(data, offset + 20).ok()?,
        sh_link: read_u32(data, offset + 24).ok()?,
        sh_info: read_u32(data, offset + 28).ok()?,
        sh_entsize: read_u32(data, offset + 36).ok()?,
    })
}

fn symbol(
    data: &[u8],
    e_shoff: usize,
    e_shentsize: usize,
    symtab_idx: usize,
    sym_idx: usize,
) -> Option<Symbol> {
    let symtab = section_header(data, e_shoff, e_shentsize, symtab_idx)?;
    let entsize = usize::try_from(symtab.sh_entsize).ok()?;
    let offset = usize::try_from(symtab.sh_offset)
        .ok()?
        .checked_add(sym_idx.checked_mul(entsize)?)?;
    Some(Symbol {
        st_value: read_u32(data, offset + 4).ok()?,
    })
}

fn section_file_offset(section: SectionHeader, addr: u32) -> Option<usize> {
    let rel = addr.checked_sub(section.sh_addr)?;
    usize::try_from(section.sh_offset.checked_add(rel)?).ok()
}

fn normalize_objdump_addresses(disassembly: &str) -> String {
    let mut lines = Vec::new();
    for line in disassembly.lines() {
        if let Some((prefix, suffix)) = line.split_once(':') {
            let trimmed = prefix.trim_start();
            if !trimmed.is_empty() && trimmed.chars().all(|ch| ch.is_ascii_hexdigit()) {
                let indent_len = prefix.len() - trimmed.len();
                let indent = &prefix[..indent_len];
                let addr = trimmed.trim_start_matches('0');
                let addr = if addr.is_empty() { "0" } else { addr };
                lines.push(format!("{indent}{addr}:{suffix}"));
                continue;
            }
        }
        lines.push(line.to_owned());
    }
    lines.join("\n")
}

fn firmware_cache() -> &'static Mutex<HashMap<String, FirmwareCacheEntry>> {
    FIRMWARE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn kernel_cache() -> &'static Mutex<HashMap<String, KernelCacheEntry>> {
    KERNEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::CBConfig;

    #[test]
    fn device_defines_match_blackhole_shapes() {
        let p100 = device_defines(7, 120, (14, 2), (14, 3));
        assert!(p100.contains(&"-DNUM_DRAM_BANKS=7".to_owned()));
        assert!(p100.contains(&"-DNUM_L1_BANKS=120".to_owned()));
        assert!(p100.contains(&"-DIS_NOT_POW2_NUM_DRAM_BANKS=1".to_owned()));

        let p150 = device_defines(8, 140, (16, 2), (16, 3));
        assert!(p150.contains(&"-DLOG_BASE_2_OF_NUM_DRAM_BANKS=3".to_owned()));
    }

    #[test]
    fn ckernel_headers_reflect_program_formats() {
        let program = Program {
            cbs: vec![CBConfig {
                index: 1,
                dtype: DType::UInt8,
                tiles: 4,
            }],
            approx: true,
            dst_accum_mode: true,
            dst_full_sync: true,
            math_fidelity: MathFidelity::LoFi,
            ..Program::default()
        };

        let headers = ckernel_headers(&program);
        assert!(headers["chlkc_unpack_data_format.h"].contains("30"));
        assert!(headers["chlkc_dst_accum_mode.h"].contains("true"));
        assert!(headers["chlkc_dst_sync_mode.h"].contains("DstSync::SyncFull"));
        assert!(headers["chlkc_math_fidelity.h"].contains("0"));
        assert!(headers["chlkc_math_approx_mode.h"].contains("true"));
    }

    #[test]
    fn pack_xip_elf_merges_l1_segments() {
        let elf = synthetic_elf(&[
            (0x1000, 0x1000, &[1u8, 2, 3, 4][..], 8, 1),
            (0x1010, 0x1010, &[5u8, 6][..], 4, 0),
            (
                TensixL1::SIZE + 0x10,
                TensixL1::SIZE + 0x10,
                &[9u8][..],
                1,
                0,
            ),
        ]);

        let (xip, text_bytes) = pack_xip_elf(&elf, false).expect("pack_xip_elf");
        assert_eq!(text_bytes, 4);
        assert_eq!(&xip[..4], &[1, 2, 3, 4]);
        assert_eq!(&xip[0x10..0x12], &[5, 6]);
        assert_eq!(xip.len(), 0x14);
    }

    fn synthetic_elf(segments: &[(u32, u32, &[u8], u32, u32)]) -> Vec<u8> {
        let e_phoff = 52usize;
        let ph_size = 32usize;
        let data_offset = e_phoff + segments.len() * ph_size;
        let mut elf = vec![0u8; data_offset];
        elf[0..4].copy_from_slice(b"\x7fELF");
        elf[4] = 1;
        elf[5] = 1;
        elf[6] = 1;
        elf[16..18].copy_from_slice(&2u16.to_le_bytes());
        elf[18..20].copy_from_slice(&243u16.to_le_bytes());
        elf[20..24].copy_from_slice(&1u32.to_le_bytes());
        elf[28..32].copy_from_slice(&(e_phoff as u32).to_le_bytes());
        elf[40..42].copy_from_slice(&52u16.to_le_bytes());
        elf[42..44].copy_from_slice(&(ph_size as u16).to_le_bytes());
        elf[44..46].copy_from_slice(&(segments.len() as u16).to_le_bytes());

        let mut payload_offset = data_offset;
        for (index, (vaddr, paddr, data, memsz, flags)) in segments.iter().enumerate() {
            let ph = e_phoff + index * ph_size;
            elf[ph..ph + 4].copy_from_slice(&1u32.to_le_bytes());
            elf[ph + 4..ph + 8].copy_from_slice(&(payload_offset as u32).to_le_bytes());
            elf[ph + 8..ph + 12].copy_from_slice(&vaddr.to_le_bytes());
            elf[ph + 12..ph + 16].copy_from_slice(&paddr.to_le_bytes());
            elf[ph + 16..ph + 20].copy_from_slice(&(data.len() as u32).to_le_bytes());
            elf[ph + 20..ph + 24].copy_from_slice(&memsz.to_le_bytes());
            elf[ph + 24..ph + 28].copy_from_slice(&flags.to_le_bytes());
            elf.resize(payload_offset + data.len(), 0);
            elf[payload_offset..payload_offset + data.len()].copy_from_slice(data);
            payload_offset += data.len();
        }
        elf
    }
}
