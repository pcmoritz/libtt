use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(libtt_mlir_frontend)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=mlir/CMakeLists.txt");
    println!("cargo:rerun-if-changed=mlir/tt_mlir_frontend.cc");
    println!("cargo:rerun-if-changed=scripts/setup_deps.sh");
    println!("cargo:rerun-if-changed=scripts/setup_deps_common.sh");
    println!("cargo:rerun-if-changed=scripts/setup_deps_llvm.sh");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_MLIR_FRONTEND");
    println!("cargo:rerun-if-env-changed=CMAKE_PREFIX_PATH");
    println!("cargo:rerun-if-env-changed=LIBTT_MLIR_PREFIX");

    if env::var_os("CARGO_FEATURE_MLIR_FRONTEND").is_none() {
        return;
    }

    if let Err(err) = try_build_mlir_frontend() {
        panic!("failed to build MLIR frontend: {err}");
    }
}

fn try_build_mlir_frontend() -> Result<(), String> {
    let cmake = find_tool("cmake").ok_or_else(|| "cmake not found in PATH".to_owned())?;
    let prefix_paths = prefix_paths();
    let prefix = prefix_paths
        .iter()
        .find(|path| {
            path.join("lib/cmake/mlir/MLIRConfig.cmake").is_file()
                && path
                    .join("include/stablehlo/dialect/StablehloOps.h")
                    .is_file()
        })
        .cloned()
        .ok_or_else(|| {
            format!(
                "no MLIR/StableHLO install found under {}",
                prefix_paths
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;

    let out_dir =
        PathBuf::from(env::var("OUT_DIR").map_err(|err| format!("OUT_DIR missing: {err}"))?);
    let build_dir = out_dir.join("mlir-build");
    let source_dir = PathBuf::from("mlir");

    run(Command::new(&cmake)
        .arg("-S")
        .arg(&source_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!("-DCMAKE_PREFIX_PATH={}", prefix.display())))?;
    run(Command::new(&cmake)
        .arg("--build")
        .arg(&build_dir)
        .arg("--target")
        .arg("tt_mlir_frontend"))?;

    let lib_dir = build_dir.join("lib");
    if !lib_dir.join(shared_lib_name("tt_mlir_frontend")).is_file() {
        return Err(format!(
            "expected shared library at {}",
            lib_dir.join(shared_lib_name("tt_mlir_frontend")).display()
        ));
    }

    println!("cargo:rustc-cfg=libtt_mlir_frontend");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=tt_mlir_frontend");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    Ok(())
}

fn run(command: &mut Command) -> Result<(), String> {
    let status = command
        .status()
        .map_err(|err| format!("failed to run {:?}: {err}", command))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("command {:?} failed with {status}", command))
    }
}

fn find_tool(name: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path).find_map(|entry| {
            let candidate = entry.join(name);
            candidate.is_file().then_some(candidate)
        })
    })
}

fn prefix_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(prefix) = env::var("LIBTT_MLIR_PREFIX") {
        let trimmed = prefix.trim();
        if !trimmed.is_empty() {
            paths.push(PathBuf::from(trimmed));
        }
    }
    if let Ok(raw) = env::var("CMAKE_PREFIX_PATH") {
        for part in raw.split([';', ':']) {
            let trimmed = part.trim();
            if !trimmed.is_empty() {
                paths.push(PathBuf::from(trimmed));
            }
        }
    }
    if let Ok(home) = env::var("HOME") {
        paths.push(Path::new(&home).join(".local/libtt-deps"));
    }
    paths
}

fn shared_lib_name(name: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{name}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{name}.dylib")
    } else {
        format!("lib{name}.so")
    }
}
