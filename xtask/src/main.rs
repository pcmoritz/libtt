use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

fn main() {
    if let Err(err) = run() {
        eprintln!("xtask failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("update-pjrt-bindings") => {
            if let Some(extra) = args.next() {
                return Err(format!("unexpected argument: {extra}").into());
            }
            update_pjrt_bindings()
        }
        Some("help") | None => {
            print_help();
            Ok(())
        }
        Some(cmd) => Err(format!("unknown xtask command: {cmd}").into()),
    }
}

fn print_help() {
    println!("Available commands:");
    println!("  cargo run --manifest-path xtask/Cargo.toml -- update-pjrt-bindings");
}

fn update_pjrt_bindings() -> Result<(), Box<dyn Error>> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("xtask manifest should live under the repo root")?
        .to_path_buf();

    let header = repo_root.join("third_party/openxla/xla/pjrt/c/pjrt_c_api.h");
    let layouts_header =
        repo_root.join("third_party/openxla/xla/pjrt/c/pjrt_c_api_layouts_extension.h");
    let output = repo_root.join("src/pjrt_bindings.rs");

    if !header.is_file() {
        return Err(format!("missing PJRT header: {}", header.display()).into());
    }
    if !layouts_header.is_file() {
        return Err(format!("missing PJRT layouts header: {}", layouts_header.display()).into());
    }

    let bindings = bindgen::Builder::default()
        .header_contents(
            "pjrt_bindings_wrapper.h",
            "#include \"xla/pjrt/c/pjrt_c_api.h\"\n\
             #include \"xla/pjrt/c/pjrt_c_api_layouts_extension.h\"\n",
        )
        .clang_arg(format!(
            "-I{}",
            repo_root.join("third_party/openxla").display()
        ))
        .blocklist_type("PJRT_Error")
        .blocklist_type("PJRT_DeviceDescription")
        .blocklist_type("PJRT_TopologyDescription")
        .blocklist_type("PJRT_Memory")
        .blocklist_type("PJRT_Device")
        .blocklist_type("PJRT_Event")
        .blocklist_type("PJRT_Buffer")
        .blocklist_type("PJRT_Client")
        .blocklist_type("PJRT_Executable")
        .blocklist_type("PJRT_LoadedExecutable")
        .blocklist_type("PJRT_Layouts_MemoryLayout")
        .blocklist_type("PJRT_Layouts_SerializedLayout")
        .allowlist_item("PJRT_.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .layout_tests(false)
        .generate()
        .map_err(|_| "failed to generate PJRT bindings")?;

    let mut contents = String::from(
        "// Generated from third_party/openxla/xla/pjrt/c/pjrt_c_api.h and\n\
         // third_party/openxla/xla/pjrt/c/pjrt_c_api_layouts_extension.h.\n\
         // Regenerate with `cargo run --manifest-path xtask/Cargo.toml -- update-pjrt-bindings`.\n\n",
    );
    contents.push_str(&bindings.to_string());

    fs::write(&output, contents)?;
    println!("updated {}", output.display());
    Ok(())
}
