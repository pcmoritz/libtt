fn main() {
    println!("cargo:rustc-check-cfg=cfg(libtt_mlir_frontend)");
    println!("cargo:rerun-if-changed=build.rs");
}
