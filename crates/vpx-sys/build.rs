//! Finds libvpx with pkg-config, compiles the small C shim, and generates the bindings.

use std::env;
use std::path::PathBuf;

fn main() {
    let lib = pkg_config::Config::new()
        .atleast_version("1.8")
        .probe("vpx")
        .expect("libvpx not found via pkg-config (install it: `brew install libvpx` / `apt install libvpx-dev`)");
    let mut cc = cc::Build::new();
    cc.file("shim.c").include(".");
    for p in &lib.include_paths {
        cc.include(p);
    }
    cc.compile("fflv_vpx_shim");

    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .allowlist_function("vpx_.*|fflv_.*")
        .allowlist_type("vpx_.*|vp8e_.*|vp8_.*|VP8.*|vp9.*")
        .allowlist_var("VPX_.*|VP8.*|VP9.*")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .prepend_enum_name(false)
        .generate_comments(false)
        .derive_default(true)
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    for p in &lib.include_paths {
        builder = builder.clang_arg(format!("-I{}", p.display()));
    }
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    builder.generate().expect("bindgen failed on the libvpx headers").write_to_file(out).unwrap();
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=shim.c");
}
