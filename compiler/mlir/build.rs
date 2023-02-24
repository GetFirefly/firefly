extern crate cmake;

use std::env;
use std::path::PathBuf;

const ENV_LLVM_INCLUDE: &'static str = "DEP_LLVM_INCLUDE";
const ENV_LLVM_PREFIX: &'static str = "LLVM_PREFIX";
const ENV_LLVM_USE_SANITIZER: &'static str = "LLVM_USE_SANITIZER";
const ENV_LLVM_LINK_LLVM_DYLIB: &'static str = "LLVM_LINK_LLVM_DYLIB";
const ENV_FIREFLY_BUILD_TYPE: &'static str = "FIREFLY_BUILD_TYPE";

fn main() {
    let firefly_llvm_include_dir = env::var(ENV_LLVM_INCLUDE).unwrap();
    let llvm_prefix = PathBuf::from(env::var(ENV_LLVM_PREFIX).unwrap());
    let mlir_dir = llvm_prefix.join("lib/cmake/mlir");
    let llvm_dir = llvm_prefix.join("lib/cmake/llvm");
    let lit_dir = llvm_prefix.join("bin/llvm-lit");

    println!("cargo:rerun-if-changed=c_src");
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_INCLUDE);
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_PREFIX);
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_USE_SANITIZER);
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_LINK_LLVM_DYLIB);
    println!("cargo:rerun-if-env-changed={}", ENV_FIREFLY_BUILD_TYPE);

    // Build and link our MLIR dialects + extensions
    let mut config = cmake::Config::new("c_src");

    if let Ok(_) = which::which("ninja") {
        config.generator("Ninja");
    } else {
        warn(
            "Unable to locate Ninja, your CMake builds may take unncessarily long.\n\
             It is highly recommended that you install Ninja.",
        );
    }

    config
        .env("LLVM_PREFIX", &llvm_prefix)
        .define("LLVM_DIR", llvm_dir)
        .define("MLIR_DIR", mlir_dir)
        .define("LLVM_EXTERNAL_LIT", lit_dir)
        .cxxflag(&format!("-I{}", firefly_llvm_include_dir))
        .configure_arg("-Wno-dev");

    if env::var_os("LLVM_NDEBUG").is_some() {
        config.define("NDEBUG", "1");
    }

    if let Ok(sanitizer) = env::var(ENV_LLVM_USE_SANITIZER) {
        config.define("LLVM_USE_SANITIZER", sanitizer);
    }

    let output_path = config.build();
    let search_path = output_path.join("lib");

    println!("cargo:rustc-link-search=native={}", search_path.display());

    let build_type = env::var("FIREFLY_BUILD_TYPE").unwrap_or("static".to_owned());
    if build_type == "static" {
        println!("cargo:rustc-link-lib=static=CIRStatic");
    } else {
        println!("cargo:rustc-link-lib=dylib=CIRDynamic");
    }
}

fn warn(s: &str) {
    println!("cargo:warning={}", s);
}
