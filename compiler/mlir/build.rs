extern crate cc;
extern crate walkdir;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use walkdir::{DirEntry, WalkDir};

const ENV_LLVM_PREFIX: &'static str = "LLVM_SYS_90_PREFIX";
const ENV_LLVM_BUILD_STATIC: &'static str = "LLVM_BUILD_STATIC";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let outdir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let llvm_prefix_env = env::var(ENV_LLVM_PREFIX).expect(ENV_LLVM_PREFIX);
    let llvm_prefix = PathBuf::from(llvm_prefix_env.as_str());

    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_PREFIX);
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_BUILD_STATIC);

    let mut cfg = cc::Build::new();
    cfg.warnings(false);

    // Compile our MLIR shims with the same flags as LLVM
    let cxxflags = env::var("DEP_LUMEN_LLVM_CORE_CXXFLAGS").unwrap();
    for flag in cxxflags.split(";") {
        cfg.flag(flag);
    }

    if env::var_os("LLVM_NDEBUG").is_some() {
        cfg.define("NDEBUG", None);
        cfg.debug(false);
    }

    rerun_if_changed_anything_in_dir(Path::new("c_src"));

    let cwd = env::current_dir().unwrap();
    let include_dir = outdir.join("include");
    let include_mlir_dir = include_dir.join("lumen/mlir");
    fs::create_dir_all(include_mlir_dir.as_path()).unwrap();
    for entry in fs::read_dir(cwd.join("c_src/include/lumen/mlir")).unwrap() {
        let entry = entry.unwrap();
        let file = entry.path();
        let basename = entry.file_name();
        fs::copy(file, include_mlir_dir.join(basename)).unwrap();
    }

    println!("cargo:include={}", include_dir.display());

    let lumen_llvm_include_dir = env::var("DEP_LUMEN_LLVM_CORE_INCLUDE").unwrap();
    cfg.file("c_src/MLIR.cpp")
       .file("c_src/Diagnostics.cpp")
       .file("c_src/ModuleReader.cpp")
       .file("c_src/ModuleWriter.cpp")
       .file("c_src/ConvertToLLVM.cpp")
       .include(llvm_prefix.join("include"))
       .include(lumen_llvm_include_dir)
       .include(include_dir)
       .cpp(true)
       .cpp_link_stdlib(None) // we handle this below
       .compile("lumen_mlir_core");

    link_libs(&[
        "MLIRAnalysis",
        "MLIRAffine",
        "MLIRCallInterfaces",
        "MLIRControlFlowInterfaces",
        "MLIRDerivedAttributeOpInterface",
        "MLIRDialect",
        "MLIREDSC",
        "MLIREDSCInterface",
        "MLIRExecutionEngine",
        "MLIRIR",
        "MLIRInferTypeOpInterface",
        "MLIRLLVMIR",
        "MLIRLLVMIRTransforms",
        "MLIRLoopAnalysis",
        "MLIRLoopOps",
        "MLIRLoopLikeInterface",
        "MLIROpenMP",
        "MLIRParser",
        "MLIRPass",
        "MLIRSideEffects",
        "MLIRStandardOps",
        "MLIRStandardToLLVM",
        "MLIRSupport",
        "MLIRTargetLLVMIR",
        "MLIRTargetLLVMIRModuleTranslation",
        "MLIRTransformUtils",
        "MLIRTransforms",
        "MLIRTranslateClParser",
        "MLIRTranslation",
    ]);

    let ldflags = env::var("DEP_LUMEN_LLVM_CORE_LDFLAGS").unwrap();
    for flag in ldflags.split(";") {
        println!("cargo:rustc-link-search=native={}", flag);
    }
}

pub fn output(cmd: &mut Command) -> String {
    let output = match cmd.stderr(Stdio::inherit()).output() {
        Ok(status) => status,
        Err(e) => fail(&format!(
            "failed to execute command: {:?}\nerror: {}",
            cmd, e
        )),
    };
    if !output.status.success() {
        panic!(
            "command did not execute successfully: {:?}\n\
             expected success, got: {}",
            cmd, output.status
        );
    }
    String::from_utf8(output.stdout).unwrap()
}

fn rerun_if_changed_anything_in_dir(dir: &Path) {
    let walker = WalkDir::new(dir).into_iter();
    for entry in walker.filter_entry(|e| !ignore_changes(e)) {
        let entry = entry.unwrap();
        let path = entry.path();
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn ignore_changes(entry: &DirEntry) -> bool {
    let ty = entry.file_type();
    if ty.is_dir() {
        return false;
    }
    let path = entry.path();
    if path.starts_with(".") {
        return true;
    }
    false
}

fn link_libs(libs: &[&str]) {
    if env::var_os(ENV_LLVM_BUILD_STATIC).is_none() {
        link_libs_dylib(libs);
    } else {
        link_libs_static(libs);
    }
}

#[inline]
fn link_libs_static(libs: &[&str]) {
    for lib in libs {
        link_lib_static(lib);
    }
}

#[inline]
fn link_libs_dylib(libs: &[&str]) {
    for lib in libs {
        link_lib_dylib(lib);
    }
}

#[inline]
fn link_lib_static(lib: &str) {
    println!("cargo:rustc-link-lib=static={}", lib);
}

#[inline]
fn link_lib_dylib(lib: &str) {
    println!("cargo:rustc-link-lib=dylib={}", lib);
}

fn fail(s: &str) -> ! {
    panic!("\n{}\n\nbuild script failed, must exit now", s)
}
