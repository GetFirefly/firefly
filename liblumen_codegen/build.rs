extern crate cc;
extern crate cmake;
extern crate which;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const ENV_LLVM_PREFIX: &'static str = "LLVM_SYS_90_PREFIX";

const MLIR_LINK_LIBRARIES: &'static [&'static str] = &[
    "MLIRAffineOps", 
    "MLIRAffineToStandard", 
    "MLIRAnalysis", 
    "MLIRDialect", 
    "MLIREDSC", 
    "MLIREDSCInterface", 
    "MLIRExecutionEngine", 
    "MLIRIR", 
    "MLIRLLVMIR", 
    "MLIRLoopOps", 
    "MLIRLoopToStandard", 
    "MLIRParser", 
    "MLIRPass", 
    "MLIRStandardOps", 
    "MLIRStandardToLLVM", 
    "MLIRSupport", 
    "MLIRTargetLLVMIR", 
    "MLIRTargetLLVMIRModuleTranslation", 
    "MLIRTransformUtils", 
    "MLIRTransforms", 
    "MLIRTranslateClParser", 
    "MLIRTranslation",
];

fn main() {
    // Emit custom cfg types:
    //     cargo:rustc-cfg=has_foo
    // Can then be used as `#[cfg(has_foo)]` when emitted

    // Emit custom env data:
    //     cargo:rustc-env=foo=bar
    // Can then be fetched with `env!("foo")`

    // LLVM
    /*
    let target = env::var("TARGET").expect("TARGET was not set");
    let host = env::var("HOST").expect("HOST was not set");
    let is_crossed = target != host;
    */

    let llvm_prefix = env::var_os(ENV_LLVM_PREFIX).expect(ENV_LLVM_PREFIX);

    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_PREFIX);

    let llvm_config = env::var_os("LLVM_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(&llvm_prefix)
                .join("bin/llvm-config")
                .to_path_buf()
        });
    println!("cargo:rerun-if-changed={}", llvm_config.display());
    println!("cargo:rerun-if-env-changed=LLVM_CONFIG");

    if let Err(_) = which::which("cmake") {
        fail(
            "Unable to locate CMake!\n\
            It is required for the build, make sure you have a recent version installed."
        );
    }

    let mut use_ninja = true;
    if let Err(_) = which::which("ninja") {
        use_ninja = false;
        warn(
            "Unable to locate Ninja, your CMake builds may take unncessarily long.\n\
            It is highly recommended that you install Ninja."
        );
    }

    println!("cargo:rerun-if-changed=use_ninja={}", use_ninja);

    let mut config = &mut cmake::Config::new("lib");
    if use_ninja {
        config = config.generator("Ninja");
    }
    let outdir = config
        .define("CMAKE_EXPORT_COMPILE_COMMANDS", "YES")
        .define("LLVM_PREFIX", llvm_prefix.clone())
        .env("LLVM_PREFIX", llvm_prefix)
        .always_configure(true)
        .no_build_target(true)
        .very_verbose(false)
        .build();

    let cwd = env::current_dir()
        .expect("unable to access current directory");

    let compile_commands_src = outdir
        .join("build")
        .join("compile_commands.json");
    let compile_commands_dest = cwd
        .join("lib")
        .join("compile_commands.json");

    fs::copy(compile_commands_src, compile_commands_dest)
        .expect("unable to copy compile_commands.json!");

    println!("cargo:rustc-link-lib=c++");
    //println!("cargo:rustc-link-lib=static=lldConfig");
    println!("cargo:rustc-link-search=native={}/build/lib", outdir.display());
    println!("cargo:rustc-link-lib=static=Lumen");
    for link_lib in MLIR_LINK_LIBRARIES {
        println!("cargo:rustc-link-lib=static={}", link_lib);
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

pub fn rerun_if_changed_anything_in_dir(dir: &Path) {
    let mut stack = dir
        .read_dir()
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| &*e.file_name() != ".git")
        .collect::<Vec<_>>();
    while let Some(entry) = stack.pop() {
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            stack.extend(path.read_dir().unwrap().map(|e| e.unwrap()));
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn warn(s: &str) {
    println!("cargo:warning={}", s);
}

fn fail(s: &str) -> ! {
    panic!("\n{}\n\nbuild script failed, must exit now", s)
}
