extern crate cc;

use std::env;
use std::path::{Path,PathBuf};
use std::process::{Command, Stdio};

fn main() {
    // Emit custom cfg types:
    //     cargo:rustc-cfg=has_foo
    // Can then be used as `#[cfg(has_foo)]` when emitted

    // Emit custom env data:
    //     cargo:rustc-env=foo=bar
    // Can then be fetched with `env!("foo")`

    // LLVM
    if env::var_os("RUST_CHECK").is_some() {
        // If we're just running `check`, there's no need for LLVM to be built.
        println!("cargo:rerun-if-env-changed=RUST_CHECK");
        return;
    }

    let home = env::var_os("HOME").map(PathBuf::from).expect("HOME was not set");
    let target = env::var("TARGET").expect("TARGET was not set");
    let host = env::var("HOST").expect("HOST was not set");
    let is_crossed = target != host;

    let llvm_prefix = env::var_os("LLVM_SYS_70_PREFIX")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let mut cmd = Command::new("llvmenv");
            cmd.arg("prefix");
            let prefix = output(&mut cmd);
            PathBuf::from(prefix.trim_end())
        });
    println!("cargo:rerun-if-env-changed=LLVM_SYS_70_PREFIX");

    let llvm_include = Path::new(&llvm_prefix).join("llvm");
    let lld_include = Path::new(&llvm_prefix).join("lld");

    let llvm_config = env::var_os("LLVM_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(&llvm_prefix).join("bin/llvm-config").to_path_buf()
        });

    println!("cargo:rerun-if-changed={}", llvm_config.display());
    println!("cargo:rerun-if-env-changed=LLVM_CONFIG");

    let mut cmd = Command::new(&llvm_config);
    cmd.arg("--cxxflags");
    let cxxflags = output(&mut cmd);
    let mut build = cc::Build::new();
    for flag in cxxflags.split_whitespace() {
        // Ignore flags like `-m64` when we're doing a cross build
        if is_crossed && flag.starts_with("-m") {
            continue;
        }

        // -Wdate-time is not supported by the netbsd cross compiler
        if is_crossed && target.contains("netbsd") && flag.contains("date-time") {
            continue;
        }
        build.flag(flag);
    }

    rerun_if_changed_anything_in_dir(Path::new("c_src"));
    build.file("c_src/LLDWrapper.cpp")
         .include(llvm_include)
         .include(lld_include)
         .cpp(true)
         .cpp_link_stdlib(None)
         .compile("liblumen_lld");

    // LLD specific flags, needed in order to use lld functions via FFI
    //println!("cargo:rustc-link-lib=static=lldConfig");
    //println!("cargo:rustc-link-lib=static=lldELF");
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
    let mut stack = dir.read_dir()
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

fn fail(s: &str) -> ! {
    println!("\n\n{}\n\n", s);
    std::process::exit(1);
}
