use std::env;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const ENV_LLVM_PREFIX: &'static str = "DEP_LUMEN_LLVM_CORE_PREFIX";

fn main() {
    // Emit custom cfg types:
    //     cargo:rustc-cfg=has_foo
    // Can then be used as `#[cfg(has_foo)]` when emitted

    // Emit custom env data:
    //     cargo:rustc-env=foo=bar
    // Can then be fetched with `env!("foo")`

    let target = env::var("TARGET").unwrap();
    println!("TARGET: {}", &target);
    let llvm_prefix = PathBuf::from(env::var(ENV_LLVM_PREFIX).unwrap());
    println!("LLVM_PREFIX: {}", llvm_prefix.display());

    // Get demangled lang_start_internal name
    // First, get the location of the Rust sysroot for the current compiler
    let mut sysroot_cmd = Command::new("rustc");
    let mut sysroot_cmd = sysroot_cmd.args(&["--print", "sysroot"]);
    let sysroot = PathBuf::from(output(&mut sysroot_cmd).trim());
    println!("Found sysroot at {}", sysroot.display());
    // Search through all of the libs bundled with the toolchain for libstd-<hash>.rlib
    let toolchain_libs = sysroot.join("lib/rustlib").join(target).join("lib");
    println!("Searching for libstd rlib in {}", toolchain_libs.display());
    let libstd_rlib = toolchain_libs
        .read_dir()
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| {
            let path = e.path();
            let extension = path.extension().and_then(|p| p.to_str());
            if extension.is_none() {
                return false;
            }
            let filename = path.file_name().and_then(|p| p.to_str());
            if filename.is_none() {
                return false;
            }
            extension.unwrap() == "rlib" && filename.unwrap().starts_with("libstd-")
        })
        .take(1)
        .next()
        .map(|e| e.path().to_string_lossy().into_owned())
        .expect("unable to find libstd rlib in toolchain directory!");
    println!("Found libstd rlib: {}", &libstd_rlib);
    // Then, run llvm-nm on the rlib we found and dump all of the symbols it contains
    let llvm_nm = llvm_prefix.join("bin/llvm-nm");
    let mut nm_cmd = Command::new(llvm_nm);
    // Dump only extern, non-weak symbol names
    // Don't print an error if no symbols are found, don't bother sorting, and skip LLVM bitcode
    let nm_cmd = nm_cmd
        .args(&[
            "-g",
            "--format=just-symbols",
            "--no-llvm-bc",
            "--no-sort",
            "--no-weak",
            "--quiet",
        ])
        .arg(&libstd_rlib)
        .stdout(Stdio::piped());
    let symbols = output(nm_cmd);
    // Find the first line containing the mangled symbol for std::rt::lang_start_internal, and postprocess it
    let lang_start_symbol = symbols
        .lines()
        .map(|line| line.trim())
        .find_map(|line| {
            if line.is_empty() || !line.contains("lang_start_internal") {
                None
            } else {
                Some(line)
            }
        })
        .map(postprocess_lang_start_symbol_name)
        .expect("unable to locate lang_start_symbol in libstd rlib, has it been removed?");
    // Success!
    println!(
        "cargo:rustc-env=LANG_START_SYMBOL_NAME={}",
        lang_start_symbol
    );
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

#[cfg(target_os = "macos")]
fn postprocess_lang_start_symbol_name(lang_start_symbol: &str) -> &str {
    // Strip off leading `_` when printing symbol name
    &lang_start_symbol[1..]
}

#[cfg(not(target_os = "macos"))]
fn postprocess_lang_start_symbol_name(lang_start_symbol: &str) -> &str {
    lang_start_symbol
}

fn fail(s: &str) -> ! {
    panic!("\n{}\n\nbuild script failed, must exit now", s)
}
