extern crate cmake;
extern crate which;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const ENV_LLVM_PREFIX: &'static str = "LLVM_SYS_90_PREFIX";
const ENV_LLVM_BUILD_STATIC: &'static str = "LLVM_BUILD_STATIC";

fn main() {
    // Emit custom cfg types:
    //     cargo:rustc-cfg=has_foo
    // Can then be used as `#[cfg(has_foo)]` when emitted

    // Emit custom env data:
    //     cargo:rustc-env=foo=bar
    // Can then be fetched with `env!("foo")`

    // LLVM
    let target = env::var("TARGET").unwrap();

    let cwd = env::current_dir().expect("unable to access current directory");

    let llvm_prefix_env = env::var(ENV_LLVM_PREFIX).expect(ENV_LLVM_PREFIX);
    let llvm_prefix = PathBuf::from(llvm_prefix_env.as_str());

    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_PREFIX);
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_BUILD_STATIC);

    if let Err(_) = which::which("cmake") {
        fail(
            "Unable to locate CMake!\n\
             It is required for the build, make sure you have a recent version installed.",
        );
    }

    let mut use_ninja = true;
    if let Err(_) = which::which("ninja") {
        use_ninja = false;
        warn(
            "Unable to locate Ninja, your CMake builds may take unncessarily long.\n\
             It is highly recommended that you install Ninja.",
        );
    }

    println!("cargo:rerun-if-changed=use_ninja={}", use_ninja);

    let mut config = &mut cmake::Config::new("lib");
    if use_ninja {
        config = config.generator("Ninja");
    }
    let build_shared = if env::var_os(ENV_LLVM_BUILD_STATIC).is_some() {
        "OFF"
    } else {
        "ON"
    };

    let lumen_llvm_include_dir = env::var("DEP_LUMEN_LLVM_CORE_INCLUDE").unwrap();
    let lumen_mlir_include_dir = env::var("DEP_LUMEN_MLIR_CORE_INCLUDE").unwrap();
    let lumen_term_include_dir = env::var("DEP_LUMEN_TERM_CORE_INCLUDE").unwrap();

    rerun_if_changed_anything_in_dir(&cwd.join("lib"));

    let outdir = config
        .define("LUMEN_BUILD_COMPILER", "ON")
        .define("LUMEN_BUILD_TESTS", "OFF")
        .define("BUILD_SHARED_LIBS", build_shared)
        .define("LLVM_PREFIX", llvm_prefix_env.as_str())
        .env("LLVM_PREFIX", llvm_prefix_env.as_str())
        .cxxflag(&format!("-I{}", lumen_llvm_include_dir))
        .cxxflag(&format!("-I{}", lumen_mlir_include_dir))
        .cxxflag(&format!("-I{}", lumen_term_include_dir))
        .always_configure(true)
        .build_target("install")
        .very_verbose(false)
        .build();

    let lumen_term_output_dir = env::var("DEP_LUMEN_TERM_CORE_OUTPUT_DIR").unwrap();
    println!(
        "cargo:rustc-env=TERM_LIB_OUTPUT_DIR={}",
        lumen_term_output_dir
    );

    let compile_commands_src = outdir.join("build").join("compile_commands.json");
    let compile_commands_dest = cwd.join("lib").join("compile_commands.json");

    fs::copy(compile_commands_src, compile_commands_dest)
        .expect("unable to copy compile_commands.json!");

    println!("cargo:rustc-link-search=native={}/lib", outdir.display());

    link_libs(&[
        "lumen_compiler_Dialect_EIR_IR_IR",
        "lumen_compiler_Dialect_EIR_Conversion_EIRToLLVM_EIRToLLVM",
        "lumen_compiler_Dialect_EIR_Transforms_Transforms",
        "lumen_compiler_Target_Target",
        "lumen_compiler_Translation_Translation",
    ]);

    // Get demangled lang_start_internal name

    let mut sysroot_cmd = Command::new("rustc");
    let mut sysroot_cmd = sysroot_cmd.args(&["--print", "sysroot"]);
    let sysroot = PathBuf::from(output(&mut sysroot_cmd).trim());
    let toolchain_libs = sysroot.join("lib/rustlib").join(target).join("lib");
    let libstd_rlib = toolchain_libs
        .read_dir()
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| {
            let path = e.path();
            let filename = path.file_name().map(|s| s.to_string_lossy());
            if let Some(fname) = filename {
                if fname.starts_with("libstd") && fname.ends_with(".rlib") {
                    return true;
                }
            }
            false
        })
        .take(1)
        .next()
        .map(|e| e.path().to_string_lossy().into_owned())
        .expect("unable to find libstd rlib in toolchain directory!");

    let llvm_objdump = llvm_prefix.join("bin/llvm-objdump");
    let mut objdump_cmd = Command::new(llvm_objdump);
    let objdump_cmd = objdump_cmd
        .args(&["--syms", &libstd_rlib])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to open llvm-objdump");
    let mut grep_cmd = Command::new("grep");
    let grep_cmd = grep_cmd
        .args(&["lang_start_internal"])
        .stdin(objdump_cmd.stdout.unwrap())
        .stderr(Stdio::inherit());

    let results = output(grep_cmd);
    let lang_start_symbol = results
        .trim()
        .split(' ')
        .last()
        .expect("expected non-empty lang_start_symbol result");

    println!(
        "cargo:rustc-env=LANG_START_SYMBOL_NAME={}",
        lang_start_symbol_name(lang_start_symbol)
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

pub fn rerun_if_changed_anything_in_dir(dir: &Path) {
    let mut stack = dir
        .read_dir()
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| !ignore_changes(Path::new(&*e.file_name())))
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

fn ignore_changes(name: &Path) -> bool {
    return name
        .file_name()
        .map(|f| {
            let name = f.to_string_lossy();
            if name.starts_with(".") {
                return true;
            }
            if name == "CMakeLists.txt" {
                return false;
            }
            if name.ends_with(".cpp") || name.ends_with(".h") || name.ends_with(".td") {
                return false;
            }
            true
        })
        .unwrap_or(false);
}

#[cfg(target_os = "macos")]
fn lang_start_symbol_name(lang_start_symbol: &str) -> &str {
    // Strip off leading `_` when printing symbol name
    &lang_start_symbol[1..]
}

#[cfg(not(target_os = "macos"))]
fn lang_start_symbol_name(lang_start_symbol: &str) -> &str {
    lang_start_symbol
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

fn warn(s: &str) {
    println!("cargo:warning={}", s);
}

fn fail(s: &str) -> ! {
    panic!("\n{}\n\nbuild script failed, must exit now", s)
}
