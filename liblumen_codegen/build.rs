extern crate cc;
extern crate cmake;
extern crate which;

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const ENV_LLVM_PREFIX: &'static str = "LLVM_SYS_90_PREFIX";

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

    let target = env::var("TARGET").expect("TARGET was not set");
    //let host = env::var("HOST").expect("HOST was not set");
    //let is_crossed = target != host;

    let llvm_prefix_env = env::var(ENV_LLVM_PREFIX).expect(ENV_LLVM_PREFIX);
    let llvm_prefix = PathBuf::from(llvm_prefix_env.as_str());

    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_PREFIX);

    let llvm_config = env::var_os("LLVM_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| llvm_prefix.as_path().join("bin/llvm-config").to_path_buf());
    println!("cargo:rerun-if-changed={}", llvm_config.display());
    println!("cargo:rerun-if-env-changed=LLVM_CONFIG");

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
    let outdir = config
        .define("CMAKE_EXPORT_COMPILE_COMMANDS", "YES")
        .define("LLVM_PREFIX", llvm_prefix_env.as_str())
        .env("LLVM_PREFIX", llvm_prefix_env.as_str())
        .always_configure(true)
        .no_build_target(true)
        .very_verbose(false)
        .build();

    let cwd = env::current_dir().expect("unable to access current directory");

    rerun_if_changed_anything_in_dir(&cwd.join("lib"));

    let link_libs = read_link_libs(llvm_config.as_path(), cwd.as_path());

    let compile_commands_src = outdir.join("build").join("compile_commands.json");
    let compile_commands_dest = cwd.join("lib").join("compile_commands.json");

    fs::copy(compile_commands_src, compile_commands_dest)
        .expect("unable to copy compile_commands.json!");

    println!(
        "cargo:rustc-link-search=native={}",
        llvm_prefix.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}/build/lib",
        outdir.display()
    );
    println!("cargo:rustc-link-lib=static=Lumen");

    for link_lib in link_libs {
        if link_lib.contains('=') {
            println!("cargo:rustc-link-lib={}", link_lib);
        } else {
            println!("cargo:rustc-link-lib=static={}", link_lib);
        }
    }

    // Some LLVM linker flags (-L and -l) may be needed even when linking
    // liblumen_codegen, for example when using static libc++, we may need to
    // manually specify the library search path and -ldl -lpthread as link
    // dependencies.
    let llvm_linker_flags = env::var_os("LLVM_LINKER_FLAGS");
    if let Some(s) = llvm_linker_flags {
        for lib in s.into_string().unwrap().split_whitespace() {
            if lib.starts_with("-l") {
                println!("cargo:rustc-link-lib={}", &lib[2..]);
            } else if lib.starts_with("-L") {
                println!("cargo:rustc-link-search=native={}", &lib[2..]);
            }
        }
    }

    print_libcpp_flags(llvm_config.as_path(), target.as_str());
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

/// Invoke the specified binary as llvm-config for a dylib build
fn llvm_config_dylib<S: AsRef<OsStr>>(binary: S, args: &[&str]) -> io::Result<String> {
    Command::new(binary)
        .arg("--link-shared")
        .args(args)
        .output()
        .map(|output| {
            String::from_utf8(output.stdout).expect("Output from llvm-config was not valid UTF-8")
        })
}

/// Invoke the specified binary as llvm-config for a static build
fn llvm_config_static<S: AsRef<OsStr>>(binary: S, args: &[&str]) -> io::Result<String> {
    Command::new(binary)
        .arg("--link-static")
        .args(args)
        .output()
        .map(|output| {
            String::from_utf8(output.stdout).expect("Output from llvm-config was not valid UTF-8")
        })
}

fn has_dylib_components(llvm_config: &Path) -> bool {
    if let Ok(_) = Command::new(llvm_config).arg("--link-shared").status() {
        return true;
    }
    false
}

fn cleanup_link_lib(lib: &str) -> Option<&str> {
    let result = if lib.starts_with("-l") {
        Some(&lib[2..])
    } else if lib.starts_with("-") {
        Some(&lib[1..])
    } else if Path::new(lib).exists() {
        // On MSVC llvm-config will print the full name to libraries, but
        // we're only interested in the name part
        let name = Path::new(lib).file_name().unwrap().to_str().unwrap();
        Some(name.trim_end_matches(".lib"))
    } else if lib.ends_with(".lib") {
        // Some MSVC libraries just come up with `.lib` tacked on, so trim it
        Some(lib.trim_end_matches(".lib"))
    } else {
        None
    };

    // We don't need or want this lib
    if let Some("LLVMLineEditor") = result {
        None
    } else {
        result
    }
}

fn read_link_libs(llvm_config: &Path, cwd: &Path) -> Vec<String> {
    let mut link_libs = Vec::new();

    // LLVM
    let libargs = &[
        "--system-libs",
        "--libs",
        "core",
        "support",
        "all-targets",
        "executionengine",
        "linker",
    ];
    if env::var_os("LLVM_BUILD_STATIC").is_some() || !has_dylib_components(llvm_config) {
        let libs = llvm_config_static(llvm_config, libargs).unwrap();
        for l in libs.split(' ') {
            if let Some(lib) = cleanup_link_lib(l) {
                if lib.starts_with("LLVM") {
                    link_libs.push(lib.to_string());
                } else {
                    // System libraries must be dynamically linked
                    link_libs.push(format!("dylib={}", lib));
                }
            }
        }
    } else {
        let libs = llvm_config_dylib(llvm_config, libargs).unwrap();
        for l in libs.split(' ') {
            if let Some(lib) = cleanup_link_lib(l) {
                link_libs.push(format!("dylib={}", lib));
            }
        }
    }

    // LLD
    for lib in &[
        "lldCore",
        "lldCommon",
        "lldYaml",
        "lldReaderWriter",
        "lldELF",
        "lldWasm",
        "lldCOFF",
        "lldMachO",
        "lldMinGW",
        "lldDriver",
    ] {
        link_libs.push(lib.to_string());
    }

    // MLIR
    let liblumen_cmakelists_path = cwd.join("lib").join("lib").join("CMakeLists.txt");
    let liblumen_cmake_lists = fs::read_to_string(liblumen_cmakelists_path).unwrap();
    let mut started = false;
    for line in liblumen_cmake_lists.lines() {
        if !started && line.starts_with("set(Lumen_mlir_libs") {
            started = true;
            continue;
        }
        if !started {
            continue;
        }
        if line.starts_with(")") {
            break;
        }
        if line.starts_with("#") {
            continue;
        }
        let lib = line.trim();
        if lib.is_empty() {
            continue;
        }
        link_libs.push(lib.to_string());
    }

    link_libs
}

fn print_libcpp_flags(llvm_config: &Path, target: &str) {
    let cxxflags = output(&mut Command::new(llvm_config).arg("--cxxflags"));

    let llvm_static_stdcpp = env::var_os("LLVM_STATIC_STDCPP");
    let llvm_use_libcxx = env::var_os("LLVM_USE_LIBCXX");

    let stdcppname = if target.contains("openbsd") {
        if target.contains("sparc64") {
            "estdc++"
        } else {
            "c++"
        }
    } else if target.contains("freebsd") {
        "c++"
    } else if target.contains("darwin") {
        "c++"
    } else if target.contains("netbsd") && llvm_static_stdcpp.is_some() {
        // NetBSD uses a separate library when relocation is required
        "stdc++_pic"
    } else if llvm_use_libcxx.is_some() {
        "c++"
    } else {
        "stdc++"
    };

    // C++ runtime library
    if !target.contains("msvc") {
        if let Some(s) = llvm_static_stdcpp {
            assert!(!cxxflags.contains("stdlib=libc++"));
            let path = PathBuf::from(s);
            println!(
                "cargo:rustc-link-search=native={}",
                path.parent().unwrap().display()
            );
            if target.contains("windows") {
                println!("cargo:rustc-link-lib=static-nobundle={}", stdcppname);
            } else {
                println!("cargo:rustc-link-lib=static={}", stdcppname);
            }
        } else if cxxflags.contains("stdlib=libc++") {
            println!("cargo:rustc-link-lib=c++");
        } else {
            println!("cargo:rustc-link-lib={}", stdcppname);
        }
    }

    // LLVM requires symbols from this library, but apparently they're not printed
    // during llvm-config?
    if target.contains("windows-gnu") {
        println!("cargo:rustc-link-lib=static-nobundle=gcc_s");
        println!("cargo:rustc-link-lib=static-nobundle=pthread");
        println!("cargo:rustc-link-lib=dylib=uuid");
    }
}

fn warn(s: &str) {
    println!("cargo:warning={}", s);
}

fn fail(s: &str) -> ! {
    panic!("\n{}\n\nbuild script failed, must exit now", s)
}
