extern crate cc;
extern crate which;

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const ENV_LLVM_PREFIX: &'static str = "LLVM_PREFIX";
const ENV_FIREFLY_BUILD_TYPE: &'static str = "FIREFLY_BUILD_TYPE";
const ENV_LLVM_LINK_LLVM_DYLIB: &'static str = "LLVM_LINK_LLVM_DYLIB";
const ENV_LLVM_USE_SANITIZER: &'static str = "LLVM_USE_SANITIZER";

fn main() {
    let cwd = env::current_dir().unwrap();
    let llvm_prefix = detect_llvm_prefix();
    let llvm_lib_dir = llvm_prefix.join("lib");

    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_PREFIX);
    println!("cargo:rerun-if-env-changed={}", ENV_FIREFLY_BUILD_TYPE);
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_LINK_LLVM_DYLIB);
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_USE_SANITIZER);
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=c_src");
    println!("cargo:prefix={}", llvm_prefix.display());

    let target = env::var("TARGET").expect("TARGET was not set");
    let host = env::var("HOST").expect("HOST was not set");
    let is_crossed = target != host;

    let llvm_link_llvm_dylib = env::var(ENV_LLVM_LINK_LLVM_DYLIB).unwrap_or("OFF".to_owned());
    let llvm_config = llvm_prefix.as_path().join("bin/llvm-config").to_path_buf();

    let optional_components = vec![
        "x86",
        //"arm",
        "aarch64",
        //"amdgpu",
        //"mips",
        //"powerpc",
        //"systemz",
        "webassembly",
        //"msp430",
        //"sparc",
        //"nvptx",
        //"hexagon",
        //"riscv",
    ];

    let required_components = &[
        "core",
        "ipo",
        "bitreader",
        "bitwriter",
        "linker",
        "asmparser",
        "lto",
        "instrumentation",
        //"orcjit",
    ];

    let components = output(Command::new(&llvm_config).arg("--components"));
    let mut components = components.split_whitespace().collect::<Vec<_>>();
    components.retain(|c| optional_components.contains(c) || required_components.contains(c));

    for component in required_components {
        if !components.contains(component) {
            panic!("require llvm component {} but wasn't found", component);
        }
    }

    // Link in our own LLVM shims, compiled with the same flags as LLVM
    let mut cmd = Command::new(&llvm_config);
    cmd.arg("--cxxflags");
    let cxxflags = output(&mut cmd);
    let mut shared_cxxflags = Vec::with_capacity(cxxflags.len());

    let mut cfg = cc::Build::new();
    cfg.warnings(false);

    for flag in cxxflags.split_whitespace() {
        // Ignore flags like `-m64` when we're doing a cross build
        if is_crossed && flag.starts_with("-m") {
            continue;
        }

        if flag.starts_with("-flto") {
            continue;
        }

        // -Wdate-time is not supported by the netbsd cross compiler
        if is_crossed && target.contains("netbsd") && flag.contains("date-time") {
            continue;
        }

        // Ignore -std=c++14 if present
        if flag.starts_with("-std=c++") {
            continue;
        }

        shared_cxxflags.push(flag);
        cfg.flag(flag);
    }

    println!("cargo:cxxflags={}", shared_cxxflags.as_slice().join(";"));

    for component in &components {
        let mut flag = String::from("LLVM_COMPONENT_");
        flag.push_str(&component.to_uppercase());
        cfg.define(&flag, None);
    }

    if env::var("FIREFLY_LLVM_LTO").unwrap_or("OFF".to_string()) == "ON" {
        println!("cargo:lto=true");
        cfg.flag("-flto=thin");
    } else {
        println!("cargo:lto=false");
    }

    if let Ok(mut sanitizer) = env::var(ENV_LLVM_USE_SANITIZER) {
        if !sanitizer.is_empty() {
            sanitizer.make_ascii_lowercase();
            cfg.flag(&format!("-fsanitize={}", sanitizer));
        }
    }

    if env::var_os("LLVM_NDEBUG").is_some() {
        cfg.define("NDEBUG", None);
        cfg.debug(false);
    }

    let include_dir = cwd.join("c_src/include");
    println!("cargo:include={}", include_dir.display());

    if cfg!(windows) {
        cfg.file("c_src/raw_win32_handle_ostream.cpp");
    }
    cfg.file("c_src/lib/CAPI/Archives.cpp")
       .file("c_src/lib/CAPI/Diagnostics.cpp")
       .file("c_src/lib/CAPI/ErrorHandling.cpp")
       .file("c_src/lib/CAPI/IR.cpp")
       .file("c_src/lib/CAPI/ModuleWriter.cpp")
       .file("c_src/lib/CAPI/Passes.cpp")
       .file("c_src/lib/CAPI/Target.cpp")
       .include(include_dir)
       .shared_flag(false)
       .static_flag(true)
       .cpp(true)
       .cpp_link_stdlib(None) // we handle this below
       .flag("-std=c++17") // Force minimum required C++ version
       .compile("firefly_llvm_core");

    let (llvm_kind, llvm_link_arg) = detect_llvm_link();
    let link_static = llvm_kind == "static";
    let link_llvm_dylib = llvm_link_llvm_dylib == "ON";
    println!("cargo:link_static={}", &link_static);
    println!("cargo:link_llvm_dylib={}", &llvm_link_llvm_dylib);
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath={}",
        llvm_lib_dir.as_path().display()
    );

    if !link_static && link_llvm_dylib {
        println!("cargo:rustc-link-lib=dylib=LLVM");
    } else {
        // Link in all LLVM libraries
        let mut cmd = Command::new(&llvm_config);
        cmd.arg(llvm_link_arg).arg("--libs");

        if !is_crossed {
            cmd.arg("--system-libs");
        }
        cmd.args(&components);

        for lib in output(&mut cmd).split_whitespace() {
            let name = if lib.starts_with("-l") {
                &lib[2..]
            } else if lib.starts_with('-') {
                &lib[1..]
            } else if Path::new(lib).exists() {
                // On MSVC llvm-config will print the full name to libraries, but
                // we're only interested in the name part
                let name = Path::new(lib).file_name().unwrap().to_str().unwrap();
                name.trim_end_matches(".lib")
            } else if lib.ends_with(".lib") {
                // Some MSVC libraries just come up with `.lib` tacked on, so chop
                // that off
                lib.trim_end_matches(".lib")
            } else {
                continue;
            };

            // Don't need or want these libraries, but LLVM's CMake build system
            // doesn't provide a way to disable it, so filter it here even though we
            // may or may not have built it. We don't reference anything from these
            // libraries and it otherwise may just pull in extra dependencies on
            // libedit which we don't want
            if name == "LLVMLineEditor" || name == "Polly" || name == "PollyISL" {
                continue;
            }

            let kind = if name.starts_with("LLVM") {
                llvm_kind
            } else {
                "dylib"
            };
            println!("cargo:rustc-link-lib={}={}", kind, name);
        }
    }

    // LLVM ldflags
    //
    // If we're a cross-compile of LLVM then unfortunately we can't trust these
    // ldflags (largely where all the LLVM libs are located). Currently just
    // hack around this by replacing the host triple with the target and pray
    // that those -L directories are the same!
    let mut cmd = Command::new(&llvm_config);
    cmd.arg(llvm_link_arg).arg("--ldflags");
    let ldflags = output(&mut cmd);
    let mut shared_ldflags = Vec::new();
    for lib in ldflags.split_whitespace() {
        if is_crossed {
            if lib.starts_with("-LIBPATH:") {
                let path = &lib[9..].replace(&host, &target);
                println!("cargo:rustc-link-search=native={}", path);
                shared_ldflags.push(path.to_owned());
            } else if lib.starts_with("-L") {
                let path = &lib[2..].replace(&host, &target);
                println!("cargo:rustc-link-search=native={}", path);
                shared_ldflags.push(path.to_owned());
            }
        } else if lib.starts_with("-LIBPATH:") {
            let path = &lib[9..];
            println!("cargo:rustc-link-search=native={}", path);
            shared_ldflags.push(path.to_owned());
        } else if lib.starts_with("-l") {
            println!("cargo:rustc-link-lib={}", &lib[2..]);
        } else if lib.starts_with("-L") {
            let path = &lib[2..];
            println!("cargo:rustc-link-search=native={}", path);
            shared_ldflags.push(path.to_owned());
        }
    }

    // Some LLVM linker flags (-L and -l) may be needed even when linking
    // firefly_llvm, for example when using static libc++, we may need to
    // manually specify the library search path and -ldl -lpthread as link
    // dependencies.
    let llvm_linker_flags = env::var_os("LLVM_LINKER_FLAGS");
    if let Some(s) = llvm_linker_flags {
        for lib in s.into_string().unwrap().split_whitespace() {
            if lib.starts_with("-l") {
                println!("cargo:rustc-link-lib={}", &lib[2..]);
            } else if lib.starts_with("-L") {
                let path = &lib[2..];
                println!("cargo:rustc-link-search=native={}", path);
                shared_ldflags.push(path.to_owned());
            }
        }
    }
    println!("cargo:ldflags={}", shared_ldflags.as_slice().join(";"));

    let llvm_static_stdcpp = env::var_os("LLVM_STATIC_STDCPP");

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

fn detect_llvm_prefix() -> PathBuf {
    if let Ok(prefix) = env::var(ENV_LLVM_PREFIX) {
        return PathBuf::from(prefix);
    }

    if let Ok(llvm_config) = which::which("llvm-config") {
        let mut cmd = Command::new(llvm_config);
        cmd.arg("--prefix");
        return PathBuf::from(output(&mut cmd));
    }

    let mut llvm_prefix = env::var("XDG_DATA_HOME")
        .map(|s| PathBuf::from(s))
        .unwrap_or_else(|_| {
            let mut home = PathBuf::from(env::var("HOME").expect("HOME not defined"));
            home.push(".local/share");
            home
        });
    llvm_prefix.push("llvm");
    if llvm_prefix.exists() {
        // Make sure its actually the prefix and not a root
        let llvm_bin = llvm_prefix.as_path().join("bin");
        if llvm_bin.exists() {
            return llvm_prefix;
        }
        let firefly = llvm_prefix.as_path().join("firefly");
        if firefly.exists() {
            return firefly.to_path_buf();
        }
    }

    fail("LLVM_PREFIX is not defined and unable to locate LLVM to build with");
}

fn output(cmd: &mut Command) -> String {
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

fn detect_llvm_link() -> (&'static str, &'static str) {
    // Force the link mode we want, preferring static by default, but
    match env::var_os(ENV_FIREFLY_BUILD_TYPE) {
        Some(val) if val == "dynamic" => ("dylib", "--link-shared"),
        _ => ("static", "--link-static"),
    }
}

fn fail(s: &str) -> ! {
    panic!("\n{}\n\nbuild script failed, must exit now", s)
}
