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
    let target = env::var("TARGET").expect("TARGET was not set");
    let host = env::var("HOST").expect("HOST was not set");
    let is_crossed = target != host;

    let llvm_prefix_env = env::var(ENV_LLVM_PREFIX).expect(ENV_LLVM_PREFIX);
    let llvm_prefix = PathBuf::from(llvm_prefix_env.as_str());

    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_PREFIX);

    let llvm_config_path = env::var_os("LLVM_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| llvm_prefix.as_path().join("bin/llvm-config").to_path_buf());
    println!("cargo:rerun-if-changed={}", llvm_config_path.display());
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

    let link_libs = read_link_libs(llvm_config_path.as_path(), outdir.as_path());
    for link_lib in link_libs {
        if link_lib.contains('=') {
            println!("cargo:rustc-link-lib={}", link_lib);
        } else {
            println!("cargo:rustc-link-lib=static={}", link_lib);
        }
    }

    // LLVM ldflags
    //
    // If we're a cross-compile of LLVM then unfortunately we can't trust these
    // ldflags (largely where all the LLVM libs are located). Currently just
    // hack around this by replacing the host triple with the target and pray
    // that those -L directories are the same!
    let ldflags_raw = llvm_config(llvm_config_path.as_path(), &["--ldflags"]).unwrap();
    for lib in ldflags_raw.split_whitespace() {
        if lib.starts_with("-LIBPATH:") {
            println!("cargo:rustc-link-search=native={}", &lib[9..]);
        } else if is_crossed {
            if lib.starts_with("-L") {
                println!(
                    "cargo:rustc-link-search=native={}",
                    lib[2..].replace(&host, &target)
                );
            }
        } else if lib.starts_with("-l") {
            println!("cargo:rustc-link-lib={}", &lib[2..]);
        } else if lib.starts_with("-L") {
            println!("cargo:rustc-link-search=native={}", &lib[2..]);
        }
    }

    // Some LLVM linker flags (-L and -l) may be needed even when linking
    // liblumen_codegen, for example when using static libc++, we may need to
    // manually specify the library search path and -ldl -lpthread as link
    // dependencies.
    let llvm_linker_flags = env::var_os("LLVM_LINKER_FLAGS");
    println!("cargo:rerun-if-env-changed=LLVM_LINKER_FLAGS");
    if let Some(s) = llvm_linker_flags {
        for lib in s.into_string().unwrap().split_whitespace() {
            if lib.starts_with("-l") {
                println!("cargo:rustc-link-lib={}", &lib[2..]);
            } else if lib.starts_with("-L") {
                println!("cargo:rustc-link-search=native={}", &lib[2..]);
            }
        }
    }

    print_libcpp_flags(llvm_config_path.as_path(), target.as_str());
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
        .map(|f| f.to_string_lossy().starts_with("."))
        .unwrap_or(false);
}

fn llvm_config<S>(binary: S, args: &[&str]) -> io::Result<String>
where
    S: AsRef<OsStr>,
{
    match Command::new(binary).args(args).output() {
        Ok(output) => match String::from_utf8(output.stdout) {
            Ok(s) => Ok(s.trim_end().to_owned()),
            Err(e) => Err(io::Error::new(io::ErrorKind::Other, e)),
        },
        Err(e) => Err(e),
    }
}

fn cleanup_link_lib(lib: &str) -> Option<&str> {
    println!("lib: {}", lib);
    let result = if lib.starts_with("-l") {
        Some(&lib[2..])
    } else if lib.starts_with("-") {
        Some(&lib[1..])
    } else if Path::new(lib).exists() {
        // On MSVC llvm-config will print the full name to libraries, but
        // we're only interested in the name part
        let path = Path::new(lib);
        println!(
            "cargo:rustc-link-search=native={}",
            path.parent().unwrap().to_str().unwrap()
        );
        let name = path.file_name().unwrap().to_str().unwrap();
        Some(
            name.trim_start_matches("lib")
                .trim_end_matches(".lib")
                .trim_end_matches(".dylib"),
        )
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

fn read_link_libs(llvm_config_path: &Path, outdir: &Path) -> Vec<String> {
    let mut link_libs = Vec::new();

    if let Some(_) = env::var_os("LLVM_BUILD_STATIC") {
        read_link_libs_static(outdir, &mut link_libs);
    } else {
        read_link_libs_shared(&mut link_libs);
    }

    // System libs
    let syslibs = llvm_config(llvm_config_path, &["--system-libs"]).unwrap();
    for l in syslibs.split(' ') {
        if l.ends_with("libxml2.dylib") {
            // This library is only required during LLVM build
            continue;
        }
        if let Some(lib) = cleanup_link_lib(l) {
            // System libraries must be dynamically linked
            link_libs.push(format!("dylib={}", lib));
        }
    }

    link_libs
}

fn read_link_libs_shared(link_libs: &mut Vec<String>) {
    // If not statically linking, we can just link against the combined dylib
    link_libs.push(format!("dylib=LLVMcpp"));
}

fn read_link_libs_static(outdir: &Path, link_libs: &mut Vec<String>) {
    // If statically linking, we need to link against the same libs as libLumen
    let lumen_libs_txt = outdir.join("build").join("Lumen_deps.txt");
    let lumen_libs = fs::read_to_string(lumen_libs_txt).unwrap();

    // LLVM
    for l in lumen_libs.lines() {
        link_libs.push(l.to_string());
    }
}

fn print_libcpp_flags(llvm_config_path: &Path, target: &str) {
    let cxxflags = llvm_config(llvm_config_path, &["-cxxflags"]).unwrap();

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
