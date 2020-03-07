extern crate cmake;
extern crate which;

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io;
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
    let target = env::var("TARGET").expect("TARGET was not set");
    let host = env::var("HOST").expect("HOST was not set");
    let is_crossed = target != host;
    let profile = env::var("PROFILE").expect("PROFILE was not set");

    let cwd = env::current_dir().expect("unable to access current directory");
    let target_dir = cwd.parent().unwrap().join(&format!("target/{}", &profile));

    let llvm_prefix_env = env::var(ENV_LLVM_PREFIX).expect(ENV_LLVM_PREFIX);
    let llvm_prefix = PathBuf::from(llvm_prefix_env.as_str());

    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_PREFIX);
    println!("cargo:rerun-if-env-changed={}", ENV_LLVM_BUILD_STATIC);

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
    let build_shared = if env::var_os(ENV_LLVM_BUILD_STATIC).is_some() {
        "OFF"
    } else {
        "ON"
    };
    let outdir = config
        .define("LUMEN_BUILD_COMPILER", "ON")
        .define("LUMEN_BUILD_TESTS", "OFF")
        .define("BUILD_SHARED_LIBS", build_shared)
        .define("LLVM_PREFIX", llvm_prefix_env.as_str())
        .env("LLVM_PREFIX", llvm_prefix_env.as_str())
        .define("CARGO_TARGET_DIR", &target_dir)
        .always_configure(true)
        .build_target("install")
        .very_verbose(false)
        .build();

    rerun_if_changed_anything_in_dir(&cwd.join("lib"));

    let compile_commands_src = outdir.join("build").join("compile_commands.json");
    let compile_commands_dest = cwd.join("lib").join("compile_commands.json");

    fs::copy(compile_commands_src, compile_commands_dest)
        .expect("unable to copy compile_commands.json!");

    println!(
        "cargo:rustc-link-search=native={}",
        llvm_prefix.join("lib").display()
    );
    println!("cargo:rustc-link-search=native={}/lib", outdir.display());

    if build_shared == "ON" {
        link_lib_dylib("lumen_compiler_LumenCodegen");
    } else {
        link_libs(&[
            "lumen_compiler_Diagnostics_Diagnostics",
            "lumen_compiler_Dialect_EIR_IR_IR",
            "lumen_compiler_Dialect_EIR_Conversion_EIRToLLVM_EIRToLLVM",
            "lumen_compiler_Dialect_EIR_Transforms_Transforms",
            "lumen_compiler_Support_Support",
            "lumen_compiler_Target_Target",
            "lumen_compiler_Translation_Translation",
            "MLIRAnalysis",
            "MLIRAffineOps",
            "MLIRDialect",
            "MLIREDSC",
            "MLIREDSCInterface",
            "MLIRExecutionEngine",
            "MLIRIR",
            "MLIRLLVMIR",
            "MLIRLoopAnalysis",
            "MLIRLoopOps",
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
        ]);

        let link_libs = read_link_libs(llvm_config_path.as_path(), outdir.as_path());
        for link_lib in link_libs {
            if link_lib.contains('=') {
                println!("cargo:rustc-link-lib={}", link_lib);
            } else {
                println!("cargo:rustc-link-lib=static={}", link_lib);
            }
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

    // Get demangled lang_start_internal name

    let mut sysroot_cmd = Command::new("rustc");
    let mut sysroot_cmd = sysroot_cmd.args(&["--print", "sysroot"]);
    let sysroot = PathBuf::from(output(&mut sysroot_cmd).trim());
    let toolchain_libs = sysroot.join("lib/rustlib").join(target).join("lib");
    println!("toolchain_libs: {:?}", &toolchain_libs);
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

    if let Some(_) = env::var_os(ENV_LLVM_BUILD_STATIC) {
        read_link_libs_static(outdir, &mut link_libs);
    } else {
        read_link_libs_shared(llvm_config_path, &mut link_libs);
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

fn read_link_libs_shared(llvm_config_path: &Path, link_libs: &mut Vec<String>) {
    if let Ok(libs) = llvm_config(
        llvm_config_path,
        &["--link-shared", "--ignore-libllvm", "--libs"],
    ) {
        for l in libs.split(' ') {
            if let Some(lib) = cleanup_link_lib(l) {
                link_libs.push(format!("dylib={}", lib));
            }
        }
    } else {
        // Try linking against components dylib
        link_libs.push(format!("dylib=LLVM"));
    }
}

fn read_link_libs_static(outdir: &Path, link_libs: &mut Vec<String>) {
    // If statically linking, we need to link against the same libs as libLumen
    let lumen_libs_txt = outdir.join("build").join("llvm_deps.txt");
    println!("cargo:rerun-if-changed={}", lumen_libs_txt.display());
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
                link_lib_static(stdcppname);
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
