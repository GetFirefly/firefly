//! ```cargo
//! [dependencies]
//! serde_json = "1.0"
//! walkdir = "*"
//!
//! [dependencies.serde]
//! version = "1.0"
//! features = ["derive"]
//! ```
#![feature(drain_filter)]
#![allow(non_snake_case)]

extern crate serde;
extern crate serde_json;
extern crate walkdir;

use std::collections::HashMap;
use std::env;
use std::fmt;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Deserialize;
use walkdir::{DirEntry, WalkDir};

#[derive(Deserialize)]
#[serde(tag = "reason")]
enum Item {
    #[serde(rename = "compiler-message")]
    Message { message: Message },
    #[serde(rename = "compiler-artifact")]
    Artifact {
        target: Target,
        filenames: Vec<String>,
    },
    #[serde(rename = "build-script-executed")]
    #[allow(unused)]
    BuildScriptExecuted { package_id: String, out_dir: String },
    #[serde(rename = "build-finished")]
    #[allow(unused)]
    BuildFinished { success: bool },
    #[serde(other)]
    #[allow(unused)]
    Ignore,
}

#[derive(Deserialize)]
struct Message {
    rendered: String,
}
impl fmt::Display for &Message {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.rendered.as_str())
    }
}

#[derive(Deserialize)]
struct Target {
    name: String,
}
impl PartialEq<str> for Target {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.name == other
    }
}

#[derive(Deserialize)]
struct TargetSpec {
    #[serde(rename = "llvm-target")]
    llvm_target: String,
}

fn main() {
    let cargo_profile = env::var("LUMEN_BUILD_PROFILE").unwrap();
    let toolchain_name = env::var("CARGO_MAKE_TOOLCHAIN").unwrap();
    let rust_target_triple = env::var("CARGO_MAKE_RUST_TARGET_TRIPLE").unwrap();
    let target_triple = get_llvm_target(&rust_target_triple);
    let target_vendor = env::var("CARGO_MAKE_RUST_TARGET_VENDOR").unwrap();
    let target_os = env::var("CARGO_MAKE_RUST_TARGET_OS").unwrap();
    let build_type = env::var("LUMEN_BUILD_TYPE").unwrap();
    let target_dir = PathBuf::from(env::var("CARGO_TARGET_DIR").unwrap());
    let bin_dir = PathBuf::from(&env::var("LUMEN_BIN_DIR").unwrap());
    let install_dir = PathBuf::from(&env::var("LUMEN_INSTALL_DIR").unwrap());
    let install_bin_dir = install_dir.join("bin");
    let install_host_lib_dir = install_dir.join("lib");
    let install_target_lib_dir = install_dir.join(&format!("lib/lumenlib/{}/lib", &target_triple));
    let rust_sysroot = get_rust_sysroot();
    let toolchain_target_dir = rust_sysroot.join("lib/rustlib").join(&rust_target_triple);
    let llvm_prefix = PathBuf::from(&env::var("LLVM_PREFIX").unwrap());

    let enable_lto = env::var("LUMEN_BUILD_LTO").unwrap_or(String::new()) == "true";

    let mut build_link_args = vec!["-Wl".to_owned()];
    let mut extra_rustc_flags = vec![];
    let mut extra_cargo_flags = vec![];

    let verbose_flags = env::var("CARGO_MAKE_CARGO_VERBOSE_FLAGS");
    if let Ok(f) = verbose_flags {
        if !f.is_empty() {
            extra_cargo_flags.push(f.to_owned());
        }
    }

    let is_darwin = target_vendor == "apple";
    let is_linux = target_os != "macos" && target_os != "windows";

    if is_linux && build_type != "static" {
        build_link_args.push("-rpath".to_owned());
        build_link_args.push("$ORIGIN/../lib".to_owned());
    }

    match cargo_profile.as_str() {
        "release" => {
            extra_cargo_flags.push("--release".to_owned());
        }
        "dev" | _ => {
            extra_rustc_flags.push("-C".to_owned());
            extra_rustc_flags.push("opt-level=0".to_owned());
        }
    }

    if build_type == "static" {
        extra_rustc_flags.push("-C".to_owned());
        extra_rustc_flags.push("prefer-dynamic=no".to_owned());
    }

    if is_darwin {
        build_link_args.push("-headerpad_max_install_names".to_owned());
    }

    let rustflags = {
        let flags = env::var("RUSTFLAGS").unwrap_or(String::new());
        if enable_lto {
            build_link_args.push("-flto=thin".to_owned());
            extra_rustc_flags.push("-C".to_owned());
            extra_rustc_flags.push("embed-bitcode=yes".to_owned());
            extra_rustc_flags.push("-C".to_owned());
            extra_rustc_flags.push("lto=thin".to_owned());
            format!("-C embed-bitcode=yes {}", &flags)
        } else {
            flags
        }
    };

    let link_args = build_link_args.join(",");
    let link_args_string = format!("-Clink-args={}", &link_args);
    let cargo_args = extra_cargo_flags.iter().collect::<Vec<_>>();
    let rustc_args = extra_rustc_flags.iter().collect::<Vec<_>>();

    let path_var = env::var("PATH").unwrap();
    let path = format!("{}/bin:{}", llvm_prefix.display(), &path_var);

    let mut cargo_cmd = Command::new("rustup");
    let cargo_cmd = cargo_cmd
        .arg("run")
        .arg(&toolchain_name)
        .args(&["cargo", "rustc"])
        .args(&["-p", "lumen"])
        .args(&["--message-format=json", "--color=never"])
        .args(cargo_args.as_slice())
        .arg("--")
        .arg(link_args_string.as_str())
        .args(rustc_args.as_slice())
        .env("PATH", path.as_str())
        .env("RUSTFLAGS", rustflags.as_str());

    let cmd = format!("{:?}", &cargo_cmd);
    let mut child = cargo_cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut deps: HashMap<String, Vec<String>> = HashMap::new();
    {
        let stdout = child.stdout.as_mut().unwrap();
        let stdout_reader = BufReader::new(stdout);

        for line in stdout_reader.lines() {
            if let Ok(line) = line {
                match serde_json::from_str(&line).unwrap() {
                    Item::Message { ref message } => print!("{}", message),
                    Item::Artifact {
                        target,
                        mut filenames,
                    } => {
                        let files = filenames
                            .drain_filter(|f| f.ends_with(".a") || f.ends_with(".rlib"))
                            .collect::<Vec<_>>();
                        if !files.is_empty() {
                            deps.insert(target.name, files);
                        }
                    }
                    _ => continue,
                }
            }
        }
    }

    let status = child.wait().unwrap();
    if !status.success() {
        let stderr = child.stderr.as_mut().unwrap();
        let stderr_reader = BufReader::new(stderr);
        for line in stderr_reader.lines() {
            if let Ok(line) = line {
                io::stdout().write_all(&line.into_bytes()).unwrap();
            }
        }
        io::stdout().write(b"\n").unwrap();
        panic!(
            "command did not execute successfully: {}\n\
             expected success, got: {}",
            cmd, status
        );
    }

    println!("Preparing to install Lumen to {}", install_dir.display());

    if !install_bin_dir.exists() {
        fs::create_dir_all(&install_bin_dir).expect("failed to create install bin directory");
    }
    if !install_host_lib_dir.exists() {
        fs::create_dir_all(&install_host_lib_dir)
            .expect("failed to create install host libs directory");
    }
    if !install_target_lib_dir.exists() {
        fs::create_dir_all(&install_target_lib_dir)
            .expect("failed to create install target libs directory");
    }

    let walker = WalkDir::new(&install_host_lib_dir).into_iter();
    for entry in walker.filter_entry(|e| is_dir_or_library_file(e)) {
        let entry = entry.unwrap();
        let ty = entry.file_type();
        if ty.is_dir() {
            continue;
        }
        fs::remove_file(entry.path()).unwrap();
    }

    println!("Installing Lumen..");

    let src_lumen_exe = target_dir.join(&cargo_profile).join("lumen");
    if !src_lumen_exe.exists() {
        panic!(
            "Expected build to place Lumen executable at {}",
            src_lumen_exe.display()
        );
    }

    let lumen_exe = install_bin_dir.join("lumen");
    fs::copy(src_lumen_exe, &lumen_exe).unwrap();

    symlink(&lumen_exe, &bin_dir.join("lumen"));

    if is_darwin {
        println!("Patching runtime path..");

        let mut install_name_tool_cmd = Command::new("install_name_tool");
        let install_name_tool_cmd = install_name_tool_cmd
            .args(&["-add_rpath", "@executable_path/../lib"])
            .arg(&format!("{}", lumen_exe.display()));

        let cmd = install_name_tool_cmd.stdin(Stdio::null()).output().unwrap();
        if !cmd.status.success() {
            io::stderr().write_all(&cmd.stderr).unwrap();
            io::stdout().write_all(&cmd.stdout).unwrap();
            panic!(
                "command did not execute successfully: {:?}\n\
                expected success, got: {}",
                install_name_tool_cmd, cmd.status
            );
        }
    }

    println!("Installing runtime dependencies..");

    let rustlibs = &["libpanic_abort", "libpanic_unwind"];
    let walker = WalkDir::new(toolchain_target_dir.join("lib")).into_iter();
    for entry in walker.filter_entry(|e| is_dir_or_matching_rlib(e, rustlibs)) {
        let entry = entry.unwrap();
        let ty = entry.file_type();
        if ty.is_dir() {
            continue;
        }
        let path = entry.path().canonicalize().unwrap();
        let stem = path.file_stem().unwrap().to_str().unwrap();
        for lib in &rustlibs[..] {
            if stem.starts_with(lib) {
                fs::copy(&path, install_target_lib_dir.join(&format!("{}.rlib", lib))).unwrap();
            }
        }
    }

    println!("Installing runtime libraries..");

    let lumenlibs = &["lumen_rt_minimal", "liblumen_otp"];
    for lib in lumenlibs.iter().copied() {
        if let Some(files) = deps.get(lib) {
            for file in files.iter() {
                let path = Path::new(file).canonicalize().unwrap();
                let extension = path.extension().unwrap().to_str().unwrap();
                let target_path = install_target_lib_dir.join(&format!("lib{}.{}", lib, extension));
                fs::copy(path, target_path).unwrap();
            }
        } else {
            panic!("Unable to find archive (.a/.rlib) for dependency: {}", lib);
        }
    }

    if build_type != "static" {
        match env::var_os("LLVM_LINK_LLVM_DYLIB") {
            Some(val) if val == "ON" => {
                let walker = WalkDir::new(llvm_prefix.join("lib")).into_iter();
                let mut symlinks = HashMap::new();
                for entry in walker.filter_entry(|e| is_dir_or_llvm_lib(e)) {
                    let entry = entry.unwrap();
                    let ty = entry.file_type();
                    if ty.is_dir() {
                        continue;
                    }
                    let path = entry.path();
                    let filename = path.file_name().unwrap().to_str().unwrap();
                    if entry.path_is_symlink() {
                        // Replicate symlink in target
                        let real_path = fs::read_link(&path).unwrap();
                        let real_name = real_path.file_name().unwrap().to_str().unwrap();
                        symlinks.insert(filename.to_owned(), real_name.to_owned());
                        continue;
                    }

                    let target_path = install_host_lib_dir.join(filename);

                    if !target_path.exists() {
                        fs::copy(&path, &install_host_lib_dir.join(filename)).unwrap();
                    } else {
                        let src_metadata = entry.metadata().unwrap();
                        let dst_metadata = target_path.metadata().unwrap();
                        let src_ctime = src_metadata.created().ok();
                        let dst_ctime = dst_metadata.created().ok();
                        // Skip unchanged files
                        if src_ctime.is_some() && dst_ctime.is_some() && src_ctime == dst_ctime {
                            continue;
                        }
                        fs::copy(&path, &install_host_lib_dir.join(filename)).unwrap();
                    }
                }

                for (link_name, file_name) in symlinks.iter() {
                    let src = install_host_lib_dir.join(file_name);
                    let dst = install_host_lib_dir.join(link_name);
                    symlink(&src, &dst);
                }
            }
            _ => {}
        }
    }

    println!("Install complete!");
}

fn get_llvm_target(target: &str) -> String {
    let mut rustc_cmd = Command::new("rustc");
    let rustc_cmd = rustc_cmd
        .args(&[
            "-Z",
            "unstable-options",
            "--print",
            "target-spec-json",
            "--target",
        ])
        .arg(target);

    let output = rustc_cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .unwrap();

    if !output.status.success() {
        panic!("unable to determine llvm target triple!");
    }

    let spec: TargetSpec = serde_json::from_slice(output.stdout.as_slice()).unwrap();
    spec.llvm_target
}

fn get_rust_sysroot() -> PathBuf {
    let mut rustc_cmd = Command::new("rustc");
    let rustc_cmd = rustc_cmd.args(&["--print", "sysroot"]);
    let output = rustc_cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .unwrap();
    if !output.status.success() {
        panic!("unable to determine rust sysroot!");
    }

    let s = String::from_utf8(output.stdout).unwrap();
    PathBuf::from(s.trim().to_owned())
}

fn is_dir_or_llvm_lib(entry: &DirEntry) -> bool {
    let ty = entry.file_type();
    if ty.is_dir() {
        return true;
    }

    let path = entry.path();
    let filename = path.file_name().unwrap().to_str().unwrap();
    filename.starts_with("libMLIR.")
        || filename.starts_with("libLLVM.")
        || filename.starts_with("libLLVM-")
}

fn is_dir_or_matching_rlib(entry: &DirEntry, libs: &[&str]) -> bool {
    // Recurse into subdirectories
    let ty = entry.file_type();
    if ty.is_dir() {
        return true;
    }
    let path = entry.path();
    if let Some(ext) = path.extension() {
        if ext != "rlib" {
            return false;
        }

        let filename = path.file_name().unwrap().to_str().unwrap();
        for lib in &libs[..] {
            if filename.starts_with(lib) {
                return true;
            }
        }
    }

    false
}

fn is_dir_or_library_file(entry: &DirEntry) -> bool {
    // Recurse into subdirectories
    let ty = entry.file_type();
    if ty.is_dir() {
        return true;
    }
    let path = entry.path();
    if let Some(ext) = path.extension() {
        ext == "dylib" || ext == "so" || ext == "a" || ext == "rlib"
    } else {
        false
    }
}

#[cfg(unix)]
fn symlink(src: &Path, dst: &Path) {
    use std::os::unix;

    fs::remove_file(dst).ok();
    unix::fs::symlink(src, dst).unwrap()
}

#[cfg(windows)]
fn symlink(src: &Path, dst: &Path) {
    use std::os::windows::fs::symlink_file;

    fs::remove_file(dst).ok();
    symlink_file(src, dst).unwrap()
}
