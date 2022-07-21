//! ```cargo
//! [dependencies]
//! cargo_metadata = "0.15"
//! serde = { version = "1.0", features = ["derive"] }
//! serde_json = "1.0"
//! walkdir = "*"
//! ```
#![feature(drain_filter)]
#![feature(slice_internals)]
#![feature(slice_concat_trait)]
#![allow(non_snake_case)]

extern crate core;
extern crate serde;
extern crate serde_json;
extern crate walkdir;
extern crate cargo_metadata;

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{self, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Deserialize;
use walkdir::{DirEntry, WalkDir};
use cargo_metadata::{Message, MetadataCommand};

#[derive(Deserialize)]
struct TargetSpec {
    #[serde(rename = "llvm-target")]
    llvm_target: String,
}

fn main() -> Result<(), ()> {
    let cargo_profile = env::var("LUMEN_BUILD_PROFILE").unwrap();
    let toolchain_name = env::var("CARGO_MAKE_TOOLCHAIN").unwrap();
    let rust_target_triple = env::var("CARGO_MAKE_RUST_TARGET_TRIPLE").unwrap();
    let target_triple = get_llvm_target(&toolchain_name, &rust_target_triple);
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
    let verbose = env::var("VERBOSE").is_ok();
    let cwd = env::var("CARGO_MAKE_WORKING_DIRECTORY").unwrap();

    let mut build_link_args = vec!["-Wl".to_owned()];
    let mut extra_cargo_flags = vec![];
    let mut rustflags = env::var("RUSTFLAGS").unwrap_or(String::new())
        .split(' ')
        .map(|flag| flag.to_string())
        .collect::<Vec<_>>();


    let verbose_flags = env::var("CARGO_MAKE_CARGO_VERBOSE_FLAGS");
    if let Ok(f) = verbose_flags {
        if !f.is_empty() {
            extra_cargo_flags.push(f.to_owned());
        }
    }

    let sanitizer = env::var("SANITIZER");
    if let Ok(sanitizer) = sanitizer {
        if !sanitizer.is_empty() {
            rustflags.push("-Z".to_owned());
            rustflags.push(format!("sanitizer={}", sanitizer));
        }
    }

    let is_darwin = target_vendor == "apple";
    let is_linux = target_os != "macos" && target_os != "windows";

    if is_linux && build_type != "static" {
        build_link_args.push("-rpath".to_owned());
        build_link_args.push("$ORIGIN/../lib".to_owned());
    }

    let target_subdir = match cargo_profile.as_str() {
        "release" => {
            extra_cargo_flags.push("--release".to_owned());
            "release"
        }
        "dev" | _ => {
            rustflags.push("-C".to_owned());
            rustflags.push("opt-level=0".to_owned());
            rustflags.push("-C".to_owned());
            rustflags.push("debuginfo=2".to_owned());
            "debug"
        }
    };

    if build_type == "static" {
        rustflags.push("-C".to_owned());
        rustflags.push("prefer-dynamic=no".to_owned());
    }

    if is_darwin {
        build_link_args.push("-headerpad_max_install_names".to_owned());
    }

    if enable_lto {
        build_link_args.push("-flto=thin".to_owned());
        rustflags.push("-C".to_owned());
        rustflags.push("embed-bitcode=yes".to_owned());
        rustflags.push("-C".to_owned());
        rustflags.push("lto=thin".to_owned());
    }

    rustflags.push("-Z".to_owned());
    rustflags.push("remap-cwd-prefix=.".to_owned());

    build_link_args.push("-v".to_string());
    let link_args = build_link_args.join(",");
    let link_args_string = format!("-Clink-args={}", &link_args);
    let cargo_args = extra_cargo_flags.iter().collect::<Vec<_>>();
    let rustflags = rustflags.as_slice().join(" ");

    let path_var = env::var("PATH").unwrap();
    let path = format!("{}/bin:{}", llvm_prefix.display(), &path_var);

    println!("Starting build..");

    let metadata = MetadataCommand::new()
        .exec()
        .unwrap();

    let workspace_members = metadata.workspace_members.iter().cloned().collect::<HashSet<_>>();

    let mut cargo_cmd = Command::new("rustup");
    let cargo_cmd = cargo_cmd
        .arg("run")
        .arg(&toolchain_name)
        .args(&["cargo", "rustc"])
        .args(&["-p", "lumen"])
        .arg("--target")
        .arg(rust_target_triple.as_str())
        .args(&["--message-format=json-diagnostic-rendered-ansi", "-vv"])
        .args(cargo_args.as_slice())
        .arg("--")
        .arg("--remap-path-prefix")
        .arg(&format!("{}=.", &cwd))
        .arg(link_args_string.as_str())
        .env("PATH", path.as_str())
        .env("RUSTFLAGS", rustflags.as_str());

    let cmd = format!("{:?}", &cargo_cmd);
    let mut child = cargo_cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();

    let mut deps: HashMap<String, Vec<String>> = HashMap::new();
    {
        let child_stdout = child.stdout.take().unwrap();
        let child_stdout_reader = BufReader::new(child_stdout);
        let stdout = io::stdout();
        let mut handle = stdout.lock();

        for message in Message::parse_stream(child_stdout_reader) {
            match message.unwrap() {
                Message::CompilerMessage(msg) => {
                    use cargo_metadata::diagnostic::DiagnosticLevel;

                    match msg.message.level {
                        DiagnosticLevel::Ice
                        | DiagnosticLevel::Error
                        | DiagnosticLevel::FailureNote => {
                          if let Some(msg) = msg.message.rendered.as_ref() {
                              handle.write_all(msg.as_bytes()).unwrap();
                          }
                        }
                        _ if workspace_members.contains(&msg.package_id) || verbose => {
                            // This message is relevant to one of our crates
                            if let Some(msg) = msg.message.rendered.as_ref() {
                                handle.write_all(msg.as_bytes()).unwrap();
                            }
                        }
                        _ => continue,
                    }
                },
                Message::CompilerArtifact(artifact) if artifact.target.name == "build-script-build" => {
                    let message = format!("Building {}\n", &artifact.package_id.repr);
                    handle.write_all(message.as_bytes()).unwrap();
                },
                Message::CompilerArtifact(mut artifact) => {
                    let message = format!("Compiled {}\n", &artifact.package_id.repr);
                    handle.write_all(message.as_bytes()).unwrap();
                    // Track the artifacts for workspace members as we need them later
                    if workspace_members.contains(&artifact.package_id) {
                        let files = artifact.filenames
                            .drain_filter(|f| {
                                let p = f.as_path();
                                let ext = p.extension();
                                ext == Some("a") || ext == Some("rlib")
                            }).map(|f| f.into_string()).collect::<Vec<_>>();
                        if !files.is_empty() {
                            deps.insert(artifact.target.name.clone(), files);
                        }
                    }
                }
                Message::BuildScriptExecuted(_script) => {
                    continue;
                },
                Message::BuildFinished(result) if result.success => {
                    handle.write_all(b"Build completed successfully!\n").unwrap();
                },
                Message::BuildFinished(_) => {
                    handle.write_all(b"Build finished with errors!\n").unwrap();
                },
                Message::TextLine(s) => {
                    // Unknown message content
                    handle.write_all(s.as_bytes()).unwrap();
                    handle.write_all(b"\n").unwrap();
                }
                // Unhandled message type (this enum is non-exhaustive)
                _ => continue,
            }
        }
    }

    println!("Build command completed, waiting for exit..");

    let output = child.wait().unwrap();
    if !output.success() {
        eprintln!(
            "command did not execute successfully: {}\n\
            expected success, got: {}",
            cmd, output
        );
        return Err(());
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

    let src_lumen_exe = target_dir.join(&rust_target_triple).join(target_subdir).join("lumen");
    if !src_lumen_exe.exists() {
        panic!(
            "Expected build to place Lumen executable at {}",
            src_lumen_exe.display()
        );
    }

    let lumen_exe = install_bin_dir.join("lumen");
    if lumen_exe.exists() {
        fs::remove_file(&lumen_exe).expect("failed to remove existing lumen executable");
    }
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

    let lumenlibs = &["lumen_rt_tiny", "panic", "unwind"];
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
    return Ok(());
}

fn get_llvm_target(toolchain_name: &str, target: &str) -> String {
   let mut rustc_cmd = Command::new("rustup");
    let rustc_cmd = rustc_cmd
        .arg("run")
        .arg(toolchain_name)
        .args(&["rustc"])
        .args(&["-Z", "unstable-options"])
        .args(&["--print", "target-spec-json", "--target"])
        .arg(target);

    let output = rustc_cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();

    if !output.status.success() {
        panic!("unable to determine llvm target triple!: {}", String::from_utf8(output.stderr).unwrap());
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
