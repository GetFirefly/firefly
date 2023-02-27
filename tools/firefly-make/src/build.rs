use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{self, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use anyhow::bail;
use cargo_metadata::{Message, MetadataCommand};
use clap::Args;
use walkdir::{DirEntry, WalkDir};

use crate::util;

#[derive(Args)]
pub struct Config {
    /// The working directory for the build
    #[clap(hide(true), long, env("CARGO_MAKE_WORKING_DIRECTORY"))]
    cwd: Option<PathBuf>,
    /// The path to the root of your LLVM installation, e.g. ~/.local/share/llvm/
    ///
    /// The given path should contain include/ and lib/ directories
    #[clap(long("llvm"), alias("llvm-prefix"), env("LLVM_PREFIX"))]
    llvm_prefix: PathBuf,
    /// Enables more informational output during the build
    ///
    /// This is enabled by default in CI
    #[clap(short, long, env("VERBOSE"))]
    verbose: bool,
    /// When true, this build is being run under CI
    #[clap(hide(true), long, env("CI"))]
    ci: bool,
    /// Enables link-time optimization of the build
    #[clap(long, env("FIREFLY_BUILD_LTO"))]
    lto: bool,
    /// The cargo profile to build with
    #[clap(long, env("FIREFLY_BUILD_PROFILE"), default_value = "debug")]
    profile: String,
    #[clap(hide(true), long, env("FIREFLY_BUILD_TYPE"), default_value = "dynamic")]
    build_type: String,
    /// Whether this build should be statically linked
    #[clap(long("static"))]
    link_static: bool,
    /// Whether this build should be dynamically linked
    #[clap(long("dynamic"), conflicts_with("link-static"))]
    link_dynamic: bool,
    /// If provided, enables building the compiler with the given sanitizer
    #[clap(long, env("SANITIZER"))]
    sanitizer: Option<String>,
    /// The name of the cargo toolchain to use
    #[clap(long, env("CARGO_MAKE_TOOLCHAIN"), default_value = "nightly")]
    toolchain: String,
    /// The name of the target platform to build for
    #[clap(long, env("CARGO_MAKE_RUST_TARGET_TRIPLE"))]
    target_triple: String,
    /// The vendor value of the current Rust target
    #[clap(long, env("CARGO_MAKE_RUST_TARGET_VENDOR"))]
    target_vendor: Option<String>,
    /// The os value of the current Rust target
    #[clap(long, env("CARGO_MAKE_RUST_TARGET_OS"))]
    target_os: Option<String>,
    /// The directory in which cargo will produce its build output
    #[clap(long, env("CARGO_TARGET_DIR"))]
    target_dir: Option<PathBuf>,
    /// The location where the compiler binaries should be symlinked
    #[clap(long, env("FIREFLY_BIN_DIR"), default_value = "./bin")]
    bin_dir: PathBuf,
    /// The location where the compiler toolchain should be installed
    #[clap(long, env("FIREFLY_INSTALL_DIR"), default_value = "./_build")]
    install_dir: PathBuf,
}
impl Config {
    pub fn working_directory(&self) -> PathBuf {
        self.cwd
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap())
    }

    pub fn llvm_prefix(&self) -> &Path {
        self.llvm_prefix.as_path()
    }

    pub fn verbose(&self) -> bool {
        self.verbose || self.ci
    }

    pub fn lto(&self) -> bool {
        self.lto
    }

    pub fn profile(&self) -> &str {
        self.profile.as_str()
    }

    pub fn link_static(&self) -> bool {
        self.link_static || self.build_type == "static" || !self.link_dynamic
    }

    pub fn link_dynamic(&self) -> bool {
        self.link_dynamic || self.build_type == "dynamic"
    }

    pub fn sanitizer(&self) -> Option<&str> {
        self.sanitizer.as_deref()
    }

    pub fn toolchain(&self) -> &str {
        self.toolchain.as_str()
    }

    pub fn rust_target(&self) -> &str {
        self.target_triple.as_str()
    }

    pub fn llvm_target(&self) -> &str {
        util::get_llvm_target(self.toolchain(), self.rust_target())
    }

    pub fn is_darwin(&self) -> bool {
        let is_apple = self
            .target_vendor
            .as_ref()
            .map(|v| v == "apple")
            .unwrap_or_else(|| self.rust_target().contains("apple"));
        let is_macos = self
            .target_os
            .as_ref()
            .map(|os| os == "macos")
            .unwrap_or_else(|| self.rust_target().contains("macos"));
        is_apple || is_macos
    }

    pub fn is_windows(&self) -> bool {
        self.target_os
            .as_ref()
            .map(|os| os == "windows")
            .unwrap_or_else(|| self.rust_target().contains("windows"))
    }

    pub fn is_linux(&self) -> bool {
        !self.is_darwin() && !self.is_windows()
    }

    pub fn bin_dir(&self) -> &Path {
        self.bin_dir.as_path()
    }

    pub fn install_dir(&self) -> &Path {
        self.install_dir.as_path()
    }

    pub fn sysroot(&self) -> &Path {
        get_rust_sysroot()
    }

    pub fn toolchain_target_dir(&self) -> PathBuf {
        self.sysroot().join("lib/rustlib").join(self.rust_target())
    }
}

pub fn run(config: &Config) -> anyhow::Result<()> {
    let cwd = config.working_directory();
    let target_dir = config
        .target_dir
        .clone()
        .unwrap_or_else(|| cwd.join("target"));

    let mut build_link_args = vec!["-Wl".to_owned()];
    let mut extra_cargo_flags = vec![];
    let mut rustflags = env::var("RUSTFLAGS")
        .unwrap_or(String::new())
        .split(' ')
        .map(|flag| flag.to_string())
        .collect::<Vec<_>>();

    if let Ok(f) = env::var("CARGO_MAKE_CARGO_VERBOSE_FLAGS") {
        if !f.is_empty() {
            extra_cargo_flags.push(f.to_owned());
        }
    }

    if let Some(sanitizer) = config.sanitizer() {
        if !sanitizer.is_empty() {
            rustflags.push("-Z".to_owned());
            rustflags.push(format!("sanitizer={}", sanitizer));
        }
    }

    if config.is_linux() && config.link_dynamic() {
        build_link_args.push("-rpath".to_owned());
        build_link_args.push("$ORIGIN/../lib".to_owned());
    }

    let target_subdir = match config.profile() {
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

    if config.link_static() {
        rustflags.push("-C".to_owned());
        rustflags.push("prefer-dynamic=no".to_owned());
    }

    if config.is_darwin() {
        build_link_args.push("-headerpad_max_install_names".to_owned());
    }

    if config.lto() {
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

    println!("Starting build..");

    let metadata = MetadataCommand::new().exec().unwrap();

    let workspace_members = metadata
        .workspace_members
        .iter()
        .cloned()
        .collect::<HashSet<_>>();

    let path_var = env::var("PATH").unwrap();
    let path = format!("{}/bin:{}", config.llvm_prefix().display(), &path_var);

    let mut cargo_cmd = Command::new("rustup");
    let cargo_cmd = cargo_cmd
        .arg("run")
        .arg(config.toolchain())
        .args(&["cargo", "rustc"])
        .args(&["-p", "firefly"])
        .arg("--target")
        .arg(config.rust_target())
        .args(&["--message-format=json-diagnostic-rendered-ansi", "-vv"])
        .args(cargo_args.as_slice())
        .arg("--")
        .arg("--remap-path-prefix")
        .arg(&format!("{}=.", cwd.display()))
        .arg(link_args_string.as_str())
        .env("PATH", path.as_str())
        .env("RUSTFLAGS", rustflags.as_str());

    let cmd = format!("{:?}", &cargo_cmd);

    // Print more verbose output when requested/in CI
    let verbose = config.verbose();
    cargo_cmd.stdout(Stdio::piped());
    if !verbose {
        cargo_cmd.stderr(Stdio::null());
    }
    let mut child = cargo_cmd.spawn().unwrap();

    let mut deps: HashMap<String, Vec<String>> = HashMap::new();
    let mut dylibs: HashMap<String, PathBuf> = HashMap::new();
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
                }
                Message::CompilerArtifact(artifact)
                    if artifact.target.name == "build-script-build" =>
                {
                    let message = format!("Building {}\n", &artifact.package_id.repr);
                    handle.write_all(message.as_bytes()).unwrap();
                }
                Message::CompilerArtifact(mut artifact) => {
                    let message = format!("Compiled {}\n", &artifact.package_id.repr);
                    handle.write_all(message.as_bytes()).unwrap();
                    // Track the artifacts for workspace members as we need them later
                    if workspace_members.contains(&artifact.package_id) {
                        let files = artifact
                            .filenames
                            .drain_filter(|f| {
                                let p = f.as_path();
                                let ext = p.extension();
                                ext == Some("a") || ext == Some("rlib")
                            })
                            .map(|f| f.into_string())
                            .collect::<Vec<_>>();
                        if !files.is_empty() {
                            deps.insert(artifact.target.name.clone(), files);
                        }
                    }
                }
                Message::BuildScriptExecuted(mut script) => {
                    // Track dynamic libraries built by crates in our workspace so
                    // that they can be added to the libs folder in the toolchain
                    if workspace_members.contains(&script.package_id) {
                        let out_dir = script.out_dir.into_std_path_buf().join("lib");
                        for ll in script.linked_libs.drain(..) {
                            let ll = ll.into_string();
                            match ll.split_once('=') {
                                Some(("dylib", lib)) => {
                                    let name = format!("lib{}.dylib", &lib);
                                    let path = out_dir.join(&name);
                                    if path.exists() {
                                        dylibs.insert(lib.to_string(), path);
                                    }
                                }
                                _ => continue,
                            }
                        }
                    }
                    continue;
                }
                Message::BuildFinished(result) if result.success => {
                    handle
                        .write_all(b"Build completed successfully!\n")
                        .unwrap();
                }
                Message::BuildFinished(_) => {
                    handle.write_all(b"Build finished with errors!\n").unwrap();
                }
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
        bail!(
            "command did not execute successfully: {}\n\
            expected success, got: {}",
            cmd,
            output
        );
    }

    let llvm_target = config.llvm_target();
    let install_dir = config.install_dir();
    let install_bin_dir = install_dir.join("bin");
    let install_host_lib_dir = install_dir.join("lib");
    let install_target_lib_dir = install_dir.join(&format!("lib/fireflylib/{}/lib", &llvm_target));
    let install_target_bin_dir = install_dir.join(&format!("lib/fireflylib/{}/bin", &llvm_target));

    println!("Preparing to install Firefly to {}", install_dir.display());

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
    if !install_target_bin_dir.exists() {
        fs::create_dir_all(&install_target_bin_dir)
            .expect("failed to create install target bin directory");
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

    println!("Installing Firefly..");

    let src_firefly_exe = target_dir
        .join(config.rust_target())
        .join(target_subdir)
        .join("firefly");
    if !src_firefly_exe.exists() {
        panic!(
            "Expected build to place Firefly executable at {}",
            src_firefly_exe.display()
        );
    }

    let firefly_exe = install_bin_dir.join("firefly");
    if firefly_exe.exists() {
        fs::remove_file(&firefly_exe).expect("failed to remove existing firefly executable");
    }
    fs::copy(src_firefly_exe, &firefly_exe).unwrap();

    symlink(&firefly_exe, config.bin_dir().join("firefly"));

    if config.is_darwin() {
        println!("Patching runtime path..");

        let mut install_name_tool_cmd = Command::new("install_name_tool");
        let install_name_tool_cmd = install_name_tool_cmd
            .args(&["-add_rpath", "@executable_path/../lib"])
            .arg(&format!("{}", firefly_exe.display()));

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

    let rustlibs = &["libpanic_abort", "libpanic_unwind", "libunwind"];
    let walker = WalkDir::new(config.toolchain_target_dir().join("lib")).into_iter();
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

    let firefly_libs = &["firefly_emulator"];
    for lib in firefly_libs.iter().copied() {
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

    println!("Installing toolchain binaries..");
    let llvm_bins = config.llvm_prefix().join("bin");
    let lld_src = llvm_bins.join(exe("lld", &llvm_target));
    let lld_dest = install_target_bin_dir.join(exe("firefly-lld", &llvm_target));
    fs::copy(lld_src, lld_dest).expect("could not copy lld");
    // for `-C gcc-ld=lld`
    let lld_wrapper = target_dir.join(exe("firefly-lld", &llvm_target));
    let gcc_ld_dir = install_target_bin_dir.join("gcc-ld");
    fs::create_dir_all(&gcc_ld_dir).unwrap();

    for lld_name in &["ld.lld", "ld64.lld", "lld-link", "wasm-ld"] {
        let dest = gcc_ld_dir.join(exe(lld_name, &llvm_target));
        fs::copy(lld_wrapper.as_path(), dest).expect("could not copy firefly-lld wrapper");
    }

    if config.link_dynamic() {
        // Copy all dylibs built by our workspace crates into the lib directory
        for (_link_name, path) in dylibs.iter() {
            let filename = path.file_name().unwrap().to_str().unwrap();
            let target_path = install_host_lib_dir.join(filename);
            if !target_path.exists() {
                fs::copy(&path, &target_path).unwrap();
            } else {
                let src_metadata = path.metadata().unwrap();
                let dst_metadata = target_path.metadata().unwrap();
                let src_ctime = src_metadata.created().ok();
                let dst_ctime = dst_metadata.created().ok();
                // Skip unchanged files
                if src_ctime.is_some() && dst_ctime.is_some() && src_ctime == dst_ctime {
                    continue;
                }
                fs::copy(&path, &target_path).unwrap();
            }
        }

        match env::var_os("LLVM_LINK_LLVM_DYLIB") {
            Some(val) if val == "ON" => {
                let walker = WalkDir::new(config.llvm_prefix().join("lib")).into_iter();
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
                        fs::copy(&path, &target_path).unwrap();
                    } else {
                        let src_metadata = entry.metadata().unwrap();
                        let dst_metadata = target_path.metadata().unwrap();
                        let src_ctime = src_metadata.created().ok();
                        let dst_ctime = dst_metadata.created().ok();
                        // Skip unchanged files
                        if src_ctime.is_some() && dst_ctime.is_some() && src_ctime == dst_ctime {
                            continue;
                        }
                        fs::copy(&path, &target_path).unwrap();
                    }
                }

                for (link_name, file_name) in symlinks.iter() {
                    let src = install_host_lib_dir.join(file_name);
                    let dst = install_host_lib_dir.join(link_name);
                    symlink(&src, dst);
                }
            }
            _ => {}
        }
    }

    println!("Install complete!");
    Ok(())
}

static RUST_SYSROOT: OnceLock<PathBuf> = OnceLock::new();

fn get_rust_sysroot() -> &'static Path {
    let path = RUST_SYSROOT.get_or_init(|| {
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
    });
    path.as_path()
}

fn is_dir_or_llvm_lib(entry: &DirEntry) -> bool {
    let ty = entry.file_type();
    if ty.is_dir() {
        return true;
    }

    let path = entry.path();
    let filename = path.file_name().unwrap().to_str().unwrap();
    filename.starts_with("libMLIR.")
        || filename.starts_with("libMLIR-.")
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
fn symlink(src: &Path, dst: PathBuf) {
    use std::os::unix;

    fs::remove_file(dst.as_path()).ok();
    unix::fs::symlink(src, dst.as_path()).unwrap()
}

#[cfg(windows)]
fn symlink(src: &Path, dst: PathBuf) {
    use std::os::windows::fs::symlink_file;

    fs::remove_file(dst.as_path()).ok();
    symlink_file(src, dst.as_path()).unwrap()
}

fn exe(name: &str, target: &str) -> String {
    if target.contains("windows") {
        format!("{}.exe", name)
    } else {
        name.to_string()
    }
}
