pub(crate) mod archive;
mod command;
pub(crate) mod link;
mod rpath;

use std::borrow::Borrow;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::mem;
use std::path::{Path, PathBuf};

use log::{debug, warn};

use liblumen_session::{DebugInfo, LinkerPluginLto, OptLevel, Options, ProjectType, Strip};
use liblumen_target::spec::LinkOutputKind;
use liblumen_target::{LinkerFlavor, LldFlavor};
use liblumen_util::diagnostics::DiagnosticsHandler;

use self::command::Command;
pub use self::link::link_binary;

// The third parameter is for env vars, used on windows to set up the
// path for MSVC to find its DLLs, and gcc to find its bundled
// toolchain
pub fn get_linker<'a>(
    options: &'a Options,
    diagnostics: &'a DiagnosticsHandler,
    linker: &Path,
    flavor: LinkerFlavor,
    self_contained: bool,
    target_cpu: &'a str,
) -> Box<dyn Linker + 'a> {
    let msvc_tool = cc::windows_registry::find_tool(&options.target.triple(), "link.exe");

    // If our linker looks like a batch script on Windows then to execute this
    // we'll need to spawn `cmd` explicitly. This is primarily done to handle
    // emscripten where the linker is `emcc.bat` and needs to be spawned as
    // `cmd /c emcc.bat ...`.
    //
    // This worked historically but is needed manually since #42436 (regression
    // was tagged as #42791) and some more info can be found on #44443 for
    // emscripten itself.
    let mut cmd = match linker.to_str() {
        Some(linker) if cfg!(windows) && linker.ends_with(".bat") => Command::bat_script(linker),
        _ => match flavor {
            LinkerFlavor::Lld(f) => Command::lld(linker, f),
            LinkerFlavor::Msvc
                if options.codegen_opts.linker.is_none()
                    && options.target.options.linker.is_none() =>
            {
                Command::new(msvc_tool.as_ref().map_or(linker, |t| t.path()))
            }
            _ => Command::new(linker),
        },
    };

    // UWP apps have API restrictions enforced during Store submissions.
    // To comply with the Windows App Certification Kit,
    // MSVC needs to link with the Store versions of the runtime libraries (vcruntime, msvcrt, etc).
    let t = &options.target;
    if (flavor == LinkerFlavor::Msvc || flavor == LinkerFlavor::Lld(LldFlavor::Link))
        && t.options.vendor == "uwp"
    {
        if let Some(ref tool) = msvc_tool {
            let original_path = tool.path();
            if let Some(ref root_lib_path) = original_path.ancestors().skip(4).next() {
                let arch = match t.arch.borrow() {
                    "x86_64" => Some("x64".to_string()),
                    "x86" => Some("x86".to_string()),
                    "aarch64" => Some("arm64".to_string()),
                    "arm" => Some("arm".to_string()),
                    _ => None,
                };
                if let Some(ref a) = arch {
                    let mut arg = OsString::from("/LIBPATH:");
                    arg.push(format!(
                        "{}\\lib\\{}\\store",
                        root_lib_path.display(),
                        a.to_string()
                    ));
                    cmd.arg(&arg);
                } else {
                    warn!("arch is not supported");
                }
            } else {
                warn!("MSVC root path lib location not found");
            }
        } else {
            warn!("link.exe not found");
        }
    }

    // The compiler's sysroot often has some bundled tools, so add it to the
    // PATH for the child.
    let mut new_path = options.get_tools_search_paths(self_contained);
    let mut msvc_changed_path = false;
    if options.target.options.is_like_msvc {
        if let Some(ref tool) = msvc_tool {
            cmd.args(tool.args());
            for &(ref k, ref v) in tool.env() {
                if k == "PATH" {
                    new_path.extend(env::split_paths(v));
                    msvc_changed_path = true;
                } else {
                    cmd.env(k, v);
                }
            }
        }
    }

    if !msvc_changed_path {
        if let Some(path) = env::var_os("PATH") {
            new_path.extend(env::split_paths(&path));
        }
    }
    cmd.env("PATH", env::join_paths(new_path).unwrap());

    assert!(cmd.get_args().is_empty() || options.target.options.vendor == "uwp");
    match flavor {
        LinkerFlavor::Lld(LldFlavor::Link) | LinkerFlavor::Msvc => Box::new(MsvcLinker {
            cmd,
            options,
            diagnostics,
        }) as Box<dyn Linker>,
        LinkerFlavor::Em => Box::new(EmLinker {
            cmd,
            options,
            diagnostics,
        }) as Box<dyn Linker>,
        LinkerFlavor::Gcc => Box::new(GccLinker {
            cmd,
            options,
            diagnostics,
            target_cpu,
            hinted_static: false,
            is_ld: false,
        }) as Box<dyn Linker>,

        LinkerFlavor::Lld(LldFlavor::Ld)
        | LinkerFlavor::Lld(LldFlavor::Ld64)
        | LinkerFlavor::Ld => Box::new(GccLinker {
            cmd,
            options,
            diagnostics,
            target_cpu,
            hinted_static: false,
            is_ld: true,
        }) as Box<dyn Linker>,

        LinkerFlavor::Lld(LldFlavor::Wasm) => {
            Box::new(WasmLd::new(cmd, options, diagnostics)) as Box<dyn Linker>
        }

        other => panic!("unsupported linker flavor: {}", other),
    }
}

/// Linker abstraction used by `back::link` to build up the command to invoke a
/// linker.
///
/// This trait is the total list of requirements needed by `back::link` and
/// represents the meaning of each option being passed down. This trait is then
/// used to dispatch on whether a GNU-like linker (generally `ld.exe`) or an
/// MSVC linker (e.g., `link.exe`) is being used.
pub trait Linker {
    fn cmd(&mut self) -> &mut Command;
    fn is_ld(&self) -> bool {
        false
    }
    fn set_output_kind(&mut self, output_kind: LinkOutputKind, out_filename: &Path);
    fn link_dylib(&mut self, lib: &str, verbatim: bool, as_needed: bool);
    fn link_rust_dylib(&mut self, lib: &str, path: &Path);
    fn link_framework(&mut self, framework: &str, as_needed: bool);
    fn link_staticlib(&mut self, lib: &str, verbatim: bool);
    fn link_rlib(&mut self, lib: &Path);
    fn link_whole_rlib(&mut self, lib: &Path);
    fn link_whole_staticlib(&mut self, lib: &str, verbatim: bool, search_path: &[PathBuf]);
    fn include_path(&mut self, path: &Path);
    fn framework_path(&mut self, path: &Path);
    fn output_filename(&mut self, path: &Path);
    fn add_object(&mut self, path: &Path);
    fn gc_sections(&mut self, keep_metadata: bool);
    fn no_gc_sections(&mut self);
    fn full_relro(&mut self);
    fn partial_relro(&mut self);
    fn no_relro(&mut self);
    fn optimize(&mut self);
    fn pgo_gen(&mut self);
    fn control_flow_guard(&mut self);
    fn debuginfo(&mut self, strip: Strip);
    fn no_crt_objects(&mut self);
    fn no_default_libraries(&mut self);
    fn export_symbols(&mut self, tmpdir: &Path, project_type: ProjectType, symbols: &[String]);
    fn exported_symbol_means_used_symbol(&self) -> bool {
        true
    }
    fn subsystem(&mut self, subsystem: &str);
    fn group_start(&mut self);
    fn group_end(&mut self);
    fn linker_plugin_lto(&mut self);
    fn add_eh_frame_header(&mut self) {}
    fn add_no_exec(&mut self) {}
    fn add_as_needed(&mut self) {}
    fn reset_per_library_state(&mut self) {}
}

impl dyn Linker + '_ {
    pub fn arg(&mut self, arg: impl AsRef<OsStr>) {
        self.cmd().arg(arg);
    }

    pub fn args(&mut self, args: impl IntoIterator<Item: AsRef<OsStr>>) {
        self.cmd().args(args);
    }

    pub fn take_cmd(&mut self) -> Command {
        mem::replace(self.cmd(), Command::new(""))
    }
}

pub struct GccLinker<'a> {
    cmd: Command,
    options: &'a Options,
    diagnostics: &'a DiagnosticsHandler,
    target_cpu: &'a str,
    hinted_static: bool, // Keeps track of the current hinting mode.
    // Link as ld
    is_ld: bool,
}
impl<'a> GccLinker<'a> {
    /// Passes an argument directly to the linker.
    ///
    /// When the linker is not ld-like such as when using a compiler as a linker, the argument is
    /// prepended by `-Wl,`.
    fn linker_arg<S>(&mut self, arg: S) -> &mut Self
    where
        S: AsRef<OsStr>,
    {
        self.linker_args(&[arg]);
        self
    }

    /// Passes a series of arguments directly to the linker.
    ///
    /// When the linker is ld-like, the arguments are simply appended to the command. When the
    /// linker is not ld-like such as when using a compiler as a linker, the arguments are joined by
    /// commas to form an argument that is then prepended with `-Wl`. In this situation, only a
    /// single argument is appended to the command to ensure that the order of the arguments is
    /// preserved by the compiler.
    fn linker_args(&mut self, args: &[impl AsRef<OsStr>]) -> &mut Self {
        if self.is_ld {
            args.into_iter().for_each(|a| {
                self.cmd.arg(a);
            });
        } else {
            if !args.is_empty() {
                let mut os = OsString::from("-Wl");
                for a in args {
                    os.push(",");
                    os.push(a);
                }
                self.cmd.arg(os);
            }
        }
        self
    }

    fn takes_hints(&self) -> bool {
        // Really this function only returns true if the underlying linker
        // configured for a compiler is binutils `ld.bfd` and `ld.gold`. We
        // don't really have a foolproof way to detect that, so rule out some
        // platforms where currently this is guaranteed to *not* be the case:
        //
        // * On OSX they have their own linker, not binutils'
        // * For WebAssembly the only functional linker is LLD, which doesn't support hint flags
        !self.options.target.options.is_like_osx && self.options.target.options.is_like_wasm
    }

    // Some platforms take hints about whether a library is static or dynamic.
    // For those that support this, we ensure we pass the option if the library
    // was flagged "static" (most defaults are dynamic) to ensure that if
    // libfoo.a and libfoo.so both exist that the right one is chosen.
    fn hint_static(&mut self) {
        if !self.takes_hints() {
            return;
        }
        if !self.hinted_static {
            self.linker_arg("-Bstatic");
            self.hinted_static = true;
        }
    }

    fn hint_dynamic(&mut self) {
        if !self.takes_hints() {
            return;
        }
        if self.hinted_static {
            self.linker_arg("-Bdynamic");
            self.hinted_static = false;
        }
    }

    fn push_linker_plugin_lto_args(&mut self, plugin_path: Option<&OsStr>) {
        if let Some(plugin_path) = plugin_path {
            let mut arg = OsString::from("-plugin=");
            arg.push(plugin_path);
            self.linker_arg(&arg);
        }
        // NOTE: We previously short-circuit returned if no plugin path was set due to an error from the linker

        let opt_level = match self.options.opt_level {
            OptLevel::No => "O0",
            OptLevel::Less => "O1",
            OptLevel::Default | OptLevel::Size | OptLevel::SizeMin => "O2",
            OptLevel::Aggressive => "O3",
        };

        self.linker_args(&[
            &format!("-plugin-opt={}", opt_level),
            &format!("-plugin-opt=mcpu={}", self.target_cpu),
        ]);
    }

    fn build_dylib(&mut self, out_filename: &Path) {
        // On mac we need to tell the linker to let this library be rpathed
        if self.options.target.options.is_like_osx {
            if !self.is_ld {
                self.cmd.arg("-dynamiclib");
            }

            self.linker_arg("-dylib");

            // Note that the `osx_rpath_install_name` option here is a hack
            // purely to support rustbuild right now, we should get a more
            // principled solution at some point to force the compiler to pass
            // the right `-Wl,-install_name` with an `@rpath` in it.
            if self.options.codegen_opts.rpath || self.options.codegen_opts.osx_rpath_install_name {
                let mut v = OsString::from("@rpath/");
                v.push(out_filename.file_name().unwrap());
                self.linker_args(&[OsString::from("-install_name"), v]);
            }
        } else {
            self.cmd.arg("-shared");
            if self.options.target.options.is_like_windows {
                // The output filename already contains `dll_suffix` so
                // the resulting import library will have a name in the
                // form of libfoo.dll.a
                let implib_name =
                    out_filename
                        .file_name()
                        .and_then(|file| file.to_str())
                        .map(|file| {
                            format!(
                                "{}{}{}",
                                self.options.target.options.staticlib_prefix,
                                file,
                                self.options.target.options.staticlib_suffix
                            )
                        });
                if let Some(implib_name) = implib_name {
                    let implib = out_filename.parent().map(|dir| dir.join(&implib_name));
                    if let Some(implib) = implib {
                        self.linker_arg(&format!("--out-implib={}", (*implib).to_str().unwrap()));
                    }
                }
            }
        }
    }
}
impl<'a> Linker for GccLinker<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn is_ld(&self) -> bool {
        self.is_ld
    }

    fn set_output_kind(&mut self, output_kind: LinkOutputKind, output_file: &Path) {
        match output_kind {
            LinkOutputKind::DynamicNoPicExe => {
                if !self.is_ld && self.options.target.options.linker_is_gnu {
                    self.cmd.arg("-no-pie");
                }
            }
            LinkOutputKind::DynamicPicExe => {
                // `-pie` works for both gcc wrapper and ld.
                if !self.options.target.options.is_like_windows {
                    // `-pie` works for both gcc wrapper and ld.
                    self.cmd.arg("-pie");
                }
            }
            LinkOutputKind::StaticNoPicExe => {
                // `-static` works for both gcc wrapper and ld.
                self.cmd.arg("-static");
                if !self.is_ld && self.options.target.options.linker_is_gnu {
                    self.cmd.arg("-no-pie");
                }
            }
            LinkOutputKind::StaticPicExe => {
                if !self.is_ld {
                    // Note that combination `-static -pie` doesn't work as expected
                    // for the gcc wrapper, `-static` in that case suppresses `-pie`.
                    self.cmd.arg("-static-pie");
                } else {
                    // `--no-dynamic-linker` and `-z text` are not strictly necessary for producing
                    // a static pie, but currently passed because gcc and clang pass them.
                    // The former suppresses the `INTERP` ELF header specifying dynamic linker,
                    // which is otherwise implicitly injected by ld (but not lld).
                    // The latter doesn't change anything, only ensures that everything is pic.
                    self.cmd
                        .args(&["-static", "-pie", "--no-dynamic-linker", "-z", "text"]);
                }
            }
            LinkOutputKind::DynamicDylib => self.build_dylib(output_file),
            LinkOutputKind::StaticDylib => {
                self.cmd.arg("-static");
                self.build_dylib(output_file);
            }
            LinkOutputKind::WasiReactorExe => {
                self.linker_args(&["--entry", "_initialize"]);
            }
        }
        // VxWorks compiler driver introduced `--static-crt` flag specifically for rustc,
        // it switches linking for libc and similar system libraries to static without using
        // any `#[link]` attributes in the `libc` crate, see #72782 for details.
        // FIXME: Switch to using `#[link]` attributes in the `libc` crate
        // similarly to other targets.
        if self.options.target.options.os == "vxworks"
            && matches!(
                output_kind,
                LinkOutputKind::StaticNoPicExe
                    | LinkOutputKind::StaticPicExe
                    | LinkOutputKind::StaticDylib
            )
        {
            self.cmd.arg("--static-crt");
        }
    }

    fn link_dylib(&mut self, lib: &str, verbatim: bool, as_needed: bool) {
        if self.options.target.options.os == "illumos" && lib == "c" {
            // libc will be added via late_link_args on illumos so that it will appear
            // last in the library search order.
            // FIXME: This should be replaced by a more complete and generic mechanism
            // for controlling the order of library arguments passed to the linker.
            return;
        }
        if !as_needed {
            if self.options.target.options.is_like_osx {
                // FIXME(81490): ld64 doesn't support these flags but macOS 11
                // has -needed-l{} / -needed_library {}
                // but we have no way to detect that here.
                self.diagnostics
                    .warn("`as-needed` modifier not implemented yet for ld64");
            } else if self.options.target.options.linker_is_gnu
                && !self.options.target.options.is_like_windows
            {
                self.linker_arg("--no-as-needed");
            } else {
                self.diagnostics
                    .warn("`as-needed` modifier not supported for current linker");
            }
        }
        self.hint_dynamic();
        self.cmd
            .arg(format!("-l{}{}", if verbatim { ":" } else { "" }, lib));
        if !as_needed {
            if self.options.target.options.is_like_osx {
                // See above FIXME comment
            } else if self.options.target.options.linker_is_gnu
                && !self.options.target.options.is_like_windows
            {
                self.linker_arg("--as-needed");
            }
        }
    }
    fn link_staticlib(&mut self, lib: &str, verbatim: bool) {
        self.hint_static();
        self.cmd
            .arg(format!("-l{}{}", if verbatim { ":" } else { "" }, lib));
    }
    fn link_rlib(&mut self, lib: &Path) {
        self.hint_static();
        self.cmd.arg(lib);
    }
    fn include_path(&mut self, path: &Path) {
        self.cmd.arg("-L").arg(path);
    }
    fn framework_path(&mut self, path: &Path) {
        self.cmd.arg("-F").arg(path);
    }
    fn output_filename(&mut self, path: &Path) {
        self.cmd.arg("-o").arg(path);
    }
    fn add_object(&mut self, path: &Path) {
        self.cmd.arg(path);
    }
    fn full_relro(&mut self) {
        self.linker_args(&["-zrelro", "-znow"]);
    }
    fn partial_relro(&mut self) {
        self.linker_arg("-zrelro");
    }
    fn no_relro(&mut self) {
        self.linker_arg("-znorelro");
    }
    fn link_rust_dylib(&mut self, lib: &str, _path: &Path) {
        self.hint_dynamic();
        self.cmd.arg(format!("-l{}", lib));
    }

    fn link_framework(&mut self, framework: &str, as_needed: bool) {
        self.hint_dynamic();
        if !as_needed {
            // FIXME(81490): ld64 as of macOS 11 supports the -needed_framework
            // flag but we have no way to detect that here.
            // self.cmd.arg("-needed_framework").sym_arg(framework);
            self.diagnostics
                .warn("`as-needed` modifier not implemented yet for ld64");
        }
        self.cmd.arg("-framework").arg(framework);
    }

    // Here we explicitly ask that the entire archive is included into the
    // result artifact. For more details see #15460, but the gist is that
    // the linker will strip away any unused objects in the archive if we
    // don't otherwise explicitly reference them. This can occur for
    // libraries which are just providing bindings, libraries with generic
    // functions, etc.
    fn link_whole_staticlib(&mut self, lib: &str, verbatim: bool, search_path: &[PathBuf]) {
        self.hint_static();
        let target = &self.options.target;
        if !target.options.is_like_osx {
            self.linker_arg("--whole-archive").cmd.arg(format!(
                "-l{}{}",
                if verbatim { ":" } else { "" },
                lib
            ));
            self.linker_arg("--no-whole-archive");
        } else {
            // -force_load is the macOS equivalent of --whole-archive, but it
            // involves passing the full path to the library to link.
            self.linker_arg("-force_load");
            match archive::find_library(lib, verbatim, search_path, &self.options) {
                Ok(ref lib) => {
                    self.linker_arg(lib);
                }
                Err(err) => self.diagnostics.fatal(format!("{}", err)).raise(),
            }
        }
    }

    fn link_whole_rlib(&mut self, lib: &Path) {
        self.hint_static();
        if self.options.target.options.is_like_osx {
            self.linker_arg("-force_load");
            self.linker_arg(&lib);
        } else {
            self.linker_arg("--whole-archive").cmd.arg(lib);
            self.linker_arg("--no-whole-archive");
        }
    }

    fn gc_sections(&mut self, keep_metadata: bool) {
        // The dead_strip option to the linker specifies that functions and data
        // unreachable by the entry point will be removed. This is quite useful
        // with Rust's compilation model of compiling libraries at a time into
        // one object file. For example, this brings hello world from 1.7MB to
        // 458K.
        //
        // Note that this is done for both executables and dynamic libraries. We
        // won't get much benefit from dylibs because LLVM will have already
        // stripped away as much as it could. This has not been seen to impact
        // link times negatively.
        //
        // -dead_strip can't be part of the pre_link_args because it's also used
        // for partial linking when using multiple codegen units (-r).  So we
        // insert it here.
        if self.options.target.options.is_like_osx {
            self.linker_arg("-dead_strip");

        // If we're building a dylib, we don't use --gc-sections because LLVM
        // has already done the best it can do, and we also don't want to
        // eliminate the metadata. If we're building an executable, however,
        // --gc-sections drops the size of hello world from 1.8MB to 597K, a 67%
        // reduction.
        } else if (self.options.target.options.linker_is_gnu
            || self.options.target.options.is_like_wasm)
            && !keep_metadata
        {
            self.linker_arg("--gc-sections");
        }
    }

    fn no_gc_sections(&mut self) {
        if self.options.target.options.is_like_osx {
            self.linker_arg("-no_dead_strip");
        } else if self.options.target.options.linker_is_gnu
            || self.options.target.options.is_like_wasm
        {
            self.linker_arg("--no-gc-sections");
        }
    }

    fn optimize(&mut self) {
        if !self.options.target.options.linker_is_gnu && !self.options.target.options.is_like_wasm {
            return;
        }

        // GNU-style linkers support optimization with -O. GNU ld doesn't
        // need a numeric argument, but other linkers do.
        if self.options.opt_level == OptLevel::Default
            || self.options.opt_level == OptLevel::Aggressive
        {
            self.linker_arg("-O1");
        }
    }

    fn pgo_gen(&mut self) {
        if !self.options.target.options.linker_is_gnu {
            return;
        }

        // If we're doing PGO generation stuff and on a GNU-like linker, use the
        // "-u" flag to properly pull in the profiler runtime bits.
        //
        // This is because LLVM otherwise won't add the needed initialization
        // for us on Linux (though the extra flag should be harmless if it
        // does).
        //
        // See https://reviews.llvm.org/D14033 and https://reviews.llvm.org/D14030.
        //
        // Though it may be worth to try to revert those changes upstream, since
        // the overhead of the initialization should be minor.
        self.cmd.arg("-u");
        self.cmd.arg("__llvm_profile_runtime");
    }

    fn control_flow_guard(&mut self) {}

    fn debuginfo(&mut self, strip: Strip) {
        // MacOS linker doesn't support stripping symbols directly anymore.
        if self.options.target.options.is_like_osx {
            return;
        }

        match strip {
            Strip::None => {}
            Strip::DebugInfo => {
                self.linker_arg("--strip-debug");
            }
            Strip::Symbols => {
                self.linker_arg("--strip-all");
            }
        }
    }

    fn no_crt_objects(&mut self) {
        if !self.is_ld {
            self.cmd.arg("-nostartfiles");
        }
    }

    fn no_default_libraries(&mut self) {
        if !self.is_ld {
            self.cmd.arg("-nodefaultlibs");
        }
    }

    fn export_symbols(&mut self, tmpdir: &Path, project_type: ProjectType, symbols: &[String]) {
        // Symbol visibility in object files typically takes care of this.
        if project_type == ProjectType::Executable
            && self
                .options
                .target
                .options
                .override_export_symbols
                .is_none()
        {
            return;
        }

        // We manually create a list of exported symbols to ensure we don't expose any more.
        // The object files have far more public symbols than we actually want to export,
        // so we hide them all here.

        if !self.options.target.options.limit_rdylib_exports {
            return;
        }

        let is_windows = self.options.target.options.is_like_windows;
        let path = tmpdir.join(if is_windows { "list.def" } else { "list" });

        debug!("EXPORTED SYMBOLS:");

        if self.options.target.options.is_like_osx {
            // Write a plain, newline-separated list of symbols
            let res: io::Result<()> = try {
                let mut f = BufWriter::new(File::create(&path)?);
                for sym in symbols {
                    debug!("  _{}", sym);
                    writeln!(f, "_{}", sym)?;
                }
            };
            if let Err(e) = res {
                self.diagnostics
                    .fatal(format!("failed to write lib.def file: {}", e))
                    .raise();
            }
        } else if is_windows {
            let res: io::Result<()> = try {
                let mut f = BufWriter::new(File::create(&path)?);

                // .def file similar to MSVC one but without LIBRARY section
                // because LD doesn't like when it's empty
                writeln!(f, "EXPORTS")?;
                for symbol in symbols {
                    debug!("  _{}", symbol);
                    writeln!(f, "  {}", symbol)?;
                }
            };
            if let Err(e) = res {
                self.diagnostics
                    .fatal(format!("failed to write list.def file: {}", e))
                    .raise();
            }
        } else {
            // Write an LD version script
            let res: io::Result<()> = try {
                let mut f = BufWriter::new(File::create(&path)?);
                writeln!(f, "{{")?;
                if !symbols.is_empty() {
                    writeln!(f, "  global:")?;
                    for sym in symbols {
                        debug!("    {};", sym);
                        writeln!(f, "    {};", sym)?;
                    }
                }
                writeln!(f, "\n  local:\n    *;\n}};")?;
            };
            if let Err(e) = res {
                self.diagnostics
                    .fatal(format!("failed to write version script: {}", e))
                    .raise();
            }
        }

        if self.options.target.options.is_like_osx {
            self.linker_args(&[OsString::from("-exported_symbols_list"), path.into()]);
        } else if self.options.target.options.is_like_solaris {
            self.linker_args(&[OsString::from("-M"), path.into()]);
        } else {
            if is_windows {
                self.linker_arg(path);
            } else {
                let mut arg = OsString::from("--version-script=");
                arg.push(path);
                self.linker_arg(arg);
            }
        }
    }

    fn exported_symbol_means_used_symbol(&self) -> bool {
        self.options.target.options.is_like_windows || self.options.target.options.is_like_osx
    }

    fn subsystem(&mut self, subsystem: &str) {
        self.linker_arg("--subsystem");
        self.linker_arg(&subsystem);
    }

    fn reset_per_library_state(&mut self) {
        self.hint_dynamic(); // Reset to default before returning the composed command line.
    }

    fn group_start(&mut self) {
        if self.takes_hints() {
            self.linker_arg("--start-group");
        }
    }

    fn group_end(&mut self) {
        if self.takes_hints() {
            self.linker_arg("--end-group");
        }
    }

    fn linker_plugin_lto(&mut self) {
        match self.options.codegen_opts.linker_plugin_lto {
            LinkerPluginLto::Disabled => {
                // Nothing to do
            }
            LinkerPluginLto::Auto => {
                self.push_linker_plugin_lto_args(None);
            }
            LinkerPluginLto::Plugin(ref path) => {
                self.push_linker_plugin_lto_args(Some(path.as_os_str()));
            }
        }
    }

    // Add the `GNU_EH_FRAME` program header which is required to locate unwinding information.
    // Some versions of `gcc` add it implicitly, some (e.g. `musl-gcc`) don't,
    // so we just always add it.
    fn add_eh_frame_header(&mut self) {
        self.linker_arg("--eh-frame-hdr");
    }

    fn add_no_exec(&mut self) {
        if self.options.target.options.is_like_windows {
            self.linker_arg("--nxcompat");
        } else if self.options.target.options.linker_is_gnu {
            self.linker_arg("-znoexecstack");
        }
    }

    fn add_as_needed(&mut self) {
        if self.options.target.options.linker_is_gnu && !self.options.target.options.is_like_windows
        {
            self.linker_arg("--as-needed");
        } else if self.options.target.options.is_like_solaris {
            // -z ignore is the Solaris equivalent to the GNU ld --as-needed option
            self.linker_args(&["-z", "ignore"]);
        }
    }
}

pub struct MsvcLinker<'a> {
    cmd: Command,
    options: &'a Options,
    diagnostics: &'a DiagnosticsHandler,
}
impl<'a> Linker for MsvcLinker<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn set_output_kind(&mut self, output_kind: LinkOutputKind, out_filename: &Path) {
        match output_kind {
            LinkOutputKind::DynamicNoPicExe
            | LinkOutputKind::DynamicPicExe
            | LinkOutputKind::StaticNoPicExe
            | LinkOutputKind::StaticPicExe => {}
            LinkOutputKind::DynamicDylib | LinkOutputKind::StaticDylib => {
                self.cmd.arg("/DLL");
                let mut arg: OsString = "/IMPLIB:".into();
                arg.push(out_filename.with_extension("dll.lib"));
                self.cmd.arg(arg);
            }
            LinkOutputKind::WasiReactorExe => {
                panic!("can't link as reactor on non-wasi target");
            }
        }
    }

    fn link_rlib(&mut self, lib: &Path) {
        self.cmd.arg(lib);
    }

    fn add_object(&mut self, path: &Path) {
        self.cmd.arg(path);
    }

    fn gc_sections(&mut self, _keep_metadata: bool) {
        // MSVC's ICF (Identical COMDAT Folding) link optimization is
        // slow for Rust and thus we disable it by default when not in
        // optimization build.
        if self.options.opt_level != OptLevel::No {
            self.cmd.arg("/OPT:REF,ICF");
        } else {
            // It is necessary to specify NOICF here, because /OPT:REF
            // implies ICF by default.
            self.cmd.arg("/OPT:REF,NOICF");
        }
    }

    fn no_gc_sections(&mut self) {
        self.cmd.arg("/OPT:NOREF,NOICF");
    }

    fn link_dylib(&mut self, lib: &str, verbatim: bool, _as_needed: bool) {
        self.cmd
            .arg(format!("{}{}", lib, if verbatim { "" } else { ".lib" }));
    }

    fn link_rust_dylib(&mut self, lib: &str, path: &Path) {
        // When producing a dll, the MSVC linker may not actually emit a
        // `foo.lib` file if the dll doesn't actually export any symbols, so we
        // check to see if the file is there and just omit linking to it if it's
        // not present.
        let name = format!("{}.dll.lib", lib);
        if path.join(&name).exists() {
            self.cmd.arg(name);
        }
    }

    fn link_staticlib(&mut self, lib: &str, verbatim: bool) {
        self.cmd
            .arg(format!("{}{}", lib, if verbatim { "" } else { ".lib" }));
    }

    fn full_relro(&mut self) {
        // noop
    }

    fn partial_relro(&mut self) {
        // noop
    }

    fn no_relro(&mut self) {
        // noop
    }

    fn no_crt_objects(&mut self) {
        // noop
    }

    fn no_default_libraries(&mut self) {
        self.cmd.arg("/NODEFAULTLIB");
    }

    fn include_path(&mut self, path: &Path) {
        let mut arg = OsString::from("/LIBPATH:");
        arg.push(path);
        self.cmd.arg(&arg);
    }

    fn output_filename(&mut self, path: &Path) {
        let mut arg = OsString::from("/OUT:");
        arg.push(path);
        self.cmd.arg(&arg);
    }

    fn framework_path(&mut self, _path: &Path) {
        panic!("frameworks are not supported on windows")
    }

    fn link_framework(&mut self, _framework: &str, _as_needed: bool) {
        panic!("frameworks are not supported on windows")
    }

    fn link_whole_staticlib(&mut self, lib: &str, verbatim: bool, _search_path: &[PathBuf]) {
        self.cmd.arg(format!(
            "/WHOLEARCHIVE:{}{}",
            lib,
            if verbatim { "" } else { ".lib" }
        ));
    }

    fn link_whole_rlib(&mut self, path: &Path) {
        let mut arg = OsString::from("/WHOLEARCHIVE:");
        arg.push(path);
        self.cmd.arg(arg);
    }

    fn optimize(&mut self) {
        // Needs more investigation of `/OPT` arguments
    }

    fn pgo_gen(&mut self) {
        // Nothing needed here.
    }

    fn control_flow_guard(&mut self) {
        self.cmd.arg("/guard:cf");
    }

    fn debuginfo(&mut self, strip: Strip) {
        match strip {
            Strip::None => {
                // This will cause the Microsoft linker to generate a PDB file
                // from the CodeView line tables in the object files.
                self.cmd.arg("/DEBUG");

                // This will cause the Microsoft linker to embed .natvis info into the PDB file
                let natvis_dir_path = self.options.sysroot.join("lib\\lumenlib\\etc");
                if let Ok(natvis_dir) = fs::read_dir(&natvis_dir_path) {
                    for entry in natvis_dir {
                        match entry {
                            Ok(entry) => {
                                let path = entry.path();
                                if path.extension() == Some("natvis".as_ref()) {
                                    let mut arg = OsString::from("/NATVIS:");
                                    arg.push(path);
                                    self.cmd.arg(arg);
                                }
                            }
                            Err(err) => {
                                self.diagnostics
                                    .warn(&format!("error enumerating natvis directory: {}", err));
                            }
                        }
                    }
                }
            }
            Strip::DebugInfo | Strip::Symbols => {
                self.cmd.arg("/DEBUG:NONE");
            }
        }
    }

    // Currently the compiler doesn't use `dllexport` (an LLVM attribute) to
    // export symbols from a dynamic library. When building a dynamic library,
    // however, we're going to want some symbols exported, so this function
    // generates a DEF file which lists all the symbols.
    //
    // The linker will read this `*.def` file and export all the symbols from
    // the dynamic library. Note that this is not as simple as just exporting
    // all the symbols in the current crate (as specified by `codegen.reachable`)
    // but rather we also need to possibly export the symbols of upstream
    // crates. Upstream rlibs may be linked statically to this dynamic library,
    // in which case they may continue to transitively be used and hence need
    // their symbols exported.
    fn export_symbols(&mut self, tmpdir: &Path, project_type: ProjectType, symbols: &[String]) {
        // Symbol visibility takes care of this typically
        if project_type == ProjectType::Executable {
            return;
        }

        let path = tmpdir.join("lib.def");
        let res: io::Result<()> = try {
            let mut f = BufWriter::new(File::create(&path)?);

            // Start off with the standard module name header and then go
            // straight to exports.
            writeln!(f, "LIBRARY")?;
            writeln!(f, "EXPORTS")?;
            for symbol in symbols {
                debug!("  _{}", symbol);
                writeln!(f, "  {}", symbol)?;
            }
        };
        if let Err(e) = res {
            self.diagnostics
                .fatal(format!("failed to write lib.def file: {}", e))
                .raise();
        }
        let mut arg = OsString::from("/DEF:");
        arg.push(path);
        self.cmd.arg(&arg);
    }

    fn subsystem(&mut self, subsystem: &str) {
        // Note that previous passes of the compiler validated this subsystem,
        // so we just blindly pass it to the linker.
        self.cmd.arg(&format!("/SUBSYSTEM:{}", subsystem));

        // Windows has two subsystems we're interested in right now, the console
        // and windows subsystems. These both implicitly have different entry
        // points (starting symbols). The console entry point starts with
        // `mainCRTStartup` and the windows entry point starts with
        // `WinMainCRTStartup`. These entry points, defined in system libraries,
        // will then later probe for either `main` or `WinMain`, respectively to
        // start the application.
        //
        // In Rust we just always generate a `main` function so we want control
        // to always start there, so we force the entry point on the windows
        // subsystem to be `mainCRTStartup` to get everything booted up
        // correctly.
        //
        // For more information see RFC #1665
        if subsystem == "windows" {
            self.cmd.arg("/ENTRY:mainCRTStartup");
        }
    }

    // MSVC doesn't need group indicators
    fn group_start(&mut self) {}
    fn group_end(&mut self) {}

    fn linker_plugin_lto(&mut self) {
        // Do nothing
    }

    fn add_no_exec(&mut self) {
        self.cmd.arg("/NXCOMPAT");
    }
}

pub struct EmLinker<'a> {
    cmd: Command,
    options: &'a Options,
    #[allow(unused)]
    diagnostics: &'a DiagnosticsHandler,
}
impl<'a> Linker for EmLinker<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn set_output_kind(&mut self, _output_kind: LinkOutputKind, _out_filename: &Path) {}

    fn include_path(&mut self, path: &Path) {
        self.cmd.arg("-L").arg(path);
    }

    fn link_staticlib(&mut self, lib: &str, _verbatim: bool) {
        self.cmd.arg("-l").arg(lib);
    }

    fn output_filename(&mut self, path: &Path) {
        self.cmd.arg("-o").arg(path);
    }

    fn add_object(&mut self, path: &Path) {
        self.cmd.arg(path);
    }

    fn link_dylib(&mut self, lib: &str, verbatim: bool, _as_needed: bool) {
        // Emscripten always links statically
        self.link_staticlib(lib, verbatim);
    }

    fn link_whole_staticlib(&mut self, lib: &str, verbatim: bool, _search_path: &[PathBuf]) {
        // not supported?
        self.link_staticlib(lib, verbatim);
    }

    fn link_whole_rlib(&mut self, lib: &Path) {
        // not supported?
        self.link_rlib(lib);
    }

    fn link_rust_dylib(&mut self, lib: &str, _path: &Path) {
        self.link_dylib(lib, false, true);
    }

    fn link_rlib(&mut self, lib: &Path) {
        self.add_object(lib);
    }

    fn full_relro(&mut self) {
        // noop
    }

    fn partial_relro(&mut self) {
        // noop
    }

    fn no_relro(&mut self) {
        // noop
    }

    fn framework_path(&mut self, _path: &Path) {
        panic!("frameworks are not supported on Emscripten")
    }

    fn link_framework(&mut self, _framework: &str, _as_needed: bool) {
        panic!("frameworks are not supported on Emscripten")
    }

    fn gc_sections(&mut self, _keep_metadata: bool) {
        // noop
    }

    fn no_gc_sections(&mut self) {
        // noop
    }

    fn optimize(&mut self) {
        // Emscripten performs own optimizations
        self.cmd.arg(match self.options.opt_level {
            OptLevel::No => "-O0",
            OptLevel::Less => "-O1",
            OptLevel::Default => "-O2",
            OptLevel::Aggressive => "-O3",
            OptLevel::Size => "-Os",
            OptLevel::SizeMin => "-Oz",
        });
        // Unusable until https://github.com/rust-lang/rust/issues/38454 is resolved
        self.cmd.args(&["--memory-init-file", "0"]);
    }

    fn pgo_gen(&mut self) {
        // noop, but maybe we need something like the gnu linker?
    }

    fn control_flow_guard(&mut self) {}

    fn debuginfo(&mut self, _strip: Strip) {
        // Preserve names or generate source maps depending on debug info
        self.cmd.arg(match self.options.debug_info {
            DebugInfo::None => "-g0",
            DebugInfo::Limited => "-g3",
            DebugInfo::Full => "-g4",
        });
    }

    fn no_crt_objects(&mut self) {}

    fn no_default_libraries(&mut self) {
        self.cmd
            .args(&["-s", "DEFAULT_LIBRARY_FUNCS_TO_INCLUDE=[]"]);
    }

    fn export_symbols(&mut self, _tmpdir: &Path, _project_type: ProjectType, symbols: &[String]) {
        debug!("EXPORTED SYMBOLS:");

        self.cmd.arg("-s");

        let mut arg = OsString::from("EXPORTED_FUNCTIONS=");
        let mut encoded = String::new();

        {
            let end = symbols.len();
            encoded.push('[');
            encoded.push('\n');

            for (i, sym) in symbols.iter().enumerate() {
                let is_last = i + 1 == end;
                encoded += "  \"";
                encoded.push('_');
                encoded += sym;
                encoded.push('"');
                if !is_last {
                    encoded.push(',');
                }
                encoded.push('\n');
            }

            encoded.push(']');
        }
        debug!("{}", encoded);
        arg.push(encoded);

        self.cmd.arg(arg);
    }

    fn subsystem(&mut self, _subsystem: &str) {
        // noop
    }

    // Appears not necessary on Emscripten
    fn group_start(&mut self) {}
    fn group_end(&mut self) {}

    fn linker_plugin_lto(&mut self) {
        // Do nothing
    }
}

pub struct WasmLd<'a> {
    cmd: Command,
    options: &'a Options,
    #[allow(unused)]
    diagnostics: &'a DiagnosticsHandler,
}
impl<'a> WasmLd<'a> {
    fn new(
        mut cmd: Command,
        options: &'a Options,
        diagnostics: &'a DiagnosticsHandler,
    ) -> WasmLd<'a> {
        // If the atomics feature is enabled for wasm then we need a whole bunch
        // of flags:
        //
        // * `--shared-memory` - the link won't even succeed without this, flags the one linear
        //   memory as `shared`
        //
        // * `--max-memory=1G` - when specifying a shared memory this must also be specified. We
        //   conservatively choose 1GB but users should be able to override this with `-C link-arg`.
        //
        // * `--import-memory` - it doesn't make much sense for memory to be exported in a threaded
        //   module because typically you're sharing memory and instantiating the module multiple
        //   times. As a result if it were exported then we'd just have no sharing.
        //
        // * `--export=__wasm_init_memory` - when using `--passive-segments` the linker will
        //   synthesize this function, and so we need to make sure that our usage of `--export`
        //   below won't accidentally cause this function to get deleted.
        //
        // * `--export=*tls*` - when `#[thread_local]` symbols are used these symbols are how the
        //   TLS segments are initialized and configured.
        let target_features = options
            .codegen_opts
            .target_features
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("");
        let atomics = target_features.contains("+atomics")
            || options.target.options.features.contains("+atomics");
        if atomics {
            cmd.arg("--shared-memory");
            cmd.arg("--max-memory=1073741824");
            cmd.arg("--import-memory");
            cmd.arg("--export=__wasm_init_memory");
            cmd.arg("--export=__wasm_init_tls");
            cmd.arg("--export=__tls_size");
            cmd.arg("--export=__tls_align");
            cmd.arg("--export=__tls_base");
        }
        WasmLd {
            cmd,
            options,
            diagnostics,
        }
    }
}
impl<'a> Linker for WasmLd<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn set_output_kind(&mut self, output_kind: LinkOutputKind, _out_filename: &Path) {
        match output_kind {
            LinkOutputKind::DynamicNoPicExe
            | LinkOutputKind::DynamicPicExe
            | LinkOutputKind::StaticNoPicExe
            | LinkOutputKind::StaticPicExe => {}
            LinkOutputKind::DynamicDylib | LinkOutputKind::StaticDylib => {
                self.cmd.arg("--no-entry");
            }
            LinkOutputKind::WasiReactorExe => {
                self.cmd.arg("--entry");
                self.cmd.arg("_initialize");
            }
        }
    }

    fn link_dylib(&mut self, lib: &str, _verbatim: bool, _as_needed: bool) {
        self.cmd.arg("-l").arg(lib);
    }

    fn link_staticlib(&mut self, lib: &str, _verbatim: bool) {
        self.cmd.arg("-l").arg(lib);
    }

    fn link_rlib(&mut self, lib: &Path) {
        self.cmd.arg(lib);
    }

    fn include_path(&mut self, path: &Path) {
        self.cmd.arg("-L").arg(path);
    }

    fn framework_path(&mut self, _path: &Path) {
        panic!("frameworks not supported")
    }

    fn output_filename(&mut self, path: &Path) {
        self.cmd.arg("-o").arg(path);
    }

    fn add_object(&mut self, path: &Path) {
        self.cmd.arg(path);
    }

    fn full_relro(&mut self) {}

    fn partial_relro(&mut self) {}

    fn no_relro(&mut self) {}

    fn link_rust_dylib(&mut self, lib: &str, _path: &Path) {
        self.cmd.arg("-l").arg(lib);
    }

    fn link_framework(&mut self, _framework: &str, _as_needed: bool) {
        panic!("frameworks not supported")
    }

    fn link_whole_staticlib(&mut self, lib: &str, _verbatim: bool, _search_path: &[PathBuf]) {
        self.cmd.arg("-l").arg(lib);
    }

    fn link_whole_rlib(&mut self, lib: &Path) {
        self.cmd.arg(lib);
    }

    fn gc_sections(&mut self, _keep_metadata: bool) {
        self.cmd.arg("--gc-sections");
    }

    fn no_gc_sections(&mut self) {
        self.cmd.arg("--no-gc-sections");
    }

    fn optimize(&mut self) {
        self.cmd.arg(match self.options.opt_level {
            OptLevel::No => "-O0",
            OptLevel::Less => "-O1",
            OptLevel::Default => "-O2",
            OptLevel::Aggressive => "-O3",
            // Currently LLD doesn't support `Os` and `Oz`, so pass through `O2`
            // instead.
            OptLevel::Size => "-O2",
            OptLevel::SizeMin => "-O2",
        });
    }

    fn pgo_gen(&mut self) {}

    fn debuginfo(&mut self, strip: Strip) {
        match strip {
            Strip::None => {}
            Strip::DebugInfo => {
                self.cmd.arg("--strip-debug");
            }
            Strip::Symbols => {
                self.cmd.arg("--strip-all");
            }
        }
    }

    fn control_flow_guard(&mut self) {}

    fn no_crt_objects(&mut self) {}

    fn no_default_libraries(&mut self) {}

    fn export_symbols(&mut self, _tmpdir: &Path, _project_type: ProjectType, symbols: &[String]) {
        for sym in symbols {
            self.cmd.arg("--export").arg(&sym);
        }

        // LLD will hide these otherwise-internal symbols since our `--export`
        // list above is a whitelist of what to export. Various bits and pieces
        // of tooling use this, so be sure these symbols make their way out of
        // the linker as well.
        self.cmd.arg("--export=__heap_base");
        self.cmd.arg("--export=__data_end");
    }

    fn subsystem(&mut self, _subsystem: &str) {}

    // Not needed for now with LLD
    fn group_start(&mut self) {}
    fn group_end(&mut self) {}

    fn linker_plugin_lto(&mut self) {
        // Do nothing for now
    }
}

pub(crate) fn exported_symbols(options: &Options) -> Vec<String> {
    if let Some(ref exports) = options.target.options.override_export_symbols {
        return exports.iter().map(ToString::to_string).collect();
    }

    vec![]
}
