use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use log::debug;

use firefly_session::{LinkerPluginLto, OptLevel, Options, ProjectType, Strip};
use firefly_target::spec::LinkOutputKind;
use firefly_util::diagnostics::DiagnosticsHandler;

use crate::archive;
use crate::{Command, Linker};

pub struct GccLinker<'a> {
    pub cmd: Command,
    pub options: &'a Options,
    pub diagnostics: &'a DiagnosticsHandler,
    pub target_cpu: &'a str,
    pub hinted_static: bool, // Keeps track of the current hinting mode.
    // Link as ld
    pub is_ld: bool,
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
