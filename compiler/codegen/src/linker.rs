pub(crate) mod archive;
mod command;
pub(crate) mod link;
mod rpath;

use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::mem;
use std::path::{Path, PathBuf};

use fxhash::FxHashMap;

use log::{debug, warn};

use liblumen_session::{DebugInfo, LinkerPluginLto, Lto, OptLevel, Options, ProjectType, Strip};
use liblumen_target::spec::LinkOutputKind;
use liblumen_target::{LinkerFlavor, LldFlavor};
use liblumen_util::diagnostics::DiagnosticsHandler;

use super::meta::LibSource;

use self::command::Command;
pub use self::link::link_binary;

/// For all the linkers we support, and information they might
/// need out of the shared crate context before we get rid of it.
#[derive(Debug)]
pub struct LinkerInfo {
    exports: FxHashMap<ProjectType, Vec<String>>,
}

impl LinkerInfo {
    pub fn new() -> LinkerInfo {
        Self {
            exports: Default::default(),
        }
    }

    pub fn to_linker<'a>(
        &'a self,
        cmd: Command,
        options: &'a Options,
        diagnostics: &'a DiagnosticsHandler,
        flavor: LinkerFlavor,
    ) -> Box<dyn Linker + 'a> {
        match flavor {
            LinkerFlavor::Lld(LldFlavor::Link) | LinkerFlavor::Msvc => Box::new(MsvcLinker {
                cmd,
                options,
                diagnostics,
                info: self,
            })
                as Box<dyn Linker>,
            LinkerFlavor::Em => Box::new(EmLinker {
                cmd,
                options,
                diagnostics,
                info: self,
            }) as Box<dyn Linker>,
            LinkerFlavor::Gcc => Box::new(GccLinker {
                cmd,
                options,
                diagnostics,
                info: self,
                hinted_static: false,
                is_ld: false,
            }) as Box<dyn Linker>,

            LinkerFlavor::Lld(LldFlavor::Ld)
            | LinkerFlavor::Lld(LldFlavor::Ld64)
            | LinkerFlavor::Ld => Box::new(GccLinker {
                cmd,
                options,
                diagnostics,
                info: self,
                hinted_static: false,
                is_ld: true,
            }) as Box<dyn Linker>,

            LinkerFlavor::Lld(LldFlavor::Wasm) => {
                Box::new(WasmLd::new(cmd, options, diagnostics, self)) as Box<dyn Linker>
            }

            LinkerFlavor::PtxLinker => Box::new(PtxLinker {
                cmd,
                options,
                diagnostics,
            }) as Box<dyn Linker>,
        }
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
    fn set_output_kind(&mut self, output_kind: LinkOutputKind, out_filename: &Path);
    fn link_dylib(&mut self, lib: &str);
    fn link_rust_dylib(&mut self, lib: &str, path: &Path);
    fn link_framework(&mut self, framework: &str);
    fn link_staticlib(&mut self, lib: &str);
    fn link_rlib(&mut self, lib: &Path);
    fn link_whole_rlib(&mut self, lib: &Path);
    fn link_whole_staticlib(&mut self, lib: &str, search_path: &[PathBuf]);
    fn include_path(&mut self, path: &Path);
    fn framework_path(&mut self, path: &Path);
    fn output_filename(&mut self, path: &Path);
    fn add_object(&mut self, path: &Path);
    fn gc_sections(&mut self, keep_metadata: bool);
    fn full_relro(&mut self);
    fn partial_relro(&mut self);
    fn no_relro(&mut self);
    fn optimize(&mut self);
    fn pgo_gen(&mut self);
    fn control_flow_guard(&mut self);
    fn debuginfo(&mut self, strip: Strip);
    fn no_crt_objects(&mut self);
    fn no_default_libraries(&mut self);
    fn export_symbols(&mut self, tmpdir: &Path, project_type: ProjectType);
    fn subsystem(&mut self, subsystem: &str);
    fn group_start(&mut self);
    fn group_end(&mut self);
    fn linker_plugin_lto(&mut self);
    fn add_eh_frame_header(&mut self) {}
    fn finalize(&mut self);
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
    #[allow(unused)]
    info: &'a LinkerInfo,
    hinted_static: bool, // Keeps track of the current hinting mode.
    // Link as ld
    is_ld: bool,
}

impl<'a> GccLinker<'a> {
    /// Argument that must be passed *directly* to the linker
    ///
    /// These arguments need to be prepended with `-Wl`, when a GCC-style linker is used.
    fn linker_arg<S>(&mut self, arg: S) -> &mut Self
    where
        S: AsRef<OsStr>,
    {
        if !self.is_ld {
            let mut os = OsString::from("-Wl,");
            os.push(arg.as_ref());
            self.cmd.arg(os);
        } else {
            self.cmd.arg(arg);
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
        !self.options.target.options.is_like_osx && self.options.target.arch != "wasm32"
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
        } else {
            return;
        }

        let opt_level = match self.options.opt_level {
            OptLevel::No => "O0",
            OptLevel::Less => "O1",
            OptLevel::Default => "O2",
            OptLevel::Aggressive => "O3",
            OptLevel::Size => "Os",
            OptLevel::SizeMin => "Oz",
        };

        self.linker_arg(&format!("-plugin-opt={}", opt_level));
        if let Some(ref target_cpu) = self.options.codegen_opts.target_cpu {
            self.linker_arg(&format!("-plugin-opt=mcpu={}", target_cpu.as_str()));
        }
    }

    fn build_dylib(&mut self, out_filename: &Path) {
        // On mac we need to tell the linker to let this library be rpathed
        if self.options.target.options.is_like_osx {
            self.cmd.arg("-dynamiclib");
            self.linker_arg("-dylib");

            // Note that the `osx_rpath_install_name` option here is a hack
            // purely to support rustbuild right now, we should get a more
            // principled solution at some point to force the compiler to pass
            // the right `-Wl,-install_name` with an `@rpath` in it.
            if self.options.codegen_opts.rpath || self.options.debugging_opts.osx_rpath_install_name
            {
                self.linker_arg("-install_name");
                let mut v = OsString::from("@rpath/");
                v.push(out_filename.file_name().unwrap());
                self.linker_arg(&v);
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
                        self.linker_arg(&format!("--out-implib,{}", (*implib).to_str().unwrap()));
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

    fn set_output_kind(&mut self, output_kind: LinkOutputKind, output_file: &Path) {
        match output_kind {
            LinkOutputKind::DynamicNoPicExe => {
                if !self.is_ld && self.options.target.options.linker_is_gnu {
                    self.cmd.arg("-no-pie");
                }
            }
            LinkOutputKind::DynamicPicExe => {
                // `-pie` works for both gcc wrapper and ld.
                self.cmd.arg("-pie");
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
        }
        // VxWorks compiler driver introduced `--static-crt` flag specifically for rustc,
        // it switches linking for libc and similar system libraries to static without using
        // any `#[link]` attributes in the `libc` crate, see #72782 for details.
        // FIXME: Switch to using `#[link]` attributes in the `libc` crate
        // similarly to other targets.
        if self.options.target.target_os == "vxworks"
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

    fn link_dylib(&mut self, lib: &str) {
        self.hint_dynamic();
        self.cmd.arg(format!("-l{}", lib));
    }
    fn link_staticlib(&mut self, lib: &str) {
        self.hint_static();
        self.cmd.arg(format!("-l{}", lib));
    }
    fn link_rlib(&mut self, lib: &Path) {
        self.hint_static();
        self.cmd.arg(lib);
    }
    fn include_path(&mut self, path: &Path) {
        if !path.exists() {
            warn!(
                "invalid include path, not found: {}",
                path.to_string_lossy()
            );
            return;
        }
        if !path.is_dir() {
            warn!(
                "invalid include path, not a directory: {}",
                path.to_string_lossy()
            );
            return;
        }
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
        self.linker_arg("-zrelro");
        self.linker_arg("-znow");
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

    fn link_framework(&mut self, framework: &str) {
        self.hint_dynamic();
        self.cmd.arg("-framework").arg(framework);
    }

    // Here we explicitly ask that the entire archive is included into the
    // result artifact. For more details see #15460, but the gist is that
    // the linker will strip away any unused objects in the archive if we
    // don't otherwise explicitly reference them. This can occur for
    // libraries which are just providing bindings, libraries with generic
    // functions, etc.
    fn link_whole_staticlib(&mut self, lib: &str, search_path: &[PathBuf]) {
        self.hint_static();
        let target = &self.options.target;
        if !target.options.is_like_osx {
            self.linker_arg("--whole-archive")
                .cmd
                .arg(format!("-l{}", lib));
            self.linker_arg("--no-whole-archive");
        } else {
            // -force_load is the macOS equivalent of --whole-archive, but it
            // involves passing the full path to the library to link.
            self.linker_arg("-force_load");
            match archive::find_library(lib, search_path, &self.options) {
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
        } else if self.options.target.options.is_like_solaris {
            self.linker_arg("-zignore");

        // If we're building a dylib, we don't use --gc-sections because LLVM
        // has already done the best it can do, and we also don't want to
        // eliminate the metadata. If we're building an executable, however,
        // --gc-sections drops the size of hello world from 1.8MB to 597K, a 67%
        // reduction.
        } else if !keep_metadata {
            self.linker_arg("--gc-sections");
        }
    }

    fn optimize(&mut self) {
        if !self.options.target.options.linker_is_gnu {
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
        match strip {
            Strip::None => {}
            Strip::DebugInfo => {
                // MacOS linker does not support longhand argument --strip-debug
                self.linker_arg("-S");
            }
            Strip::Symbols => {
                // MacOS linker does not support longhand argument --strip-all
                self.linker_arg("-s");
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

    fn export_symbols(&mut self, tmpdir: &Path, project_type: ProjectType) {
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

        let mut arg = OsString::new();
        let path = tmpdir.join("list");

        debug!("EXPORTED SYMBOLS:");

        if self.options.target.options.is_like_osx {
            // Write a plain, newline-separated list of symbols
            let res: io::Result<()> = try {
                let mut f = BufWriter::new(File::create(&path)?);
                for sym in self.info.exports[&project_type].iter() {
                    debug!("  _{}", sym);
                    writeln!(f, "_{}", sym)?;
                }
            };
            if let Err(e) = res {
                self.diagnostics
                    .fatal(format!("failed to write lib.def file: {}", e))
                    .raise();
            }
        } else {
            // Write an LD version script
            let res: io::Result<()> = try {
                let mut f = BufWriter::new(File::create(&path)?);
                writeln!(f, "{{")?;
                if !self.info.exports[&project_type].is_empty() {
                    writeln!(f, "  global:")?;
                    for sym in self.info.exports[&project_type].iter() {
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
            if !self.is_ld {
                arg.push("-Wl,")
            }
            arg.push("-exported_symbols_list,");
        } else if self.options.target.options.is_like_solaris {
            if !self.is_ld {
                arg.push("-Wl,")
            }
            arg.push("-M,");
        } else {
            if !self.is_ld {
                arg.push("-Wl,")
            }
            arg.push("--version-script=");
        }

        arg.push(&path);
        self.cmd.arg(arg);
    }

    fn subsystem(&mut self, subsystem: &str) {
        self.linker_arg("--subsystem");
        self.linker_arg(&subsystem);
    }

    fn finalize(&mut self) {
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
}

pub struct MsvcLinker<'a> {
    cmd: Command,
    options: &'a Options,
    diagnostics: &'a DiagnosticsHandler,
    info: &'a LinkerInfo,
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

    fn link_dylib(&mut self, lib: &str) {
        self.cmd.arg(&format!("{}.lib", lib));
    }

    fn link_rust_dylib(&mut self, lib: &str, path: &Path) {
        // When producing a dll, the MSVC linker may not actually emit a
        // `foo.lib` file if the dll doesn't actually export any symbols, so we
        // check to see if the file is there and just omit linking to it if it's
        // not present.
        let name = format!("{}.dll.lib", lib);
        if fs::metadata(&path.join(&name)).is_ok() {
            self.cmd.arg(name);
        }
    }

    fn link_staticlib(&mut self, lib: &str) {
        self.cmd.arg(&format!("{}.lib", lib));
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

    fn link_framework(&mut self, _framework: &str) {
        panic!("frameworks are not supported on windows")
    }

    fn link_whole_staticlib(&mut self, lib: &str, _search_path: &[PathBuf]) {
        // not supported?
        self.link_staticlib(lib);
        self.cmd.arg(format!("/WHOLEARCHIVE:{}.lib", lib));
    }

    fn link_whole_rlib(&mut self, path: &Path) {
        // not supported?
        self.link_rlib(path);
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
                let natvis_dir_path = self.options.sysroot.join("lib\\rustlib\\etc");
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
    fn export_symbols(&mut self, tmpdir: &Path, project_type: ProjectType) {
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
            for symbol in self.info.exports[&project_type].iter() {
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

    fn finalize(&mut self) {}

    // MSVC doesn't need group indicators
    fn group_start(&mut self) {}
    fn group_end(&mut self) {}

    fn linker_plugin_lto(&mut self) {
        // Do nothing
    }
}

pub struct EmLinker<'a> {
    cmd: Command,
    options: &'a Options,
    #[allow(dead_code)]
    diagnostics: &'a DiagnosticsHandler,
    info: &'a LinkerInfo,
}

impl<'a> Linker for EmLinker<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn set_output_kind(&mut self, _output_kind: LinkOutputKind, _out_filename: &Path) {}

    fn include_path(&mut self, path: &Path) {
        self.cmd.arg("-L").arg(path);
    }

    fn link_staticlib(&mut self, lib: &str) {
        self.cmd.arg("-l").arg(lib);
    }

    fn output_filename(&mut self, path: &Path) {
        self.cmd.arg("-o").arg(path);
    }

    fn add_object(&mut self, path: &Path) {
        self.cmd.arg(path);
    }

    fn link_dylib(&mut self, lib: &str) {
        // Emscripten always links statically
        self.link_staticlib(lib);
    }

    fn link_whole_staticlib(&mut self, lib: &str, _search_path: &[PathBuf]) {
        // not supported?
        self.link_staticlib(lib);
    }

    fn link_whole_rlib(&mut self, lib: &Path) {
        // not supported?
        self.link_rlib(lib);
    }

    fn link_rust_dylib(&mut self, lib: &str, _path: &Path) {
        self.link_dylib(lib);
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

    fn link_framework(&mut self, _framework: &str) {
        panic!("frameworks are not supported on Emscripten")
    }

    fn gc_sections(&mut self, _keep_metadata: bool) {
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

    fn export_symbols(&mut self, _tmpdir: &Path, project_type: ProjectType) {
        let symbols = &self.info.exports[&project_type];

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
                encoded.push('"');
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

    fn finalize(&mut self) {}

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
    #[allow(dead_code)]
    diagnostics: &'a DiagnosticsHandler,
    info: &'a LinkerInfo,
}

impl<'a> WasmLd<'a> {
    fn new(
        mut cmd: Command,
        options: &'a Options,
        diagnostics: &'a DiagnosticsHandler,
        info: &'a LinkerInfo,
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
            info,
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
        }
    }

    fn link_dylib(&mut self, lib: &str) {
        self.cmd.arg("-l").arg(lib);
    }

    fn link_staticlib(&mut self, lib: &str) {
        self.cmd.arg("-l").arg(lib);
    }

    fn link_rlib(&mut self, lib: &Path) {
        self.cmd.arg(lib);
    }

    fn include_path(&mut self, path: &Path) {
        if !path.exists() {
            warn!(
                "invalid include path, not found: {}",
                path.to_string_lossy()
            );
            return;
        }
        if !path.is_dir() {
            warn!(
                "invalid include path, not a directory: {}",
                path.to_string_lossy()
            );
            return;
        }
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

    fn link_framework(&mut self, _framework: &str) {
        panic!("frameworks not supported")
    }

    fn link_whole_staticlib(&mut self, lib: &str, _search_path: &[PathBuf]) {
        self.cmd.arg("-l").arg(lib);
    }

    fn link_whole_rlib(&mut self, lib: &Path) {
        self.cmd.arg(lib);
    }

    fn gc_sections(&mut self, _keep_metadata: bool) {
        self.cmd.arg("--gc-sections");
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

    fn no_crt_objects(&mut self) {}

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

    fn no_default_libraries(&mut self) {}

    fn export_symbols(&mut self, _tmpdir: &Path, project_type: ProjectType) {
        for sym in self.info.exports[&project_type].iter() {
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

    fn finalize(&mut self) {}

    // Not needed for now with LLD
    fn group_start(&mut self) {}
    fn group_end(&mut self) {}

    fn linker_plugin_lto(&mut self) {
        // Do nothing for now
    }
}

/// Much simplified and explicit CLI for the NVPTX linker. The linker operates
/// with bitcode and uses LLVM backend to generate a PTX assembly.
pub struct PtxLinker<'a> {
    cmd: Command,
    options: &'a Options,
    #[allow(dead_code)]
    diagnostics: &'a DiagnosticsHandler,
}

impl<'a> Linker for PtxLinker<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn set_output_kind(&mut self, _output_kind: LinkOutputKind, _out_filename: &Path) {}

    fn link_rlib(&mut self, path: &Path) {
        self.cmd.arg("--rlib").arg(path);
    }

    fn link_whole_rlib(&mut self, path: &Path) {
        self.cmd.arg("--rlib").arg(path);
    }

    fn include_path(&mut self, path: &Path) {
        self.cmd.arg("-L").arg(path);
    }

    fn debuginfo(&mut self, _strip: Strip) {
        self.cmd.arg("--debug");
    }

    fn add_object(&mut self, path: &Path) {
        self.cmd.arg("--bitcode").arg(path);
    }

    fn optimize(&mut self) {
        match self.options.lto() {
            Lto::Thin | Lto::Fat | Lto::ThinLocal => {
                self.cmd.arg("-Olto");
            }

            Lto::No => {}
        };
    }

    fn output_filename(&mut self, path: &Path) {
        self.cmd.arg("-o").arg(path);
    }

    fn finalize(&mut self) {
        // Provide the linker with fallback to internal `target-cpu`.
        self.cmd
            .arg("--fallback-arch")
            .arg(match self.options.codegen_opts.target_cpu {
                Some(ref s) => s,
                None => &self.options.target.options.cpu,
            });
    }

    fn link_dylib(&mut self, _lib: &str) {
        panic!("external dylibs not supported")
    }

    fn link_rust_dylib(&mut self, _lib: &str, _path: &Path) {
        panic!("external dylibs not supported")
    }

    fn link_staticlib(&mut self, _lib: &str) {
        panic!("staticlibs not supported")
    }

    fn link_whole_staticlib(&mut self, _lib: &str, _search_path: &[PathBuf]) {
        panic!("staticlibs not supported")
    }

    fn framework_path(&mut self, _path: &Path) {
        panic!("frameworks not supported")
    }

    fn link_framework(&mut self, _framework: &str) {
        panic!("frameworks not supported")
    }

    fn full_relro(&mut self) {}

    fn partial_relro(&mut self) {}

    fn no_relro(&mut self) {}

    fn gc_sections(&mut self, _keep_metadata: bool) {}

    fn pgo_gen(&mut self) {}

    fn no_crt_objects(&mut self) {}

    fn no_default_libraries(&mut self) {}

    fn control_flow_guard(&mut self) {}

    fn export_symbols(&mut self, _tmpdir: &Path, _project_type: ProjectType) {}

    fn subsystem(&mut self, _subsystem: &str) {}

    fn group_start(&mut self) {}

    fn group_end(&mut self) {}

    fn linker_plugin_lto(&mut self) {}
}
