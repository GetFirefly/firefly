mod command;
mod archive;
mod link;
mod rpath;
mod builtin;

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use log::warn;

use liblumen_session::{ProjectType, LinkerPluginLto, OptLevel, DebugInfo, DiagnosticsHandler, Options};
use liblumen_target::{LinkerFlavor, LldFlavor};

use self::command::Command;

pub use self::link::link_binary;

#[derive(PartialEq, Clone, Debug)]
pub enum LibSource {
    Some(PathBuf),
    None,
}

impl LibSource {
    pub fn is_some(&self) -> bool {
        if let LibSource::Some(_) = *self {
            true
        } else {
            false
        }
    }

    pub fn option(&self) -> Option<PathBuf> {
        match *self {
            LibSource::Some(ref p) => Some(p.clone()),
            LibSource::None => None,
        }
    }
}

/// For all the linkers we support, and information they might
/// need out of the shared crate context before we get rid of it.
#[derive(Debug)]
pub struct LinkerInfo {
    exports: Vec<String>,
}

impl LinkerInfo {
    pub fn new() -> LinkerInfo {
        Self {
            exports: vec![],
        }
    }

    pub fn to_linker<'a>(
        &'a self,
        cmd: Command,
        options: &'a Options,
        diagnostics: &'a DiagnosticsHandler,
        flavor: LinkerFlavor,
    ) -> Box<dyn Linker+'a> {
        match flavor {
            LinkerFlavor::Gcc =>  {
                Box::new(GccLinker {
                    cmd,
                    options,
                    diagnostics,
                    info: self,
                    hinted_static: false,
                    is_ld: false,
                }) as Box<dyn Linker>
            }

            LinkerFlavor::Lld(LldFlavor::Ld) |
            LinkerFlavor::Lld(LldFlavor::Ld64) |
            LinkerFlavor::Ld => {
                Box::new(GccLinker {
                    cmd,
                    options,
                    diagnostics,
                    info: self,
                    hinted_static: false,
                    is_ld: true,
                }) as Box<dyn Linker>
            }

            LinkerFlavor::Lld(LldFlavor::Wasm) => {
                Box::new(WasmLd::new(cmd, options, self)) as Box<dyn Linker>
            }

            LinkerFlavor::Lld(LldFlavor::Link)
            | LinkerFlavor::Msvc
            | LinkerFlavor::Em
            | LinkerFlavor::PtxLinker => { unimplemented!("unsupported linker flavor") },
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
    fn position_independent_executable(&mut self);
    fn no_position_independent_executable(&mut self);
    fn full_relro(&mut self);
    fn partial_relro(&mut self);
    fn no_relro(&mut self);
    fn optimize(&mut self);
    fn pgo_gen(&mut self);
    fn debuginfo(&mut self);
    fn no_default_libraries(&mut self);
    fn build_dylib(&mut self, out_filename: &Path);
    fn build_static_executable(&mut self);
    fn args(&mut self, args: &[String]);
    fn export_symbols(&mut self, tmpdir: &Path, project_type: ProjectType);
    fn subsystem(&mut self, subsystem: &str);
    fn group_start(&mut self);
    fn group_end(&mut self);
    fn linker_plugin_lto(&mut self);
    // Should have been finalize(self), but we don't support self-by-value on trait objects (yet?).
    fn finalize(&mut self) -> Command;
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
        where S: AsRef<OsStr>
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
        // * For WebAssembly the only functional linker is LLD, which doesn't
        //   support hint flags
        !self.options.target.options.is_like_osx &&
            self.options.target.arch != "wasm32"
    }

    // Some platforms take hints about whether a library is static or dynamic.
    // For those that support this, we ensure we pass the option if the library
    // was flagged "static" (most defaults are dynamic) to ensure that if
    // libfoo.a and libfoo.so both exist that the right one is chosen.
    fn hint_static(&mut self) {
        if !self.takes_hints() { return }
        if !self.hinted_static {
            self.linker_arg("-Bstatic");
            self.hinted_static = true;
        }
    }

    fn hint_dynamic(&mut self) {
        if !self.takes_hints() { return }
        if self.hinted_static {
            self.linker_arg("-Bdynamic");
            self.hinted_static = false;
        }
    }

    fn push_linker_plugin_lto_args(&mut self, plugin_path: Option<&OsStr>) {
        // TODO: Figure out why this isn't a problem for Rust;
        // ld64 doesn't support the -plugin or -plugin-opt flags as far
        // as I can tell
        if self.options.target.options.is_like_osx {
            return;
        }

        if let Some(plugin_path) = plugin_path {
            let mut arg = OsString::from("-plugin=");
            arg.push(plugin_path);
            self.linker_arg(&arg);
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
}

impl<'a> Linker for GccLinker<'a> {
    fn link_dylib(&mut self, lib: &str) {
        self.hint_dynamic();
        self.cmd.arg(format!("-l{}", lib));
    }
    fn link_staticlib(&mut self, lib: &str) {
        self.hint_static();
        self.cmd.arg(format!("-l{}", lib));
    }
    fn link_rlib(&mut self, lib: &Path) { self.hint_static(); self.cmd.arg(lib); }
    fn include_path(&mut self, path: &Path) {
        if !path.exists() {
            warn!("invalid include path, not found: {}", path.to_string_lossy());
            return;
        }
        if !path.is_dir() {
            warn!("invalid include path, not a directory: {}", path.to_string_lossy());
            return;
        }
        self.cmd.arg("-L").arg(path);
    }
    fn framework_path(&mut self, path: &Path) { self.cmd.arg("-F").arg(path); }
    fn output_filename(&mut self, path: &Path) { self.cmd.arg("-o").arg(path); }
    fn add_object(&mut self, path: &Path) { self.cmd.arg(path); }
    fn position_independent_executable(&mut self) { self.cmd.arg("-pie"); }
    fn no_position_independent_executable(&mut self) { self.cmd.arg("-no-pie"); }
    fn full_relro(&mut self) { self.linker_arg("-zrelro"); self.linker_arg("-znow"); }
    fn partial_relro(&mut self) { self.linker_arg("-zrelro"); }
    fn no_relro(&mut self) { self.linker_arg("-znorelro"); }
    fn build_static_executable(&mut self) { self.cmd.arg("-static"); }
    fn args(&mut self, args: &[String]) { self.cmd.args(args); }

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
            self.linker_arg("--whole-archive").cmd.arg(format!("-l{}", lib));
            self.linker_arg("--no-whole-archive");
        } else {
            // -force_load is the macOS equivalent of --whole-archive, but it
            // involves passing the full path to the library to link.
            self.linker_arg("-force_load");
            match archive::find_library(lib, search_path, &self.options) {
                Ok(ref lib) => { self.linker_arg(lib); },
                Err(err) => self.diagnostics.fatal(err).raise(),
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
        if !self.options.target.options.linker_is_gnu { return }

        // GNU-style linkers support optimization with -O. GNU ld doesn't
        // need a numeric argument, but other linkers do.
        if self.options.opt_level == OptLevel::Default ||
           self.options.opt_level == OptLevel::Aggressive {
            self.linker_arg("-O1");
        }
    }

    fn pgo_gen(&mut self) {
        if !self.options.target.options.linker_is_gnu { return }

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

    fn debuginfo(&mut self) {
        if let DebugInfo::None = self.options.debug_info {
            // If we are building without debuginfo enabled and we were called with
            // `-Zstrip-debuginfo-if-disabled=yes`, tell the linker to strip any debuginfo
            // found when linking to get rid of symbols from libstd.
            if let Some(true) = self.options.debugging_opts.strip_debuginfo_if_disabled {
                self.linker_arg("-S");
            }
        };
    }

    fn no_default_libraries(&mut self) {
        if !self.is_ld {
            self.cmd.arg("-nodefaultlibs");
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
            if self.options.codegen_opts.rpath || self.options.debugging_opts.osx_rpath_install_name {
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
                let implib_name = out_filename
                    .file_name()
                    .and_then(|file| file.to_str())
                    .map(|file| format!("{}{}{}",
                         self.options.target.options.staticlib_prefix,
                         file,
                         self.options.target.options.staticlib_suffix));
                if let Some(implib_name) = implib_name {
                    let implib = out_filename
                        .parent()
                        .map(|dir| dir.join(&implib_name));
                    if let Some(implib) = implib {
                        self.linker_arg(&format!("--out-implib,{}", (*implib).to_str().unwrap()));
                    }
                }
            }
        }
    }

    fn subsystem(&mut self, subsystem: &str) {
        self.linker_arg("--subsystem");
        self.linker_arg(&subsystem);
    }

    fn export_symbols(&mut self, _tmpdir: &Path, _project_type: ProjectType) {
        unimplemented!();
    }

    fn finalize(&mut self) -> Command {
        self.hint_dynamic(); // Reset to default before returning the composed command line.

        ::std::mem::replace(&mut self.cmd, Command::new(""))
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
}


pub struct WasmLd<'a> {
    cmd: Command,
    options: &'a Options,
    info: &'a LinkerInfo,
}

impl<'a> WasmLd<'a> {
    fn new(mut cmd: Command, options: &'a Options, info: &'a LinkerInfo) -> WasmLd<'a> {
        // If the atomics feature is enabled for wasm then we need a whole bunch
        // of flags:
        //
        // * `--shared-memory` - the link won't even succeed without this, flags
        //   the one linear memory as `shared`
        //
        // * `--max-memory=1G` - when specifying a shared memory this must also
        //   be specified. We conservatively choose 1GB but users should be able
        //   to override this with `-C link-arg`.
        //
        // * `--import-memory` - it doesn't make much sense for memory to be
        //   exported in a threaded module because typically you're
        //   sharing memory and instantiating the module multiple times. As a
        //   result if it were exported then we'd just have no sharing.
        //
        // * `--passive-segments` - all memory segments should be passive to
        //   prevent each module instantiation from reinitializing memory.
        //
        // * `--export=__wasm_init_memory` - when using `--passive-segments` the
        //   linker will synthesize this function, and so we need to make sure
        //   that our usage of `--export` below won't accidentally cause this
        //   function to get deleted.
        //
        // * `--export=*tls*` - when `#[thread_local]` symbols are used these
        //   symbols are how the TLS segments are initialized and configured.
        let target_features = options.codegen_opts.target_features.as_ref().map(|s| s.as_str()).unwrap_or("");
        let atomics = target_features.contains("+atomics") || options.target.options.features.contains("+atomics");
        if atomics {
            cmd.arg("--shared-memory");
            cmd.arg("--max-memory=1073741824");
            cmd.arg("--import-memory");
            cmd.arg("--passive-segments");
            cmd.arg("--export=__wasm_init_memory");
            cmd.arg("--export=__wasm_init_tls");
            cmd.arg("--export=__tls_size");
            cmd.arg("--export=__tls_align");
            cmd.arg("--export=__tls_base");
        }
        WasmLd { cmd, options, info }
    }
}

impl<'a> Linker for WasmLd<'a> {
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

    fn position_independent_executable(&mut self) {
    }

    fn full_relro(&mut self) {
    }

    fn partial_relro(&mut self) {
    }

    fn no_relro(&mut self) {
    }

    fn build_static_executable(&mut self) {
    }

    fn args(&mut self, args: &[String]) {
        self.cmd.args(args);
    }

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
            OptLevel::SizeMin => "-O2"
        });
    }

    fn pgo_gen(&mut self) {
    }

    fn debuginfo(&mut self) {
    }

    fn no_default_libraries(&mut self) {
    }

    fn build_dylib(&mut self, _out_filename: &Path) {
        self.cmd.arg("--no-entry");
    }

    fn export_symbols(&mut self, _tmpdir: &Path, _project_type: ProjectType) {
        for sym in self.info.exports.iter() {
            self.cmd.arg("--export").arg(&sym);
        }

        // LLD will hide these otherwise-internal symbols since our `--export`
        // list above is a whitelist of what to export. Various bits and pieces
        // of tooling use this, so be sure these symbols make their way out of
        // the linker as well.
        self.cmd.arg("--export=__heap_base");
        self.cmd.arg("--export=__data_end");
    }

    fn subsystem(&mut self, _subsystem: &str) {
    }

    fn no_position_independent_executable(&mut self) {
    }

    fn finalize(&mut self) -> Command {
        ::std::mem::replace(&mut self.cmd, Command::new(""))
    }

    // Not needed for now with LLD
    fn group_start(&mut self) {}
    fn group_end(&mut self) {}

    fn linker_plugin_lto(&mut self) {
        // Do nothing for now
    }
}
