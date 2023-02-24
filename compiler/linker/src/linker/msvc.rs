use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use log::debug;

use firefly_session::{OptLevel, Options, ProjectType, Strip};
use firefly_target::spec::LinkOutputKind;
use firefly_util::diagnostics::DiagnosticsHandler;

use crate::{Command, Linker};

pub struct MsvcLinker<'a> {
    pub cmd: Command,
    pub options: &'a Options,
    pub diagnostics: &'a DiagnosticsHandler,
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
                let natvis_dir_path = self.options.sysroot.join("lib\\fireflylib\\etc");
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
