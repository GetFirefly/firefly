use std::ffi::OsString;
use std::path::{Path, PathBuf};

use log::debug;

use firefly_session::{DebugInfo, OptLevel, Options, ProjectType, Strip};
use firefly_target::spec::LinkOutputKind;

use crate::{Command, Linker};

pub struct EmLinker<'a> {
    pub cmd: Command,
    pub options: &'a Options,
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
