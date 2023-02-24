use std::path::{Path, PathBuf};

use firefly_session::{OptLevel, Options, ProjectType, Strip};
use firefly_target::spec::LinkOutputKind;

use crate::{Command, Linker};

pub struct WasmLinker<'a> {
    pub cmd: Command,
    pub options: &'a Options,
}
impl<'a> WasmLinker<'a> {
    pub fn new(mut cmd: Command, options: &'a Options) -> Self {
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
        Self { cmd, options }
    }
}
impl<'a> Linker for WasmLinker<'a> {
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
