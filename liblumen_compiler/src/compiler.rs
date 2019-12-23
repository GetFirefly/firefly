mod intern;
mod queries;
mod query_groups;

use std::sync::{Arc, Mutex};
use std::path::PathBuf;

use log::debug;

use salsa::Snapshot;
pub use salsa::ParallelDatabase;

use libeir_diagnostics::{CodeMap, Diagnostic};

pub use liblumen_incremental::{ParserDatabase, ParserDatabaseBase};
use liblumen_incremental::{InternedInput, InternerStorage};
use liblumen_incremental::{ParserStorage, QueryResult};
use liblumen_session::{DiagnosticsHandler, Emit, Options, OutputType};

pub(crate) mod prelude {
    pub use super::query_groups::*;
}

use self::prelude::*;

#[salsa::database(CodegenStorage, ParserStorage, InternerStorage, StringInternerStorage)]
pub struct CompilerDatabase {
    runtime: salsa::Runtime<CompilerDatabase>,
    diagnostics: DiagnosticsHandler,
    codemap: Arc<Mutex<CodeMap>>,
}
impl CompilerDatabase {
    pub fn new(codemap: Arc<Mutex<CodeMap>>, diagnostics: DiagnosticsHandler) -> Self {
        Self {
            runtime: Default::default(),
            codemap,
            diagnostics,
        }
    }
}
impl salsa::Database for CompilerDatabase {
    fn salsa_runtime(&self) -> &salsa::Runtime<Self> {
        &self.runtime
    }

    fn salsa_runtime_mut(&mut self) -> &mut salsa::Runtime<Self> {
        &mut self.runtime
    }
}
impl salsa::ParallelDatabase for CompilerDatabase {
    fn snapshot(&self) -> Snapshot<Self> {
        Snapshot::new(
            CompilerDatabase {
                runtime: self.runtime.snapshot(self),
                codemap: self.codemap.clone(),
                diagnostics: self.diagnostics.clone(),
            }
        )
    }
}

impl ParserDatabaseBase for CompilerDatabase {
    fn diagnostics(&self) -> &DiagnosticsHandler {
        &self.diagnostics
    }

    fn diagnostic(&self, diagnostic: &Diagnostic) {
        self.diagnostics.diagnostic(diagnostic);
    }

    fn codemap(&self) -> &Arc<Mutex<CodeMap>> {
        &self.codemap
    }

    fn maybe_emit_file<E>(&self, input: InternedInput, output: &E) -> QueryResult<Option<PathBuf>>
    where
        E: Emit,
    {
        let options = self.options();
        self.maybe_emit_file_with_opts(&options, input, output)
    }

    fn maybe_emit_file_with_opts<E>(
        &self,
        options: &Options,
        input: InternedInput,
        output: &E,
    ) -> QueryResult<Option<PathBuf>>
    where
        E: Emit,
    {
        use liblumen_incremental::InternerDatabase;

        let input = self.lookup_intern_input(input);
        if let Some(filename) = options.output_types.maybe_emit(&input, E::TYPE) {
            debug!("emitting {} for {:?}", E::TYPE, input);
            let output_dir = self.output_dir();
            let outfile = output_dir.join(filename);
            Ok(Some(self.emit_file(outfile, output)?))
        } else {
            Ok(None)
        }
    }

    fn maybe_emit_file_with_callback<F>(
        &self,
        input: InternedInput,
        output_type: OutputType,
        callback: F,
    ) -> QueryResult<Option<PathBuf>>
    where
        F: FnOnce(&mut std::fs::File) -> anyhow::Result<()>,
    {
        let options = self.options();
        self.maybe_emit_file_with_callback_and_opts(&options, input, output_type, callback)
    }

    fn maybe_emit_file_with_callback_and_opts<F>(
        &self,
        options: &Options,
        input: InternedInput,
        output_type: OutputType,
        callback: F,
    ) -> QueryResult<Option<PathBuf>>
    where
        F: FnOnce(&mut std::fs::File) -> anyhow::Result<()>,
    {
        use liblumen_incremental::InternerDatabase;

        let input = self.lookup_intern_input(input);
        if let Some(filename) = options.output_types.maybe_emit(&input, output_type) {
            debug!("emitting {} for {:?}", output_type, input);
            let output_dir = self.output_dir();
            let outfile = output_dir.join(filename);
            Ok(Some(self.emit_file_with_callback(outfile, callback)?))
        } else {
            Ok(None)
        }
    }
}

impl CodegenDatabaseBase for CompilerDatabase {}
