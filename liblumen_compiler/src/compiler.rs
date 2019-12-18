mod intern;
mod queries;
mod query_groups;

use std::sync::{Arc, Mutex};

use libeir_diagnostics::{CodeMap, Diagnostic};

use liblumen_incremental::{InternedInput, InternerStorage};
use liblumen_incremental::{ParserDatabase, ParserDatabaseBase, ParserStorage, QueryResult};
use liblumen_session::{DiagnosticsHandler, Emit, Options};

use self::query_groups::CodegenStorage;
pub use self::query_groups::{CodegenDatabase, StringInternerDatabase, StringInternerStorage};

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

    fn maybe_emit_file<E>(&self, input: InternedInput, output: &E) -> QueryResult<()>
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
    ) -> QueryResult<()>
    where
        E: Emit,
    {
        use liblumen_incremental::InternerDatabase;

        let input = self.lookup_intern_input(input);
        if let Some(filename) = options.output_types.maybe_emit(&input, E::TYPE) {
            let output_dir = self.output_dir();
            let outfile = output_dir.join(filename);
            self.emit_file(outfile, output)
        } else {
            Ok(())
        }
    }
}
