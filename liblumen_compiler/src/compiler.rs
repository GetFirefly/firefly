mod intern;
mod queries;
mod query_groups;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use log::debug;

pub use salsa::ParallelDatabase;
use salsa::Snapshot;

use libeir_diagnostics::{CodeMap, Diagnostic};
use libeir_intern::Symbol;

use liblumen_core::symbols::FunctionSymbol;
use liblumen_incremental::{InternedInput, InternerStorage};
pub use liblumen_incremental::{ParserDatabase, ParserDatabaseBase};
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
    codemap: Arc<RwLock<CodeMap>>,
    atoms: Arc<Mutex<HashSet<Symbol>>>,
    symbols: Arc<Mutex<HashSet<FunctionSymbol>>>,
}
impl CompilerDatabase {
    pub fn new(codemap: Arc<RwLock<CodeMap>>, diagnostics: DiagnosticsHandler) -> Self {
        let mut atoms = HashSet::default();
        atoms.insert(Symbol::intern("false"));
        atoms.insert(Symbol::intern("true"));
        Self {
            runtime: Default::default(),
            diagnostics,
            codemap,
            atoms: Arc::new(Mutex::new(atoms)),
            symbols: Arc::new(Mutex::new(HashSet::default())),
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
        Snapshot::new(CompilerDatabase {
            runtime: self.runtime.snapshot(self),
            diagnostics: self.diagnostics.clone(),
            codemap: self.codemap.clone(),
            atoms: self.atoms.clone(),
            symbols: self.symbols.clone(),
        })
    }
}

impl ParserDatabaseBase for CompilerDatabase {
    fn diagnostics(&self) -> &DiagnosticsHandler {
        &self.diagnostics
    }

    fn diagnostic(&self, diagnostic: &Diagnostic) {
        self.diagnostics.diagnostic(diagnostic);
    }

    fn codemap(&self) -> &Arc<RwLock<CodeMap>> {
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
        let output_type = output.emit_output_type();
        if let Some(filename) = options.output_types.maybe_emit(&input, output_type) {
            debug!("emitting {} for {:?}", output_type, input);
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

impl CodegenDatabaseBase for CompilerDatabase {
    fn take_atoms(&mut self) -> HashSet<Symbol> {
        let atoms = Arc::get_mut(&mut self.atoms).unwrap().get_mut().unwrap();
        let empty = HashSet::default();
        core::mem::replace(atoms, empty)
    }

    fn add_atoms<'a, I>(&self, atoms: I)
    where
        I: Iterator<Item = &'a Symbol>,
    {
        let mut locked = self.atoms.lock().unwrap();
        for i in atoms {
            locked.insert(*i);
        }
    }

    fn take_symbols(&mut self) -> HashSet<FunctionSymbol> {
        let symbols = Arc::get_mut(&mut self.symbols).unwrap().get_mut().unwrap();
        let empty = HashSet::default();
        core::mem::replace(symbols, empty)
    }

    fn add_symbols<'a, I>(&self, symbols: I)
    where
        I: Iterator<Item = &'a FunctionSymbol>,
    {
        let mut locked = self.symbols.lock().unwrap();
        for i in symbols {
            locked.insert(*i);
        }
    }
}
