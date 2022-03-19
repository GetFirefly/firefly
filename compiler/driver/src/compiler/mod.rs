mod queries;
mod query_groups;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;

use log::debug;

use salsa::Snapshot;

use liblumen_core::symbols::FunctionSymbol;
use liblumen_intern::Symbol;
use liblumen_session::{Options, OutputType};
use liblumen_util::diagnostics::{CodeMap, DiagnosticsHandler};
use liblumen_util::emit::Emit;

use crate::diagnostics::*;
use crate::interner::{InternedInput, Interner, InternerStorage};
use crate::output::CompilerOutput;
use crate::parser::{Parser, ParserStorage};

use self::query_groups::CompilerStorage;

pub(crate) mod prelude {
    pub use super::query_groups::Compiler;
    pub use crate::diagnostics::*;
    pub use crate::interner::{InternedInput, Interner};
    pub use crate::output::CompilerOutput;
    pub use crate::parser::Parser;
    pub use salsa::ParallelDatabase;
}

#[salsa::database(CompilerStorage, ParserStorage, InternerStorage)]
pub struct Compiler {
    runtime: salsa::Runtime<Compiler>,
    diagnostics: Arc<DiagnosticsHandler>,
    codemap: Arc<CodeMap>,
    atoms: Arc<Mutex<HashSet<Symbol>>>,
    symbols: Arc<Mutex<HashSet<FunctionSymbol>>>,
}
impl Compiler {
    pub fn new(codemap: Arc<CodeMap>, diagnostics: Arc<DiagnosticsHandler>) -> Self {
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
impl salsa::Database for Compiler {
    fn salsa_runtime(&self) -> &salsa::Runtime<Self> {
        &self.runtime
    }

    fn salsa_runtime_mut(&mut self) -> &mut salsa::Runtime<Self> {
        &mut self.runtime
    }
}
impl salsa::ParallelDatabase for Compiler {
    fn snapshot(&self) -> Snapshot<Self> {
        Snapshot::new(Self {
            runtime: self.runtime.snapshot(self),
            diagnostics: self.diagnostics.clone(),
            codemap: self.codemap.clone(),
            atoms: self.atoms.clone(),
            symbols: self.symbols.clone(),
        })
    }
}

impl CompilerDiagnostics for Compiler {
    #[inline]
    fn diagnostics(&self) -> &Arc<DiagnosticsHandler> {
        &self.diagnostics
    }

    fn codemap(&self) -> &Arc<CodeMap> {
        &self.codemap
    }
}

impl CompilerOutput for Compiler {
    fn maybe_emit_file<E>(
        &self,
        input: InternedInput,
        output: &E,
    ) -> Result<Option<PathBuf>, ErrorReported>
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
    ) -> Result<Option<PathBuf>, ErrorReported>
    where
        E: Emit,
    {
        use std::str::FromStr;

        let input = self.lookup_intern_input(input);
        let extension = output.file_type().unwrap();
        let output_type =
            OutputType::from_str(extension).expect("unrecognized file type extension");
        if let Some(filename) = options.maybe_emit(&input, output_type) {
            debug!("emitting {} for {:?}", output_type, input);
            Ok(Some(self.emit_file(filename, output)?))
        } else {
            Ok(None)
        }
    }

    fn maybe_emit_file_with_callback<F>(
        &self,
        input: InternedInput,
        output_type: OutputType,
        callback: F,
    ) -> Result<Option<PathBuf>, ErrorReported>
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
    ) -> Result<Option<PathBuf>, ErrorReported>
    where
        F: FnOnce(&mut std::fs::File) -> anyhow::Result<()>,
    {
        let input = self.lookup_intern_input(input);
        if let Some(filename) = options.maybe_emit(&input, output_type) {
            debug!("emitting {} for {:?}", output_type, input);
            Ok(Some(self.emit_file_with_callback(filename, callback)?))
        } else {
            Ok(None)
        }
    }
}
