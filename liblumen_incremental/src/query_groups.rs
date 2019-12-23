use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use liblumen_session::{DiagnosticsHandler, Emit, Input, InputType, OutputType, Options};
use liblumen_session::{IRModule, ParsedModule};
use liblumen_util::seq::Seq;

use libeir_diagnostics::{CodeMap, Diagnostic};
use libeir_syntax_erl::ParseConfig;

use crate::intern::InternedInput;
use crate::queries;
use crate::QueryResult;

#[salsa::query_group(ParserStorage)]
pub trait ParserDatabase: ParserDatabaseBase {
    #[salsa::input]
    fn options(&self) -> Arc<Options>;

    #[salsa::invoke(queries::output_dir)]
    fn output_dir(&self) -> PathBuf;

    #[salsa::invoke(queries::inputs)]
    fn inputs(&self) -> QueryResult<Arc<Seq<InternedInput>>>;

    #[salsa::invoke(queries::input_type)]
    fn input_type(&self, input: InternedInput) -> InputType;

    #[salsa::invoke(queries::parse_config)]
    fn parse_config(&self) -> ParseConfig;

    #[salsa::invoke(queries::input_parsed)]
    fn input_parsed(&self, input: InternedInput) -> QueryResult<ParsedModule>;

    #[salsa::invoke(queries::input_eir)]
    fn input_eir(&self, input: InternedInput) -> QueryResult<IRModule>;
}

#[salsa::query_group(InternerStorage)]
pub trait InternerDatabase: salsa::Database {
    #[salsa::interned]
    fn intern_input(&self, input: Input) -> InternedInput;
}

pub trait ParserDatabaseBase: InternerDatabase {
    fn diagnostics(&self) -> &DiagnosticsHandler;

    fn diagnostic(&self, diag: &Diagnostic) {
        self.diagnostics().diagnostic(diag);
    }

    fn codemap(&self) -> &Arc<Mutex<CodeMap>>;

    fn maybe_emit_file<E>(&self, input: InternedInput, emit: &E) -> QueryResult<Option<PathBuf>>
    where
        E: Emit;

    fn maybe_emit_file_with_callback<F>(
        &self,
        input: InternedInput,
        output_type: OutputType,
        callback: F,
    ) -> QueryResult<Option<PathBuf>>
    where
        F: FnOnce(&mut std::fs::File) -> anyhow::Result<()>;

    fn maybe_emit_file_with_opts<E>(
        &self,
        options: &Options,
        input: InternedInput,
        emit: &E,
    ) -> QueryResult<Option<PathBuf>>
    where
        E: Emit;

    fn maybe_emit_file_with_callback_and_opts<F>(
        &self,
        options: &Options,
        input: InternedInput,
        output_type: OutputType,
        callback: F,
    ) -> QueryResult<Option<PathBuf>>
    where
        F: FnOnce(&mut std::fs::File) -> anyhow::Result<()>;

    fn emit_file_with_callback<F>(
        &self,
        outfile: PathBuf,
        callback: F,
    ) -> QueryResult<PathBuf>
    where
        F: FnOnce(&mut std::fs::File) -> anyhow::Result<()>,
    {
        use std::fs::File;

        match File::create(outfile.as_path()) {
            Err(err) => {
                self.diagnostics().io_error(err);
                Err(())
            }
            Ok(mut f) => {
                if let Err(err) = callback(&mut f) {
                    self.diagnostics().error(err);
                    return Err(());
                }
                Ok(outfile)
            }
        }
    }

    fn emit_file<E>(&self, outfile: PathBuf, output: &E) -> QueryResult<PathBuf>
    where
        E: Emit,
    {
        self.emit_file_with_callback(outfile, |f| output.emit(f))
    }
}
