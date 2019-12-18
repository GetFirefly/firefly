use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use liblumen_session::{DiagnosticsHandler, Emit, Input, Options};
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

    #[salsa::invoke(queries::calculate_inputs)]
    fn inputs(&self) -> QueryResult<Seq<InternedInput>>;

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

    fn maybe_emit_file<E>(&self, input: InternedInput, emit: &E) -> QueryResult<()>
    where
        E: Emit;

    fn maybe_emit_file_with_opts<E>(
        &self,
        options: &Options,
        input: InternedInput,
        emit: &E,
    ) -> QueryResult<()>
    where
        E: Emit;

    fn emit_file<E>(&self, outfile: PathBuf, output: &E) -> QueryResult<()>
    where
        E: Emit,
    {
        use std::fs::File;

        match File::create(outfile) {
            Err(err) => {
                self.diagnostics().io_error(err);
                Err(())
            }
            Ok(mut f) => {
                if let Err(err) = output.emit(&mut f) {
                    self.diagnostics().error(err);
                    return Err(());
                }
                Ok(())
            }
        }
    }
}
