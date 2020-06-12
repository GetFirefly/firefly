use std::path::PathBuf;

use liblumen_session::{Emit, Options, OutputType};

use crate::diagnostics::*;
use crate::interner::{InternedInput, Interner};

pub trait CompilerOutput: CompilerDiagnostics + Interner {
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

    fn emit_file_with_callback<F>(&self, outfile: PathBuf, callback: F) -> QueryResult<PathBuf>
    where
        F: FnOnce(&mut std::fs::File) -> anyhow::Result<()>,
    {
        use std::fs::File;

        let mut f = self.to_query_result(File::create(outfile.as_path()).map_err(|e| e.into()))?;
        self.to_query_result(callback(&mut f))?;
        Ok(outfile)
    }

    fn emit_file<E>(&self, outfile: PathBuf, output: &E) -> QueryResult<PathBuf>
    where
        E: Emit,
    {
        self.emit_file_with_callback(outfile, |f| output.emit(f))
    }
}
