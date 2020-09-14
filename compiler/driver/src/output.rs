use std::fs::create_dir_all;
use std::path::PathBuf;

use anyhow::*;

use thiserror::private::PathAsDisplay;

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

        let result = outfile
            .parent()
            .with_context(|| format!("{} does not have a parent directory", outfile.as_display()))
            .and_then(|outdir| {
                create_dir_all(outdir).with_context(|| {
                    format!(
                        "Could not create parent directories ({}) of file ({})",
                        outdir.as_display(),
                        outfile.as_display()
                    )
                })
            })
            .and_then(|()| {
                File::create(outfile.as_path())
                    .with_context(|| format!("Could not create file ({})", outfile.as_display()))
            })
            .and_then(|mut f| callback(&mut f))
            .map(|()| outfile);

        self.to_query_result(result)
    }

    fn emit_file<E>(&self, outfile: PathBuf, output: &E) -> QueryResult<PathBuf>
    where
        E: Emit,
    {
        self.emit_file_with_callback(outfile, |f| output.emit(f))
    }
}
