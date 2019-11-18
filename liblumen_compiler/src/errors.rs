use std::sync::{Arc, Mutex};

use thiserror::Error;

use libeir_diagnostics::{CodeMap, Diagnostic};

/// Represents various compilation errors to compiler consumers
#[derive(Error, Debug)]
pub enum CompilerError {
    #[error("i/o error: {0}")]
    IO(#[from] std::io::Error),

    #[error("parsing failed")]
    Parser {
        codemap: Arc<Mutex<CodeMap>>,
        errs: Vec<Diagnostic>,
    },

    #[error("compilation failed")]
    Failed,

    #[error("invalid file type: '{0}'")]
    FileType(String),
}

unsafe impl Send for CompilerError {}
unsafe impl Sync for CompilerError {}
