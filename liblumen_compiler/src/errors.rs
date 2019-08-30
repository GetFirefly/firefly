use std::convert::From;
use std::sync::{Arc, Mutex};

use failure::Fail;

use libeir_diagnostics::CodeMap;
use libeir_syntax_erl::ParserError;

/// Represents various compilation errors to compiler consumers
#[derive(Fail, Debug)]
pub enum CompilerError {
    #[fail(display = "{}", _0)]
    IO(#[fail(cause)] std::io::Error),

    #[fail(display = "parsing failed")]
    Parser {
        codemap: Arc<Mutex<CodeMap>>,
        errs: Vec<ParserError>,
    },

    #[fail(display = "compilation failed")]
    Failed
}
impl From<std::io::Error> for CompilerError {
    fn from(err: std::io::Error) -> Self {
        CompilerError::IO(err)
    }
}
