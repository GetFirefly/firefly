use std::convert::From;
use std::sync::{Arc, Mutex};

use failure::Fail;

use liblumen_syntax::ParserError;
use liblumen_diagnostics::CodeMap;

use liblumen_beam::FromBeamError;

use liblumen_codegen::CodeGenError;

/// Represents various compilation errors to compiler consumers
#[derive(Fail, Debug)]
pub enum CompilerError {
    #[fail(display = "{}", _0)]
    IO(#[fail(cause)] std::io::Error),

    #[fail(display = "parsing failed")]
    Parser { codemap: Arc<Mutex<CodeMap>>, errs: Vec<ParserError> },

    #[fail(display = "invalid beam source")]
    FromBeam(#[fail(cause)] FromBeamError),

    #[fail(display = "codegen failed")]
    CodeGenerator(#[fail(cause)] CodeGenError),
}
impl From<std::io::Error> for CompilerError {
    fn from(err: std::io::Error) -> Self {
        CompilerError::IO(err)
    }
}
impl From<FromBeamError> for CompilerError {
    fn from(err: FromBeamError) -> Self {
        CompilerError::FromBeam(err)
    }
}
impl From<CodeGenError> for CompilerError {
    fn from(err: CodeGenError) -> Self {
        CompilerError::CodeGenerator(err)
    }
}
