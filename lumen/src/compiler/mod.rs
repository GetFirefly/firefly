pub mod codegen;

use std::ffi::OsStr;
use std::path::Path;

use clap::ArgMatches;

use crate::syntax;
use crate::syntax::ast::AST;
use crate::syntax::parser::{Parser, TokenReader};
use crate::syntax::preprocessor::Preprocessor;
use crate::syntax::tokenizer::Lexer;

use super::CommandError;

use self::codegen::CodeGenError;

/// Represents the type of file given to the compiler
pub enum FileType {
    Beam(std::ffi::OsString),
    Erlang(std::ffi::OsString),
}

/// Represents various compilation errors to compiler consumers
#[derive(Debug)]
pub enum CompileError {
    LoadFailed(std::io::Error),
    ParseFailed(syntax::parser::Error),
    InvalidBeam(String),
    InvalidModule(String),
    InvalidForm(String),
    CodeGenFailed(CodeGenError),
}
impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use self::CompileError::*;
        match *self {
            LoadFailed(ref e) => e.fmt(f),
            ParseFailed(ref e) => e.fmt(f),
            CodeGenFailed(ref e) => e.fmt(f),
            InvalidBeam(ref e) => write!(f, "Invalid BEAM file: {}", e),
            InvalidModule(ref e) => write!(f, "Invalid module definition: {}", e),
            InvalidForm(ref e) => write!(f, "Invalid form: {}", e),
        }
    }
}
impl std::error::Error for CompileError {
    fn description(&self) -> &str {
        use self::CompileError::*;
        match *self {
            LoadFailed(ref e) => e.description(),
            ParseFailed(ref e) => e.description(),
            CodeGenFailed(ref e) => e.description(),
            InvalidBeam(ref e) => e,
            InvalidModule(ref e) => e,
            InvalidForm(ref e) => e,
        }
    }
    fn cause(&self) -> Option<&std::error::Error> {
        use self::CompileError::*;
        match *self {
            LoadFailed(ref e) => Some(e),
            ParseFailed(ref e) => Some(e),
            CodeGenFailed(ref e) => Some(e),
            _ => None,
        }
    }
}
impl std::convert::From<std::io::Error> for CompileError {
    fn from(err: std::io::Error) -> Self {
        CompileError::LoadFailed(err)
    }
}
impl std::convert::From<syntax::ast::error::FromBeamError> for CompileError {
    fn from(err: syntax::ast::error::FromBeamError) -> Self {
        use crate::beam::reader::ReadError;
        use crate::syntax::ast::error::FromBeamError;
        use std::error::Error;
        match err {
            FromBeamError::Io(e) => CompileError::LoadFailed(e),
            FromBeamError::BeamFile(ReadError::FileError(e)) => CompileError::LoadFailed(e),
            FromBeamError::BeamFile(ReadError::InvalidString(e)) => {
                CompileError::InvalidBeam(e.description().to_string())
            }
            FromBeamError::BeamFile(ReadError::UnexpectedMagicNumber(_)) => {
                CompileError::InvalidBeam("unexpected magic number".to_string())
            }
            FromBeamError::BeamFile(ReadError::UnexpectedFormType(tag)) => {
                let e = format!("unexpected form type {:?}", tag);
                CompileError::InvalidBeam(e.to_string())
            }
            FromBeamError::BeamFile(ReadError::UnexpectedChunk { id, expected }) => {
                let e = format!("unexpected chunk {:?}, expected {:?}", id, expected);
                CompileError::InvalidBeam(e.to_string())
            }
            FromBeamError::TermDecode(e) => CompileError::InvalidBeam(e.description().to_string()),
            FromBeamError::NoDebugInfo => CompileError::InvalidBeam(
                "source .beam was compiled without debug info".to_string(),
            ),
            FromBeamError::NoModuleAttribute => {
                CompileError::InvalidModule("no module attribute".to_string())
            }
            FromBeamError::UnexpectedTerm(_) => {
                CompileError::InvalidForm("encountered unexpected term".to_string())
            }
        }
    }
}
impl std::convert::From<CodeGenError> for CompileError {
    fn from(err: CodeGenError) -> Self {
        CompileError::CodeGenFailed(err)
    }
}

pub enum Artifact {
    Module(crate::syntax::ast::ast::ModuleDecl),
    Expr(crate::syntax::parser::cst::Expr),
}

pub type CompileResult = Result<Artifact, CompileError>;

/// Dispatches command-line arguments to the compiler backend
pub fn dispatch(args: &ArgMatches) -> Result<(), CommandError> {
    codegen::initialize();

    let file = args.value_of_os("file").unwrap();
    let result = match detect_compiler(file)? {
        FileType::Beam(ref path) => compile_beam(path),
        FileType::Erlang(ref path) => compile_erl(path),
    };
    match result {
        Err(err) => Err(CommandError::CompilationFailed(err)),
        Ok(Artifact::Expr(_)) => {
            println!("Parsing succeeded, but compilation skipped");
            Ok(())
        }
        Ok(Artifact::Module(module)) => {
            match codegen::generate_to_file(&vec![module], Path::new(file), codegen::OutputType::IR)
            {
                Err(err) => Err(CommandError::from(err)),
                Ok(_) => {
                    println!("Compilation successful!");
                    Ok(())
                }
            }
        }
    }
}

/// Compiles a BEAM file to a Module
pub fn compile_beam(file: &OsStr) -> CompileResult {
    let path = Path::new(file);
    match AST::from_beam_file(path) {
        Err(err) => Err(CompileError::from(err)),
        Ok(AST { module }) => Ok(Artifact::Module(module)),
    }
}

/// Compiles a .erl file to Erlang AST
pub fn compile_erl(file: &OsStr) -> CompileResult {
    match std::fs::read_to_string(file) {
        Err(reason) => Err(CompileError::LoadFailed(reason)),
        Ok(contents) => compile_forms(&contents),
    }
}

/// Compiles an arbitrary string of Erlang forms to Erlang AST
pub fn compile_forms(expr: &str) -> CompileResult {
    let lexer = Lexer::new(expr);
    let pp = Preprocessor::new(lexer);
    let reader = TokenReader::new(pp);
    let mut parser = Parser::new(reader);
    match parser.parse::<crate::syntax::parser::cst::ModuleDecl>() {
        Err(_) => {
            // This isn't a module, try parsing an expression
            match parser.parse::<crate::syntax::parser::cst::Expr>() {
                Err(err) => Err(CompileError::ParseFailed(err)),
                Ok(expr) => Ok(Artifact::Expr(expr)),
            }
        }
        Ok(module) => {
            let module = crate::syntax::ast::ast::ModuleDecl::from(module);
            Ok(Artifact::Module(module))
        }
    }
}

/// Determines which type of compiler we're dispatching to
fn detect_compiler(f: &OsStr) -> Result<FileType, CommandError> {
    let path = Path::new(f);
    match path.extension() {
        None => Err(CommandError::badarg("file type must be .beam or .erl")),
        Some(ext) => match ext.to_str().unwrap() {
            "beam" => Ok(FileType::Beam(f.to_os_string())),
            "erl" => Ok(FileType::Erlang(f.to_os_string())),
            _ => Err(CommandError::badarg("file type must be .beam or .erl")),
        },
    }
}
