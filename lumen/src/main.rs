#![feature(uniform_paths)]
#![feature(custom_attribute)]
#![feature(stmt_expr_attributes)]

mod beam;
mod compiler;
mod serialization;
mod syntax;

use std::process;

use clap::{crate_description, crate_name, crate_version};
use clap::{App, Arg, SubCommand};

/// Represents errors at the top-level
#[derive(Debug)]
pub enum CommandError {
    /// An invalid argument was found during command execution
    ArgumentError(String),
    /// Compilation failed
    CompilationFailed(compiler::CompileError),
}
impl CommandError {
    /// Builds a `CommandError::ArgumentError`
    pub fn badarg(s: &str) -> Self {
        CommandError::ArgumentError(s.to_string())
    }
}
impl std::fmt::Display for CommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use self::CommandError::*;
        match *self {
            ArgumentError(ref e) => write!(f, "Invalid argument: {}", e),
            CompilationFailed(ref e) => e.fmt(f),
        }
    }
}
impl std::error::Error for CommandError {
    fn description(&self) -> &str {
        use self::CommandError::*;
        match *self {
            ArgumentError(ref e) => e,
            CompilationFailed(ref e) => e.description(),
        }
    }
    fn cause(&self) -> Option<&std::error::Error> {
        use self::CommandError::*;
        match *self {
            CompilationFailed(ref e) => e.cause(),
            _ => None,
        }
    }
}
impl std::convert::From<compiler::CompileError> for CommandError {
    fn from(err: compiler::CompileError) -> Self {
        CommandError::CompilationFailed(err)
    }
}
impl std::convert::From<compiler::codegen::CodeGenError> for CommandError {
    fn from(err: compiler::codegen::CodeGenError) -> Self {
        CommandError::from(compiler::CompileError::from(err))
    }
}

fn main() {
    let matches = App::new(crate_name!())
        .version(crate_version!())
        .about(crate_description!())
        .subcommand(
            SubCommand::with_name("compile")
                .about("Compiles a .erl, or .beam file, to a static binary")
                .arg(
                    Arg::with_name("file")
                        .help("The path to the file you wish to compile")
                        .index(1)
                        .takes_value(true)
                        .required(true),
                ),
        )
        .get_matches();

    // Dispatch commands
    let result: Result<(), CommandError> = match matches.subcommand() {
        ("compile", Some(args)) => compiler::dispatch(&args),
        _ => Ok(()),
    };

    // Handle success/failure
    match result {
        Err(err) => {
            eprintln!("{}", err);
            process::exit(2);
        }
        _ => return,
    };
}
