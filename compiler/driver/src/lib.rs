//#![deny(warnings)]
#![feature(iterator_try_collect)]

mod argparser;
mod commands;
mod compiler;

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::anyhow;
use clap::crate_version;

use firefly_diagnostics::CodeMap;
use firefly_session::{
    CodegenOptions, DebuggingOptions, OptionGroup, Options, ShowOptionGroupHelp,
};
use firefly_util::diagnostics::Emitter;
use firefly_util::error::HelpRequested;

use self::commands::{compile, print};

pub const FIREFLY_RELEASE: &'static str = crate_version!();
pub const FIREFLY_COMMIT_HASH: &'static str = env!("FIREFLY_COMMIT_HASH");
pub const FIREFLY_COMMIT_DATE: &'static str = env!("FIREFLY_COMMIT_DATE");

/// Runs the compiler using the provided working directory, args iterator, and default emitter
///
/// Returns the exit code for the compiler if successful (in some cases it is non-zero), otherwise an error
pub fn run_compiler(cwd: PathBuf, args: impl Iterator<Item = OsString>) -> anyhow::Result<i32> {
    run_compiler_with_emitter(cwd, args, None)
}

/// Runs the compiler using the provided working directory, args iterator, and emittter
///
/// Returns the exit code for the compiler if successful (in some cases it is non-zero), otherwise an error
pub fn run_compiler_with_emitter(
    cwd: PathBuf,
    args: impl Iterator<Item = OsString>,
    emitter: Option<Arc<dyn Emitter>>,
) -> anyhow::Result<i32> {
    // Parse arguments
    let matches = argparser::parse(args)?;

    // Parse option groups first, as they can produce usage
    let c_opts = match parse_option_group::<CodegenOptions>(&matches)? {
        Ok(opts) => opts,
        Err(code) => return Ok(code),
    };
    let z_opts = match parse_option_group::<DebuggingOptions>(&matches)? {
        Ok(opts) => opts,
        Err(code) => return Ok(code),
    };

    // Dispatch to the command implementation
    match matches.subcommand() {
        ("print", matches) => {
            // Initialize options from current context and arguments
            let codemap = Arc::new(CodeMap::new());
            let matches = matches.unwrap();
            let options = print::configure(codemap.clone(), c_opts, z_opts, cwd, matches)?;
            // Initialize LLVM/MLIR backends
            init(&options)?;
            // Dispatch
            print::handle_command(options, matches).map(|_| 0)
        }
        ("compile", matches) => {
            // Initialize options from current context and arguments
            let codemap = Arc::new(CodeMap::new());
            let options =
                compile::configure(codemap.clone(), c_opts, z_opts, cwd, matches.unwrap())?;
            // Initialize LLVM/MLIR backends
            init(&options)?;
            // Dispatch
            compile::handle_command(options, codemap, emitter).map(|_| 0)
        }
        (subcommand, _) => Err(anyhow!(format!("Unrecognized subcommand '{}'", subcommand))),
    }
}

fn parse_option_group<'a, G: OptionGroup + Default>(
    matches: &clap::ArgMatches<'a>,
) -> anyhow::Result<Result<G, i32>> {
    match G::parse_option_group(matches) {
        Ok(None) => Ok(Ok(Default::default())),
        Ok(Some(opts)) => Ok(Ok(opts)),
        Err(err) => {
            if let Some(err) = err.downcast_ref::<HelpRequested>() {
                argparser::command_help(err.primary());
                return Ok(Err(2));
            }
            if let Some(err) = err.downcast_ref::<ShowOptionGroupHelp>() {
                err.print_help();
                return Ok(Err(2));
            }
            Err(err)
        }
    }
}

/// Perform initialization of MLIR/LLVM for code generation
#[cfg(feature = "native-compilation")]
fn init(options: &Options) -> anyhow::Result<()> {
    firefly_mlir::init(options)?;
    firefly_llvm::init(options)?;

    Ok(())
}

/// Perform initialization of LLVM for code generation
#[cfg(not(feature = "native-compilation"))]
fn init(options: &Options) -> anyhow::Result<()> {
    firefly_llvm::init(options)?;

    Ok(())
}
