#![deny(warnings)]

mod argparser;
mod commands;
mod compiler;
mod diagnostics;
mod interner;
mod output;
mod parser;
pub(crate) mod task;

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::anyhow;
use clap::crate_version;
use liblumen_session::{CodegenOptions, DebuggingOptions, OptionGroup, ShowOptionGroupHelp};
use liblumen_util::diagnostics::Emitter;
use liblumen_util::error::HelpRequested;

pub const LUMEN_RELEASE: &'static str = crate_version!();
pub const LUMEN_COMMIT_HASH: &'static str = env!("LUMEN_COMMIT_HASH");
pub const LUMEN_COMMIT_DATE: &'static str = env!("LUMEN_COMMIT_DATE");

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
        ("print", subcommand_matches) => {
            commands::print::handle_command(c_opts, z_opts, subcommand_matches.unwrap(), cwd)
                .map(|_| 0)
        }
        ("compile", subcommand_matches) => commands::compile::handle_command(
            c_opts,
            z_opts,
            subcommand_matches.unwrap(),
            cwd,
            emitter,
        )
        .map(|_| 0),
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
