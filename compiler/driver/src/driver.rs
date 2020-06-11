use std::env::ArgsOs;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::anyhow;

use liblumen_session::{CodegenOptions, DebuggingOptions};
use liblumen_util::diagnostics::Emitter;

use crate::argparser;
use crate::commands;

pub fn run_compiler(cwd: PathBuf, args: ArgsOs) -> anyhow::Result<()> {
    run_compiler_with_emitter(cwd, args, None)
}

pub fn run_compiler_with_emitter(
    cwd: PathBuf,
    args: ArgsOs,
    emitter: Option<Arc<dyn Emitter>>,
) -> anyhow::Result<()> {
    use liblumen_session::OptionGroup;

    // Parse arguments
    let matches = argparser::parse(args)?;

    // Parse option groups first, as they can produce usage
    let c_opts = CodegenOptions::parse_option_group(&matches)?.unwrap_or_else(Default::default);
    let z_opts = DebuggingOptions::parse_option_group(&matches)?.unwrap_or_else(Default::default);

    // Dispatch to the command implementation
    match matches.subcommand() {
        ("print", subcommand_matches) => commands::print::handle_command(
            c_opts,
            z_opts,
            subcommand_matches.unwrap(),
            cwd,
            emitter,
        ),
        ("compile", subcommand_matches) => commands::compile::handle_command(
            c_opts,
            z_opts,
            subcommand_matches.unwrap(),
            cwd,
            emitter,
        ),
        (subcommand, _) => Err(anyhow!(format!("Unrecognized subcommand '{}'", subcommand))),
    }
}
