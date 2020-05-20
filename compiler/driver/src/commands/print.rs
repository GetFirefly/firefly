use std::path::PathBuf;
use std::sync::Arc;

use clap::ArgMatches;

use liblumen_util::diagnostics::Emitter;

use liblumen_codegen as codegen;
use liblumen_llvm as llvm;
use liblumen_session::{CodegenOptions, DebuggingOptions, Options};
use liblumen_target::{self as target, Target};

use crate::commands::*;

/// The main entry point for the 'print' command
pub fn handle_command<'a>(
    c_opts: CodegenOptions,
    z_opts: DebuggingOptions,
    matches: &ArgMatches<'a>,
    cwd: PathBuf,
    emitter: Option<Arc<dyn Emitter>>,
) -> anyhow::Result<()> {
    match matches.subcommand() {
        ("version", subcommand_matches) => {
            let verbose = subcommand_matches
                .map(|m| m.is_present("verbose"))
                .unwrap_or_else(|| matches.is_present("verbose"));
            if verbose {
                println!("release:     {}", crate::LUMEN_RELEASE);
                println!("commit-hash: {}", crate::LUMEN_COMMIT_HASH);
                println!("commit-date: {}", crate::LUMEN_COMMIT_DATE);
                println!("host:        {}", target::host_triple());
                println!("llvm:        {}", llvm::version());
            } else {
                println!("{}", crate::LUMEN_RELEASE);
            }
        }
        ("project-name", _) => {
            let basename = cwd.file_name().unwrap();
            println!("{}", basename.to_str().unwrap());
        }
        ("targets", _) => {
            for target in Target::all() {
                println!("{}", target);
            }
        }
        ("target-features", subcommand_matches) => {
            let options =
                Options::new_with_defaults(c_opts, z_opts, cwd, subcommand_matches.unwrap())?;
            let diagnostics = default_diagnostics_handler(&options, emitter);
            codegen::init(&options)?;
            llvm::target::print_target_features(&options, &diagnostics);
        }
        ("target-cpus", subcommand_matches) => {
            let options =
                Options::new_with_defaults(c_opts, z_opts, cwd, subcommand_matches.unwrap())?;
            let diagnostics = default_diagnostics_handler(&options, emitter);
            codegen::init(&options)?;
            llvm::target::print_target_cpus(&options, &diagnostics);
        }
        ("passes", _subcommand_matches) => {
            llvm::passes::print();
        }
        (subcommand, _) => unimplemented!("print subcommand '{}' is not implemented", subcommand),
    }

    Ok(())
}
