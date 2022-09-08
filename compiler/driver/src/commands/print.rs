use std::path::PathBuf;
use std::sync::Arc;

use clap::ArgMatches;

use firefly_codegen as codegen;
use firefly_diagnostics::{CodeMap, Reporter};
use firefly_llvm::{self as llvm, target::TargetMachine};
use firefly_session::{CodegenOptions, DebuggingOptions, Options};
use firefly_target::{self as target, Target};

/// The main entry point for the 'print' command
pub fn handle_command<'a>(
    c_opts: CodegenOptions,
    z_opts: DebuggingOptions,
    matches: &ArgMatches<'a>,
    cwd: PathBuf,
) -> anyhow::Result<()> {
    match matches.subcommand() {
        ("version", subcommand_matches) => {
            let verbose = subcommand_matches
                .map(|m| m.is_present("verbose"))
                .unwrap_or_else(|| matches.is_present("verbose"));
            if verbose {
                println!("release:     {}", crate::FIREFLY_RELEASE);
                println!("commit-hash: {}", crate::FIREFLY_COMMIT_HASH);
                println!("commit-date: {}", crate::FIREFLY_COMMIT_DATE);
                println!("host:        {}", target::host_triple());
                println!("llvm:        {}", llvm::version());
            } else {
                println!("{}", crate::FIREFLY_RELEASE);
            }
        }
        ("current-target", _) => {
            let triple = target::host_triple();
            let target = Target::search(triple)?;
            println!("{:#?}", &target)
        }
        ("targets", _) => {
            for target in Target::all() {
                println!("{}", target);
            }
        }
        ("target-features", subcommand_matches) => {
            let reporter = Reporter::new();
            let codemap = Arc::new(CodeMap::new());
            let options = Options::new_with_defaults(
                &reporter,
                codemap,
                c_opts,
                z_opts,
                cwd,
                subcommand_matches.unwrap(),
            )?;
            codegen::init(&options)?;
            let target_machine = TargetMachine::create(&options)?;
            target_machine.print_target_features();
        }
        ("target-cpus", subcommand_matches) => {
            let reporter = Reporter::new();
            let codemap = Arc::new(CodeMap::new());
            let options = Options::new_with_defaults(
                &reporter,
                codemap,
                c_opts,
                z_opts,
                cwd,
                subcommand_matches.unwrap(),
            )?;
            codegen::init(&options)?;
            let target_machine = TargetMachine::create(&options)?;
            target_machine.print_target_cpus();
        }
        ("passes", _subcommand_matches) => {
            llvm::passes::print();
        }
        (subcommand, _) => unimplemented!("print subcommand '{}' is not implemented", subcommand),
    }

    Ok(())
}
