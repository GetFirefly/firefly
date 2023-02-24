use std::path::PathBuf;
use std::sync::Arc;

use clap::ArgMatches;

use firefly_llvm::{self as llvm, target::TargetMachine};
use firefly_session::{CodegenOptions, DebuggingOptions, Options};
use firefly_target::{self as target, Target};
use firefly_util::diagnostics::CodeMap;
use firefly_util::error::Verbosity;

pub fn configure<'a>(
    codemap: Arc<CodeMap>,
    c_opts: CodegenOptions,
    z_opts: DebuggingOptions,
    cwd: PathBuf,
    matches: &ArgMatches<'a>,
) -> anyhow::Result<Arc<Options>> {
    let args = matches.subcommand().1.unwrap();
    Options::new_with_defaults(None, codemap, c_opts, z_opts, cwd, &args).map(Arc::new)
}

/// The main entry point for the 'print' command
pub fn handle_command<'a>(options: Arc<Options>, matches: &ArgMatches<'a>) -> anyhow::Result<()> {
    match matches.subcommand() {
        ("version", _) => {
            if options.verbosity < Verbosity::Warning {
                println!("release:     {}", crate::FIREFLY_RELEASE);
                println!("commit-hash: {}", crate::FIREFLY_COMMIT_HASH);
                println!("commit-date: {}", crate::FIREFLY_COMMIT_DATE);
                println!("host:        {}", target::host_triple());
                //TODO: Set env var during build to display LLVM version we compiled with
                //println!("llvm:        {}", llvm::version());
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
        ("target-features", _) => {
            let target_machine = TargetMachine::create(&options)?;
            target_machine.print_target_features();
        }
        ("target-cpus", _) => {
            let target_machine = TargetMachine::create(&options)?;
            target_machine.print_target_cpus();
        }
        ("passes", _) => llvm::passes::print(),
        (subcommand, _) => unimplemented!("print subcommand '{}' is not implemented", subcommand),
    }

    Ok(())
}
