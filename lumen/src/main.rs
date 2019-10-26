mod compiler;

use std::process;

use clap::{crate_description, crate_name, crate_version};
use clap::{App, Arg, SubCommand, ArgMatches};

use libeir_diagnostics::{ColorChoice, Emitter, StandardStreamEmitter};
use liblumen_compiler::CompilerError;

fn main() -> anyhow::Result<()> {
    human_panic::setup_panic!();

    let emitter = StandardStreamEmitter::new(ColorChoice::Auto);

    // Get current working directory
    let cwd = std::env::current_dir()?;
    let output_dir = cwd.join("_build/target");

    // Build argument parser
    let matches = App::new(crate_name!())
        .version(crate_version!())
        .about(crate_description!())
        .subcommand(
            SubCommand::with_name("compile")
                .about("Compiles Erlang to an executable or shared library")
                .arg(
                    Arg::with_name("path")
                        .help("The path to the file or directory of files you wish to compile")
                        .index(1)
                        .takes_value(true)
                        .value_name("FILE_OR_DIR")
                        .default_value_os(cwd.as_os_str())
                        .required(true),
                )
                .arg(
                    Arg::with_name("compiler")
                        .help("The type of compiler to use")
                        .takes_value(true)
                        .value_name("TYPE")
                        .possible_values(&["beam", "erl"])
                        .default_value("erl")
                        .required(true),
                )
                .arg(
                    Arg::with_name("output")
                        .help("The directory to place compiler output")
                        .short("o")
                        .long("output")
                        .value_name("DIR")
                        .default_value_os(output_dir.as_os_str()),
                )
                .arg(
                    Arg::with_name("define")
                        .help("Define a macro, e.g. -DTEST")
                        .short("D")
                        .long("define")
                        .value_name("NAME")
                        .takes_value(true)
                        .multiple(true),
                )
                .arg(
                    Arg::with_name("warnings-as-errors")
                        .help("Causes the compiler to treat all warnings as errors")
                        .long("warnings-as-errors"),
                )
                .arg(
                    Arg::with_name("no-warnings")
                        .help("Disable warnings")
                        .long("no-warnings")
                        .conflicts_with("warnings-as-errors"),
                )
                .arg(
                    Arg::with_name("verbose")
                        .help("Set verbosity level")
                        .short("v")
                        .multiple(true),
                )
                .arg(
                    Arg::with_name("append-path")
                        .help("Appends a path to the code path")
                        .short("pz")
                        .long("append-path")
                        .value_name("PATH")
                        .takes_value(true)
                        .multiple(true),
                )
                .arg(
                    Arg::with_name("prepend-path")
                        .help("Prepends a path to the code path")
                        .short("pa")
                        .long("prepend-path")
                        .value_name("PATH")
                        .takes_value(true)
                        .multiple(true),
                ),
        )
        .get_matches();

    // Handle success/failure
    if let Err(err) = self::dispatch(matches) {
        match err.downcast_ref::<CompilerError>() {
            Some(CompilerError::Parser { codemap, errs }) => {
                let emitter = emitter.set_codemap(codemap.clone());
                for err in errs.iter() {
                    emitter
                        .diagnostic(&err)
                        .expect("stdout failed");
                }
                process::exit(2);
            }
            _ => return Err(err),
        }
    }

    Ok(())
}

#[inline]
fn dispatch(matches: ArgMatches) -> anyhow::Result<()> {
    match matches.subcommand() {
        ("compile", Some(args)) => compiler::dispatch(&args),
        _ => Ok(()),
    }
}
