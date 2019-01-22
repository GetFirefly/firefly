mod compiler;

use std::process;

use clap::{crate_description, crate_name, crate_version};
use clap::{App, Arg, SubCommand};
use failure::Error;

use liblumen_compiler::CompilerError;
use liblumen_diagnostics::{ColorChoice, Emitter, StandardStreamEmitter};

fn main() {
    human_panic::setup_panic!();

    let emitter = StandardStreamEmitter::new(ColorChoice::Auto);

    // Get current working directory
    let cwd = match std::env::current_dir() {
        Ok(path) => path,
        Err(err) => {
            emitter.error(err.into()).unwrap();
            process::exit(2);
        }
    };

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

    // Dispatch commands
    let result: Result<(), Error> = match matches.subcommand() {
        ("compile", Some(args)) => compiler::dispatch(&args),
        _ => Ok(()),
    };

    // Handle success/failure
    match result {
        Err(err) => {
            match err.downcast::<CompilerError>() {
                Ok(CompilerError::Parser { codemap, errs }) => {
                    let emitter = emitter.set_codemap(codemap);
                    for err in errs.iter() {
                        emitter
                            .diagnostic(&err.to_diagnostic())
                            .expect("stdout failed");
                    }
                }
                Ok(err) => {
                    emitter.error(err.into()).unwrap();
                }
                Err(err) => {
                    emitter.error(err).unwrap();
                }
            }
            process::exit(2);
        }
        _ => return,
    };
}
