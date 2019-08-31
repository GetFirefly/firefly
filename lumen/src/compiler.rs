use std::collections::VecDeque;
use std::convert::From;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use clap::{value_t, ArgMatches};
use failure::Error;

use libeir_diagnostics::{CodeMap, ColorChoice};
use liblumen_compiler::{Compiler, CompilerMode, CompilerSettings, Verbosity};

/// Dispatches command-line arguments to the compiler backend
pub fn dispatch<'a>(args: &'a ArgMatches) -> Result<(), Error> {
    let config = configure(args)?;
    let mut compiler = Compiler::new(config);

    compiler.compile()?;

    Ok(())
}

/// Create a CompilerSettings struct from ArgMatches produced by clap
fn configure<'a>(args: &'a ArgMatches) -> Result<CompilerSettings, Error> {
    let codemap = Arc::new(Mutex::new(CodeMap::new()));
    let mode = value_t!(args, "compiler", CompilerMode).unwrap_or_else(|e| e.exit());
    let source_dir = args.value_of_os("path").map(PathBuf::from).unwrap();
    let output_dir = args.value_of_os("output").map(PathBuf::from).unwrap();
    let warnings_as_errors = args.is_present("warnings-as-errors");
    let no_warn = args.is_present("no_warn");
    let verbosity = Verbosity::from_level(args.occurrences_of("verbose") as isize);
    let include_path = VecDeque::new();
    let mut code_path = match args.values_of_os("prepend-path") {
        None => Vec::new(),
        Some(values) => values.map(PathBuf::from).collect(),
    };
    let mut append_dirs = match args.values_of_os("append-path") {
        None => Vec::new(),
        Some(values) => values.map(PathBuf::from).collect(),
    };
    code_path.append(&mut append_dirs);
    Ok(CompilerSettings {
        mode,
        color: ColorChoice::Auto,
        source_dir,
        output_dir,
        warnings_as_errors,
        no_warn,
        verbosity,
        code_path,
        include_path,
        codemap,
    })
}
