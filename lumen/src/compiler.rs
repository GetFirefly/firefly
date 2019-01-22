use std::borrow::Cow;
use std::collections::HashMap;
use std::convert::From;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use clap::{value_t, ArgMatches};
use failure::{format_err, Error};

use liblumen_compiler::{Compiler, CompilerMode, CompilerSettings, Verbosity};
use liblumen_diagnostics::{CodeMap, ColorChoice, FileName};
use liblumen_syntax::{FileMapSource, Lexer, MacroDef, Scanner, Symbol};

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
    let defines = match args.values_of("defines") {
        None => HashMap::new(),
        Some(values) => parse_defines(codemap.clone(), values.collect())?,
    };
    let warnings_as_errors = args.is_present("warnings-as-errors");
    let no_warn = args.is_present("no_warn");
    let verbosity = Verbosity::from_level(args.occurrences_of("verbose") as isize);
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
        defines,
        warnings_as_errors,
        no_warn,
        verbosity,
        code_path,
        codemap,
    })
}

fn parse_defines<'a>(
    codemap: Arc<Mutex<CodeMap>>,
    values: Vec<&'a str>,
) -> Result<HashMap<Symbol, MacroDef>, Error> {
    let mut result = HashMap::new();
    for value in values.iter() {
        let parts: Vec<_> = value.split('=').collect();
        let name = unsafe { parts.get_unchecked(0) };
        // Validate that macro name is a valid bare atom or identifier
        if !name.starts_with(|c: char| c.is_ascii_alphabetic() && !(c == '_')) {
            return Err(format_err!(
                "invalid macro definition `{}`, macro names must begin with [A-Za-z_]",
                value
            ));
        }
        if name.contains(|c: char| (c.is_ascii_alphanumeric() || c == '_' || c == '@') == false) {
            return Err(format_err!("invalid macro definition `{}`, macro names must be a valid unquoted atom or identifier", value));
        }

        let num_parts = parts.len();
        match num_parts {
            1 => {
                result.insert(Symbol::intern(name), MacroDef::Boolean(true));
            }
            _ => {
                // Lex the input string so we can provide syntax errors
                let src = parts[1..].join("=");
                let filemap = {
                    codemap
                        .lock()
                        .unwrap()
                        .add_filemap(FileName::Virtual(Cow::Borrowed("nofile")), src)
                };
                let source = FileMapSource::new(filemap);
                let scanner = Scanner::new(source);
                let lexer = Lexer::new(scanner);
                let mut tokens = Vec::new();
                for result in lexer {
                    match result {
                        Ok(token) => tokens.push(token),
                        Err(err) => return Err(err.into()),
                    }
                }
                result.insert(Symbol::intern(name), MacroDef::Dynamic(tokens));
            }
        }
    }
    Ok(result)
}
