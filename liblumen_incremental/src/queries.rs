use std::ops::Deref;
use std::path::{Path, PathBuf};

use liblumen_session::{IRModule, Input, ParsedModule};
use liblumen_util::{seq, seq::Seq};

use libeir_diagnostics::{Diagnostic, FileName};
use libeir_syntax_erl::{self as syntax, ParseConfig};

use crate::intern::InternedInput;
use crate::query_groups::ParserDatabase;
use crate::QueryResult;

const ERL_EXT: &'static str = "erl";

pub(crate) fn output_dir<P>(db: &P) -> PathBuf
where
    P: ParserDatabase,
{
    db.options().output_dir()
}

pub(crate) fn calculate_inputs<P>(db: &P) -> QueryResult<Seq<InternedInput>>
where
    P: ParserDatabase,
{
    use std::io::Read;
    use std::io::{self, Error as IoError, ErrorKind as IoErrorKind};

    let options = db.options();

    // Handle case where input is empty, indicating to compile the current working directory
    if options.input_file.is_none() {
        return find_sources(db, &options.current_dir);
    }

    // We can get three types of input:
    //
    // 1. `-` for standard input
    // 2. `path/to/file.erl` for a single file
    // 3. `path/to/dir` for a directory of files
    match options.input_file.as_ref().unwrap() {
        // Read from standard input
        &FileName::Virtual(ref name) if name == "-" => {
            let mut source = String::new();
            if io::stdin().read_to_string(&mut source).is_err() {
                db.diagnostics().io_error(IoError::new(
                    IoErrorKind::InvalidData,
                    "couldn't read from stdin, invalid UTF-8",
                ));
                return Err(());
            }
            let input = Input::new(name.clone(), source);
            let interned = db.intern_input(input);
            Ok(seq![interned])
        }
        // Read from a single file
        &FileName::Real(ref path) if path.exists() && path.is_file() => {
            let input = Input::File(path.clone());
            let interned = db.intern_input(input);
            Ok(seq![interned])
        }
        // Read all files in a directory
        &FileName::Real(ref path) if path.exists() && path.is_dir() => find_sources(db, path),
        // Invalid virtual file
        &FileName::Virtual(_) => {
            db.diagnostics().io_error(IoError::new(
                IoErrorKind::InvalidInput,
                "invalid input file, expected `-`, a file path, or a directory",
            ));
            Err(())
        }
        // Invalid file/directory path
        &FileName::Real(_) => {
            db.diagnostics().io_error(IoError::new(
                IoErrorKind::InvalidInput,
                "invalid input file, not a file or directory",
            ));
            Err(())
        }
    }
}

pub(crate) fn parse_config<P>(db: &P) -> ParseConfig
where
    P: ParserDatabase,
{
    let options = db.options();
    let codemap = db.codemap().clone();
    let mut parse_config = ParseConfig::new(codemap);
    parse_config.warnings_as_errors = options.warnings_as_errors;
    parse_config.no_warn = options.no_warn;
    parse_config.include_paths = options.include_path.clone();
    parse_config.code_paths = options.code_path.clone();
    parse_config
}

pub(crate) fn input_parsed<P>(db: &P, input: InternedInput) -> QueryResult<ParsedModule>
where
    P: ParserDatabase,
{
    use syntax::{ParseResult, Parser};

    let options = db.options();
    let parser = Parser::new(db.parse_config());
    let result: ParseResult<syntax::ast::Module> = match db.lookup_intern_input(input) {
        Input::File(ref path) => parser.parse_file(path),
        Input::Str { ref input, .. } => parser.parse_string(input),
    };
    match result {
        Ok(module) => {
            db.maybe_emit_file_with_opts(&options, input, &module)?;
            Ok(module.into())
        }
        Err(errors) => {
            for ref diagnostic in errors.iter().map(|e| e.to_diagnostic()) {
                db.diagnostic(diagnostic);
            }
            Err(())
        }
    }
}

pub(crate) fn input_eir<P>(db: &P, input: InternedInput) -> QueryResult<IRModule>
where
    P: ParserDatabase,
{
    let module = db.input_parsed(input)?;
    let parse_config = db.parse_config();
    let codemap = parse_config.codemap.lock().unwrap();
    match syntax::lower_module(codemap.deref(), module.as_ref()) {
        (Ok(ir_module), _) => {
            db.maybe_emit_file(input, &ir_module)?;
            Ok(ir_module.into())
        }
        (_, errors) => {
            for ref diagnostic in errors.iter().map(|e| e.to_diagnostic()) {
                db.diagnostic(diagnostic);
            }
            Err(())
        }
    }
}

pub fn find_sources<D, P>(db: &D, dir: P) -> QueryResult<Seq<InternedInput>>
where
    D: ParserDatabase,
    P: AsRef<Path>,
{
    use walkdir::{DirEntry, WalkDir};

    fn is_hidden(entry: &DirEntry) -> bool {
        entry
            .path()
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.starts_with('.'))
            .unwrap_or(false)
    }

    fn is_source_file(entry: &DirEntry, extension: &str) -> bool {
        let path = entry.path();
        if !path.is_file() {
            return false;
        }
        match path.extension() {
            None => false,
            Some(ext) => ext == extension,
        }
    }

    let walker = WalkDir::new(dir.as_ref()).follow_links(false).into_iter();

    let mut inputs = Vec::new();

    let mut has_errors = false;
    for maybe_entry in walker.filter_entry(|e| !is_hidden(e) && is_source_file(e, ERL_EXT)) {
        match maybe_entry {
            Ok(entry) if !has_errors => {
                let input = Input::from(entry.path());
                let interned = db.intern_input(input);
                inputs.push(interned);
            }
            Ok(_) => continue,
            Err(err) => {
                let diagnostic = Diagnostic::new_error(err.to_string());
                db.diagnostic(&diagnostic);
                has_errors = true;
            }
        }
    }

    if !has_errors {
        Ok(inputs.into())
    } else {
        Err(())
    }
}
