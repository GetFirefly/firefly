use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::anyhow;

use libeir_frontend::{AnyFrontend, DynFrontend};
use libeir_syntax_erl::ParseConfig;

use liblumen_session::{IRModule, Input, InputType};
use liblumen_util::diagnostics::FileName;
use liblumen_util::{seq, seq::Seq};

use super::prelude::*;

pub(crate) fn output_dir<P>(db: &P) -> PathBuf
where
    P: Parser,
{
    db.options().output_dir()
}

pub(crate) fn inputs<P>(db: &P) -> QueryResult<Arc<Seq<InternedInput>>>
where
    P: Parser,
{
    use std::io::{self, Read};

    let options = db.options();

    // Handle case where input is empty, indicating to compile the current working directory
    if options.input_file.is_none() {
        return db.to_query_result(find_sources(db, &options.current_dir));
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
            db.to_query_result(
                io::stdin()
                    .read_to_string(&mut source)
                    .map_err(|e| e.into()),
            )?;
            let input = Input::new(name.clone(), source);
            let interned = db.intern_input(input);
            Ok(Arc::new(seq![interned]))
        }
        // Read from a single file
        &FileName::Real(ref path) if path.exists() && path.is_file() => {
            let input = Input::File(path.clone());
            let interned = db.intern_input(input);
            Ok(Arc::new(seq![interned]))
        }
        // Read all files in a directory
        &FileName::Real(ref path) if path.exists() && path.is_dir() => {
            db.to_query_result(find_sources(db, path))
        }
        // Invalid virtual file
        &FileName::Virtual(_) => {
            db.report_error("invalid input file, expected `-`, a file path, or a directory");
            Err(ErrorReported)
        }
        // Invalid file/directory path
        &FileName::Real(_) => {
            db.report_error("invalid input file, not a file or directory");
            Err(ErrorReported)
        }
    }
}

pub(crate) fn input_type<P>(db: &P, input: InternedInput) -> InputType
where
    P: Parser,
{
    let input_info = db.lookup_intern_input(input);
    input_info.get_type()
}

pub(crate) fn parse_config<P>(db: &P) -> ParseConfig
where
    P: Parser,
{
    let options = db.options();
    let mut parse_config = ParseConfig::new();
    parse_config.warnings_as_errors = options.warnings_as_errors;
    parse_config.no_warn = options.no_warn;
    parse_config.include_paths = options.include_path.clone();
    parse_config.code_paths = options.code_path.clone();
    parse_config
}

pub(crate) fn input_parsed<P>(db: &P, input: InternedInput) -> QueryResult<IRModule>
where
    P: Parser,
{
    use libeir_frontend::abstr_erlang::AbstrErlangFrontend;
    use libeir_frontend::eir::EirFrontend;
    use libeir_frontend::erlang::ErlangFrontend;

    let codemap = db.codemap().clone();
    let frontend: AnyFrontend = match db.input_type(input) {
        InputType::Erlang => ErlangFrontend::new(db.parse_config(), codemap).into(),
        InputType::AbstractErlang => AbstrErlangFrontend::new(codemap).into(),
        InputType::EIR => EirFrontend::new(codemap).into(),
        ty => {
            db.report_error(format!("invalid input type: {}", ty));
            return Err(ErrorReported);
        }
    };

    let codemap = db.codemap().clone();

    let (result, diags) = match db.lookup_intern_input(input) {
        Input::File(ref path) => frontend.parse_file_dyn(path),
        Input::Str { ref input, .. } => frontend.parse_string_dyn(input),
    };

    for ref diagnostic in diags.iter() {
        db.diagnostic(diagnostic);
    }

    match result {
        Ok(module) => {
            let options = db.options();
            db.maybe_emit_file_with_opts(&options, input, &module)?;
            Ok(module.into())
        }
        Err(_) => {
            db.report_error("parsing failed");
            Err(ErrorReported)
        }
    }
}

pub(crate) fn input_eir<P>(db: &P, input: InternedInput) -> QueryResult<IRModule>
where
    P: Parser,
{
    use libeir_passes::PassManager;

    let module = db.input_parsed(input)?;
    let mut ir_module: libeir_ir::Module = module.as_ref().clone();

    let mut pass_manager = PassManager::default();
    pass_manager.run(&mut ir_module);

    let new_module = IRModule::new(ir_module);
    db.maybe_emit_file(input, &new_module)?;
    Ok(new_module)
}

pub fn find_sources<D, P>(db: &D, dir: P) -> anyhow::Result<Arc<Seq<InternedInput>>>
where
    D: Parser,
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

    fn is_valid_entry(entry: &DirEntry) -> bool {
        // Recursively enter subdirectories
        if entry.path().is_dir() {
            return true;
        }
        if is_hidden(entry) {
            return false;
        }
        InputType::is_valid(entry.path())
    }

    let walker = WalkDir::new(dir.as_ref()).follow_links(false).into_iter();

    let mut inputs = Vec::new();

    for maybe_entry in walker.filter_entry(is_valid_entry) {
        let entry = maybe_entry?;
        if entry.path().is_file() {
            let input = Input::from(entry.path());
            let interned = db.intern_input(input);
            inputs.push(interned);
        }
    }

    Ok(Arc::new(inputs.into()))
}
