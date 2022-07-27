use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::ThreadId;

use log::debug;

use liblumen_diagnostics::Reporter;
use liblumen_llvm as llvm;
use liblumen_mlir as mlir;
use liblumen_session::{Input, InputType};
use liblumen_syntax_core as syntax_core;
use liblumen_syntax_erl::{self as syntax_erl, ParseConfig};
use liblumen_util::diagnostics::FileName;

use super::prelude::*;

macro_rules! unwrap_or_bail {
    ($db:ident, $e:expr) => {
        match $e {
            Ok(result) => result,
            Err(ref e) => {
                bail!($db, "{}", e);
            }
        }
    };
}

macro_rules! bail {
    ($db:ident, $e:expr) => {{
        $db.report_error($e);
        return Err(ErrorReported);
    }};

    ($db:ident, $fmt:literal, $($arg:expr),*) => {{
        $db.report_error(format!($fmt, $($arg),*));
        return Err(ErrorReported);
    }}
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
    parse_config.code_paths = Default::default();
    parse_config
}

pub(crate) fn output_dir<P>(db: &P) -> PathBuf
where
    P: Parser,
{
    db.options().output_dir()
}

pub(super) fn llvm_context<P>(db: &P, thread_id: ThreadId) -> Arc<llvm::OwnedContext>
where
    P: Parser,
{
    debug!("constructing new llvm context for thread {:?}", thread_id);
    let options = db.options().clone();
    let diagnostics = db.diagnostics().clone();
    Arc::new(llvm::OwnedContext::new(options, diagnostics))
}

pub(super) fn target_machine<P>(
    db: &P,
    thread_id: ThreadId,
) -> Arc<llvm::target::OwnedTargetMachine>
where
    P: Parser,
{
    use llvm::target::TargetMachine;

    debug!("constructing new target machine for thread {:?}", thread_id);
    let options = db.options();
    let target_machine = TargetMachine::create(&options).unwrap();
    Arc::new(target_machine)
}

pub(super) fn mlir_context<P>(db: &P, thread_id: ThreadId) -> Arc<mlir::OwnedContext>
where
    P: Parser,
{
    debug!("constructing new mlir context for thread {:?}", thread_id);

    let options = db.options();
    let diagnostics = db.diagnostics().clone();

    Arc::new(mlir::OwnedContext::new(&options, &diagnostics))
}

pub(crate) fn inputs<P>(db: &P) -> Result<Vec<InternedInput>, ErrorReported>
where
    P: Parser,
{
    use std::io::{self, Read};

    // Fetch all of the inputs associated with the current application
    let options = db.options();
    let mut inputs: Vec<InternedInput> = Vec::new();

    for input in options.input_files.iter() {
        // We can get three types of input:
        //
        // 1. `stdin` for standard input
        // 2. `path/to/file.erl` for a single file
        // 3. `path/to/dir` for a directory containing a standard Erlang application
        match input {
            // Read from standard input
            &FileName::Virtual(ref name) if name == "stdin" => {
                let mut source = String::new();
                db.to_query_result(
                    io::stdin()
                        .read_to_string(&mut source)
                        .map_err(|e| e.into()),
                )?;
                let input = Input::new(name.clone(), source);
                let interned = db.intern_input(input);
                inputs.push(interned)
            }
            // Read from a single file
            &FileName::Real(ref path) if path.exists() && path.is_file() => {
                let input = Input::File(path.clone());
                let interned = db.intern_input(input);
                inputs.push(interned)
            }
            // Load sources from <dir>
            &FileName::Real(ref path) if path.exists() && path.is_dir() => {
                let sources = unwrap_or_bail!(db, find_sources(db, path));
                inputs.extend_from_slice(&sources);
            }
            // Invalid virtual file
            &FileName::Virtual(_) => {
                bail!(
                    db,
                    "invalid input file, expected `-`, a file path, or a directory"
                );
            }
            // Invalid file/directory path
            &FileName::Real(ref path_buf) => {
                bail!(
                    db,
                    "invalid input file ({}), not a file or directory",
                    path_buf.to_string_lossy()
                );
            }
        }
    }

    Ok(inputs)
}

pub(crate) fn input_type<P>(db: &P, input: InternedInput) -> InputType
where
    P: Parser,
{
    let input_info = db.lookup_intern_input(input);
    input_info.get_type()
}

pub(crate) fn input_ast<P>(
    db: &P,
    input: InternedInput,
) -> Result<syntax_erl::ast::Module, ErrorReported>
where
    P: Parser,
{
    use liblumen_parser as parse;

    let codemap = db.codemap().clone();
    let config = db.parse_config();

    let result = match db.input_type(input) {
        InputType::Erlang => {
            let reporter = if config.warnings_as_errors {
                Reporter::strict()
            } else {
                Reporter::new()
            };
            let parser = parse::Parser::new(config, codemap);
            match db.lookup_intern_input(input) {
                Input::File(ref path) => {
                    parser.parse_file::<syntax_erl::ast::Module, &Path>(reporter, path)
                }
                Input::Str { ref input, .. } => {
                    parser.parse_string::<syntax_erl::ast::Module, _>(reporter, input)
                }
            }
        }
        ty => bail!(db, "invalid input type: {}", ty),
    };

    match result {
        Ok(module) => {
            let options = db.options();
            db.maybe_emit_file_with_opts(&options, input, &module)?;
            Ok(module)
        }
        Err(_) => bail!(db, "parsing failed"),
    }
}

pub(crate) fn input_cst<P>(
    db: &P,
    input: InternedInput,
) -> Result<syntax_erl::cst::Module, ErrorReported>
where
    P: Parser,
{
    use liblumen_pass::Pass;
    use liblumen_syntax_erl::passes::AstToCst;

    // Get Erlang AST
    let ast = db.input_ast(input)?;

    // Run lowering passes
    let options = db.options();
    let reporter = if options.warnings_as_errors {
        Reporter::strict()
    } else {
        Reporter::new()
    };
    let mut passes = AstToCst::new(reporter.clone());
    let module = unwrap_or_bail!(db, passes.run(ast));

    db.maybe_emit_file(input, &module)?;

    Ok(module)
}

pub(crate) fn input_syntax_core<P>(
    db: &P,
    input: InternedInput,
) -> Result<syntax_core::Module, ErrorReported>
where
    P: Parser,
{
    use liblumen_pass::Pass;
    use liblumen_syntax_erl::passes::CstToCore;

    // Get Erlang CST
    let cst = db.input_cst(input)?;

    // Run lowering passes
    let options = db.options();
    let reporter = if options.warnings_as_errors {
        Reporter::strict()
    } else {
        Reporter::new()
    };

    let mut passes = CstToCore::new(reporter);
    let module = unwrap_or_bail!(db, passes.run(cst));

    db.maybe_emit_file(input, &module)?;

    Ok(module)
}

pub(crate) fn input_mlir<P>(
    db: &P,
    thread_id: ThreadId,
    input: InternedInput,
) -> Result<mlir::OwnedModule, ErrorReported>
where
    P: Parser,
{
    use liblumen_codegen::passes::CoreToMlir;
    use liblumen_pass::Pass;

    let options = db.options();
    let module = match db.input_type(input) {
        InputType::MLIR => {
            let context = db.mlir_context(thread_id);
            match db.lookup_intern_input(input) {
                Input::File(ref path) => {
                    debug!("parsing mlir from file for {:?} on {:?}", input, thread_id);
                    unwrap_or_bail!(db, context.parse_file(path))
                }
                Input::Str { ref input, .. } => {
                    debug!(
                        "parsing mlir from string for {:?} on {:?}",
                        input, thread_id
                    );
                    unwrap_or_bail!(db, context.parse_string(input.as_ref()))
                }
            }
        }
        InputType::Erlang | InputType::AbstractErlang | InputType::CoreIR => {
            debug!("generating mlir for {:?} on {:?}", input, thread_id);
            let module = db.input_syntax_core(input)?;
            let codemap = db.codemap();
            let context = db.mlir_context(thread_id);

            let mut passes = CoreToMlir::new(&context, &codemap, &options);
            match unwrap_or_bail!(db, passes.run(module)) {
                Ok(mlir_module) => mlir_module,
                Err(mlir_module) => {
                    db.maybe_emit_file_with_opts(&options, input, &mlir_module)?;
                    bail!(db, "mlir module verification failed");
                }
            }
        }
        ty => bail!(db, "invalid input type: {}", ty),
    };

    db.maybe_emit_file_with_opts(&options, input, &module)?;

    Ok(module)
}

pub fn find_sources<D, P>(db: &D, dir: P) -> anyhow::Result<Vec<InternedInput>>
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

    fn is_valid_entry(root: &Path, entry: &DirEntry) -> bool {
        if is_hidden(entry) {
            return false;
        }
        // Recurse into the root directory, and nested src directory, no others
        let path = entry.path();
        if entry.file_type().is_dir() {
            return path == root || path.file_name().unwrap().to_str().unwrap() == "src";
        }
        InputType::Erlang.validate(path)
    }

    let root = dir.as_ref();
    let walker = WalkDir::new(root)
        .max_depth(2)
        .follow_links(false)
        .into_iter();

    let mut inputs = Vec::new();

    for maybe_entry in walker.filter_entry(|e| is_valid_entry(root, e)) {
        let entry = maybe_entry?;
        if entry.path().is_file() {
            let input = Input::from(entry.path());
            let interned = db.intern_input(input);
            inputs.push(interned);
        }
    }

    Ok(inputs)
}
