use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::ThreadId;

use log::debug;

use firefly_diagnostics::{Reporter, ToDiagnostic};
use firefly_intern::{symbols, Symbol};
use firefly_llvm as llvm;
use firefly_mlir as mlir;
use firefly_session::{Input, InputType};
use firefly_syntax_base::ApplicationMetadata;
use firefly_syntax_core as syntax_core;
use firefly_syntax_erl::{self as syntax_erl, ParseConfig};
use firefly_syntax_kernel as syntax_kernel;
use firefly_syntax_ssa as syntax_ssa;
use firefly_util::diagnostics::FileName;

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

    ($db:ident, $reporter:expr, $codemap:expr, $e:expr) => {
        match $e {
            Ok(result) => {
                $reporter.print($codemap);
                result
            }
            Err(ref e) => {
                $reporter.print($codemap);
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
    parse_config.define(symbols::VSN, crate::FIREFLY_RELEASE);
    parse_config.define(symbols::COMPILER_VSN, crate::FIREFLY_RELEASE);
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

pub(crate) fn inputs<P>(db: &P, app: Symbol) -> Result<Vec<InternedInput>, ErrorReported>
where
    P: Parser,
{
    use std::io::{self, Read};

    // Fetch all of the inputs associated with the given application
    let options = db.options();
    let mut inputs: Vec<InternedInput> = Vec::new();

    let input_files = options.input_files.get(&app).unwrap();

    for input in input_files.iter() {
        // We can get three types of input:
        //
        // 1. `stdin` for standard input
        // 2. `path/to/file.erl` for a single file
        // 3. `path/to/dir` for a directory containing Erlang sources
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
) -> Result<syntax_erl::Module, ErrorReported>
where
    P: Parser,
{
    use firefly_beam::AbstractCode;
    use firefly_parser as parse;
    use firefly_pass::Pass;
    use firefly_syntax_erl::passes::AbstractErlangToAst;
    use firefly_syntax_pp as syntax_pp;

    let options = db.options();
    let codemap = db.codemap().clone();
    let config = db.parse_config();
    let reporter = if config.warnings_as_errors {
        Reporter::strict()
    } else {
        Reporter::new()
    };

    let input_type = db.input_type(input);

    // For standard Erlang sources, we need only parse the source file
    if input_type == InputType::Erlang {
        let parser = parse::Parser::new(config, codemap.clone());
        let result = match db.lookup_intern_input(input) {
            Input::File(ref path) => {
                parser.parse_file::<syntax_erl::Module, &Path, _>(reporter.clone(), path)
            }
            Input::Str { ref input, .. } => {
                parser.parse_string::<syntax_erl::Module, _, _>(reporter.clone(), input)
            }
        };

        match result {
            Ok(module) => {
                reporter.print(&codemap);
                db.maybe_emit_file_with_opts(&options, input, &module)?;
                return Ok(module);
            }
            Err(e) => {
                reporter.diagnostic(e.to_diagnostic());
                reporter.print(&codemap);
                bail!(db, "parsing failed, see diagnostics for details");
            }
        }
    }

    // For Abstract Erlang, either in source form or from a BEAM, we need to obtain the
    // Abstract Erlang syntax tree, and then convert it to our normal Erlang syntax tree
    let ast = match input_type {
        InputType::AbstractErlang => {
            let parser = parse::Parser::new((), codemap.clone());
            let result = match db.lookup_intern_input(input) {
                Input::File(ref path) => {
                    parser.parse_file::<syntax_pp::ast::Ast, &Path, _>(reporter.clone(), path)
                }
                Input::Str { ref input, .. } => {
                    parser.parse_string::<syntax_pp::ast::Ast, _, _>(reporter.clone(), input)
                }
            };
            unwrap_or_bail!(db, reporter, &codemap, result)
        }
        InputType::BEAM => {
            let result = match db.lookup_intern_input(input) {
                Input::File(ref path) => AbstractCode::from_beam_file(path).map(|code| code.into()),
                Input::Str { .. } => {
                    bail!(db, "beam parsing is only supported on files");
                }
            };
            unwrap_or_bail!(db, reporter, &codemap, result)
        }
        ty => bail!(db, "invalid input type: {}", ty),
    };

    let mut passes = AbstractErlangToAst::new(reporter.clone(), codemap.clone());
    match passes.run(ast) {
        Ok(module) => {
            reporter.print(&codemap);
            db.maybe_emit_file_with_opts(&options, input, &module)?;
            Ok(module)
        }
        Err(ref e) => {
            reporter.print(&codemap);
            bail!(db, format!("{}", e));
        }
    }
}

pub(crate) fn input_core<P>(
    db: &P,
    input: InternedInput,
    app: Arc<ApplicationMetadata>,
) -> Result<syntax_core::Module, ErrorReported>
where
    P: Parser,
{
    use firefly_pass::Pass;
    use firefly_syntax_erl::passes::{AstToCore, CanonicalizeSyntax, SemanticAnalysis};

    // Get Erlang AST
    let ast = db.input_ast(input)?;

    // Run lowering passes
    let options = db.options();
    let codemap = db.codemap().clone();
    let reporter = if options.warnings_as_errors {
        Reporter::strict()
    } else {
        Reporter::new()
    };

    let mut passes = SemanticAnalysis::new(reporter.clone(), &app)
        .chain(CanonicalizeSyntax::new(reporter.clone(), codemap.clone()))
        .chain(AstToCore::new(reporter.clone()));

    let module = unwrap_or_bail!(db, reporter, &codemap, passes.run(ast));

    db.maybe_emit_file(input, &module)?;

    Ok(module)
}

pub(crate) fn input_kernel<P>(
    db: &P,
    input: InternedInput,
    app: Arc<ApplicationMetadata>,
) -> Result<syntax_kernel::Module, ErrorReported>
where
    P: Parser,
{
    use firefly_pass::Pass;
    use firefly_syntax_kernel::passes::CoreToKernel;

    // Get Core AST
    let ast = db.input_core(input, app)?;

    // Run lowering passes
    let options = db.options();
    let codemap = db.codemap().clone();
    let reporter = if options.warnings_as_errors {
        Reporter::strict()
    } else {
        Reporter::new()
    };
    let mut passes = CoreToKernel::new(reporter.clone());
    let module = unwrap_or_bail!(db, reporter, &codemap, passes.run(ast));

    db.maybe_emit_file(input, &module)?;

    Ok(module)
}

pub(crate) fn input_ssa<P>(
    db: &P,
    input: InternedInput,
    app: Arc<ApplicationMetadata>,
) -> Result<syntax_ssa::Module, ErrorReported>
where
    P: Parser,
{
    use firefly_pass::Pass;
    use firefly_syntax_kernel::passes::KernelToSsa;

    // Get Kernel Erlang module
    let cst = db.input_kernel(input, app)?;

    // Run lowering passes
    let options = db.options();
    let codemap = db.codemap().clone();
    let reporter = if options.warnings_as_errors {
        Reporter::strict()
    } else {
        Reporter::new()
    };

    let mut passes = KernelToSsa::new(reporter.clone());
    let module = unwrap_or_bail!(db, reporter, &codemap, passes.run(cst));

    db.maybe_emit_file(input, &module)?;

    Ok(module)
}

pub(crate) fn input_mlir<P>(
    db: &P,
    thread_id: ThreadId,
    input: InternedInput,
    app: Arc<ApplicationMetadata>,
) -> Result<mlir::OwnedModule, ErrorReported>
where
    P: Parser,
{
    use firefly_codegen::passes::SsaToMlir;
    use firefly_pass::Pass;

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
        InputType::Erlang | InputType::AbstractErlang => {
            debug!("generating mlir for {:?} on {:?}", input, thread_id);
            let module = db.input_ssa(input, app)?;
            let codemap = db.codemap();
            let context = db.mlir_context(thread_id);

            let mut passes = SsaToMlir::new(&context, &codemap, &options);
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
