use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use clap::ArgMatches;
use log::debug;
use salsa::{ParallelDatabase, Snapshot};

use firefly_codegen as codegen;
use firefly_codegen::linker;
use firefly_codegen::meta::{CodegenResults, CompiledModule, ProjectInfo};
use firefly_diagnostics::{CodeMap, Diagnostic, Label};
use firefly_session::{CodegenOptions, DebuggingOptions, Options};
use firefly_syntax_base::{ApplicationMetadata, Deprecation, FunctionName, ModuleMetadata};
use firefly_util::diagnostics::Emitter;
use firefly_util::time::HumanDuration;

use crate::commands::*;
use crate::compiler::prelude::{Compiler as CompilerQueryGroup, *};
use crate::compiler::Compiler;
use crate::parser::prelude::Parser as ParserQueryGroup;
use crate::task;

pub fn handle_command<'a>(
    c_opts: CodegenOptions,
    z_opts: DebuggingOptions,
    matches: &ArgMatches<'a>,
    cwd: PathBuf,
    emitter: Option<Arc<dyn Emitter>>,
) -> anyhow::Result<()> {
    // Extract options from provided arguments
    let options = Options::new(c_opts, z_opts, cwd, &matches)?;
    // Construct empty code map for use in compilation
    let codemap = Arc::new(CodeMap::new());
    // Set up diagnostics
    let diagnostics = create_diagnostics_handler(&options, codemap.clone(), emitter);

    // Initialize codegen backend
    codegen::init(&options)?;

    // Build query database
    let mut db = Compiler::new(codemap, diagnostics);

    // The core of the query system is the initial set of options provided to the compiler
    //
    // The query system will use these options to construct the set of inputs on demand
    db.set_options(Arc::new(options));

    let inputs = db.inputs().unwrap_or_else(abort_on_err);
    let num_inputs = inputs.len();
    if num_inputs < 1 {
        db.diagnostics().fatal("No input sources found!").raise();
    }

    let start = Instant::now();

    // Spawn tasks to do initial parsing, semantic analysis and metadata gathering
    let mut tasks = inputs
        .iter()
        .copied()
        .map(|input| {
            let snapshot = db.snapshot();
            task::spawn(async move { parse(snapshot, input) })
        })
        .collect::<Vec<_>>();

    debug!("awaiting parse results from workers ({} units)", num_inputs);

    let options = db.options();
    let diagnostics = db.diagnostics();

    let mut modules = BTreeMap::new();

    for task in tasks.drain(..) {
        match task::join(task).unwrap() {
            Ok(metadata) => {
                modules.insert(metadata.name.name, metadata);
            }
            Err(_) => (),
        }
    }

    // Do not proceed with compilation if there were frontend errors
    diagnostics.abort_if_errors();

    // do not proceed with compilation if parse_only was set
    if options.debugging_opts.parse_only {
        diagnostics.notice("Finished", "skipping compilation, -Z parse_only was set");
        return Ok(());
    }

    // Initialize application metadata for use by compilation tasks
    let app = Arc::new(ApplicationMetadata {
        name: options.app.name,
        modules,
    });

    // Spawn tasks for each input to be compiled
    let mut tasks = inputs
        .iter()
        .copied()
        .map(|input| {
            let app = app.clone();
            let snapshot = db.snapshot();
            task::spawn(async move { compile(snapshot, input, app) })
        })
        .collect::<Vec<_>>();

    debug!(
        "awaiting compilation results from workers ({} units)",
        num_inputs
    );

    // Gather compilation results
    let mut codegen_results = CodegenResults {
        app_name: options.app.name,
        modules: Vec::with_capacity(num_inputs),
        project_info: ProjectInfo::new(&options),
    };

    for task in tasks.drain(..) {
        match task::join(task).unwrap() {
            Ok(None) => continue,
            Ok(Some(module)) => {
                codegen_results.modules.push(module);
            }
            Err(_) => (),
        }
    }

    // Do not proceed to linking if there were compilation errors
    diagnostics.abort_if_errors();

    // do not proceed with compilation if analyze_only was set
    if options.debugging_opts.analyze_only {
        diagnostics.notice("Finished", "skipping link, -Z analyze_only was set");
        return Ok(());
    }

    // Do not proceed to linking if we have no codegen artifacts
    if codegen_results.modules.is_empty() {
        diagnostics.notice("Finished", "skipping link, no artifacts requested");
    } else {
        // Link all compiled objects, if requested
        if !options.should_link() {
            if options.app_type.requires_link() {
                diagnostics.notice(
                    "Linker",
                    "linker was disabled, but is required for this artifact type",
                );
            } else {
                debug!(
                    "skipping link because it was not requested and project type does not require it"
                );
            }
        } else {
            if options.app_type.requires_link() {
                linker::link_binary(&options, &diagnostics, &codegen_results)?;
            } else {
                debug!("skipping link because project type does not require it");
            }
        }
    }

    let duration = HumanDuration::since(start);
    diagnostics.success(
        "Finished",
        &format!("built {} in {:#}", options.app.name, duration),
    );
    Ok(())
}

fn parse<C>(db: Snapshot<C>, input: InternedInput) -> Result<ModuleMetadata, ErrorReported>
where
    C: ParserQueryGroup + ParallelDatabase,
{
    debug!("spawning worker for {:?}", input);

    // Generate metadata about modules read from sources provided to the compiler
    let result = db.input_ast(input);
    match result {
        Err(err) => {
            let diagnostics = db.diagnostics();
            let input_info = db.lookup_intern_input(input);
            diagnostics.failed("Failed", format!("{}", &input_info.source_name()));
            Err(err)
        }
        Ok(module) => {
            let diagnostics = db.diagnostics();
            let name = module.name;
            let exports = module.exports.iter().cloned().collect();
            let mut deprecation = module.deprecation.clone();
            let mut deprecations: BTreeMap<FunctionName, Deprecation> = BTreeMap::new();
            for dep in module.deprecations.iter().copied() {
                match dep {
                    d @ Deprecation::Module { .. } if deprecation.is_none() => {
                        deprecation = Some(d);
                        continue;
                    }
                    Deprecation::Module { .. } => continue,
                    Deprecation::Function {
                        span,
                        function,
                        flag,
                    } => {
                        if function.is_local() {
                            deprecations.insert(
                                function.resolve(name.name),
                                Deprecation::Function {
                                    span,
                                    function,
                                    flag,
                                },
                            );
                        } else {
                            let module = function.module.unwrap();
                            if module == name.name {
                                deprecations.insert(
                                    *function,
                                    Deprecation::Function {
                                        span,
                                        function,
                                        flag,
                                    },
                                );
                            } else {
                                let diagnostic = Diagnostic::warning()
                                    .with_message("invalid deprecation")
                                    .with_labels(vec![Label::primary(span.source_id(), span)
                                        .with_message(
                                            "cannot deprecate a function in another module",
                                        )]);
                                diagnostics.emit(&diagnostic);
                            }
                        }
                    }
                }
            }
            Ok(ModuleMetadata {
                name,
                exports,
                deprecation,
                deprecations,
            })
        }
    }
}

fn compile<C>(
    db: Snapshot<C>,
    input: InternedInput,
    app: Arc<ApplicationMetadata>,
) -> Result<Option<CompiledModule>, ErrorReported>
where
    C: CompilerQueryGroup + ParallelDatabase,
{
    debug!("spawning worker for {:?}", input);

    // Generate an LLVM IR module for this input, or None, if only earlier stages are requested
    let thread_id = thread::current().id();
    let result = db.compile(thread_id, input, app);
    if result.is_err() {
        let diagnostics = db.diagnostics();
        let input_info = db.lookup_intern_input(input);
        diagnostics.failed("Failed", format!("{}", &input_info.source_name()));
    }

    result
}
