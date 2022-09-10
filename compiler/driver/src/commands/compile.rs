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
use firefly_diagnostics::{CodeMap, Diagnostic, Label, Reporter, Span};
use firefly_intern::Symbol;
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
    // Construct empty code map for use in compilation
    let codemap = Arc::new(CodeMap::new());
    let options = {
        // This is used for diagnostics while parsing .app/.app.src files
        let reporter = Reporter::new();
        let result = Options::new(&reporter, codemap.clone(), c_opts, z_opts, cwd, &matches);
        match result {
            Ok(options) => options,
            Err(err) => {
                reporter.print(&codemap);
                return Err(err);
            }
        }
    };
    // Set up diagnostics
    let diagnostics = create_diagnostics_handler(&options, codemap.clone(), emitter);

    // Initialize codegen backend
    codegen::init(&options)?;

    // Build query database
    let mut db = Compiler::new(codemap, diagnostics);

    // Prefetch all of the applications to compile
    let apps = options.input_files.keys().copied().collect::<Vec<_>>();
    let num_apps = apps.len();

    // The core of the query system is the initial set of options provided to the compiler
    //
    // The query system will use these options to construct the set of inputs on demand
    db.set_options(Arc::new(options));

    if apps.is_empty() {
        db.diagnostics().fatal("No inputs found!").raise();
    }

    let start = Instant::now();

    // Spawn tasks to do initial parsing, semantic analysis and metadata gathering
    let mut tasks = apps
        .iter()
        .copied()
        .map(|app| {
            let snapshot = db.snapshot();
            task::spawn(async move { parse_all(snapshot, app) })
        })
        .collect::<Vec<_>>();

    debug!("awaiting parse results from workers ({} units)", num_apps);

    let options = db.options();
    let diagnostics = db.diagnostics();

    let mut apps = BTreeMap::new();

    for task in tasks.drain(..) {
        match task::join(task).unwrap() {
            Ok(metadata) => {
                apps.insert(metadata.name, metadata);
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

    // Spawn tasks for each input to be compiled
    let mut tasks = apps
        .iter()
        .map(|(app, meta)| {
            let app = *app;
            let meta = meta.clone();
            let snapshot = db.snapshot();
            task::spawn(async move { compile_all(snapshot, app, meta) })
        })
        .collect::<Vec<_>>();

    debug!(
        "awaiting compilation results from workers ({} units)",
        num_apps
    );

    let mut results = BTreeMap::new();

    for task in tasks.drain(..) {
        match task::join(task).unwrap() {
            Ok(cg) => {
                results.insert(cg.app_name, cg);
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
    if results.iter().all(|(_, cg)| cg.modules.is_empty()) {
        diagnostics.notice("Finished", "skipping link, no artifacts requested");
    } else {
        // Link all compiled objects, if requested
        if !options.should_link() {
            if options.project_type.requires_link() {
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
            if options.codegen_opts.no_link {
                diagnostics.notice("Linker", "skipping link as -C no_link was set");
            } else {
                if results.len() == 1 {
                    let (_, cg) = results.pop_first().unwrap();
                    linker::link_binary(&options, &diagnostics, &cg)?;
                } else {
                    // We have multiple codegen units which we want to link together
                    // Simply take the modules from all the non-root apps and append to the root app
                    let (_, mut root) = results.remove_entry(&options.app.name).unwrap();
                    for mut cg in results.into_values() {
                        root.modules.extend(cg.modules.drain(..));
                    }
                    linker::link_binary(&options, &diagnostics, &root)?;
                }
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

fn parse_all<C>(db: Snapshot<C>, app: Symbol) -> Result<Arc<ApplicationMetadata>, ErrorReported>
where
    C: ParserQueryGroup + ParallelDatabase,
{
    debug!("spawning worker for {:?}", app);

    let inputs = db.inputs(app).unwrap_or_else(abort_on_err);
    if inputs.is_empty() {
        return Ok(Arc::new(ApplicationMetadata {
            name: app,
            modules: BTreeMap::new(),
        }));
    }

    let modules = inputs
        .iter()
        .copied()
        .map(|input| parse(&db, input).map(|meta| (meta.name.name, meta)))
        .try_collect()?;
    Ok(Arc::new(ApplicationMetadata { name: app, modules }))
}

fn parse<C>(db: &Snapshot<C>, input: InternedInput) -> Result<ModuleMetadata, ErrorReported>
where
    C: ParserQueryGroup + ParallelDatabase,
{
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
                    Deprecation::FunctionAnyArity {
                        span,
                        name: deprecated_name,
                        flag,
                    } => {
                        // Search for matching functions and deprecate them
                        for function in module.functions.keys().copied() {
                            if function.function == deprecated_name {
                                deprecations.insert(
                                    function.resolve(name.name),
                                    Deprecation::Function {
                                        span,
                                        function: Span::new(span, function),
                                        flag,
                                    },
                                );
                            }
                        }
                    }
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

fn compile_all<C>(
    db: Snapshot<C>,
    app: Symbol,
    meta: Arc<ApplicationMetadata>,
) -> Result<CodegenResults, ErrorReported>
where
    C: CompilerQueryGroup + ParallelDatabase,
{
    let options = db.options();
    let project_info = if app == options.app.name {
        ProjectInfo::new(&options)
    } else {
        ProjectInfo::default()
    };

    let inputs = db.inputs(app).unwrap_or_else(abort_on_err);
    if inputs.is_empty() {
        return Ok(CodegenResults {
            app_name: app,
            modules: vec![],
            project_info,
        });
    }

    let modules = inputs
        .iter()
        .copied()
        .filter_map(|input| match compile(&db, input, meta.clone()) {
            Ok(None) => None,
            Ok(Some(m)) => Some(Ok(m)),
            Err(err) => Some(Err(err)),
        })
        .try_collect()?;

    Ok(CodegenResults {
        app_name: app,
        modules,
        project_info,
    })
}

fn compile<C>(
    db: &Snapshot<C>,
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
