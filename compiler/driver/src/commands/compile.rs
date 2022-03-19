use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use clap::ArgMatches;
use log::debug;
use salsa::{ParallelDatabase, Snapshot};

use liblumen_codegen as codegen;
use liblumen_codegen::linker::{self, LinkerInfo};
use liblumen_codegen::meta::{CodegenResults, CompiledModule, ProjectInfo};
use liblumen_session::{CodegenOptions, DebuggingOptions, Options};
use liblumen_util::diagnostics::{CodeMap, Emitter};
use liblumen_util::time::HumanDuration;

use crate::commands::*;
use crate::compiler::prelude::{Compiler as CompilerQueryGroup, *};
use crate::compiler::Compiler;
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

    // Spawn tasks for each input to be compiled
    let start = Instant::now();
    let mut tasks = inputs
        .iter()
        .copied()
        .map(|input| {
            let snapshot = db.snapshot();
            task::spawn(async move { compile(snapshot, input) })
        })
        .collect::<Vec<_>>();

    debug!("awaiting results from workers ({} units)", num_inputs);

    // Gather compilation results
    let options = db.options();
    let mut codegen_results = CodegenResults {
        app_name: options.app.name,
        modules: Vec::with_capacity(num_inputs),
        windows_subsystem: None,
        linker_info: LinkerInfo::new(),
        project_info: ProjectInfo::new(&options),
    };

    let diagnostics = db.diagnostics();
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

    // Do not proceed to linking if we have no codegen artifacts
    if codegen_results.modules.is_empty() {
        diagnostics.notice("Linker", "skipping link step, no artifacts requested");
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

fn compile<C>(
    db: Snapshot<C>,
    input: InternedInput,
) -> Result<Option<CompiledModule>, ErrorReported>
where
    C: CompilerQueryGroup + ParallelDatabase,
{
    debug!("spawning worker for {:?}", input);

    // Genereate an LLVM IR module for this input, or None, if only earlier stages are requested
    let thread_id = thread::current().id();
    let result = db.compile(thread_id, input);
    if result.is_err() {
        let diagnostics = db.diagnostics();
        let input_info = db.lookup_intern_input(input);
        diagnostics.failed("Failed", format!("{}", &input_info.source_name()));
    }

    result
}
