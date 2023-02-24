use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::ArgMatches;
use log::debug;

use firefly_session::{CodegenOptions, DebuggingOptions, Options};
use firefly_util::diagnostics::{CodeMap, Emitter};
use firefly_util::time::HumanDuration;

use crate::compiler::Compiler;

pub fn configure<'a>(
    codemap: Arc<CodeMap>,
    c_opts: CodegenOptions,
    z_opts: DebuggingOptions,
    cwd: PathBuf,
    matches: &ArgMatches<'a>,
) -> anyhow::Result<Arc<Options>> {
    // This is used for diagnostics while parsing .app/.app.src files
    Options::new(None, codemap, c_opts, z_opts, cwd, matches).map(Arc::new)
}

pub fn handle_command(
    options: Arc<Options>,
    codemap: Arc<CodeMap>,
    emitter: Option<Arc<dyn Emitter>>,
) -> anyhow::Result<()> {
    // Set up diagnostics
    let diagnostics = options.create_diagnostics_handler(codemap.clone(), emitter);

    if options.input_files.is_empty() {
        diagnostics.fatal("No inputs found!").raise();
    }

    // Track when compilation began
    let start = Instant::now();

    // Run the compiler
    let compiler = Compiler::new(options.clone(), codemap.clone(), diagnostics.clone());
    let artifacts = compiler.compile()?;

    // Do not proceed with compilation if there were frontend errors
    diagnostics.abort_if_errors();

    if options.debugging_opts.parse_only {
        diagnostics.notice("Finished", "skipping compilation, -Z parse_only was set");
        return Ok(());
    }

    if options.debugging_opts.analyze_only {
        diagnostics.notice("Finished", "skipping link, -Z analyze_only was set");
        return Ok(());
    }

    // Do not proceed to linking if we have no codegen artifacts
    if artifacts.modules.is_empty() {
        diagnostics.notice("Finished", "skipping link, no artifacts requested");
        return Ok(());
    }

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
            firefly_linker::link_binary(&options, &diagnostics, &artifacts)?;
        }
    }

    let duration = HumanDuration::since(start);
    diagnostics.success(
        "Finished",
        &format!("built {} in {:#}", options.app.name, duration),
    );
    Ok(())
}
