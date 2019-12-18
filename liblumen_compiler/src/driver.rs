use std::convert::TryInto;
use std::env::ArgsOs;
use std::error::Error;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};

use clap::ArgMatches;

use libeir_diagnostics::{CodeMap, Emitter};

use liblumen_codegen::{
    self as codegen,
    codegen::{CodegenResults, CompiledModule, ProjectInfo},
    linker::LinkerInfo,
};
use liblumen_incremental::ParserDatabase;
use liblumen_session::{
    CodegenOptions, DebuggingOptions, DiagnosticsHandler, Emit, Input, Options, OutputType,
};
use liblumen_target::{self as target, Target};

use crate::argparser;
use crate::compiler::{CodegenDatabase, CompilerDatabase};

pub fn run_compiler(cwd: PathBuf, args: ArgsOs) -> Result<()> {
    run_compiler_with_emitter(cwd, args, None)
}

pub fn run_compiler_with_emitter(
    cwd: PathBuf,
    args: ArgsOs,
    emitter: Option<Arc<dyn Emitter>>,
) -> Result<()> {
    use liblumen_session::OptionGroup;

    // Parse arguments
    let matches = argparser::parse(args)?;

    // Parse option groups first, as they can produce usage
    let c_opts = CodegenOptions::parse_option_group(&matches)?.unwrap_or_else(Default::default);
    let z_opts = DebuggingOptions::parse_option_group(&matches)?.unwrap_or_else(Default::default);

    // Get the selected subcommand
    match matches.subcommand() {
        ("print", subcommand_matches) => {
            handle_print(c_opts, z_opts, subcommand_matches.unwrap(), cwd, emitter)
        }
        ("compile", subcommand_matches) => {
            handle_compile(c_opts, z_opts, subcommand_matches.unwrap(), cwd, emitter)
        }
        ("parse", subcommand_matches) => {
            handle_parse(c_opts, z_opts, subcommand_matches.unwrap(), cwd, emitter)
        }
        (subcommand, _) => unimplemented!("subcommand '{}' is not implemented", subcommand),
    }
}

pub fn handle_print<'a>(
    c_opts: CodegenOptions,
    z_opts: DebuggingOptions,
    matches: &ArgMatches<'a>,
    cwd: PathBuf,
    emitter: Option<Arc<dyn Emitter>>,
) -> Result<()> {
    match matches.subcommand() {
        ("version", subcommand_matches) => {
            let verbose = subcommand_matches
                .map(|m| m.is_present("verbose"))
                .unwrap_or_else(|| matches.is_present("verbose"));
            if verbose {
                println!("release:     {}", crate::LUMEN_RELEASE);
                println!("commit-hash: {}", crate::LUMEN_COMMIT_HASH);
                println!("commit-date: {}", crate::LUMEN_COMMIT_DATE);
                println!("host:        {}", target::host_triple());
                println!("llvm:        {}", codegen::llvm_version());
            } else {
                println!("{}", crate::LUMEN_RELEASE);
            }
        }
        ("project-name", _) => {
            let basename = cwd.file_name().unwrap();
            println!("{}", basename.to_str().unwrap());
        }
        ("targets", _) => {
            for target in Target::all() {
                println!("{}", target);
            }
        }
        ("target-features", subcommand_matches) => {
            let options =
                Options::new_with_defaults(c_opts, z_opts, cwd, subcommand_matches.unwrap())?;
            let diagnostics = default_diagnostics_handler(&options, emitter);
            codegen::init(&options);
            codegen::print_target_features(&options, &diagnostics);
        }
        ("target-cpus", subcommand_matches) => {
            let options =
                Options::new_with_defaults(c_opts, z_opts, cwd, subcommand_matches.unwrap())?;
            let diagnostics = default_diagnostics_handler(&options, emitter);
            codegen::init(&options);
            codegen::print_target_cpus(&options, &diagnostics);
        }
        ("passes", _subcommand_matches) => {
            codegen::print_passes();
        }
        (subcommand, _) => unimplemented!("print subcommand '{}' is not implemented", subcommand),
    }

    Ok(())
}

fn default_diagnostics_handler(
    options: &Options,
    emitter: Option<Arc<dyn Emitter>>,
) -> DiagnosticsHandler {
    let codemap = Arc::new(Mutex::new(CodeMap::new()));
    create_diagnostics_handler(options, codemap, emitter)
}

fn create_diagnostics_handler(
    options: &Options,
    codemap: Arc<Mutex<CodeMap>>,
    emitter: Option<Arc<dyn Emitter>>,
) -> DiagnosticsHandler {
    let emitter = emitter.unwrap_or_else(|| default_emitter(codemap.clone(), &options));
    DiagnosticsHandler::new(emitter)
        .warnings_as_errors(options.warnings_as_errors)
        .no_warn(options.no_warn)
}

pub fn handle_compile<'a>(
    c_opts: CodegenOptions,
    z_opts: DebuggingOptions,
    matches: &ArgMatches<'a>,
    cwd: PathBuf,
    emitter: Option<Arc<dyn Emitter>>,
) -> Result<()> {
    // Extract options from provided arguments
    let mut options = Options::new(c_opts, z_opts, cwd, &matches)?;
    // Construct empty code map for use in compilation
    let codemap = Arc::new(Mutex::new(CodeMap::new()));
    // Set up diagnostics
    let diagnostics = create_diagnostics_handler(&options, codemap.clone(), emitter);

    let host = Target::search(target::host_triple()).unwrap_or_else(|e| {
        diagnostics
            .fatal_str(&format!(
                "Unable to load host specification: {}",
                e.description()
            ))
            .raise()
    });

    options.set_host_target(host);

    // Build query database
    let mut db = CompilerDatabase::new(codemap, diagnostics);

    // The core of the query system is the initial set of options provided to the compiler
    //
    // The query system will use these options to construct the set of inputs on demand
    db.set_options(Arc::new(options));

    // Run compilation, this is entirely demand-driven, so queries executed by the compiler
    // will either reuse cached artifacts, or perform the necessary work to construct fresh
    // artifacts.
    let inputs = db.inputs().unwrap_or_else(abort_on_err);
    for interned in inputs.iter().cloned() {
        let _llvm_ir = db.input_llvm_ir(interned).unwrap_or_else(abort_on_err);
        // TODO: This is just basic filler, needs to account for a lot more things
    }

    Ok(())
}

pub fn handle_parse<'a>(
    c_opts: CodegenOptions,
    z_opts: DebuggingOptions,
    matches: &ArgMatches<'a>,
    cwd: PathBuf,
    emitter: Option<Arc<dyn Emitter>>,
) -> Result<()> {
    use codegen::{
        mlir::{Context, Dialect},
        CodegenError,
    };
    use std::fs::File;

    // Extract options from provided arguments
    let mut options = Options::new(c_opts, z_opts, cwd, &matches)?;
    let diagnostics = default_diagnostics_handler(&options, emitter);

    let host = Target::search(target::host_triple()).unwrap_or_else(|e| {
        diagnostics
            .fatal_str(&format!(
                "Unable to load host specification: {}",
                e.description()
            ))
            .raise()
    });

    options.set_host_target(host);

    let input: Input = options
        .input_file
        .as_ref()
        .and_then(|filename| filename.try_into().ok())
        .ok_or_else(|| anyhow!("expected an input file"))?;

    // Initialize codegen backend
    codegen::init(&options);

    // Create context for MLIR
    let mut context = Context::new();

    // Parse MLIR source file
    let mlir_path = input.as_path().unwrap();
    let mut module = context.parse_file(mlir_path)?;

    // Lower to LLVM dialect
    let (opt, size) = codegen::ffi::util::to_llvm_opt_settings(options.opt_level);
    module.lower(&mut context, Dialect::LLVM, opt)?;

    // Emit LLVM dialect
    let mlir_output_name = options
        .output_types
        .always_emit(&input, OutputType::LLVMDialect);
    let mlir_output_path = options.output_dir().join(mlir_output_name);
    let mut f = File::create(mlir_output_path)?;
    if let Err(err) = module.emit(&mut f) {
        diagnostics.error(err);
        return Err(anyhow::Error::new(CodegenError).context("failed to emit MLIR (LLVM dialect)"));
    }

    // Convert to LLVM IR
    let target_machine = codegen::target::create_target_machine(&options, &diagnostics, false);
    let llvm_module = module.lower_to_llvm_ir(opt, size, target_machine)?;

    // Emit LLVM IR
    let llvm_output_name = options
        .output_types
        .always_emit(&input, OutputType::LLVMAssembly);
    let llvm_output_path = options.output_dir().join(llvm_output_name);
    let mut f = File::create(llvm_output_path)?;
    if let Err(err) = llvm_module.emit_ir(&mut f) {
        diagnostics.error(err);
        return Err(anyhow::Error::new(CodegenError).context("failed to emit LLVM IR"));
    }

    /*
    // Emit LLVM bitcode
    let bc_output_name = options.output_types.always_emit(&input, OutputType::LLVMBitcode);
    let bc_output_path = options.output_dir().join(bc_output_name);
    let mut f = File::create(bc_output_path.clone())?;
    if let Err(err) = llvm_module.emit_ir(&mut f) {
        diagnostics.error(err);
        return Err(anyhow::Error::new(CodegenError)
            .context("failed to emit LLVM bitcode"));
    }
    */

    // Emit object file
    let obj_output_name = options.output_types.always_emit(&input, OutputType::Object);
    let obj_output_path = options.output_dir().join(obj_output_name);
    let mut f = File::create(obj_output_path.clone())?;
    if let Err(err) = llvm_module.emit_obj(&mut f) {
        diagnostics.error(err);
        return Err(anyhow::Error::new(CodegenError).context("failed to emit object file"));
    }

    // Link binary
    let compiled = CompiledModule {
        name: input.file_stem().to_string_lossy().into_owned(),
        object: Some(obj_output_path),
        bytecode: None,
    };
    let results = CodegenResults {
        project_name: options.project_name.clone(),
        modules: vec![compiled],
        windows_subsystem: None,
        linker_info: LinkerInfo::new(),
        project_info: ProjectInfo::new(&options),
    };

    if let Err(err) = codegen::linker::link_binary(&options, &diagnostics, &results) {
        diagnostics.error(err);
        return Err(anyhow::Error::new(CodegenError).context("failed to link binary"));
    }

    Ok(())
}

fn default_emitter(codemap: Arc<Mutex<CodeMap>>, options: &Options) -> Arc<dyn Emitter> {
    use libeir_diagnostics::{NullEmitter, StandardStreamEmitter};
    use liblumen_session::verbosity_to_severity;
    use liblumen_util::error::Verbosity;

    match options.verbosity {
        Verbosity::Silent => Arc::new(NullEmitter::new()),
        v => Arc::new(
            StandardStreamEmitter::new(options.use_color.into())
                .set_codemap(codemap)
                .set_min_severity(verbosity_to_severity(v)),
        ),
    }
}

fn abort_on_err<T>(_: ()) -> T {
    use liblumen_util::error::FatalError;

    FatalError.raise()
}
