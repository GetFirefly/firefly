pub mod passes;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context};

use log::debug;

use rayon::prelude::*;

use firefly_diagnostics::{CodeMap, FileName};
use firefly_intern::Symbol;
use firefly_linker::{AppArtifacts, ProjectInfo};
use firefly_pass::Pass;
use firefly_session::{Input, InputType, Options, OutputType};
use firefly_syntax_base::ApplicationMetadata;
use firefly_util::diagnostics::DiagnosticsHandler;
use firefly_util::emit::Emit;

use self::passes::{ParsePipeline, PreCodegenPipeline};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorReported;

#[derive(Clone)]
pub struct Compiler {
    pub options: Arc<Options>,
    pub codemap: Arc<CodeMap>,
    pub diagnostics: Arc<DiagnosticsHandler>,
}
impl Compiler {
    pub fn new(
        options: Arc<Options>,
        codemap: Arc<CodeMap>,
        diagnostics: Arc<DiagnosticsHandler>,
    ) -> Self {
        Self {
            options,
            codemap,
            diagnostics,
        }
    }

    #[cfg(not(feature = "native-compilation"))]
    pub fn compile(self) -> anyhow::Result<AppArtifacts> {
        // Fetch all of the inputs associated with the given application
        let files = self.options.input_files.clone();

        let mut compiler = BytecodeCompiler {
            options: self.options,
            codemap: self.codemap,
            diagnostics: self.diagnostics,
        };

        compiler.run(files)
    }
}

pub struct Artifact<T, M = ()> {
    pub input: Input,
    pub output: T,
    pub metadata: M,
}
impl<T, M> Clone for Artifact<T, M>
where
    T: Clone,
    M: Clone,
{
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            output: self.output.clone(),
            metadata: self.metadata.clone(),
        }
    }
}
impl<T, M> Artifact<T, M>
where
    T: Emit,
{
    pub fn maybe_emit_file_with_opts(&self, options: &Options) -> anyhow::Result<Option<PathBuf>> {
        use std::str::FromStr;

        let extension = self.output.file_type().unwrap();
        let output_type =
            OutputType::from_str(extension).expect("unrecognized file type extension");
        if let Some(filename) = options.maybe_emit(&self.input, output_type) {
            debug!("emitting {} for {:?}", output_type, &self.input);
            Ok(Some(self.emit_file(filename)?))
        } else {
            Ok(None)
        }
    }

    pub fn emit_file(&self, outfile: PathBuf) -> anyhow::Result<PathBuf> {
        emit_file_with_callback(outfile, |f| self.output.emit(f))
    }
}

pub struct BytecodeCompiler {
    pub options: Arc<Options>,
    pub codemap: Arc<CodeMap>,
    pub diagnostics: Arc<DiagnosticsHandler>,
}
impl BytecodeCompiler {
    fn parse(
        &self,
        inputs: HashMap<Symbol, Vec<FileName>>,
    ) -> anyhow::Result<BTreeMap<Symbol, ParsedApp>> {
        let pipeline = ParsePipeline::new(
            self.options.clone(),
            self.codemap.clone(),
            self.diagnostics.clone(),
        );
        inputs
            .into_par_iter()
            .panic_fuse()
            .map_with(pipeline, |pipeline, (app, files)| {
                parse(pipeline, app, files)
            })
            .collect()
    }

    fn lower(
        &self,
        parsed: BTreeMap<Symbol, ParsedApp>,
    ) -> anyhow::Result<Vec<Artifact<firefly_syntax_ssa::Module>>> {
        let pipeline = PreCodegenPipeline::new(
            self.options.clone(),
            self.codemap.clone(),
            self.diagnostics.clone(),
        );
        parsed
            .into_par_iter()
            .panic_fuse()
            .map_with(pipeline, |pipeline, (_app, mut parsed)| {
                let mut result = Vec::with_capacity(parsed.modules.len());

                for artifact in parsed.modules.drain(..) {
                    result.push(pipeline.run((parsed.metadata.clone(), artifact))?);
                }

                Ok(result)
            })
            .reduce(
                || Ok(vec![]),
                |acc, result| match acc {
                    err @ Err(_) => err,
                    Ok(_) if result.is_err() => result,
                    Ok(mut acc) => {
                        let mut artifacts = result.unwrap();
                        acc.append(&mut artifacts);
                        Ok(acc)
                    }
                },
            )
    }
}
impl Pass for BytecodeCompiler {
    type Input<'a> = HashMap<Symbol, Vec<FileName>>;
    type Output<'a> = AppArtifacts;

    fn run<'a>(&mut self, inputs: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        use self::passes::CompileBytecode;

        if self.options.debugging_opts.parse_only {
            self.parse(inputs)?;
            return Ok(AppArtifacts {
                name: self.options.app.name,
                modules: vec![],
                project_info: ProjectInfo::new(&self.options),
            });
        }

        if self.options.debugging_opts.analyze_only {
            let parsed = self.parse(inputs)?;
            self.lower(parsed)?;
            return Ok(AppArtifacts {
                name: self.options.app.name,
                modules: vec![],
                project_info: ProjectInfo::new(&self.options),
            });
        }

        let parsed = self.parse(inputs)?;
        let lowered = self.lower(parsed)?;

        let mut codegen = CompileBytecode::new(
            self.options.clone(),
            self.codemap.clone(),
            self.diagnostics.clone(),
        );
        let compiled = codegen.run(lowered)?;

        Ok(AppArtifacts {
            name: self.options.app.name,
            modules: vec![compiled],
            project_info: ProjectInfo::new(&self.options),
        })
    }
}

pub struct ParsedApp {
    pub metadata: Arc<ApplicationMetadata>,
    pub modules: Vec<Artifact<firefly_syntax_erl::Module>>,
}

fn parse(
    pipeline: &mut ParsePipeline,
    app: Symbol,
    files: Vec<FileName>,
) -> anyhow::Result<(Symbol, ParsedApp)> {
    // Load all of the inputs
    let mut inputs: Vec<Input> = files.into_par_iter().panic_fuse().map(read_input).reduce(
        || Ok(vec![]),
        |acc, result| match acc {
            err @ Err(_) => err,
            Ok(_) if result.is_err() => result,
            Ok(mut items) => {
                let mut result = result.unwrap();
                items.append(&mut result);
                Ok(items)
            }
        },
    )?;

    let mut app_metadata = ApplicationMetadata {
        name: app,
        modules: BTreeMap::default(),
    };
    let mut modules = Vec::with_capacity(inputs.len());

    for input in inputs.drain(..) {
        match pipeline.run(input) {
            Ok(Artifact {
                input,
                output: module,
                metadata,
            }) => {
                app_metadata.modules.insert(module.name.name, metadata);
                modules.push(Artifact {
                    input,
                    output: module,
                    metadata: (),
                });
            }
            Err(err) => return Err(err),
        }
    }

    let result = ParsedApp {
        metadata: Arc::new(app_metadata),
        modules,
    };

    Ok((app, result))
}

#[cfg(feature = "native-compilation")]
impl Compiler {
    pub fn lower(
        &self,
        app: Symbol,
        mut parsed: ParsedApp,
    ) -> Result<CodegenResults, ErrorReported> {
        let project_info = if app == self.options.app.name {
            ProjectInfo::new(&self.options)
        } else {
            ProjectInfo::default()
        };

        if parsed.modules.is_empty() {
            return Ok(CodegenResults {
                app_name: app,
                modules: vec![],
                project_info,
            });
        }

        let mut results = CodegenResults {
            app_name: app,
            modules: Vec::with_capacity(parsed.modules.len()),
            project_info,
        };

        let mut pipeline = NativeCompilerPipeline::new();
        for module in parsed.modules.drain(..) {
            if let Some(module) = pipeline.run(module)? {
                results.modules.push(module);
            }
        }

        Ok(results)
    }
}

#[cfg(feature = "native-compilation")]
pub(super) fn mlir_context(
    options: &Options,
    diagnostics: Arc<DiagnosticsHandler>,
) -> Arc<firefly_mlir::OwnedContext> {
    use firefly_mlir::OwnedContext;

    Arc::new(OwnedContext::new(options, diagnostics))
}

#[allow(unused)]
pub fn maybe_emit_file_with_callback_and_opts<F>(
    options: &Options,
    input: &Input,
    output_type: OutputType,
    callback: F,
) -> anyhow::Result<Option<PathBuf>>
where
    F: FnOnce(&mut std::fs::File) -> anyhow::Result<()>,
{
    if let Some(filename) = options.maybe_emit(input, output_type) {
        debug!("emitting {} for {:?}", output_type, input);
        Ok(Some(emit_file_with_callback(filename, callback)?))
    } else {
        Ok(None)
    }
}

pub fn emit_file_with_callback<F>(outfile: PathBuf, callback: F) -> anyhow::Result<PathBuf>
where
    F: FnOnce(&mut std::fs::File) -> anyhow::Result<()>,
{
    use std::fs::{self, File};

    outfile
        .parent()
        .with_context(|| format!("{} does not have a parent directory", outfile.display()))
        .and_then(|outdir| {
            fs::create_dir_all(outdir).with_context(|| {
                format!(
                    "Could not create parent directories ({}) of file ({})",
                    outdir.display(),
                    outfile.display()
                )
            })
        })
        .and_then(|()| {
            File::create(outfile.as_path())
                .with_context(|| format!("Could not create file ({})", outfile.display()))
        })
        .and_then(|mut f| callback(&mut f))
        .map(|_| outfile)
}

fn read_input(filename: FileName) -> anyhow::Result<Vec<Input>> {
    use std::io::{self, Read};

    // We can get three types of input:
    //
    // 1. `stdin` for standard input
    // 2. `path/to/file.erl` for a single file
    // 3. `path/to/dir` for a directory containing Erlang sources
    match filename {
        // Read from standard input
        FileName::Virtual(name) if name == "stdin" => {
            let mut source = String::new();
            io::stdin()
                .read_to_string(&mut source)
                .context("unable to read standard input")?;
            Ok(vec![Input::new(name, source)])
        }
        // Read from a single file
        FileName::Real(path) if path.exists() && path.is_file() => Ok(vec![Input::File(path)]),
        // Load sources from <dir>
        FileName::Real(ref path) if path.exists() && path.is_dir() => find_sources(path),
        // Invalid virtual file
        FileName::Virtual(_) => {
            bail!("invalid input file, expected `-`, a file path, or a directory");
        }
        // Invalid file/directory path
        FileName::Real(ref path) => {
            bail!(
                "invalid input file ({}), not a file or directory",
                path.display()
            );
        }
    }
}

fn find_sources<P>(dir: P) -> anyhow::Result<Vec<Input>>
where
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
            inputs.push(Input::from(entry.path()));
        }
    }

    Ok(inputs)
}
