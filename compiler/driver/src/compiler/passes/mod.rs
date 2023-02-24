mod bytecode;
mod lower_ast;
mod lower_core;
mod lower_kernel;
mod parse;
#[cfg(feature = "native-compilation")]
mod ssa_to_mlir;

pub use self::bytecode::CompileBytecode;
pub use self::lower_ast::LowerAst;
pub use self::lower_core::LowerCore;
pub use self::lower_kernel::LowerKernel;
pub use self::parse::ParsePipeline;
#[cfg(feature = "native-compilation")]
pub use self::ssa_to_mlir::SsaToMlir;

use std::sync::Arc;

use firefly_diagnostics::CodeMap;
use firefly_pass::Pass;
use firefly_session::Options;
use firefly_syntax_base::ApplicationMetadata;
use firefly_util::diagnostics::DiagnosticsHandler;

use super::Artifact;

/// This is a pipeline/pass which performs the transformations which compose the middle-tier of the compiler.
///
/// It's input is an Erlang AST module, and it's output is an SSA IR module, the last stage prior to code generation.
#[derive(Clone)]
pub struct PreCodegenPipeline {
    pub options: Arc<Options>,
    pub codemap: Arc<CodeMap>,
    pub diagnostics: Arc<DiagnosticsHandler>,
}
impl PreCodegenPipeline {
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
}
impl Pass for PreCodegenPipeline {
    type Input<'a> = (
        Arc<ApplicationMetadata>,
        Artifact<firefly_syntax_erl::Module>,
    );
    type Output<'a> = Artifact<firefly_syntax_ssa::Module>;

    fn run<'a>(&mut self, input: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let (app, artifact) = input;

        let mut pipeline = LowerAst {
            options: &self.options,
            codemap: self.codemap.clone(),
            diagnostics: self.diagnostics.clone(),
            app,
        }
        .chain(LowerCore {
            options: &self.options,
            codemap: self.codemap.clone(),
            diagnostics: self.diagnostics.clone(),
        })
        .chain(LowerKernel {
            options: &self.options,
            codemap: self.codemap.clone(),
            diagnostics: self.diagnostics.clone(),
        });

        pipeline.run(artifact)
    }
}

/// This is a pipeline/pass which performs all transformations in the middle-tier of the compiler,
/// as well as code generation, for the native compilation backend.
///
/// It's input is an Erlang AST module, and it's output is a set of code generation artifacts, most
/// importantly consisting of the object file generated from the input module.
#[cfg(feature = "native-compilation")]
pub struct NativeCompilerPipeline<'p> {
    diagnostics: &'p DiagnosticsHandler,
}
#[cfg(feature = "native-compilation")]
impl<'p> NativeCompilerPipeline<'p> {
    pub fn new(diagnostics: &'p DiagnosticsHandler) -> Self {
        Self { diagnostics }
    }
}
#[cfg(feature = "native-compilation")]
impl<'p> Pass for NativeCompilerPipeline<'p> {
    type Input<'a> = firefly_syntax_erl::Module;
    type Output<'a> = firefly_codegen::meta::CompiledModule;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut pipeline = PreCodegenPipeline::new(self.diagnostics).chain(SsaToMlir::new());

        pipeline.run(module)
    }
}

#[cfg(feature = "native-compilation")]
struct LowerSsaToMlir<'a> {
    parser: &'a Parser,
    app: Arc<ApplicationMetadata>,
}
#[cfg(feature = "native-compilation")]
impl<'p> Pass for LowerSsaToMlir<'p> {
    type Input<'a> = syntax_ssa::Module;
    type Output<'a> = firefly_mlir::OwnedModule;

    fn run<'a>(&mut self, input: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
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
}
