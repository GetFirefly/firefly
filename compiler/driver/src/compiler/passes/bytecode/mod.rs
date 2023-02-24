mod lower_bytecode;
mod lower_ssa;

use self::lower_bytecode::LowerBytecode;
use self::lower_ssa::LowerSsa;

use std::sync::Arc;

use firefly_diagnostics::CodeMap;
use firefly_pass::Pass;
use firefly_session::Options;
use firefly_util::diagnostics::DiagnosticsHandler;

use crate::compiler::Artifact;

pub struct CompileBytecode {
    options: Arc<Options>,
    codemap: Arc<CodeMap>,
    diagnostics: Arc<DiagnosticsHandler>,
}
impl CompileBytecode {
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
impl Pass for CompileBytecode {
    type Input<'a> = Vec<Artifact<firefly_syntax_ssa::Module>>;
    type Output<'a> = firefly_linker::ModuleArtifacts;

    fn run<'a>(&mut self, modules: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut pipeline = LowerSsa::new(&self.options, &self.diagnostics, &self.codemap).chain(
            LowerBytecode::new(self.options.clone(), self.diagnostics.clone()),
        );

        pipeline.run(modules)
    }
}
