use std::sync::Arc;

use anyhow::bail;

use firefly_diagnostics::CodeMap;
use firefly_pass::Pass;
use firefly_session::Options;
use firefly_syntax_base::ApplicationMetadata;
use firefly_util::diagnostics::DiagnosticsHandler;

use crate::compiler::Artifact;

pub struct LowerAst<'p> {
    pub options: &'p Options,
    pub codemap: Arc<CodeMap>,
    pub diagnostics: Arc<DiagnosticsHandler>,
    pub app: Arc<ApplicationMetadata>,
}
impl<'p> Pass for LowerAst<'p> {
    type Input<'a> = Artifact<firefly_syntax_erl::Module>;
    type Output<'a> = Artifact<firefly_syntax_core::Module>;

    fn run<'a>(&mut self, input: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        use firefly_syntax_erl::passes::{AstToCore, CanonicalizeSyntax, SemanticAnalysis};

        let Artifact {
            input, output: ast, ..
        } = input;

        // Run lowering passes
        let mut passes = SemanticAnalysis::new(&self.diagnostics, &self.app)
            .chain(CanonicalizeSyntax::new(self.codemap.clone()))
            .chain(AstToCore::new(self.diagnostics.clone()));

        match passes.run(ast) {
            Ok(output) => {
                if self.diagnostics.has_errors() {
                    bail!("semantic analysis failed, see diagnostics for details");
                }
                let artifact = Artifact {
                    input,
                    output,
                    metadata: (),
                };

                artifact.maybe_emit_file_with_opts(self.options)?;

                Ok(artifact)
            }
            Err(e) => Err(e),
        }
    }
}
