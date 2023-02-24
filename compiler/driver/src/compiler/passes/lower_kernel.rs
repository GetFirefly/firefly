use std::sync::Arc;

use anyhow::bail;

use firefly_diagnostics::CodeMap;
use firefly_pass::Pass;
use firefly_session::Options;
use firefly_util::diagnostics::DiagnosticsHandler;

use crate::compiler::Artifact;

pub struct LowerKernel<'p> {
    pub options: &'p Options,
    pub codemap: Arc<CodeMap>,
    pub diagnostics: Arc<DiagnosticsHandler>,
}
impl<'p> Pass for LowerKernel<'p> {
    type Input<'a> = Artifact<firefly_syntax_kernel::Module>;
    type Output<'a> = Artifact<firefly_syntax_ssa::Module>;

    fn run<'a>(&mut self, input: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        use firefly_syntax_kernel::passes::KernelToSsa;

        let Artifact {
            input,
            output: kernel,
            ..
        } = input;

        // Run lowering passes
        let mut passes = KernelToSsa::new(self.diagnostics.clone());
        match passes.run(kernel) {
            Ok(output) => {
                if self.diagnostics.has_errors() {
                    bail!("lowering to ssa ir failed, see diagnostics for details");
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
