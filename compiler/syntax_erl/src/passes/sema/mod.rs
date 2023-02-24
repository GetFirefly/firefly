mod attributes;
mod functions;
mod inject;
mod records;
mod verify;

use firefly_intern::Ident;
use firefly_pass::Pass;
use firefly_syntax_base::ApplicationMetadata;
use firefly_util::diagnostics::*;

use crate::ast;

pub use self::attributes::analyze_attribute;
pub use self::functions::analyze_function;
pub use self::records::analyze_record;

/// This pass is responsible for taking a set of top-level forms and
/// analyzing them in the context of a new module to produce a fully
/// constructed and initially validated module.
///
/// Some of the analyses run include:
///
/// * If configured to do so, warns if functions are missing type specs
/// * Warns about type specs for undefined functions
/// * Warns about redefined attributes
/// * Errors on invalid nif declarations
/// * Errors on invalid syntax in built-in attributes (e.g. -import(..))
/// * Errors on mismatched function clauses (name/arity)
/// * Errors on unterminated function clauses
/// * Errors on redefined functions
///
/// And a few other similar lints
pub struct SemanticAnalysis<'p> {
    diagnostics: &'p DiagnosticsHandler,
    app: &'p ApplicationMetadata,
}
impl<'p> SemanticAnalysis<'p> {
    pub fn new(diagnostics: &'p DiagnosticsHandler, app: &'p ApplicationMetadata) -> Self {
        Self { diagnostics, app }
    }
}
impl<'p> Pass for SemanticAnalysis<'p> {
    type Input<'a> = ast::Module;
    type Output<'a> = ast::Module;

    fn run<'a>(&mut self, mut module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut passes = inject::AddAutoImports
            .chain(verify::VerifyExports::new(self.diagnostics))
            .chain(verify::VerifyOnLoadFunctions::new(self.diagnostics))
            .chain(verify::VerifyTypeSpecs::new(self.diagnostics))
            .chain(verify::VerifyNifs::new(self.diagnostics))
            // We place this after VerifyNifs so that we have all the nifs available for module_info,
            // but before VerifyCalls so that any calls to module_info are not erroneously treated as
            // errors prior to them being defined by this pass
            .chain(inject::DefinePseudoLocals)
            .chain(verify::VerifyCalls::new(self.diagnostics, self.app));

        passes.run(&mut module)?;

        Ok(module)
    }
}
