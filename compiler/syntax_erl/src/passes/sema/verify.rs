use liblumen_diagnostics::*;
use liblumen_pass::Pass;

use crate::ast::*;

pub struct VerifyOnLoadFunctions {
    reporter: Reporter,
}
impl VerifyOnLoadFunctions {
    pub fn new(reporter: Reporter) -> Self {
        Self { reporter }
    }
}
impl Pass for VerifyOnLoadFunctions {
    type Input<'a> = &'a mut Module;
    type Output<'a> = &'a mut Module;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        // Verify on_load function exists
        if let Some(on_load_name) = module.on_load.as_ref() {
            if !module.functions.contains_key(on_load_name.as_ref()) {
                let span = on_load_name.span();
                self.reporter.diagnostic(
                    Diagnostic::error()
                        .with_message("invalid on_load function")
                        .with_labels(vec![Label::primary(span.source_id(), span)
                            .with_message("this function is not defined in this module")]),
                );
            }
        }

        Ok(module)
    }
}

pub struct VerifyTypeSpecs {
    reporter: Reporter,
}
impl VerifyTypeSpecs {
    pub fn new(reporter: Reporter) -> Self {
        Self { reporter }
    }
}
impl Pass for VerifyTypeSpecs {
    type Input<'a> = &'a mut Module;
    type Output<'a> = &'a mut Module;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        // Check for orphaned type specs
        for (spec_name, spec) in module.specs.iter() {
            let local_spec_name = spec_name.to_local();
            if !module.functions.contains_key(&local_spec_name) {
                self.reporter.diagnostic(
                    Diagnostic::warning()
                        .with_message("type spec for undefined function")
                        .with_labels(vec![Label::primary(spec.span.source_id(), spec.span)
                            .with_message(
                                "this type spec has no corresponding function definition",
                            )]),
                );
            }
        }
        Ok(module)
    }
}
