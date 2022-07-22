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
                self.reporter.show_error(
                    "invalid on_load function",
                    &[(span, "this function is not defined in this module")],
                );
            }
        }

        Ok(module)
    }
}

pub struct VerifyNifs {
    reporter: Reporter,
}
impl VerifyNifs {
    pub fn new(reporter: Reporter) -> Self {
        Self { reporter }
    }
}
impl Pass for VerifyNifs {
    type Input<'a> = &'a mut Module;
    type Output<'a> = &'a mut Module;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        // Verify that all of the nif declarations have a definition, and that the corresponding function was marked as such
        for nif in module.nifs.iter() {
            match module.functions.get(nif.as_ref()) {
                None => {
                    let span = nif.span();
                    self.reporter.show_error(
                        "invalid -nif declaration",
                        &[(
                            span,
                            "the referenced function is not defined in this module",
                        )],
                    );
                }
                Some(fun) => {
                    if !fun.is_nif {
                        let span = fun.span;
                        self.reporter.show_error(
                            "misplaced -nif declaration",
                            &[(
                                span,
                                "expected -nif declaration to precede the function it references",
                            )],
                        );
                    }
                }
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
                self.reporter.show_warning(
                    "type spec for undefined function",
                    &[(
                        spec.span,
                        "this type spec has no corresponding function definition",
                    )],
                );
            }
        }
        Ok(module)
    }
}
