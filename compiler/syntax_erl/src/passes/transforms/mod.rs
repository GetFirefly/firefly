mod expand_records;
mod expand_substitutions;
mod expand_unqualified_calls;

use std::collections::BTreeMap;
use std::sync::Arc;

use firefly_diagnostics::CodeMap;
use firefly_pass::Pass;

use crate::ast;

use self::expand_records::ExpandRecords;
use self::expand_substitutions::ExpandSubstitutions;
use self::expand_unqualified_calls::ExpandUnqualifiedCalls;

pub struct CanonicalizeSyntax {
    codemap: Arc<CodeMap>,
}
impl CanonicalizeSyntax {
    pub fn new(codemap: Arc<CodeMap>) -> Self {
        Self { codemap }
    }
}
impl Pass for CanonicalizeSyntax {
    type Input<'a> = ast::Module;
    type Output<'a> = ast::Module;

    fn run<'a>(&mut self, mut module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut functions = BTreeMap::new();
        while let Some((key, mut function)) = module.functions.pop_first() {
            // Prepare function for translation to CST
            let mut pipeline = ExpandRecords::new(&module)
                .chain(ExpandUnqualifiedCalls::new(&module))
                .chain(ExpandSubstitutions::new(module.name, &self.codemap));
            pipeline.run(&mut function)?;

            functions.insert(key, function);
        }

        module.functions = functions;

        Ok(module)
    }
}
