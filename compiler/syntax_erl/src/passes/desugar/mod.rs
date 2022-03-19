mod functions;
mod macros;
mod records;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use liblumen_pass::Pass;

use super::FunctionContext;
use crate::ast;

use self::functions::{ExpandUnqualifiedCalls, NameAnonymousClosures};
use self::macros::ExpandSubstitutions;
use self::records::ExpandRecords;

pub struct DesugarSyntax;
impl Pass for DesugarSyntax {
    type Input<'a> = ast::Module;
    type Output<'a> = ast::Module;

    fn run<'a>(&mut self, mut module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut functions = BTreeMap::new();
        while let Some((key, mut function)) = module.functions.pop_first() {
            let name = function.name;
            let arity = function.arity;
            let context = Rc::new(RefCell::new(FunctionContext::new(name, arity)));

            let mut pipeline = NameAnonymousClosures::new(context.clone())
                .chain(ExpandRecords::new(&module, context.clone()))
                .chain(ExpandUnqualifiedCalls::new(&module))
                .chain(ExpandSubstitutions::new(context.clone()));

            pipeline.run(&mut function)?;

            functions.insert(key, function);
        }

        module.functions = functions;

        Ok(module)
    }
}
