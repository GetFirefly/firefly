use std::collections::HashMap;

use failure::Error;

use liblumen_syntax::ast::Module;
use liblumen_syntax::Symbol;

use liblumen_common::compiler::Compiler;

pub fn transform<C>(_compiler: &C, _modules: HashMap<Symbol, Module>) -> Result<(), Error>
where
    C: Compiler,
{
    unimplemented!()
}
