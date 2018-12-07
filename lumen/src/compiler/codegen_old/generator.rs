use std::path::Path;

use syntax::ast::ast::ModuleDecl;
use syntax::ast::ast::form::Form;
use syntax::ast::ast::clause::Clause;
use syntax::ast::ast::form::*;

#[macro_use]
use super::*;
use super::context::Context;

pub struct CodeGenerator {
    context: Context,
}
impl CodeGenerator {
    pub fn new() -> Result<CodeGenerator, CodeGenError> {
        Ok(CodeGenerator { context: Context::new()? })
    }

    pub fn module(&self, module: ModuleDecl) -> Result<(), CodeGenError> {
        if let Form::Module(ModuleAttr { line: _line, ref name }) = module.forms[0].clone() {
            self.context.add_module(&name, module);
            Ok(())
        } else {
            Err(CodeGenError::ValidationError("missing -module() attribute".to_string()))
        }
    }

    pub fn emit_file(&self, out: &Path, out_type: OutputType) -> Result<(), CodeGenError> {
        let path = Path::new(out).with_extension(out_type.to_extension());
        self.context.emit_file(out, out_type)
    }
}
