use liblumen_syntax_core::*;

use crate::ast;

use super::*;

impl<'m> LowerFunctionToCore<'m> {
    pub(super) fn lower_fun<'a>(
        &mut self,
        _builder: &'a mut IrBuilder,
        _fun: ast::Fun,
    ) -> anyhow::Result<Value> {
        todo!()
    }
}
