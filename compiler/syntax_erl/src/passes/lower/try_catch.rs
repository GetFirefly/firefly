use liblumen_syntax_core::*;

use crate::ast;

use super::*;

impl<'m> LowerFunctionToCore<'m> {
    pub(super) fn lower_try<'a>(
        &mut self,
        _builder: &'a mut IrBuilder,
        _try_expr: ast::Try,
    ) -> anyhow::Result<Value> {
        todo!()
    }
}
