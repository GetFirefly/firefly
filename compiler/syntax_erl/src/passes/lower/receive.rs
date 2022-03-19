use liblumen_syntax_core::*;

use crate::ast;

use super::*;

impl<'m> LowerFunctionToCore<'m> {
    pub(super) fn lower_receive<'a>(
        &mut self,
        _builder: &'a mut IrBuilder,
        _receive: ast::Receive,
    ) -> anyhow::Result<Value> {
        todo!()
    }
}
