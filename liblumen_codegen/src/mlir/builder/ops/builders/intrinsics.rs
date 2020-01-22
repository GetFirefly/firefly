use super::*;

pub struct IntrinsicBuilder;

impl IntrinsicBuilder {
    pub fn build<'f, 'o>(
        _builder: &mut ScopedFunctionBuilder<'f, 'o>,
        _ir_value: Option<ir::Value>,
        op: Intrinsic,
    ) -> Result<Option<Value>> {
        todo!("intrinsic '{:?}", op.name);
    }
}
