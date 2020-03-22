use super::*;

pub struct ConsBuilder;

impl ConsBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        op: Cons,
    ) -> Result<Option<Value>> {
        let head_ref = builder.value_ref(op.head);
        let tail_ref = builder.value_ref(op.tail);

        let cons_ref = unsafe { MLIRCons(builder.as_ref(), op.loc, head_ref, tail_ref) };
        assert!(!cons_ref.is_null());

        let cons = builder.new_value(ir_value, cons_ref, ValueDef::Result(0));
        Ok(Some(cons))
    }
}
