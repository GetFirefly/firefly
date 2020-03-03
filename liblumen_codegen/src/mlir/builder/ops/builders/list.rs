use super::*;

pub struct ConsBuilder;

impl ConsBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        head: Value,
        tail: Value,
    ) -> Result<Option<Value>> {
        let head_ref = builder.value_ref(head);
        let tail_ref = builder.value_ref(tail);

        let cons_ref = unsafe { MLIRCons(builder.as_ref(), head_ref, tail_ref) };
        assert!(!cons_ref.is_null());

        let cons = builder.new_value(ir_value, cons_ref, ValueDef::Result(0));
        Ok(Some(cons))
    }
}
