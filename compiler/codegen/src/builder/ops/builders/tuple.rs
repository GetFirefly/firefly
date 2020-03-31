use super::*;

pub struct TupleBuilder;

impl TupleBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        op: Tuple,
    ) -> Result<Option<Value>> {
        let args = op
            .elements
            .iter()
            .copied()
            .map(|v| builder.value_ref(v))
            .collect::<Vec<_>>();

        let tuple_ref = unsafe {
            MLIRConstructTuple(
                builder.as_ref(),
                op.loc,
                args.as_ptr(),
                args.len() as libc::c_uint,
            )
        };
        assert!(!tuple_ref.is_null());

        let tuple = builder.new_value(ir_value, tuple_ref, ValueDef::Result(0));
        Ok(Some(tuple))
    }
}
