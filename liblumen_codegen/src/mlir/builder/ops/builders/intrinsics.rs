use super::*;

pub struct IntrinsicBuilder;

impl IntrinsicBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        op: Intrinsic,
    ) -> Result<Option<Value>> {
        match op.name.as_str().get() {
            "print" => {
                Self::build_print(builder, ir_value, op.args.as_slice())
            }
            other => {
                unimplemented!("intrinsic '{:?}'", op.name);
            }
        }
    }

    fn build_print<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        args: &[ir::Value],
    ) -> Result<Option<Value>> {
        let mut argv = Vec::with_capacity(args.len());
        for arg in args.iter().copied() {
            let v = builder.build_value(arg)?;
            argv.push(builder.value_ref(v));
        }

        let result_ref = unsafe {
            MLIRBuildPrintOp(
                builder.as_ref(),
                argv.as_ptr(),
                argv.len() as libc::c_uint
            )
        };

        let result = builder.new_value(ir_value, result_ref, ValueDef::Result(0));
        Ok(Some(result))
    }
}
