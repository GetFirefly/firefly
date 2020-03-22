use std::ffi::{CStr, CString};

use super::*;

pub struct IntrinsicBuilder;

impl IntrinsicBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        op: Intrinsic,
    ) -> Result<Option<Value>> {
        let name = CString::new(op.name.as_str().get()).unwrap();
        if is_intrinsic(name.as_c_str()) {
            let mut argv = Vec::with_capacity(op.args.len());
            for arg in op.args.iter().copied() {
                let v = builder.build_value(arg)?;
                argv.push(builder.value_ref(v));
            }

            let result_ref = unsafe {
                MLIRBuildIntrinsic(
                    builder.as_ref(),
                    op.loc,
                    name.as_ptr(),
                    argv.as_ptr(),
                    argv.len() as libc::c_uint,
                )
            };

            let result = builder.new_value(ir_value, result_ref, ValueDef::Result(0));
            Ok(Some(result))
        } else {
            unimplemented!("unknown intrinsic '{:?}'", op.name);
        }
    }
}

#[inline]
fn is_intrinsic(name: &CStr) -> bool {
    unsafe { MLIRIsIntrinsic(name.as_ptr()) }
}
