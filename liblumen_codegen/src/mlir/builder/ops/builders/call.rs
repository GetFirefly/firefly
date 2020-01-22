use std::ffi::CString;

use super::*;

pub struct CallBuilder;

impl CallBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        op: Call,
    ) -> Result<Option<Value>> {
        match op.callee {
            Callee::Static(ref ident) => {
                builder.debug(&format!("static call target is {}", ident));

                let is_local = builder.is_current_module(ident.module.name);
                let name = CString::new(ident.to_string()).unwrap();
                let args = op
                    .args
                    .iter()
                    .copied()
                    .map(|v| builder.value_ref(v))
                    .collect::<Vec<_>>();
                let result = unsafe {
                    MLIRBuildStaticCall(
                        builder.as_ref(),
                        name.as_ptr(),
                        args.as_ptr(),
                        args.len() as libc::c_uint,
                        op.is_tail,
                    )
                };

                if ir_value.is_some() {
                    assert!(!result.is_null());
                    let value = builder.new_value(ir_value, result, ValueDef::Result(0));
                    Ok(Some(value))
                } else {
                    Ok(None)
                }
            }
            callee => todo!("unimplemented call type {:?}", callee),
        }
    }
}

pub struct CalleeBuilder;

impl CalleeBuilder {
    pub fn build<'f, 'o>(
        _builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        callee: Callee,
    ) -> Result<Option<Value>> {
        todo!(
            "build function reference constant for {:?} = {:?}",
            ir_value,
            callee
        );
    }
}
