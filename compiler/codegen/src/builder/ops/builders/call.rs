use std::ffi::CString;

use super::*;

pub struct CallBuilder;

impl CallBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        _ir_value: Option<ir::Value>,
        op: Call,
    ) -> Result<Option<Value>> {
        // Construct arguments
        let args = op
            .args
            .iter()
            .copied()
            .map(|v| builder.value_ref(v))
            .collect::<Vec<_>>();
        builder.debug(&format!("call args: {:?}", args.as_slice()));

        let mut ok_args = Vec::new();
        let ok_block = match op.ok {
            CallSuccess::Branch(Branch { block, args }) => {
                for arg in args.iter().copied() {
                    ok_args.push(builder.value_ref(arg));
                }
                builder.block_ref(block)
            }
            _ => Default::default(),
        };
        builder.debug(&format!("call ok args: {:?}", ok_args.as_slice()));

        let mut err_args = Vec::new();
        let err_block = match op.err {
            CallError::Branch(Branch { block, args }) => {
                for arg in args.iter().copied() {
                    err_args.push(builder.value_ref(arg));
                }
                builder.block_ref(block)
            }
            _ => Default::default(),
        };
        builder.debug(&format!("call err args: {:?}", err_args.as_slice()));

        match op.callee {
            Callee::ClosureDynamic(closure) => {
                builder.debug("call target is closure");

                let closure_ref = builder.value_ref(closure);
                unsafe {
                    MLIRBuildClosureCall(
                        builder.as_ref(),
                        op.loc,
                        closure_ref,
                        args.as_ptr(),
                        args.len() as libc::c_uint,
                        op.is_tail,
                        ok_block,
                        ok_args.as_ptr(),
                        ok_args.len() as libc::c_uint,
                        err_block,
                        err_args.as_ptr(),
                        err_args.len() as libc::c_uint,
                    );
                }

                Ok(None)
            }
            Callee::Static(ref ident) => {
                builder.debug(&format!("static call target is {}", ident));

                let name = CString::new(ident.to_string()).unwrap();
                unsafe {
                    MLIRBuildStaticCall(
                        builder.as_ref(),
                        op.loc,
                        name.as_ptr(),
                        args.as_ptr(),
                        args.len() as libc::c_uint,
                        op.is_tail,
                        ok_block,
                        ok_args.as_ptr(),
                        ok_args.len() as libc::c_uint,
                        err_block,
                        err_args.as_ptr(),
                        err_args.len() as libc::c_uint,
                    );
                }

                Ok(None)
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
        op: FunctionRef,
    ) -> Result<Option<Value>> {
        todo!(
            "build function reference constant for {:?} = {:?}",
            ir_value,
            &op.callee
        );
    }
}
