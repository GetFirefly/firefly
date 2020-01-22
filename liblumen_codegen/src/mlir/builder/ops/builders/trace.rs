use super::*;

use crate::mlir::builder::traits::*;

/// Handles the trace_capture operation
///
/// This operation is a terminator, it takes a destination block
/// and block arguments, and branches to that block while additionally
/// passing along a handle that represents the captured trace.
pub struct TraceCaptureBuilder;
impl TraceCaptureBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        branch: Branch,
    ) -> Result<Option<Value>> {
        let Branch { block, args } = branch;

        let block_ref = builder.block_ref(block);
        let mut block_args = args
            .iter()
            .copied()
            .map(|a| builder.value_ref(a))
            .collect::<Vec<_>>();

        // Append captured trace as the only explicit argument
        let builder_ref = builder.as_ref();

        let nil = ir::AtomicTerm::Nil.as_value_ref(builder_ref, builder.options())?;
        block_args.push(nil);
        unsafe {
            MLIRBuildTraceCaptureOp(
                builder_ref,
                block_ref,
                block_args.as_ptr(),
                block_args.len() as libc::c_uint,
            );
        }
        Ok(None)
    }
}

/// Handles the trace_construct operation
///
/// This operation returns the captured trace object as a value.
/// It receives the trace handle as its sole argument.
pub struct TraceConstructBuilder;
impl TraceConstructBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        capture: Value,
    ) -> Result<Option<Value>> {
        let capture_ref = builder.value_ref(capture);

        let trace_ref = unsafe { MLIRBuildTraceConstructOp(builder.as_ref(), capture_ref) };
        if trace_ref.is_null() {
            return Err(anyhow!("failed to build trace_construct operation"));
        }

        let trace = builder.new_value(ir_value, trace_ref, ValueDef::Result(0));
        Ok(Some(trace))
    }
}
