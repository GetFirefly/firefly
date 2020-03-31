use super::*;

/// Handles the trace_capture operation
///
/// This operation is a terminator, it takes a destination block
/// and block arguments, and branches to that block while additionally
/// passing along a handle that represents the captured trace.
pub struct TraceCaptureBuilder;
impl TraceCaptureBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        op: TraceCapture,
    ) -> Result<Option<Value>> {
        let Branch { block, args } = op.dest;

        let block_ref = builder.block_ref(block);
        let block_args = args
            .iter()
            .copied()
            .map(|a| builder.value_ref(a))
            .collect::<Vec<_>>();

        // Append captured trace as the only explicit argument
        let builder_ref = builder.as_ref();

        unsafe {
            MLIRBuildTraceCaptureOp(
                builder_ref,
                op.loc,
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
        op: TraceConstruct,
    ) -> Result<Option<Value>> {
        let capture_ref = builder.value_ref(op.capture);

        let trace_ref = unsafe { MLIRBuildTraceConstructOp(builder.as_ref(), op.loc, capture_ref) };
        if trace_ref.is_null() {
            return Err(anyhow!("failed to build trace_construct operation"));
        }

        let trace = builder.new_value(ir_value, trace_ref, ValueDef::Result(0));
        Ok(Some(trace))
    }
}
