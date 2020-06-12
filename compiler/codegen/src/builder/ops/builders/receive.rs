use super::*;

pub struct ReceiveStartBuilder;

impl ReceiveStartBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        op: ReceiveStart,
    ) -> Result<Option<Value>> {
        let cont = builder.block_ref(op.cont);
        let timeout = builder.value_ref(op.timeout);

        unsafe {
            MLIRBuildReceiveStart(builder.as_ref(), op.loc, cont, timeout);
        }

        Ok(None)
    }
}

pub struct ReceiveWaitBuilder;

impl ReceiveWaitBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        op: ReceiveWait,
    ) -> Result<Option<Value>> {
        let timeout = builder.block_ref(op.timeout);
        let check = builder.block_ref(op.check);
        let receive_ref = builder.value_ref(op.receive_ref);

        unsafe {
            MLIRBuildReceiveWait(builder.as_ref(), op.loc, timeout, check, receive_ref);
        }

        Ok(None)
    }
}

pub struct ReceiveDoneBuilder;

impl ReceiveDoneBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        op: ReceiveDone,
    ) -> Result<Option<Value>> {
        let cont = builder.block_ref(op.cont);
        let receive_ref = builder.value_ref(op.receive_ref);
        let args = op
            .args
            .iter()
            .map(|v| builder.value_ref(*v))
            .collect::<Vec<_>>();

        unsafe {
            MLIRBuildReceiveDone(
                builder.as_ref(),
                op.loc,
                cont,
                receive_ref,
                args.as_ptr(),
                args.len() as libc::c_uint,
            );
        }

        Ok(None)
    }
}
