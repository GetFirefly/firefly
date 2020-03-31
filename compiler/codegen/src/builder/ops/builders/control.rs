use super::*;

pub struct ReturnBuilder;
impl ReturnBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        op: Return,
    ) -> Result<Option<Value>> {
        debug_in!(builder, "building return");
        let value_ref = op.value.map(|v| builder.value_ref(v)).unwrap_or_default();
        unsafe {
            MLIRBuildReturn(builder.as_ref(), op.loc, value_ref);
        }
        Ok(None)
    }
}

pub struct ThrowBuilder;
impl ThrowBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        op: Throw,
    ) -> Result<Option<Value>> {
        debug_in!(builder, "building throw");
        // TODO: For now we just lower to an abort until we decide on how this should work
        unsafe {
            MLIRBuildUnreachable(builder.as_ref(), op.loc);
        }

        Ok(None)
    }
}

pub struct UnreachableBuilder;
impl UnreachableBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        loc: LocationRef,
    ) -> Result<Option<Value>> {
        debug_in!(builder, "building unreachable");
        unsafe {
            MLIRBuildUnreachable(builder.as_ref(), loc);
        }

        Ok(None)
    }
}

pub struct BranchBuilder;
impl BranchBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        op: Br,
    ) -> Result<Option<Value>> {
        debug_in!(builder, "building branch");

        let Branch { block, args } = op.dest;
        let block_ref = builder.block_ref(block);
        let block_argc = args.len();
        let block_args = args
            .iter()
            .copied()
            .map(|a| builder.value_ref(a))
            .collect::<Vec<_>>();
        let block_argv = if block_argc > 0 {
            block_args.as_ptr() as *mut _
        } else {
            core::ptr::null_mut()
        };
        unsafe {
            MLIRBuildBr(
                builder.as_ref(),
                op.loc,
                block_ref,
                block_argv,
                block_argc as libc::c_uint,
            );
        }

        Ok(None)
    }
}

pub struct IfBuilder;
impl IfBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        op: If,
    ) -> Result<Option<Value>> {
        debug_in!(builder, "building if");

        let cond_ref = builder.value_ref(op.cond);
        let yes_ref = builder.block_ref(op.yes.block);
        let no_ref = builder.block_ref(op.no.block);
        let other_ref = op.otherwise.as_ref().map(|o| builder.block_ref(o.block));

        let yes_args = op
            .yes
            .args
            .iter()
            .map(|v| builder.value_ref(*v))
            .collect::<Vec<_>>();
        let no_args = op
            .no
            .args
            .iter()
            .map(|v| builder.value_ref(*v))
            .collect::<Vec<_>>();
        let other_args = op
            .otherwise
            .as_ref()
            .map(|o| {
                o.args
                    .iter()
                    .map(|v| builder.value_ref(*v))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(Vec::new);
        // 4. Generate if using mapped values
        unsafe {
            MLIRBuildIf(
                builder.as_ref(),
                op.loc,
                cond_ref,
                yes_ref,
                yes_args.as_ptr(),
                yes_args.len() as libc::c_uint,
                no_ref,
                no_args.as_ptr(),
                no_args.len() as libc::c_uint,
                other_ref.unwrap_or_default(),
                other_args.as_ptr(),
                other_args.len() as libc::c_uint,
            )
        }
        Ok(None)
    }
}
