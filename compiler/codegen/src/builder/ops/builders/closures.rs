use std::ffi::CString;

use super::*;

use crate::builder::traits::*;

pub struct ClosureBuilder;

impl ClosureBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        target: ir::Block,
    ) -> Result<Option<Value>> {
        // Find all free variables in the target block, and add them to the
        // closure environment. We determine free variables to be any values live
        // at the target block
        let env = builder
            .ir_live_at(target)
            .iter()
            .map(|v| builder.value_ref(builder.get_value(v)))
            .collect::<Vec<_>>();
        let info = builder.block_to_closure_info(target);
        let ident = info.ident;
        let name = CString::new(ident.to_string()).unwrap();

        let builder_ref = builder.as_ref();
        let loc = ir_value
            .map(|v| builder.value_location(v))
            .unwrap_or_else(|| builder.unknown_value_location());
        let module_ref = ident
            .module
            .name
            .as_attribute_ref(loc, builder_ref, builder.options())?;
        let closure = Closure {
            loc,
            module: module_ref,
            name: name.as_ptr(),
            arity: ident.arity as u8,
            index: info.index,
            old_unique: info.old_unique,
            unique: info.unique,
            env: env.as_ptr(),
            env_len: env.len() as libc::c_uint,
        };

        let result_ref = unsafe { MLIRBuildClosure(builder.as_ref(), &closure) };
        let result = builder.new_value(ir_value, result_ref, ValueDef::Result(0));
        Ok(Some(result))
    }
}
