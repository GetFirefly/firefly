use super::*;

pub struct MapBuilder;

impl MapBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        op: Map,
    ) -> Result<Option<Value>> {
        let pairs = op
            .elements
            .iter()
            .map(|(k, v)| MapEntry {
                key: builder.value_ref(*k),
                value: builder.value_ref(*v),
            })
            .collect::<Vec<_>>();

        let map_ref = unsafe {
            MLIRConstructMap(
                builder.as_ref(),
                op.loc,
                pairs.as_ptr(),
                pairs.len() as libc::c_uint,
            )
        };
        assert!(!map_ref.is_null());

        let map = builder.new_value(ir_value, map_ref, ValueDef::Result(0));
        Ok(Some(map))
    }
}

pub struct MapPutBuilder;

impl MapPutBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        mut op: MapPuts,
    ) -> Result<Option<Value>> {
        let actions = op
            .puts
            .drain(..)
            .map(|put| MapAction {
                action: put.action,
                key: builder.value_ref(put.key),
                value: builder.value_ref(put.value),
            })
            .collect::<Vec<_>>();

        let update = MapUpdate {
            loc: op.loc,
            map: builder.value_ref(op.map),
            ok: builder.block_ref(op.ok),
            err: builder.block_ref(op.err),
            actionsv: actions.as_ptr(),
            actionsc: actions.len(),
        };

        unsafe {
            MLIRBuildMapOp(builder.as_ref(), update);
        }

        Ok(None)
    }
}
