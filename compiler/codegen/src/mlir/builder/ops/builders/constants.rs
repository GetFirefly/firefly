use cranelift_entity::EntityList;

use libeir_ir::{AtomTerm, AtomicTerm, Const, ConstKind};

use liblumen_session::Options;

use crate::mlir::builder::ffi::{AttributeRef, ValueRef};
use crate::mlir::builder::traits::*;

use super::*;

pub struct ConstantBuilder;

impl ConstantBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        constant: ir::Const,
    ) -> Result<Option<Value>> {
        let const_kind = builder.const_kind(constant).clone();
        match const_kind {
            ConstKind::Atomic(AtomicTerm::Atom(AtomTerm(symbol))) => {
                let value_ref = symbol.as_value_ref(builder.as_ref(), builder.options())?;
                Self::into_value(builder, constant, ir_value, value_ref)
            }
            ConstKind::Atomic(ref atomic) => {
                let value_ref = atomic.as_value_ref(builder.as_ref(), builder.options())?;
                Self::into_value(builder, constant, ir_value, value_ref)
            }
            ConstKind::ListCell { head, tail } => {
                Self::list(builder, constant, ir_value, head, tail)
            }
            ConstKind::Tuple { ref entries } => Self::tuple(builder, constant, ir_value, entries),
            ConstKind::Map {
                ref keys,
                ref values,
            } => {
                let kvs = {
                    let ks = builder.const_entries(keys);
                    let vs = builder.const_entries(values);
                    ks.iter()
                        .copied()
                        .zip(vs.iter().copied())
                        .collect::<Vec<_>>()
                };
                Self::map(builder, constant, ir_value, kvs.as_slice())
            }
        }
    }

    fn list<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        const_value: Const,
        ir_value: Option<ir::Value>,
        head: Const,
        tail: Const,
    ) -> Result<Option<Value>> {
        debug!(
            "building constant list (head = {:?}, tail = {:?})",
            head, tail
        );
        let mut elements = Vec::with_capacity(2);
        {
            let element = AttributeBuilder::build(builder, head)?;
            elements.push(element);
        }
        let mut next = tail;
        loop {
            match builder.const_kind(next).clone() {
                ConstKind::ListCell { head: h, tail: t } => {
                    let element = AttributeBuilder::build(builder, h)?;
                    elements.push(element);
                    next = t;
                    continue;
                }
                other => {
                    debug!("tail is ({:?})", other);
                    let element = AttributeBuilder::build(builder, next)?;
                    elements.push(element);
                    break;
                }
            }
        }
        let list = ConstList(elements);
        let value_ref = list.as_value_ref(builder.as_ref(), builder.options())?;
        Self::into_value(builder, const_value, ir_value, value_ref)
    }

    fn tuple<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        const_value: Const,
        ir_value: Option<ir::Value>,
        const_elements: &EntityList<Const>,
    ) -> Result<Option<Value>> {
        let mut const_elements = builder.const_entries(&const_elements).to_vec();
        debug!("constant type is tuple (elements = {:#?})", &const_elements);
        let mut elements = Vec::with_capacity(const_elements.len());
        for c in const_elements.drain(..) {
            let element = AttributeBuilder::build(builder, c)?;
            elements.push(element);
        }
        let tuple = ConstTuple(elements);
        let value_ref = tuple.as_value_ref(builder.as_ref(), builder.options())?;
        Self::into_value(builder, const_value, ir_value, value_ref)
    }

    fn map<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        const_value: Const,
        ir_value: Option<ir::Value>,
        const_pairs: &[(Const, Const)],
    ) -> Result<Option<Value>> {
        debug!("constant type is map (values = {:#?})", const_pairs);
        let mut pairs = Vec::with_capacity(const_pairs.len());
        for (k, v) in const_pairs {
            let key = AttributeBuilder::build(builder, *k)?;
            let value = AttributeBuilder::build(builder, *v)?;
            pairs.push(KeyValuePair { key, value });
        }
        let map = ConstMap(pairs);
        let value_ref = map.as_value_ref(builder.as_ref(), builder.options())?;
        Self::into_value(builder, const_value, ir_value, value_ref)
    }

    fn into_value<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        const_value: Const,
        ir_value: Option<ir::Value>,
        value_ref: ValueRef,
    ) -> Result<Option<Value>> {
        use crate::mlir::builder::ops::OpKind;

        let value_def = ValueDef::Result(0);
        Ok(Some(builder.new_value(ir_value, value_ref, value_def)))
    }
}

pub struct AttributeBuilder;

impl AttributeBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        constant: Const,
    ) -> Result<AttributeRef> {
        match builder.const_kind(constant).clone() {
            ConstKind::Atomic(AtomicTerm::Atom(AtomTerm(symbol))) => {
                symbol.as_attribute_ref(builder.as_ref(), builder.options())
            }
            ConstKind::Atomic(ref atomic) => {
                atomic.as_attribute_ref(builder.as_ref(), builder.options())
            }
            ConstKind::ListCell { head, tail } => Self::list(builder, head, tail),
            ConstKind::Tuple { ref entries } => Self::tuple(builder, entries),
            ConstKind::Map {
                ref keys,
                ref values,
            } => {
                let kvs = {
                    let ks = builder.const_entries(keys);
                    let vs = builder.const_entries(values);
                    ks.iter()
                        .copied()
                        .zip(vs.iter().copied())
                        .collect::<Vec<_>>()
                };
                Self::map(builder, kvs.as_slice())
            }
        }
    }

    fn list<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        head: Const,
        tail: Const,
    ) -> Result<AttributeRef> {
        let mut elements = Vec::with_capacity(2);
        let head_ref = AttributeBuilder::build(builder, head)?;
        elements.push(head_ref);
        let mut next = tail;
        loop {
            match builder.const_kind(next).clone() {
                ConstKind::ListCell { head: h, tail: t } => {
                    let element = AttributeBuilder::build(builder, h)?;
                    elements.push(element);
                    next = t;
                    continue;
                }
                _other => {
                    let element = AttributeBuilder::build(builder, next)?;
                    elements.push(element);
                    break;
                }
            }
        }
        let list = ConstList(elements);
        list.as_attribute_ref(builder.as_ref(), builder.options())
    }

    fn tuple<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        const_elements: &EntityList<Const>,
    ) -> Result<AttributeRef> {
        let mut const_elements = builder.const_entries(const_elements).to_vec();
        let mut elements = Vec::with_capacity(const_elements.len());
        for c in const_elements.drain(..) {
            let element = AttributeBuilder::build(builder, c)?;
            elements.push(element);
        }
        let tuple = ConstTuple(elements);
        tuple.as_attribute_ref(builder.as_ref(), builder.options())
    }

    fn map<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        const_pairs: &[(Const, Const)],
    ) -> Result<AttributeRef> {
        let mut pairs = Vec::with_capacity(const_pairs.len());
        for (k, v) in const_pairs {
            let key = AttributeBuilder::build(builder, *k)?;
            let value = AttributeBuilder::build(builder, *v)?;
            pairs.push(KeyValuePair { key, value });
        }
        let map = ConstMap(pairs);
        map.as_attribute_ref(builder.as_ref(), builder.options())
    }
}
