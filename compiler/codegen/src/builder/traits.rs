use std::ffi::CString;

use anyhow::anyhow;

use num_bigint::BigInt;

use libeir_intern::Symbol;
use libeir_ir::{AtomTerm, AtomicTerm, BigIntTerm, BinaryTerm, FloatTerm, IntTerm};

use liblumen_alloc::erts::term::prelude::{BinaryFlags, BinaryLiteral};
use liblumen_mlir::ir::{AttributeRef, LocationRef, ValueRef};
use liblumen_session::Options;

use super::ffi::*;
use crate::Result;

macro_rules! unwrap {
    ($x:expr, $fmt:expr $(,$args:expr)*) => {{
        let unwrapped = $x;
        if unwrapped.is_null() {
            Err(anyhow!($fmt $(,$args)*))
        } else {
            Ok(unwrapped)
        }
    }}
}

// Represents a cons cell/list of constant values
#[derive(Debug)]
pub(super) struct ConstList(pub Vec<AttributeRef>);

// Represents a tuple of constant values
#[derive(Debug)]
pub(super) struct ConstTuple(pub Vec<AttributeRef>);

// Represents a map of constant key/value pairs
#[derive(Debug)]
pub(super) struct ConstMap(pub Vec<KeyValuePair>);

// This trait represents the conversion of some value to an MLIR value
pub trait AsValueRef {
    fn as_value_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        options: &Options,
    ) -> Result<ValueRef>;
}

impl AsValueRef for AtomicTerm {
    fn as_value_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        options: &Options,
    ) -> Result<ValueRef> {
        match self {
            AtomicTerm::Int(ref i) => i.as_value_ref(loc, builder, options),
            AtomicTerm::BigInt(ref i) => i.as_value_ref(loc, builder, options),
            AtomicTerm::Float(ref f) => f.as_value_ref(loc, builder, options),
            AtomicTerm::Atom(ref a) => a.as_value_ref(loc, builder, options),
            AtomicTerm::Binary(ref b) => b.as_value_ref(loc, builder, options),
            AtomicTerm::Nil => {
                let val = unsafe { MLIRBuildConstantNil(builder, loc) };
                unwrap!(val, "failed to construct constant nil")
            }
        }
    }
}
impl AsValueRef for ConstList {
    fn as_value_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<ValueRef> {
        let val = unsafe {
            let len = self.0.len() as libc::c_uint;
            MLIRBuildConstantList(builder, loc, self.0.as_ptr(), len)
        };
        unwrap!(val, "failed to construct constant list")
    }
}
impl AsValueRef for ConstTuple {
    fn as_value_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<ValueRef> {
        let val = unsafe {
            let len = self.0.len() as libc::c_uint;
            MLIRBuildConstantTuple(builder, loc, self.0.as_ptr(), len)
        };
        unwrap!(val, "failed to construct constant tuple")
    }
}
impl AsValueRef for ConstMap {
    fn as_value_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<ValueRef> {
        let val = unsafe {
            MLIRBuildConstantMap(builder, loc, self.0.as_ptr(), self.0.len() as libc::c_uint)
        };
        unwrap!(val, "failed to construct constant map")
    }
}
impl AsValueRef for IntTerm {
    #[inline]
    fn as_value_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        options: &Options,
    ) -> Result<ValueRef> {
        self.value().as_value_ref(loc, builder, options)
    }
}
impl AsValueRef for i64 {
    fn as_value_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<ValueRef> {
        let val = unsafe { MLIRBuildConstantInt(builder, loc, *self) };
        unwrap!(val, "failed to construct constant integer ({})", self)
    }
}
impl AsValueRef for BigIntTerm {
    fn as_value_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<ValueRef> {
        // Redundant, but needed, since libeir_ir is referencing a different num_bigint than us
        let bi = self.value();
        let width = bi.bits();
        let s = CString::new(bi.to_str_radix(10)).unwrap();
        let val =
            unsafe { MLIRBuildConstantBigInt(builder, loc, s.as_ptr(), width as libc::c_uint) };
        unwrap!(val, "failed to construct constant bigint ({})", bi)
    }
}
impl AsValueRef for BigInt {
    fn as_value_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<ValueRef> {
        let width = self.bits();
        let s = CString::new(self.to_str_radix(10)).unwrap();
        let val =
            unsafe { MLIRBuildConstantBigInt(builder, loc, s.as_ptr(), width as libc::c_uint) };
        unwrap!(val, "failed to construct constant bigint ({})", self)
    }
}
impl AsValueRef for FloatTerm {
    #[inline]
    fn as_value_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        options: &Options,
    ) -> Result<ValueRef> {
        self.value().as_value_ref(loc, builder, options)
    }
}
impl AsValueRef for f64 {
    fn as_value_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<ValueRef> {
        let val = unsafe { MLIRBuildConstantFloat(builder, loc, *self) };
        unwrap!(val, "failed to construct constant float ({})", self)
    }
}
impl AsValueRef for AtomTerm {
    #[inline]
    fn as_value_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        options: &Options,
    ) -> Result<ValueRef> {
        self.0.as_value_ref(loc, builder, options)
    }
}
impl AsValueRef for Symbol {
    fn as_value_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<ValueRef> {
        let i = self.as_usize() as u64;
        let s = CString::new(self.as_str().get()).unwrap();
        let val = unsafe { MLIRBuildConstantAtom(builder, loc, s.as_ptr(), i) };
        unwrap!(val, "failed to construct constant atom ({})", self)
    }
}
impl AsValueRef for BinaryTerm {
    fn as_value_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        options: &Options,
    ) -> Result<ValueRef> {
        use liblumen_term::*;

        let slice = self.value();
        let encoding_type = options.target.options.encoding;
        let pointer_width = options.target.target_pointer_width;
        let (header, flags) = match encoding_type {
            EncodingType::Encoding32 => BinaryLiteral::make_parts_from_slice::<Encoding32>(slice),
            EncodingType::Encoding64 => BinaryLiteral::make_parts_from_slice::<Encoding64>(slice),
            EncodingType::Encoding64Nanboxed => {
                BinaryLiteral::make_parts_from_slice::<Encoding64Nanboxed>(slice)
            }
            EncodingType::Default if pointer_width == 32 => {
                BinaryLiteral::make_parts_from_slice::<Encoding32>(slice)
            }
            EncodingType::Default if pointer_width == 64 => {
                BinaryLiteral::make_parts_from_slice::<Encoding64>(slice)
            }
            EncodingType::Default => unreachable!(),
        };

        let ptr = slice.as_ptr() as *const libc::c_char;
        let len = slice.len() as libc::c_uint;
        let val = unsafe { MLIRBuildConstantBinary(builder, loc, ptr, len, header, flags) };
        if val.is_null() {
            let flags = unsafe {
                if pointer_width == 32 {
                    BinaryFlags::from_u32(flags as u32)
                } else {
                    BinaryFlags::from_u64(flags)
                }
            };
            Err(anyhow!(
                "failed to construct constant binary (flags = {:#?})",
                flags
            ))
        } else {
            Ok(val)
        }
    }
}

// This trait represents the conversion of some value to an MLIR attribute value
pub trait AsAttributeRef {
    fn as_attribute_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        options: &Options,
    ) -> Result<AttributeRef>;
}

impl AsAttributeRef for AtomicTerm {
    fn as_attribute_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        options: &Options,
    ) -> Result<AttributeRef> {
        match self {
            AtomicTerm::Int(ref i) => i.as_attribute_ref(loc, builder, options),
            AtomicTerm::BigInt(ref i) => i.as_attribute_ref(loc, builder, options),
            AtomicTerm::Float(ref f) => f.as_attribute_ref(loc, builder, options),
            AtomicTerm::Atom(ref a) => a.as_attribute_ref(loc, builder, options),
            AtomicTerm::Binary(ref b) => b.as_attribute_ref(loc, builder, options),
            AtomicTerm::Nil => {
                let val = unsafe { MLIRBuildNilAttr(builder, loc) };
                unwrap!(val, "failed to construct constant nil")
            }
        }
    }
}
impl AsAttributeRef for ConstList {
    fn as_attribute_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<AttributeRef> {
        let val = unsafe {
            let len = self.0.len() as libc::c_uint;
            MLIRBuildListAttr(builder, loc, self.0.as_ptr(), len)
        };
        unwrap!(val, "failed to construct list attribute")
    }
}
impl AsAttributeRef for ConstTuple {
    fn as_attribute_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<AttributeRef> {
        let val = unsafe {
            let len = self.0.len() as libc::c_uint;
            MLIRBuildTupleAttr(builder, loc, self.0.as_ptr(), len)
        };
        unwrap!(val, "failed to construct tuple attribute")
    }
}
impl AsAttributeRef for ConstMap {
    fn as_attribute_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<AttributeRef> {
        let val = unsafe {
            MLIRBuildMapAttr(builder, loc, self.0.as_ptr(), self.0.len() as libc::c_uint)
        };
        unwrap!(val, "failed to construct map attribute")
    }
}
impl AsAttributeRef for IntTerm {
    #[inline]
    fn as_attribute_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        options: &Options,
    ) -> Result<AttributeRef> {
        self.value().as_attribute_ref(loc, builder, options)
    }
}
impl AsAttributeRef for i64 {
    fn as_attribute_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<AttributeRef> {
        let val = unsafe { MLIRBuildIntAttr(builder, loc, *self) };
        unwrap!(val, "failed to construct integer attribute ({})", self)
    }
}
impl AsAttributeRef for BigIntTerm {
    fn as_attribute_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<AttributeRef> {
        // Redundant, but needed, as libeir_ir uses a different num_bigint than us
        let bi = self.value();
        let width = bi.bits();
        let s = CString::new(bi.to_str_radix(10)).unwrap();
        let val = unsafe { MLIRBuildBigIntAttr(builder, loc, s.as_ptr(), width as libc::c_uint) };
        unwrap!(val, "failed to construct bigint attribute ({})", bi)
    }
}
impl AsAttributeRef for num_bigint::BigInt {
    fn as_attribute_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<AttributeRef> {
        let width = self.bits();
        let s = CString::new(self.to_str_radix(10)).unwrap();
        let val = unsafe { MLIRBuildBigIntAttr(builder, loc, s.as_ptr(), width as libc::c_uint) };
        unwrap!(val, "failed to construct bigint attribute ({})", self)
    }
}
impl AsAttributeRef for FloatTerm {
    #[inline]
    fn as_attribute_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        options: &Options,
    ) -> Result<AttributeRef> {
        self.value().as_attribute_ref(loc, builder, options)
    }
}
impl AsAttributeRef for f64 {
    fn as_attribute_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<AttributeRef> {
        let val = unsafe { MLIRBuildFloatAttr(builder, loc, *self) };
        unwrap!(val, "failed to construct constant float ({})", self)
    }
}
impl AsAttributeRef for AtomTerm {
    #[inline]
    fn as_attribute_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        options: &Options,
    ) -> Result<AttributeRef> {
        self.0.as_attribute_ref(loc, builder, options)
    }
}
impl AsAttributeRef for Symbol {
    fn as_attribute_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        _options: &Options,
    ) -> Result<AttributeRef> {
        let i = self.as_usize() as u64;
        let s = CString::new(self.as_str().get()).unwrap();
        let val = unsafe { MLIRBuildAtomAttr(builder, loc, s.as_ptr(), i) };
        unwrap!(val, "failed to construct atom attribute ({})", self)
    }
}
impl AsAttributeRef for BinaryTerm {
    fn as_attribute_ref(
        &self,
        loc: LocationRef,
        builder: ModuleBuilderRef,
        options: &Options,
    ) -> Result<AttributeRef> {
        use liblumen_term::*;

        let slice = self.value();
        let encoding_type = options.target.options.encoding;
        let pointer_width = options.target.target_pointer_width;
        let (header, flags) = match encoding_type {
            EncodingType::Encoding32 => BinaryLiteral::make_parts_from_slice::<Encoding32>(slice),
            EncodingType::Encoding64 => BinaryLiteral::make_parts_from_slice::<Encoding64>(slice),
            EncodingType::Encoding64Nanboxed => {
                BinaryLiteral::make_parts_from_slice::<Encoding64Nanboxed>(slice)
            }
            EncodingType::Default if pointer_width == 32 => {
                BinaryLiteral::make_parts_from_slice::<Encoding32>(slice)
            }
            EncodingType::Default if pointer_width == 64 => {
                BinaryLiteral::make_parts_from_slice::<Encoding64>(slice)
            }
            EncodingType::Default => unreachable!(),
        };

        let ptr = slice.as_ptr() as *const libc::c_char;
        let len = slice.len() as libc::c_uint;
        let val = unsafe { MLIRBuildBinaryAttr(builder, loc, ptr, len, header, flags) };
        if val.is_null() {
            let flags = unsafe {
                if pointer_width == 32 {
                    BinaryFlags::from_u32(flags as u32)
                } else {
                    BinaryFlags::from_u64(flags)
                }
            };
            Err(anyhow!(
                "failed to construct constant binary (flags = {:#?})",
                flags
            ))
        } else {
            Ok(val)
        }
    }
}
