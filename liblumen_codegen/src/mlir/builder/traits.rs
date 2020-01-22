use std::ffi::CString;

use anyhow::anyhow;

use num_bigint::BigInt;

use libeir_intern::Symbol;
use libeir_ir::{Const, ConstKind};
use libeir_ir::{
    AtomicTerm, AtomTerm, BigIntTerm, BinaryTerm, FloatTerm, IntTerm, NilTerm,
};

use liblumen_alloc::erts::term::prelude::{BinaryLiteral, BinaryFlags};
use liblumen_session::Options;
use liblumen_term::EncodingType;

use crate::Result;

use super::ffi::*;

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
    fn as_value_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<ValueRef>;
}

impl AsValueRef for AtomicTerm {
    fn as_value_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<ValueRef> {
        match self {
            AtomicTerm::Int(ref i) => i.as_value_ref(builder, options),
            AtomicTerm::BigInt(ref i) => i.as_value_ref(builder, options),
            AtomicTerm::Float(ref f) => f.as_value_ref(builder, options),
            AtomicTerm::Atom(ref a) => a.as_value_ref(builder, options),
            AtomicTerm::Binary(ref b) => b.as_value_ref(builder, options),
            AtomicTerm::Nil => {
                let pointer_width = options.target.target_pointer_width;
                let encoding = options.target.options.encoding;
                let nil = with_encoding!(encoding, pointer_width, Encoding::NIL as i64);
                let val = unsafe {
                    MLIRBuildConstantNil(builder, nil, pointer_width as libc::c_uint)
                };
                unwrap!(val, "failed to construct constant nil")
            }
        }
    }
}
impl AsValueRef for ConstList {
    fn as_value_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<ValueRef> {
        let val = unsafe {
            let len = self.0.len() as libc::c_uint;
            MLIRBuildConstantList(builder, self.0.as_ptr(), len)
        };
        unwrap!(val, "failed to construct constant list")
    }
}
impl AsValueRef for ConstTuple {
    fn as_value_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<ValueRef> {
        let val = unsafe {
            let len = self.0.len() as libc::c_uint;
            MLIRBuildConstantTuple(builder, self.0.as_ptr(), len)
        };
        unwrap!(val, "failed to construct constant tuple")
    }
}
impl AsValueRef for ConstMap {
    fn as_value_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<ValueRef> {
        let val = unsafe {
            MLIRBuildConstantMap(builder, self.0.as_ptr(), self.0.len() as libc::c_uint)
        };
        unwrap!(val, "failed to construct constant map")
    }
}
impl AsValueRef for IntTerm {
    #[inline]
    fn as_value_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<ValueRef> {
        self.value().as_value_ref(builder, options)
    }
}
impl AsValueRef for i64 {
    fn as_value_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<ValueRef> {
        let pointer_width = options.target.target_pointer_width;
        let encoding = options.target.options.encoding;
        let val = with_encoding!(encoding, pointer_width, {
            let i = *self;
            if i < Encoding::MIN_SMALLINT_VALUE as i64 || i > Encoding::MAX_SMALLINT_VALUE as i64 {
                // Too large, must use big int
                let bi: BigInt = i.into();
                return bi.as_value_ref(builder, options);
            }
            unsafe {
                MLIRBuildConstantInt(builder, i, pointer_width as libc::c_uint)
            }
        });
        unwrap!(val, "failed to construct constant integer ({}) with encoding {:#?} for target pointer width of {}", self, encoding, pointer_width)
    }
}
impl AsValueRef for BigIntTerm {
    fn as_value_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<ValueRef> {
        // Redundant, but needed, since libeir_ir is referencing a different num_bigint than us
        let bi = self.value();
        let width = bi.bits();
        let s = CString::new(bi.to_str_radix(10)).unwrap();
        let val = unsafe {
            MLIRBuildConstantBigInt(builder, s.as_ptr(), width as libc::c_uint)
        };
        unwrap!(val, "failed to construct constant bigint ({})", bi)
    }
}
impl AsValueRef for BigInt {
    fn as_value_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<ValueRef> {
        let width = self.bits();
        let s = CString::new(self.to_str_radix(10)).unwrap();
        let val = unsafe {
            MLIRBuildConstantBigInt(builder, s.as_ptr(), width as libc::c_uint)
        };
        unwrap!(val, "failed to construct constant bigint ({})", self)
    }
}
impl AsValueRef for FloatTerm {
    #[inline]
    fn as_value_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<ValueRef> {
        self.value().as_value_ref(builder, options)
    }
}
impl AsValueRef for f64 {
    fn as_value_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<ValueRef> {
        let is_packed = options.target.options.encoding != EncodingType::Encoding64Nanboxed;
        let val = unsafe {
            MLIRBuildConstantFloat(builder, *self, is_packed)
        };
        unwrap!(val, "failed to construct constant float ({}, packed = {})", self, is_packed)
    }
}
impl AsValueRef for AtomTerm {
    #[inline]
    fn as_value_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<ValueRef> {
        self.0.as_value_ref(builder, options)
    }
}
impl AsValueRef for Symbol {
    fn as_value_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<ValueRef> {
        let pointer_width = options.target.target_pointer_width;
        let encoding = options.target.options.encoding;
        let val = with_encoding!(encoding, pointer_width, {
            let i = self.as_usize() as <Encoding as TermEncoding>::Type;
            if i >= Encoding::MAX_ATOM_ID {
                return Err(anyhow!("invalid atom id ({}) in encoding {:#?} for target pointer width of {}", i, encoding, pointer_width));
            }
            let encoded = Encoding::encode_immediate(i, Encoding::TAG_ATOM);
            let s = CString::new(self.as_str().get()).unwrap();
            unsafe {
                MLIRBuildConstantAtom(builder, s.as_ptr(), encoded as u64, pointer_width as libc::c_uint)
            }
        });
        unwrap!(val, "failed to construct constant atom ({}) with encoding {:#?} for target pointer width of {}", self, encoding, pointer_width)
    }
}
impl AsValueRef for BinaryTerm {
    fn as_value_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<ValueRef> {
        let slice = self.value();
        let pointer_width = options.target.target_pointer_width;
        let (header, flags) = if pointer_width == 32 {
            let (h, f) = BinaryLiteral::make_arch32_parts_from_slice(slice);
            (h as u64, f as u64)
        } else {
            BinaryLiteral::make_arch64_parts_from_slice(slice)
        };

        let ptr = slice.as_ptr() as *const libc::c_char;
        let len = slice.len() as libc::c_uint;
        let val = unsafe {
            MLIRBuildConstantBinary(builder, ptr, len, header, flags, pointer_width as libc::c_uint)
        };
        if val.is_null() {
            let flags = unsafe {
                if pointer_width == 32 {
                    BinaryFlags::from_u32(flags as u32)
                } else {
                    BinaryFlags::from_u64(flags)
                }
            };
            Err(anyhow!("failed to construct constant binary (flags = {:#?})", flags))
        } else {
            Ok(val)
        }
    }
}

// This trait represents the conversion of some value to an MLIR attribute value
pub trait AsAttributeRef {
    fn as_attribute_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<AttributeRef>;
}

impl AsAttributeRef for AtomicTerm {
    fn as_attribute_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<AttributeRef> {
        match self {
            AtomicTerm::Int(ref i) => i.as_attribute_ref(builder, options),
            AtomicTerm::BigInt(ref i) => i.as_attribute_ref(builder, options),
            AtomicTerm::Float(ref f) => f.as_attribute_ref(builder, options),
            AtomicTerm::Atom(ref a) => a.as_attribute_ref(builder, options),
            AtomicTerm::Binary(ref b) => b.as_attribute_ref(builder, options),
            AtomicTerm::Nil => {
                let pointer_width = options.target.target_pointer_width;
                let encoding = options.target.options.encoding;
                let nil = with_encoding!(encoding, pointer_width, Encoding::NIL as i64);
                let val = unsafe {
                    MLIRBuildNilAttr(builder, nil, pointer_width as libc::c_uint)
                };
                unwrap!(val, "failed to construct constant nil")
            }
        }
    }
}
impl AsAttributeRef for ConstList {
    fn as_attribute_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<AttributeRef> {
        let val = unsafe {
            let len = self.0.len() as libc::c_uint;
            MLIRBuildListAttr(builder, self.0.as_ptr(), len)
        };
        unwrap!(val, "failed to construct list attribute")
    }
}
impl AsAttributeRef for ConstTuple {
    fn as_attribute_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<AttributeRef> {
        let val = unsafe {
            let len = self.0.len() as libc::c_uint;
            MLIRBuildTupleAttr(builder, self.0.as_ptr(), len)
        };
        unwrap!(val, "failed to construct tuple attribute")
    }
}
impl AsAttributeRef for ConstMap {
    fn as_attribute_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<AttributeRef> {
        let val = unsafe {
            MLIRBuildMapAttr(builder, self.0.as_ptr(), self.0.len() as libc::c_uint)
        };
        unwrap!(val, "failed to construct map attribute")
    }
}
impl AsAttributeRef for IntTerm {
    #[inline]
    fn as_attribute_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<AttributeRef> {
        self.value().as_attribute_ref(builder, options)
    }
}
impl AsAttributeRef for i64 {
    fn as_attribute_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<AttributeRef> {
        let pointer_width = options.target.target_pointer_width;
        let encoding = options.target.options.encoding;
        let val = with_encoding!(encoding, pointer_width, {
            let i = *self;
            if i < Encoding::MIN_SMALLINT_VALUE as i64 || i > Encoding::MAX_SMALLINT_VALUE as i64 {
                // Too large, must use big int
                let bi: BigInt = i.into();
                return bi.as_attribute_ref(builder, options);
            }
            unsafe {
                MLIRBuildIntAttr(builder, i, pointer_width as libc::c_uint)
            }
        });
        unwrap!(val, "failed to construct integer attribute ({}) with encoding {:#?} for target pointer width of {}", self, encoding, pointer_width)
    }
}
impl AsAttributeRef for BigIntTerm {
    fn as_attribute_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<AttributeRef> {
        // Redundant, but needed, as libeir_ir uses a different num_bigint than us
        let bi = self.value();
        let width = bi.bits();
        let s = CString::new(bi.to_str_radix(10)).unwrap();
        let val = unsafe {
            MLIRBuildBigIntAttr(builder, s.as_ptr(), width as libc::c_uint)
        };
        unwrap!(val, "failed to construct bigint attribute ({})", bi)
    }
}
impl AsAttributeRef for num_bigint::BigInt {
    fn as_attribute_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<AttributeRef> {
        let width = self.bits();
        let s = CString::new(self.to_str_radix(10)).unwrap();
        let val = unsafe {
            MLIRBuildBigIntAttr(builder, s.as_ptr(), width as libc::c_uint)
        };
        unwrap!(val, "failed to construct bigint attribute ({})", self)
    }
}
impl AsAttributeRef for FloatTerm {
    #[inline]
    fn as_attribute_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<AttributeRef> {
        self.value().as_attribute_ref(builder, options)
    }
}
impl AsAttributeRef for f64 {
    fn as_attribute_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<AttributeRef> {
        let is_packed = options.target.options.encoding != EncodingType::Encoding64Nanboxed;
        let val = unsafe {
            MLIRBuildFloatAttr(builder, *self, is_packed)
        };
        unwrap!(val, "failed to construct constant float ({}, packed = {})", self, is_packed)
    }
}
impl AsAttributeRef for AtomTerm {
    #[inline]
    fn as_attribute_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<AttributeRef> {
        self.0.as_attribute_ref(builder, options)
    }
}
impl AsAttributeRef for Symbol {
    fn as_attribute_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<AttributeRef> {
        let pointer_width = options.target.target_pointer_width;
        let encoding = options.target.options.encoding;
        let val = with_encoding!(encoding, pointer_width, {
            let i = self.as_usize() as <Encoding as TermEncoding>::Type;
            if i >= Encoding::MAX_ATOM_ID {
                return Err(anyhow!("invalid atom id ({}) in encoding {:#?} for target pointer width of {}", i, encoding, pointer_width));
            }
            let encoded = Encoding::encode_immediate(i, Encoding::TAG_ATOM);
            let s = CString::new(self.as_str().get()).unwrap();
            unsafe {
                MLIRBuildAtomAttr(builder, s.as_ptr(), encoded as u64, pointer_width as libc::c_uint)
            }
        });
        unwrap!(val, "failed to construct atom attribute ({}) with encoding {:#?} for target pointer width of {}", self, encoding, pointer_width)
    }
}
impl AsAttributeRef for BinaryTerm {
    fn as_attribute_ref(&self, builder: ModuleBuilderRef, options: &Options) -> Result<AttributeRef> {
        let slice = self.value();
        let pointer_width = options.target.target_pointer_width;
        let (header, flags) = if pointer_width == 32 {
            let (h, f) = BinaryLiteral::make_arch32_parts_from_slice(slice);
            (h as u64, f as u64)
        } else {
            BinaryLiteral::make_arch64_parts_from_slice(slice)
        };

        let ptr = slice.as_ptr() as *const libc::c_char;
        let len = slice.len() as libc::c_uint;
        let val = unsafe {
            MLIRBuildBinaryAttr(builder, ptr, len, header, flags, pointer_width as libc::c_uint)
        };
        if val.is_null() {
            let flags = unsafe {
                if pointer_width == 32 {
                    BinaryFlags::from_u32(flags as u32)
                } else {
                    BinaryFlags::from_u64(flags)
                }
            };
            Err(anyhow!("failed to construct constant binary (flags = {:#?})", flags))
        } else {
            Ok(val)
        }
    }
}
