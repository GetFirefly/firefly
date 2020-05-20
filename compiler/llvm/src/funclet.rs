use std::convert::AsRef;

use crate::Value;

/// A structure representing an active landing pad for the duration of a basic
/// block.
///
/// Each `Block` may contain an instance of this, indicating whether the block
/// is part of a landing pad or not. This is used to make decision about whether
/// to emit `invoke` instructions (e.g., in a landing pad we don't continue to
/// use `invoke`) and also about various function call metadata.
///
/// For GNU exceptions (`landingpad` + `resume` instructions) this structure is
/// just a bunch of `None` instances (not too interesting), but for MSVC
/// exceptions (`cleanuppad` + `cleanupret` instructions) this contains data.
/// When inside of a landing pad, each function call in LLVM IR needs to be
/// annotated with which landing pad it's a part of. This is accomplished via
/// the `OperandBundleDef` value created for MSVC landing pads.
pub struct Funclet {
    pad: Value,
    operand: OperandBundleDef,
}

impl Funclet {
    pub fn new(pad: Value) -> Self {
        Self {
            pad,
            operand: OperandBundleDef::new("funclet", &[pad]),
        }
    }

    pub fn pad(&self) -> Value {
        self.pad
    }

    pub fn bundle(&self) -> &OperandBundleDef {
        &self.operand
    }
}

pub struct OperandBundleDef {
    pub raw: ffi::OperandBundleDefRef,
}

impl OperandBundleDef {
    pub fn new(name: &str, vals: &[Value]) -> Self {
        let raw = unsafe {
            ffi::LLVMLumenBuildOperandBundleDef(
                name.as_ptr() as *const libc::c_char,
                name.len() as libc::size_t,
                vals.as_ptr(),
                vals.len() as libc::c_uint,
            )
        };
        Self { raw }
    }
}

impl AsRef<ffi::OperandBundleDef> for OperandBundleDef {
    fn as_ref(&self) -> &ffi::OperandBundleDef {
        self.raw.as_ref()
    }
}

impl Drop for OperandBundleDef {
    fn drop(&mut self) {
        unsafe {
            ffi::LLVMLumenFreeOperandBundleDef(self.raw);
        }
    }
}

pub mod ffi {
    use super::*;

    use liblumen_compiler_macros::foreign_struct;

    #[foreign_struct]
    pub struct OperandBundleDef;

    extern "C" {
        crate fn LLVMLumenBuildOperandBundleDef(
            name: *const libc::c_char,
            name_len: libc::size_t,
            vals: *const Value,
            num_vals: libc::c_uint,
        ) -> OperandBundleDefRef;

        crate fn LLVMLumenFreeOperandBundleDef(bundle: OperandBundleDefRef);
    }
}
