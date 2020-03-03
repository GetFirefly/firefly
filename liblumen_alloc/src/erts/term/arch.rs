// Provides the required trait implementations for a boxable term
//
// Intended for use internally within the `arch` module
macro_rules! impl_boxable {
    ($typ: ty, $raw: ty) => {
        impl crate::erts::term::encoding::Boxable<$raw> for $typ {}
        impl core::convert::From<*const $typ> for $raw {
            fn from(ptr: *const $typ) -> $raw {
                <$raw>::encode_box(ptr)
            }
        }
        impl core::convert::From<*mut $typ> for $raw {
            fn from(ptr: *mut $typ) -> $raw {
                <$raw>::encode_box(ptr)
            }
        }
        impl core::convert::From<&$typ> for $raw {
            fn from(ptr: &$typ) -> $raw {
                <$raw>::encode_box(ptr as *const $typ)
            }
        }
        impl core::convert::From<&mut $typ> for $raw {
            fn from(ptr: &mut $typ) -> $raw {
                <$raw>::encode_box(ptr as *mut $typ)
            }
        }
        impl core::convert::From<crate::erts::term::prelude::Boxed<$typ>> for $raw {
            fn from(ptr: crate::erts::term::prelude::Boxed<$typ>) -> $raw {
                <$raw>::encode_box(ptr.as_ptr())
            }
        }
        impl crate::erts::term::encoding::Encode<$raw> for $typ {
            fn encode(&self) -> crate::erts::exception::InternalResult<$raw> {
                Ok(<$raw>::encode_box(self as *const $typ))
            }
        }
    };
}

macro_rules! impl_unsized_boxable {
    ($typ: ty, $raw: ty) => {
        impl core::convert::From<*const $typ> for $raw {
            fn from(ptr: *const $typ) -> $raw {
                <$raw>::encode_box(ptr)
            }
        }
        impl core::convert::From<*mut $typ> for $raw {
            fn from(ptr: *mut $typ) -> $raw {
                <$raw>::encode_box(ptr)
            }
        }
        impl core::convert::From<&$typ> for $raw {
            fn from(ptr: &$typ) -> $raw {
                <$raw>::encode_box(ptr as *const $typ)
            }
        }
        impl core::convert::From<&mut $typ> for $raw {
            fn from(ptr: &mut $typ) -> $raw {
                <$raw>::encode_box(ptr as *mut $typ)
            }
        }
        impl core::convert::From<crate::erts::term::prelude::Boxed<$typ>> for $raw {
            fn from(ptr: crate::erts::term::prelude::Boxed<$typ>) -> $raw {
                <$raw>::encode_box(ptr.as_ptr())
            }
        }
        impl crate::erts::term::encoding::Encode<$raw> for $typ {
            fn encode(&self) -> crate::erts::exception::InternalResult<$raw> {
                Ok(<$raw>::encode_box(self as *const $typ))
            }
        }
    };
}

// Provides the required trait implementations for a boxable literal term
//
// Intended for use internally within the `arch` module
macro_rules! impl_literal {
    ($typ: ty, $raw: ty) => {
        impl crate::erts::term::encoding::Boxable<$raw> for $typ {}
        impl crate::erts::term::encoding::Literal<$raw> for $typ {}
        impl core::convert::From<*const $typ> for $raw {
            fn from(ptr: *const $typ) -> $raw {
                <$raw>::encode_literal(ptr)
            }
        }
        impl core::convert::From<*mut $typ> for $raw {
            fn from(ptr: *mut $typ) -> $raw {
                <$raw>::encode_literal(ptr)
            }
        }
        impl core::convert::From<&$typ> for $raw {
            fn from(ptr: &$typ) -> $raw {
                <$raw>::encode_box(ptr as *const $typ)
            }
        }
        impl core::convert::From<&mut $typ> for $raw {
            fn from(ptr: &mut $typ) -> $raw {
                <$raw>::encode_box(ptr as *mut $typ)
            }
        }
        impl core::convert::From<crate::erts::term::prelude::Boxed<$typ>> for $raw {
            fn from(ptr: crate::erts::term::prelude::Boxed<$typ>) -> $raw {
                <$raw>::encode_literal(ptr.as_ptr())
            }
        }
        impl crate::erts::term::encoding::Encode<$raw> for $typ {
            fn encode(&self) -> crate::erts::exception::InternalResult<$raw> {
                Ok(<$raw>::encode_literal(self as *const $typ))
            }
        }
    };
}

// Provides the required trait implementations for Cons
//
// Intended for use internally within the `arch` module
macro_rules! impl_list {
    ($raw: ty) => {
        impl crate::erts::term::encoding::Boxable<$raw> for Cons {}
        impl core::convert::From<*const Cons> for $raw {
            fn from(ptr: *const Cons) -> $raw {
                <$raw>::encode_list(ptr)
            }
        }
        impl core::convert::From<*mut Cons> for $raw {
            fn from(ptr: *mut Cons) -> $raw {
                <$raw>::encode_list(ptr)
            }
        }
        impl core::convert::From<&$raw> for $raw {
            fn from(ptr: &$raw) -> $raw {
                <$raw>::encode_box(ptr as *const $raw)
            }
        }
        impl core::convert::From<&mut $raw> for $raw {
            fn from(ptr: &mut $raw) -> $raw {
                <$raw>::encode_box(ptr as *mut $raw)
            }
        }
        impl core::convert::From<crate::erts::term::prelude::Boxed<Cons>> for $raw {
            fn from(ptr: crate::erts::term::prelude::Boxed<Cons>) -> $raw {
                <$raw>::encode_list(ptr.as_ptr())
            }
        }
        impl crate::erts::term::encoding::Encode<$raw> for Cons {
            fn encode(&self) -> crate::erts::exception::InternalResult<$raw> {
                Ok(<$raw>::encode_list(self as *const Cons))
            }
        }
    };
}

pub mod arch_32;
pub mod arch_64;
pub mod arch_x86_64;
pub mod repr;

use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(target_pointer_width = "32")] {
        pub use self::arch_32 as target;
    } else if #[cfg(all(target_pointer_width = "64", target_arch = "x86_64"))] {
        use liblumen_core::sys::sysconf::MIN_ALIGN;
        const_assert!(MIN_ALIGN >= 8);

        pub use self::arch_x86_64 as target;
    } else if #[cfg(target_pointer_width = "64")] {
        pub use self::arch_64 as target;
    }
}

use core::mem;

use super::prelude::{Cons, TypedTerm};

pub use self::repr::Repr;
pub use liblumen_term::Tag;

// Export the platform-specific representation for use by the higher-level Term code
pub use target::RawTerm;

pub type Word = <target::Encoding as liblumen_term::Encoding>::Type;

impl RawTerm {
    /// Dynamically casts the underlying term into an instance of the `binary::Binary` trait
    ///
    /// NOTE: This will panic if the term is not a valid binary type, and does not include
    /// `MatchContext` or `SubBinary`.
    pub(in crate::erts) unsafe fn as_binary_ptr<'a>(&self) -> *mut u8 {
        use super::prelude::{Bitstring, Encoded};

        match self.decode().unwrap() {
            TypedTerm::ProcBin(bin_ptr) => bin_ptr.as_ref().as_byte_ptr(),
            TypedTerm::BinaryLiteral(bin_ptr) => bin_ptr.as_ref().as_byte_ptr(),
            TypedTerm::HeapBinary(bin_ptr) => bin_ptr.as_ref().as_byte_ptr(),
            t => panic!("tried to cast invalid term type to binary: {:?}", t),
        }
    }

    /// Resolve a term potentially containing a move marker to the location
    /// of the forward reference, returning the "real" term there.
    ///
    /// Move markers are used in two scenarios:
    ///
    /// - For non-cons cell terms which are moved, the original location is
    /// updated with a box that points to the new location. There is no marker
    /// per se, we just treat the term as a box
    /// - For cons cells, the old cell is overwritten with a special marker
    /// cell, where the head term is the none value, and the tail term is a pointer
    /// to the new location of the cell
    ///
    /// This function does not follow boxes, it just returns them as if they had
    /// been found that way. In the case of a cons cell, the term you get back will
    /// be the top-level list term, i.e. the term which has the pointer to the head
    /// cons cell
    #[inline]
    pub(crate) fn follow_moved(self) -> RawTerm {
        use super::prelude::{Cast, Encoded};

        if self.is_boxed() {
            let ptr: *const RawTerm = self.dyn_cast();
            let boxed = unsafe { *ptr };
            if boxed.is_boxed() {
                // Moved, and `boxed` is the forwarding address
                boxed
            } else {
                // Not moved
                self
            }
        } else if self.is_non_empty_list() {
            let ptr: *const Cons = self.dyn_cast();
            let cons = unsafe { &*ptr };
            if cons.is_move_marker() {
                cons.tail
            } else {
                self
            }
        } else {
            self
        }
    }
}

// It is currently assumed in various places in the codebase that the
// size of a raw term (either immediate or header value) is equal to
// the target platform pointer width, i.e. the machine word size
const_assert_eq!(mem::size_of::<RawTerm>(), mem::size_of::<usize>());

/// The larged atom ID supported on the current platform
pub const MAX_ATOM_ID: usize = <target::Encoding as liblumen_term::Encoding>::MAX_ATOM_ID as usize;

/// The smallest signed integer value supported on the current platform
pub const MIN_SMALLINT_VALUE: isize =
    <target::Encoding as liblumen_term::Encoding>::MIN_SMALLINT_VALUE as isize;

/// The larged signed integer value supported on the current platform
pub const MAX_SMALLINT_VALUE: isize =
    <target::Encoding as liblumen_term::Encoding>::MAX_SMALLINT_VALUE as isize;
