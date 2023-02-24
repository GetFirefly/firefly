use alloc::alloc::Layout;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::mem;
use core::ops::Deref;

use smallvec::SmallVec;

use super::*;

const EMPTY_HEAP: firefly_alloc::heap::EmptyHeap = firefly_alloc::heap::EmptyHeap;

/// This trait is implemented by all term values, or values which may be represented as terms
pub trait Value {
    /// Returns true if this value can be represented as an immediate
    ///
    /// If this function returns true, `is_boxed` must return false.
    ///
    /// If you implement this function, `is_boxed` is implemented for you.
    fn is_immediate(&self) -> bool;

    /// Returns true if this value is heap-allocated
    ///
    /// If this function returns true, `is_immediate` must return false.
    #[inline]
    fn is_boxed(&self) -> bool {
        !self.is_immediate()
    }

    /// This function returns the size of a value that is heap-allocated, without consideration
    /// for the size of its transitive references.
    ///
    /// For example, a `Tuple` is effectively an array of terms. This function will return the
    /// size of just the `Tuple` struct, not including the size of any heap-allocated children
    /// present in the tuple. In other words, the size returned from this function should be sufficient
    /// to skip over the memory containing the value, without jumping into the middle of any subsequent
    /// values.
    ///
    /// Care must be taken to ensure the resulting size matches that of the underlying term,
    /// or unexpected behavior may result during garbage collection.
    ///
    /// Immediate values may simply return a size of zero. Pointer types should return the size of their
    /// pointee, not the size of the pointer itself (which is considered immediate for these purposes).
    ///
    /// By default the implementation of this function returns 0 if `is_immediate` returns
    /// true, and otherwise returns `mem::size_of_val(self)`.
    #[inline]
    fn sizeof(&self) -> usize {
        if self.is_immediate() {
            0
        } else {
            mem::size_of_val(self)
        }
    }

    #[inline]
    fn fragment_layout(&self) -> Layout {
        if self.is_immediate() {
            Layout::new::<()>()
        } else {
            Layout::for_value(self)
        }
    }
}
impl<'a, T: ?Sized + Value> Value for &'a T {
    #[inline]
    fn sizeof(&self) -> usize {
        (**self).sizeof()
    }
    #[inline]
    fn fragment_layout(&self) -> Layout {
        (**self).fragment_layout()
    }
    #[inline]
    fn is_immediate(&self) -> bool {
        (**self).is_immediate()
    }
    #[inline]
    fn is_boxed(&self) -> bool {
        (**self).is_boxed()
    }
}
impl Value for OpaqueTerm {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        let this = *self;
        let term: Term = this.into();
        term.layout()
    }

    #[inline]
    fn sizeof(&self) -> usize {
        let this = *self;
        if this.is_immediate() || this.is_rc() {
            return 0;
        }
        if this.is_nonempty_list() {
            return mem::size_of::<Cons>();
        }
        if this.is_gcbox() || this.is_tuple() {
            unsafe {
                let mut header_ptr = this.as_ptr();
                loop {
                    let header = *header_ptr.cast::<OpaqueTerm>();
                    if header.is_header() {
                        let header = header.as_header();
                        return match header.tag() {
                            Tag::BigInt => mem::size_of::<BigInt>(),
                            Tag::Pid => mem::size_of::<Pid>(),
                            Tag::Port => mem::size_of::<Port>(),
                            Tag::Reference => mem::size_of::<Reference>(),
                            Tag::Match => mem::size_of::<MatchContext>(),
                            Tag::Slice => mem::size_of::<BitSlice>(),
                            Tag::Tuple => {
                                let tup =
                                    <Tuple as Boxable>::from_raw_parts(header_ptr.cast(), header);
                                mem::size_of_val_raw(tup)
                            }
                            Tag::Map => {
                                let map =
                                    <Map as Boxable>::from_raw_parts(header_ptr.cast(), header);
                                mem::size_of_val_raw(map)
                            }
                            Tag::Closure => {
                                let closure =
                                    <Closure as Boxable>::from_raw_parts(header_ptr.cast(), header);
                                mem::size_of_val_raw(closure)
                            }
                            Tag::Binary => {
                                let bin = <BinaryData as Boxable>::from_raw_parts(
                                    header_ptr.cast(),
                                    header,
                                );
                                mem::size_of_val_raw(bin)
                            }
                        };
                    } else {
                        assert!(header.is_gcbox());
                        header_ptr = header.as_ptr();
                    }
                }
            }
        }
        assert!(this.is_header());
        let header = unsafe { this.as_header() };
        let header_ptr = self as *const OpaqueTerm as *mut OpaqueTerm;
        match header.tag() {
            Tag::BigInt => mem::size_of::<BigInt>(),
            Tag::Pid => mem::size_of::<Pid>(),
            Tag::Port => mem::size_of::<Port>(),
            Tag::Reference => mem::size_of::<Reference>(),
            Tag::Match => mem::size_of::<MatchContext>(),
            Tag::Slice => mem::size_of::<BitSlice>(),
            Tag::Tuple => unsafe {
                let tup = <Tuple as Boxable>::from_raw_parts(header_ptr.cast(), header);
                mem::size_of_val_raw(tup)
            },
            Tag::Map => unsafe {
                let map = <Map as Boxable>::from_raw_parts(header_ptr.cast(), header);
                mem::size_of_val_raw(map)
            },
            Tag::Closure => unsafe {
                let closure = <Closure as Boxable>::from_raw_parts(header_ptr.cast(), header);
                mem::size_of_val_raw(closure)
            },
            Tag::Binary => unsafe {
                let bin = <BinaryData as Boxable>::from_raw_parts(header_ptr.cast(), header);
                mem::size_of_val_raw(bin)
            },
        }
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        (*self).is_immediate() || (*self).is_special()
    }

    #[inline(always)]
    fn is_boxed(&self) -> bool {
        (*self).is_box()
    }
}
impl Value for Term {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        self.layout()
    }

    #[inline]
    fn sizeof(&self) -> usize {
        match self {
            Term::None
            | Term::Catch(_)
            | Term::Code(_)
            | Term::Nil
            | Term::Bool(_)
            | Term::Atom(_)
            | Term::Int(_)
            | Term::Float(_)
            | Term::Port(_)
            | Term::RcBinary(_)
            | Term::ConstantBinary(_) => 0,
            Term::BigInt(boxed) => boxed.sizeof(),
            Term::Cons(boxed) => boxed.sizeof(),
            Term::Tuple(boxed) => boxed.sizeof(),
            Term::Map(boxed) => boxed.sizeof(),
            Term::Closure(boxed) => boxed.sizeof(),
            Term::Pid(boxed) => boxed.sizeof(),
            Term::Reference(boxed) => boxed.sizeof(),
            Term::HeapBinary(boxed) => boxed.sizeof(),
            Term::RefBinary(boxed) => boxed.sizeof(),
        }
    }

    fn is_immediate(&self) -> bool {
        match self {
            Self::None
            | Self::Catch(_)
            | Self::Code(_)
            | Self::Nil
            | Self::Bool(_)
            | Self::Atom(_)
            | Self::Int(_)
            | Self::Float(_) => true,
            _ => false,
        }
    }

    fn is_boxed(&self) -> bool {
        match self {
            Self::BigInt(_)
            | Self::Cons(_)
            | Self::Tuple(_)
            | Self::Map(_)
            | Self::Closure(_)
            | Self::Pid(_)
            | Self::Port(_)
            | Self::Reference(_)
            | Self::HeapBinary(_)
            | Self::RcBinary(_)
            | Self::RefBinary(_)
            | Self::ConstantBinary(_) => true,
            _ => false,
        }
    }
}
impl<T: ?Sized + Value> Value for Gc<T> {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        self.deref().fragment_layout()
    }

    #[inline]
    fn sizeof(&self) -> usize {
        self.deref().sizeof()
    }

    #[inline(always)]
    fn is_boxed(&self) -> bool {
        true
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl<T: ?Sized + Value> Value for Arc<T> {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        Layout::new::<()>()
    }

    #[inline]
    fn sizeof(&self) -> usize {
        0
    }

    #[inline(always)]
    fn is_boxed(&self) -> bool {
        true
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl Value for bool {
    #[inline(always)]
    fn is_immediate(&self) -> bool {
        true
    }
}
impl Value for Atom {
    #[inline(always)]
    fn is_immediate(&self) -> bool {
        true
    }
}
impl Value for f64 {
    #[inline(always)]
    fn is_immediate(&self) -> bool {
        true
    }
}
impl Value for Float {
    #[inline(always)]
    fn is_immediate(&self) -> bool {
        true
    }
}
impl Value for i64 {
    #[inline]
    fn sizeof(&self) -> usize {
        if self.is_immediate() {
            0
        } else {
            mem::size_of::<BigInt>()
        }
    }

    #[inline]
    fn is_immediate(&self) -> bool {
        let n = *self;
        n <= Int::MAX_SMALL && n >= Int::MIN_SMALL
    }
}
impl Value for BigInt {
    #[inline(always)]
    fn sizeof(&self) -> usize {
        mem::size_of::<Self>()
    }
    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl Value for Pid {
    #[inline(always)]
    fn sizeof(&self) -> usize {
        mem::size_of::<Self>()
    }
    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl Value for Port {
    #[inline]
    fn sizeof(&self) -> usize {
        0
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl Value for Reference {
    #[inline(always)]
    fn sizeof(&self) -> usize {
        mem::size_of::<Self>()
    }
    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl Value for Map {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        self.layout_excluding_heap(&EMPTY_HEAP)
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl Value for Closure {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        self.layout_excluding_heap(&EMPTY_HEAP)
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl Value for Cons {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        self.layout_excluding_heap(&EMPTY_HEAP)
    }

    #[inline(always)]
    fn sizeof(&self) -> usize {
        mem::size_of::<Self>()
    }
    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl Value for Tuple {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        self.layout_excluding_heap(&EMPTY_HEAP)
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl Value for BitSlice {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        self.layout_excluding_heap(&EMPTY_HEAP)
    }

    #[inline]
    fn sizeof(&self) -> usize {
        if self.is_owner_refcounted() {
            mem::size_of::<Self>()
        } else {
            let placeholder: *const BinaryData = ptr::from_raw_parts(ptr::null(), self.byte_size());
            unsafe { mem::size_of_val_raw(placeholder) }
        }
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl Value for BinaryData {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        self.layout_excluding_heap(&EMPTY_HEAP)
    }

    #[inline]
    fn sizeof(&self) -> usize {
        // If too large, binary data is reference-counted
        if self.byte_size() > BinaryData::MAX_HEAP_BYTES {
            0
        } else {
            mem::size_of_val(self)
        }
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl Value for str {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        let placeholder: *const BinaryData = ptr::from_raw_parts(ptr::null(), self.byte_size());
        unsafe { Layout::for_value_raw(placeholder) }
    }

    #[inline]
    fn sizeof(&self) -> usize {
        let byte_size = self.as_bytes().len();
        if byte_size > BinaryData::MAX_HEAP_BYTES {
            0
        } else {
            let placeholder: *const BinaryData = ptr::from_raw_parts(ptr::null(), self.byte_size());
            unsafe { mem::size_of_val_raw(placeholder) }
        }
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}

impl Value for [u8] {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        let placeholder: *const BinaryData = ptr::from_raw_parts(ptr::null(), self.byte_size());
        unsafe { Layout::for_value_raw(placeholder) }
    }

    #[inline]
    fn sizeof(&self) -> usize {
        if self.len() > BinaryData::MAX_HEAP_BYTES {
            0
        } else {
            let placeholder: *const BinaryData = ptr::from_raw_parts(ptr::null(), self.byte_size());
            unsafe { mem::size_of_val_raw(placeholder) }
        }
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl<T: Value, U: Value> Value for (T, U) {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        let mut builder = LayoutBuilder::new();
        builder += self.0.fragment_layout();
        builder += self.1.fragment_layout();
        builder.build_tuple(3);
        builder.finish()
    }

    #[inline]
    fn sizeof(&self) -> usize {
        let placeholder: *const Tuple = ptr::from_raw_parts(ptr::null(), 2);
        self.0.sizeof() + self.1.sizeof() + unsafe { mem::size_of_val_raw(placeholder) }
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl<T: Value, U: Value, V: Value> Value for (T, U, V) {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        let mut builder = LayoutBuilder::new();
        builder += self.0.fragment_layout();
        builder += self.1.fragment_layout();
        builder += self.2.fragment_layout();
        builder.build_tuple(3);
        builder.finish()
    }

    #[inline]
    fn sizeof(&self) -> usize {
        let placeholder: *const Tuple = ptr::from_raw_parts(ptr::null(), 3);
        self.0.sizeof()
            + self.1.sizeof()
            + self.2.sizeof()
            + unsafe { mem::size_of_val_raw(placeholder) }
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        false
    }
}
impl<T: Value> Value for [T] {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        let mut builder = LayoutBuilder::new();
        for item in self {
            builder += item.fragment_layout();
        }
        builder.build_list(self.len());
        builder.finish()
    }

    fn sizeof(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            let size = mem::size_of::<Cons>() * self.len();
            size + self.iter().map(|t| t.sizeof()).sum::<usize>()
        }
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        self.is_empty()
    }
}
impl<const N: usize, T: Value> Value for SmallVec<[T; N]> {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        let mut builder = LayoutBuilder::new();
        for item in self.iter() {
            builder += item.fragment_layout();
        }
        builder.build_list(N);
        builder.finish()
    }

    fn sizeof(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            let size = mem::size_of::<Cons>() * self.len();
            size + self.iter().map(|t| t.sizeof()).sum::<usize>()
        }
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        self.is_empty()
    }
}
impl<T: Value> Value for Vec<T> {
    #[inline]
    fn fragment_layout(&self) -> Layout {
        let mut builder = LayoutBuilder::new();
        for item in self.iter() {
            builder += item.fragment_layout();
        }
        builder.build_list(self.len());
        builder.finish()
    }

    fn sizeof(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            let size = mem::size_of::<Cons>() * self.len();
            size + self.iter().map(|t| t.sizeof()).sum::<usize>()
        }
    }

    #[inline(always)]
    fn is_immediate(&self) -> bool {
        self.is_empty()
    }
}
