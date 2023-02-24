use alloc::alloc::Layout;
use core::ops::Deref;
use core::ptr::{self, NonNull};

use firefly_alloc::fragment::HeapFragment;

use super::*;

/// This builder is used to calculate a sufficient layout to represent some
/// set of terms and associated data on a fresh heap fragment, i.e. it does not
/// try to optimize for cases in which parts of a term may already be allocated
/// on the heap.
///
/// For allocating on the process heap, this is not generally useful; partially
/// due to the optimization mentioned above. Instead, this is intended for those
/// use cases where we need to allocate memory in a fragment, which may later be
/// copied to a process heap, but may be used for other purposes as well.
#[repr(transparent)]
#[derive(Debug)]
pub struct LayoutBuilder {
    layout: Layout,
}
impl LayoutBuilder {
    pub fn new() -> Self {
        Self {
            layout: Layout::new::<()>(),
        }
    }

    /// Consume the builder, returning the computed [`Layout`]
    pub fn finish(self) -> Layout {
        self.layout
    }

    /// Consume the builder, allocating a [`HeapFragment`] from the computed layout.
    pub fn into_fragment(self) -> Result<NonNull<HeapFragment>, AllocError> {
        HeapFragment::new(self.finish(), None)
    }

    /// Extends the layout with space for any immediate term
    pub fn build_immediate(&mut self) -> &mut Self {
        self
    }

    /// Extends the layout with space for an [`i64`] term
    pub fn build_int(&mut self) -> &mut Self {
        self
    }

    /// Extends the layout with space for a specific [`i64`] term
    /// which may be out of bounds for an immediate integer value
    pub fn build_for_i64(&mut self, i: i64) -> &mut Self {
        if i > Int::MAX_SMALL || i < Int::MIN_SMALL {
            self.build_bigint()
        } else {
            self.build_int()
        }
    }

    /// Extends the layout with space for an [`f64`] term
    pub fn build_float(&mut self) -> &mut Self {
        self
    }

    /// Extends the layout with space for an [`Atom`] term
    pub fn build_atom(&mut self) -> &mut Self {
        self
    }

    /// Extends the layout with space for a [`BigInt`] term
    pub fn build_bigint(&mut self) -> &mut Self {
        *self += Layout::new::<BigInt>();
        self
    }

    /// Extends the layout with a single [`Cons`] cell
    ///
    /// When building a list on a heap, we first allocate the value (if non-immediate),
    /// then allocate the cell corresponding to that value. Callers should ensure that they
    /// construct the layout in the correct order, as it may affect how padding is calculated.
    pub fn build_cons(&mut self) -> &mut Self {
        *self += Layout::new::<Cons>();
        self
    }

    /// Extends the layout with `len` consecutive cons cells
    ///
    /// It is expected that space for any non-immediate terms in the list are allocated
    /// in advance of this.
    pub fn build_list(&mut self, len: usize) -> &mut Self {
        if len == 0 {
            return self;
        }

        for _ in 0..len {
            *self += Layout::new::<Cons>();
        }

        // Proper lists require an extra trailing cell
        *self += Layout::new::<Cons>();

        self
    }

    /// Extends the layout with space for a [`Tuple`] term with `arity` elements
    pub fn build_tuple(&mut self, arity: usize) -> &mut Self {
        let ptr: *const Tuple = ptr::from_raw_parts(ptr::null(), arity);
        *self += unsafe { Layout::for_value_raw(ptr) };
        self
    }

    /// Extends the layout with space for an empty [`Map`] term
    ///
    /// NOTE: It is expected that the caller will have already extended the
    /// layout for each key/value pair to be stored in the map.
    pub fn build_map(&mut self, capacity: usize) -> &mut Self {
        unsafe {
            let empty: *const SmallMap = ptr::from_raw_parts(ptr::null(), capacity);
            *self += Layout::for_value_raw(empty);
        }
        self
    }

    /// Extends the layout with space for a [`Pid`] term
    pub fn build_pid(&mut self) -> &mut Self {
        *self += Layout::new::<Pid>();
        self
    }

    /// This is a no-op, as ports are reference-counted and thus not allocated on the
    /// process heap, but this allows expressing intent when defining layouts.
    pub fn build_port(&mut self) -> &mut Self {
        self
    }

    /// Extends the layout with space for a [`Reference`] term
    pub fn build_reference(&mut self) -> &mut Self {
        *self += Layout::new::<Reference>();
        self
    }

    /// Extends the layout with space for a binary term capable of holding `byte_size` bytes
    ///
    /// NOTE: For byte sizes <= 64 bytes, the layout is unchanged, as binaries of that size are
    /// allocated as reference-counted data, and as such appear as immediates on the process heap.
    pub fn build_binary(&mut self, byte_size: usize) -> &mut Self {
        if byte_size <= 64 {
            self.build_heap_binary(byte_size);
        }
        self
    }

    /// Extends the layout with space for a heap binary term capable of holding `byte_size` bytes
    ///
    /// This will extend the layout regardless of whether the binary _should_ be a ref-counted binary
    pub fn build_heap_binary(&mut self, byte_size: usize) -> &mut Self {
        unsafe {
            let empty: *const BinaryData = ptr::from_raw_parts(ptr::null(), byte_size);
            *self += Layout::for_value_raw(empty);
        }
        self
    }

    /// Extends the layout with space for a [`BitSlice`]
    pub fn build_ref_binary(&mut self) -> &mut Self {
        *self += Layout::new::<BitSlice>();
        self
    }

    /// Extends the layout with space for a [`Closure`] term with `arity` elements in its captured environment
    pub fn build_closure(&mut self, arity: usize) -> &mut Self {
        unsafe {
            let empty: *const Closure = ptr::from_raw_parts(ptr::null(), arity);
            *self += Layout::for_value_raw(empty);
        }
        self
    }

    /// Extends the layout to hold the given term, along with any data referenced by the term
    ///
    /// NOTE: For immediate terms, this has no effect on the layout, as they are not allocated on the heap.
    pub fn extend(&mut self, term: &Term) -> &mut Self {
        let layout = match term {
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
            | Term::ConstantBinary(_) => return self,
            Term::BigInt(boxed) => boxed.deref().layout(),
            Term::Map(boxed) => boxed.deref().layout(),
            Term::Closure(boxed) => boxed.deref().layout(),
            Term::Pid(boxed) => boxed.deref().layout(),
            Term::Cons(boxed) => boxed.deref().layout(),
            Term::Tuple(boxed) => boxed.deref().layout(),
            Term::Reference(boxed) => boxed.deref().layout(),
            Term::HeapBinary(boxed) => boxed.deref().layout(),
            Term::RefBinary(boxed) => boxed.deref().layout(),
        };
        *self += layout;
        self
    }
}
impl core::ops::Add<Layout> for LayoutBuilder {
    type Output = LayoutBuilder;

    fn add(self, layout: Layout) -> Self::Output {
        let (new_layout, _offset) = self.layout.extend(layout).unwrap();
        Self {
            layout: new_layout.pad_to_align(),
        }
    }
}
impl core::ops::AddAssign<Layout> for LayoutBuilder {
    fn add_assign(&mut self, layout: Layout) {
        let (new_layout, _offset) = self.layout.extend(layout).unwrap();
        self.layout = new_layout.pad_to_align();
    }
}
