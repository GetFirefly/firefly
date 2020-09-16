use core::ffi::c_void;
use core::mem;
use core::ptr::{self, NonNull};
use core::str::Chars;

use hashbrown::HashMap;

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedListLink, UnsafeRef};

use liblumen_core::alloc::prelude::*;
use liblumen_core::alloc::utils::{align_up_to, is_aligned, is_aligned_at};
use liblumen_core::sys::sysconf::MIN_ALIGN;

use crate::erts::exception::AllocResult;
use crate::erts::module_function_arity::Arity;
use crate::erts::process::alloc::{Heap, HeapAlloc, TermAlloc};
use crate::erts::term::closure::{ClosureLayout, Creator, Index, OldUnique, Unique};
use crate::erts::term::prelude::*;
use crate::scheduler;
use crate::std_alloc;
use crate::{erts, CloneToProcess};

// This adapter is used to track a list of heap fragments, attached to a process
intrusive_adapter!(pub HeapFragmentAdapter = UnsafeRef<HeapFragment>: HeapFragment { link: LinkedListLink });

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawFragment {
    size: usize,
    align: usize,
    base: NonNull<u8>,
}
impl RawFragment {
    /// Get a pointer to the data in this heap fragment
    #[inline]
    pub fn data(&self) -> NonNull<u8> {
        self.base
    }

    /// Get the layout of this heap fragment
    #[inline]
    pub fn layout(&self) -> Layout {
        unsafe { Layout::from_size_align_unchecked(self.size, self.align) }
    }
}

#[derive(Debug)]
pub struct HeapFragment {
    // Link to the intrusive list that holds all heap fragments
    pub link: LinkedListLink,
    // The memory region allocated for this fragment
    raw: RawFragment,
    // The amount of used memory in this fragment
    top: *mut u8,
}
impl HeapFragment {
    /// Returns the pointer to the data region of this fragment
    #[inline]
    pub fn data(&self) -> NonNull<u8> {
        self.raw.data()
    }

    /// Creates a new heap fragment with the given layout, allocated via `std_alloc`
    #[inline]
    pub fn new(layout: Layout) -> AllocResult<NonNull<Self>> {
        // `alloc_layout` pads to `MIN_ALIGN`, so creating the new `HeapFragment` must too
        // Ensure layout has alignment padding
        let layout = layout.align_to(MIN_ALIGN).unwrap().pad_to_align();
        let (full_layout, offset) = Layout::new::<Self>().extend(layout.clone()).unwrap();
        let size = layout.size();
        let align = layout.align();
        let block = std_alloc::alloc(full_layout, AllocInit::Uninitialized)?;
        let ptr = block.ptr.as_ptr() as *mut Self;
        let data = unsafe { (ptr as *mut u8).add(offset) };
        let top = data;
        unsafe {
            ptr::write(
                ptr,
                Self {
                    link: LinkedListLink::new(),
                    raw: RawFragment {
                        size,
                        align,
                        base: NonNull::new_unchecked(data),
                    },
                    top,
                },
            );
        }
        Ok(block.ptr.cast())
    }

    pub fn new_from_word_size(word_size: usize) -> AllocResult<NonNull<Self>> {
        let byte_size = word_size * mem::size_of::<Term>();
        let align = mem::align_of::<Term>();

        let layout = unsafe { Layout::from_size_align_unchecked(byte_size, align) };

        Self::new(layout)
    }

    pub fn new_binary_from_bytes(bytes: &[u8]) -> AllocResult<(Term, NonNull<Self>)> {
        let len = bytes.len();

        if len > HeapBin::MAX_SIZE {
            Self::new_procbin_from_bytes(bytes)
        } else {
            Self::new_heapbin_from_bytes(bytes)
        }
    }

    fn new_procbin_from_bytes(bytes: &[u8]) -> AllocResult<(Term, NonNull<Self>)> {
        let layout = Layout::new::<ProcBin>();
        let mut non_null_heap_fragment = Self::new(layout)?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        heap_fragment
            .procbin_from_bytes(bytes)
            .map(|boxed_proc_bin| (boxed_proc_bin.into(), non_null_heap_fragment))
    }

    fn new_heapbin_from_bytes(bytes: &[u8]) -> AllocResult<(Term, NonNull<Self>)> {
        let (layout, _, _) = HeapBin::layout_for(bytes);
        let mut non_null_heap_fragment = Self::new(layout)?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        heap_fragment
            .heapbin_from_bytes(bytes)
            .map(|boxed_heap_bin| (boxed_heap_bin.into(), non_null_heap_fragment))
    }

    pub fn new_binary_from_str(s: &str) -> AllocResult<(Term, NonNull<Self>)> {
        let len = s.len();

        if len > HeapBin::MAX_SIZE {
            Self::new_procbin_from_str(s)
        } else {
            Self::new_heapbin_from_str(s)
        }
    }

    fn new_procbin_from_str(s: &str) -> AllocResult<(Term, NonNull<Self>)> {
        let layout = Layout::new::<ProcBin>();
        let mut non_null_heap_fragment = Self::new(layout)?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        heap_fragment
            .procbin_from_str(s)
            .map(|boxed_proc_bin| (boxed_proc_bin.into(), non_null_heap_fragment))
    }

    fn new_heapbin_from_str(s: &str) -> AllocResult<(Term, NonNull<Self>)> {
        let (layout, _, _) = HeapBin::layout_for(s.as_bytes());
        let mut non_null_heap_fragment = Self::new(layout)?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        heap_fragment
            .heapbin_from_str(s)
            .map(|boxed_heap_bin| (boxed_heap_bin.into(), non_null_heap_fragment))
    }

    pub fn new_charlist_from_str(s: &str) -> AllocResult<(Option<Boxed<Cons>>, NonNull<Self>)> {
        Self::new_list_from_chars(s.chars())
    }

    pub fn new_copy_closure(
        closure: Boxed<Closure>,
    ) -> AllocResult<(Boxed<Closure>, NonNull<Self>)> {
        let layout = closure.layout();
        let mut non_null_heap_fragment = Self::new(layout)?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        closure
            .clone_to(heap_fragment)
            .map(|cloned_closure| (cloned_closure, non_null_heap_fragment))
    }

    pub fn new_anonymous_closure_with_env_from_slice(
        module: Atom,
        index: Index,
        old_unique: OldUnique,
        unique: Unique,
        arity: Arity,
        native: Option<NonNull<c_void>>,
        creator: Creator,
        env: &[Term],
    ) -> AllocResult<(Boxed<Closure>, NonNull<Self>)> {
        let closure_layout = ClosureLayout::for_env(env);
        let layout = closure_layout.layout();
        let mut non_null_heap_fragment = Self::new(layout.clone())?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        heap_fragment
            .anonymous_closure_with_env_from_slice(
                module, index, old_unique, unique, arity, native, creator, env,
            )
            .map(|boxed_closure| (boxed_closure, non_null_heap_fragment))
    }

    pub fn new_export_closure(
        module: Atom,
        function: Atom,
        arity: u8,
        native: Option<NonNull<c_void>>,
    ) -> AllocResult<(Boxed<Closure>, NonNull<Self>)> {
        let closure_layout = ClosureLayout::for_env_len(0);
        let layout = closure_layout.layout();
        let mut non_null_heap_fragment = Self::new(layout.clone())?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        heap_fragment
            .export_closure(module, function, arity, native)
            .map(|boxed_closure| (boxed_closure, non_null_heap_fragment))
    }

    pub fn new_cons(head: Term, tail: Term) -> AllocResult<(Boxed<Cons>, NonNull<Self>)> {
        let layout = Layout::new::<Cons>();
        let mut non_null_heap_fragment = Self::new(layout)?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        heap_fragment
            .cons(head, tail)
            .map(|boxed_cons| (boxed_cons, non_null_heap_fragment))
    }

    #[cfg(target_arch = "x86_64")]
    pub fn new_float(_f: f64) -> AllocResult<(Float, NonNull<Self>)> {
        unreachable!(
            "x86_64 should never need to store a float in a fragment as it fits in a term"
        );
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn new_float(f: f64) -> AllocResult<(Boxed<Float>, NonNull<Self>)> {
        let layout = Layout::new::<Float>();
        let mut non_null_heap_fragment = Self::new(layout)?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        heap_fragment
            .float(f)
            .map(|boxed_float| (boxed_float, non_null_heap_fragment))
    }

    pub fn new_list_from_chars(chars: Chars) -> AllocResult<(Option<Boxed<Cons>>, NonNull<Self>)> {
        let len = chars.clone().count();
        let (layout, _offset) = Layout::new::<Cons>().repeat(len).map_err(|_| alloc!())?;
        let mut non_null_heap_fragment = Self::new(layout)?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        heap_fragment
            .list_from_chars(chars)
            .map(|option_boxed_cons| (option_boxed_cons, non_null_heap_fragment))
    }

    pub fn new_list_from_iter<I>(iter: I) -> AllocResult<(Option<Boxed<Cons>>, NonNull<Self>)>
    where
        I: Clone + DoubleEndedIterator + Iterator<Item = Term>,
    {
        Self::new_improper_list_from_iter(iter, Term::NIL)
    }

    pub fn new_list_from_slice(
        slice: &[Term],
    ) -> AllocResult<(Option<Boxed<Cons>>, NonNull<Self>)> {
        Self::new_improper_list_from_slice(slice, Term::NIL)
    }

    pub fn new_improper_list_from_iter<I>(
        iter: I,
        last: Term,
    ) -> AllocResult<(Option<Boxed<Cons>>, NonNull<Self>)>
    where
        I: Clone + DoubleEndedIterator + Iterator<Item = Term>,
    {
        let init_len = iter.clone().count();

        let len = if init_len == 0 {
            if last.is_nil() {
                unreachable!("An empty list should not need a HeapFragment as it all fits in a Term, Term::NIL")
            } else if last.is_non_empty_list() {
                unreachable!("A non-empty list should not need to be reallocated")
            } else {
                1
            }
        } else {
            init_len
        };

        let (layout, _offset) = Layout::new::<Cons>().repeat(len).map_err(|_| alloc!())?;
        let mut non_null_heap_fragment = Self::new(layout)?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        heap_fragment
            .improper_list_from_iter(iter, last)
            .map(|option_boxed_cons| (option_boxed_cons, non_null_heap_fragment))
    }

    pub fn new_improper_list_from_slice(
        slice: &[Term],
        tail: Term,
    ) -> AllocResult<(Option<Boxed<Cons>>, NonNull<Self>)> {
        Self::new_improper_list_from_iter(slice.iter().copied(), tail)
    }

    pub fn new_map_from_hash_map(
        hash_map: HashMap<Term, Term>,
    ) -> AllocResult<(Boxed<Map>, NonNull<Self>)> {
        let map = Map::from_hash_map(hash_map);
        let layout = Layout::for_value(&map);
        let mut non_null_heap_fragment = Self::new(layout)?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        let term = map.clone_to_heap(heap_fragment)?;
        let boxed_map: Boxed<Map> = term.dyn_cast();

        Ok((boxed_map, non_null_heap_fragment))
    }

    pub fn new_map_from_slice(slice: &[(Term, Term)]) -> AllocResult<(Boxed<Map>, NonNull<Self>)> {
        let mut hash_map: HashMap<Term, Term> = HashMap::with_capacity(slice.len());

        for (entry_key, entry_value) in slice {
            hash_map.insert(*entry_key, *entry_value);
        }

        Self::new_map_from_hash_map(hash_map)
    }

    pub fn new_reference(
        scheduler_id: scheduler::ID,
        number: ReferenceNumber,
    ) -> AllocResult<(Boxed<Reference>, NonNull<Self>)> {
        let layout = Reference::layout();
        let mut non_null_heap_fragment = Self::new(layout)?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        heap_fragment
            .reference(scheduler_id, number)
            .map(|boxed_reference| (boxed_reference, non_null_heap_fragment))
    }

    pub fn new_resource<V: 'static>(value: V) -> AllocResult<(Boxed<Resource>, NonNull<Self>)> {
        let layout = Layout::new::<Resource>();
        let mut non_null_heap_fragment = Self::new(layout)?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        heap_fragment
            .resource(value)
            .map(|boxed_resource| (boxed_resource, non_null_heap_fragment))
    }

    pub fn new_subbinary_from_original(
        original: Term,
        byte_offset: usize,
        bit_offset: u8,
        full_byte_len: usize,
        partial_byte_bit_len: u8,
    ) -> AllocResult<(Boxed<SubBinary>, NonNull<HeapFragment>)> {
        let layout = Layout::new::<SubBinary>();
        let mut non_null_heap_fragment = Self::new(layout)?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        heap_fragment
            .subbinary_from_original(
                original,
                byte_offset,
                bit_offset,
                full_byte_len,
                partial_byte_bit_len,
            )
            .map(|boxed_subbinary| (boxed_subbinary, non_null_heap_fragment))
    }

    pub fn new_tuple_from_slice(slice: &[Term]) -> AllocResult<(Boxed<Tuple>, NonNull<Self>)> {
        let layout = Tuple::recursive_layout_for(slice);

        let mut non_null_heap_fragment = Self::new(layout)?;
        let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

        match heap_fragment.tuple_from_slice(slice) {
            Ok(boxed_tuple) => Ok((boxed_tuple, non_null_heap_fragment)),
            Err(_) => {
                let formatted_element_vec: Vec<String> =
                    slice.iter().map(|term| term.to_string()).collect();
                let formatted_elements = formatted_element_vec.join(", ");
                let formatted_slice = format!("[{}]", formatted_elements);
                panic!(
                    "Heap fragment ({:?}) could not allocate a tuple with {} elements ({})",
                    unsafe { non_null_heap_fragment.as_mut() },
                    slice.len(),
                    formatted_slice
                )
            }
        }
    }
}
impl Drop for HeapFragment {
    fn drop(&mut self) {
        assert!(!self.link.is_linked());
        // Check if the contained value needs to have its destructor run
        let ptr = self.raw.base.as_ptr() as *mut Term;
        let term = unsafe { *ptr };
        term.release();
        // Actually deallocate the memory backing this fragment
        let (layout, _offset) = Layout::new::<Self>().extend(self.raw.layout()).unwrap();
        unsafe {
            let ptr = NonNull::new_unchecked(self as *const _ as *mut u8);
            std_alloc::dealloc(ptr, layout);
        }
    }
}
impl Heap for HeapFragment {
    fn is_corrupted(&self) -> bool {
        // TODO real check
        false
    }

    #[inline]
    fn heap_start(&self) -> *mut Term {
        self.raw.base.as_ptr() as *mut Term
    }

    #[inline]
    fn heap_top(&self) -> *mut Term {
        self.top as *mut Term
    }

    #[inline]
    fn heap_end(&self) -> *mut Term {
        unsafe { self.raw.base.as_ptr().add(self.raw.size) as *mut Term }
    }
}
impl HeapAlloc for HeapFragment {
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        // Ensure layout has alignment padding
        let layout = layout.align_to(MIN_ALIGN).unwrap().pad_to_align();
        // Capture the base pointer for this allocation
        let top = self.heap_top() as *mut u8;
        // Calculate available space and fail if not enough is free
        let needed = layout.size();
        let end = self.heap_end() as *mut u8;
        if erts::to_word_size(needed) > self.heap_available() {
            return Err(alloc!());
        }
        // Calculate new top of the heap
        let new_top = top.add(needed);
        debug_assert!(new_top <= end);
        self.top = new_top;
        // Ensure base pointer for allocation fulfills minimum alignment requirements
        let align = layout.align();
        let ptr = if is_aligned_at(top, align) {
            top as *mut Term
        } else {
            align_up_to(top as *mut Term, align)
        };
        // Success!
        debug_assert!(is_aligned(ptr));
        Ok(NonNull::new_unchecked(ptr))
    }
}
