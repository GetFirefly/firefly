use std::fmt;
use std::slice;

use crate::erts::term::prelude::{Boxed, Encoded, Term};

/// This struct contains the set of roots which are to be scanned during garbage collection
///
/// The root set is effectively a vector of pointers to terms, i.e. pointers to the roots,
/// rather than the roots directly, this is because the roots are modified during garbage
/// collection to point to the new locations of the values they reference, so we need the
/// pointer to the root to perform the replacement
pub struct RootSet(Vec<Boxed<Term>>);
impl RootSet {
    pub fn new(roots: &mut [Term]) -> Self {
        let len = roots.len();
        let mut set = Vec::with_capacity(len);
        if len > 0 {
            for root in roots {
                // Skip immediates
                if root.is_immediate() {
                    continue;
                }
                set.push(unsafe { Boxed::new_unchecked(root) });
            }
        }
        Self(set)
    }

    pub fn empty() -> Self {
        Self(Vec::new())
    }

    #[inline]
    pub fn push(&mut self, root: *mut Term) {
        let root = unsafe { &mut *root };
        // Ignore immediates
        if root.is_immediate() {
            return;
        }
        self.0.push(Boxed::new(root).unwrap());
    }

    #[inline]
    pub fn push_range(&mut self, start: *mut Term, size: usize) {
        let end = unsafe { start.add(size) };
        let mut pos = start;

        while pos < end {
            let term = unsafe { &*pos };
            if term.is_boxed() || term.is_non_empty_list() {
                // Add pointer to box
                self.0.push(unsafe { Boxed::new_unchecked(pos) });
                pos = unsafe { pos.add(1) };
            } else if term.is_header() {
                // Add pointer to header
                self.0.push(unsafe { Boxed::new_unchecked(pos) });
                pos = unsafe { pos.add(term.arity()) };
            } else {
                // For all others, skip over
                pos = unsafe { pos.add(1) };
            }
        }
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<Boxed<Term>> {
        self.0.as_slice().iter()
    }
}
impl From<Vec<Boxed<Term>>> for RootSet {
    #[inline]
    fn from(roots: Vec<Boxed<Term>>) -> Self {
        Self(roots)
    }
}
// NOTE: This is for legacy code that should probably be removed
impl From<&mut [Term]> for RootSet {
    fn from(roots: &mut [Term]) -> Self {
        Self::new(roots)
    }
}
impl Default for RootSet {
    fn default() -> Self {
        Self::empty()
    }
}
impl fmt::Debug for RootSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for root in self.0.iter().map(|b| b.as_ptr()) {
            unsafe {
                let term = &*root;
                let decoded = term.decode();
                f.write_fmt(format_args!(
                    "  {:p}: {:0bit_len$b} {:?}\n",
                    root,
                    *(root as *const usize),
                    decoded,
                    bit_len = (core::mem::size_of::<usize>() * 8)
                ))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use core::ptr;

    use crate::erts::process::alloc::TermAlloc;
    use crate::erts::term::prelude::*;
    use crate::erts::testing::RegionHeap;

    use super::*;

    #[test]
    fn modified_rootset_updates_roots() {
        let mut heap = RegionHeap::default();

        let tuple = heap
            .tuple_from_slice(&[atom!("hello"), atom!("world")])
            .unwrap();
        let mut stack_ref: Term = tuple.into();

        let mut rootset = RootSet::empty();
        rootset.push(&mut stack_ref as *mut _);

        for root in rootset.iter() {
            let root_ref = root.as_ref();
            assert!(root_ref.is_boxed());
            unsafe {
                ptr::write(root.as_ptr(), Term::NIL);
            }
        }

        assert_eq!(stack_ref, Term::NIL);
    }
}
