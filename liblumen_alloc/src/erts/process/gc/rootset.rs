use core::slice;

use alloc::vec::Vec;

use crate::erts::term::prelude::{Term, Encoded};

/// This struct contains the set of roots which are to be scanned during garbage collection
///
/// The root set is effectively a vector of pointers to terms, i.e. pointers to the roots,
/// rather than the roots directly, this is because the roots are modified during garbage
/// collection to point to the new locations of the values they reference, so we need the
/// pointer to the root to perform the replacement
pub struct RootSet(Vec<*mut Term>);
impl RootSet {
    pub fn new(roots: &mut [Term]) -> Self {
        let len = roots.len();
        let mut set = Vec::with_capacity(len);
        if len > 0 {
            for root in roots {
                set.push(root as *mut _);
            }
        }
        Self(set)
    }

    pub fn empty() -> Self {
        Self(Vec::new())
    }

    #[inline]
    pub fn push(&mut self, root: *mut Term) {
        self.0.push(root);
    }

    #[inline]
    pub fn push_range(&mut self, start: *mut Term, size: usize) {
        let end = unsafe { start.add(size) as usize };
        let mut pos = start;

        while (pos as usize) < (end as usize) {
            let term = unsafe { *pos };

            if term.is_immediate() || term.is_boxed() || term.is_non_empty_list() {
                self.0.push(pos);
            }

            let skip = term.sizeof();
            pos = unsafe { pos.add(skip) };
        }
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<*mut Term> {
        self.0.as_slice().iter()
    }
}
impl Default for RootSet {
    fn default() -> Self {
        Self::empty()
    }
}
