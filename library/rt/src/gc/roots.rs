use alloc::vec::Vec;
use core::fmt;
use core::ops::AddAssign;

use log::trace;

use crate::term::{OpaqueTerm, Term};

use super::*;

#[derive(Copy, Clone)]
pub enum Root {
    Raw(*mut OpaqueTerm),
    Term(*mut Term),
}
impl fmt::Debug for Root {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Raw(ptr) => {
                let term = unsafe { *ptr };
                f.debug_struct("Raw")
                    .field("value", &format_args!("{}", &term))
                    .field("address", &format_args!("{:p}", ptr))
                    .finish()
            }
            Self::Term(ptr) => {
                let term = unsafe { &*ptr };
                f.debug_struct("Term")
                    .field("value", &format_args!("{}", &term))
                    .field("address", &format_args!("{:p}", ptr))
                    .finish()
            }
        }
    }
}

#[derive(Default)]
pub struct RootSet {
    roots: Vec<Root>,
}
impl fmt::Debug for RootSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("RootSet")
            .field("roots", &self.roots)
            .finish()
    }
}
impl AddAssign<*mut OpaqueTerm> for RootSet {
    fn add_assign(&mut self, root_ptr: *mut OpaqueTerm) {
        assert!(!root_ptr.is_null());
        let root = unsafe { *root_ptr };
        // Anything which isn't a heap-allocated term can be ignored
        if !root.is_box() {
            return;
        }
        // Literals can be ignored
        if root.is_literal() {
            return;
        }
        self.roots.push(Root::Raw(root_ptr));
    }
}
impl AddAssign<*mut Term> for RootSet {
    fn add_assign(&mut self, root: *mut Term) {
        let r = unsafe { &*root };
        if r.is_box() {
            self.roots.push(Root::Term(root));
        }
    }
}
impl RootSet {
    pub fn with_capacity(size: usize) -> Self {
        Self {
            roots: Vec::with_capacity(size),
        }
    }

    pub fn pop(&mut self) -> Option<Root> {
        self.roots.pop()
    }

    /// Sweeps all of the roots in this set to `target`.
    ///
    /// NOTE: The roots are only swept as shallow clones, it is expected that
    /// a specific collector will follow this operation up by sweeping the target
    /// heap for any references out of that heap. We could do that here, but it
    /// would make the collection sweep more expensive due to redundant work.
    ///
    /// Returns the number of bytes moved in the process.
    pub fn sweep_roots<C>(&mut self, target: &C) -> Result<usize, GcError>
    where
        C: CollectionType,
    {
        trace!(target: "process", "sweeping root set to target heap");

        // Follow roots and move values as appropriate
        let mut moved = 0;
        while let Some(mut root) = self.pop() {
            if let Move::Ok { bytes_moved, .. } = root.sweep(target)? {
                moved += bytes_moved;
            }
        }

        trace!(target: "process", "all roots have been swept, moved {} bytes", moved);

        Ok(moved)
    }
}
