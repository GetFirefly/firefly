use core::ops::DerefMut;

use crate::erts::term::prelude::{ProcBin, Term};

pub trait VirtualAlloc {
    /// Attaches a ProcBin to virtual binary heap
    fn virtual_alloc(&mut self, bin: &ProcBin) -> Term;
}

impl<A, H> VirtualAlloc for H
where
    A: VirtualAlloc,
    H: DerefMut<Target = A>,
{
    fn virtual_alloc(&mut self, bin: &ProcBin) -> Term {
        self.deref_mut().virtual_alloc(bin)
    }
}
