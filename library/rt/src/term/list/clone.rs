use alloc::alloc::{AllocError, Layout};

use firefly_alloc::heap::{EmptyHeap, Heap};

use crate::gc::Gc;
use crate::term::{LayoutBuilder, OpaqueTerm};

use super::*;

/// Support for `Boxable::clone_to_heap`
impl Cons {
    pub fn layout(&self) -> Layout {
        const EMPTY: EmptyHeap = EmptyHeap;
        self.layout_excluding_heap(&EMPTY)
    }

    pub fn layout_excluding_heap<H: ?Sized + Heap>(&self, heap: &H) -> Layout {
        let mut builder = LayoutBuilder::new();
        if !heap.contains((self as *const Self).cast()) {
            for result in self.iter() {
                match result {
                    Ok(term) => {
                        builder += term.layout_excluding_heap(heap);
                        builder.build_cons();
                    }
                    Err(improper) => {
                        builder += improper.tail.layout_excluding_heap(heap);
                    }
                }
            }
        }
        builder.build_cons();
        builder.finish()
    }

    pub fn clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Result<Gc<Self>, AllocError> {
        let layout = self.layout_excluding_heap(heap);
        if heap.heap_available() < layout.size() {
            return Err(AllocError);
        }

        Ok(unsafe { self.unsafe_clone_to_heap(heap) })
    }

    pub unsafe fn unsafe_clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Gc<Self> {
        use smallvec::SmallVec;

        let ptr = self as *const Self;
        if heap.contains(ptr.cast()) {
            unsafe { Gc::from_raw(ptr.cast_mut()) }
        } else {
            let mut items = SmallVec::<[OpaqueTerm; 4]>::new();
            let mut improper = None;
            for result in self.iter() {
                match result {
                    Ok(term) => unsafe {
                        let term = term.unsafe_clone_to_heap(heap);
                        items.push(term.into());
                    },
                    Err(improper_list) => unsafe {
                        let term = improper_list.tail.unsafe_clone_to_heap(heap);
                        improper = Some(term.into());
                    },
                }
            }
            let mut builder = match improper {
                None => ListBuilder::new(heap),
                Some(term) => ListBuilder::new_improper(term, heap),
            };
            for item in items.drain(..).rev() {
                builder.push_unsafe(item).unwrap();
            }
            builder.finish().unwrap()
        }
    }
}
