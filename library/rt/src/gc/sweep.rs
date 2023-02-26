use alloc::alloc::AllocError;
use alloc::sync::Arc;
use core::mem;
use core::ops::Deref;

use firefly_alloc::clone::WriteCloneIntoRaw;
use firefly_binary::Bitstring;

use log::trace;

use crate::term::{BinaryData, BitSlice, Closure, Cons, Map, OpaqueTerm, Term, Tuple};
use crate::term::{Boxable, Header, MatchContext, Tag};

use super::*;

pub trait Sweep<Output = Self> {
    fn sweep<C>(&mut self, collector: &C) -> Result<Move<Output>, AllocError>
    where
        C: CollectionType;
}

pub enum Move<T> {
    Ok { to: T, bytes_moved: usize },
    Skipped,
}
impl<T: Default> Default for Move<T> {
    fn default() -> Self {
        Self::Skipped
    }
}
impl<T> From<T> for Move<T> {
    fn from(to: T) -> Self {
        Self::Ok {
            to,
            bytes_moved: mem::size_of::<T>(),
        }
    }
}

impl Sweep<()> for Root {
    fn sweep<C>(&mut self, collector: &C) -> Result<Move<()>, AllocError>
    where
        C: CollectionType,
    {
        match *self {
            Self::Raw(ptr) => {
                let mut opaque = unsafe { *ptr };
                // Reference-counted terms must have their ref count incremented
                opaque.maybe_increment_refcount();
                // Skip all non-boxed or reference-counted types
                if !opaque.is_gcbox() && !opaque.is_cons_or_tuple() {
                    trace!(target: "gc", "skipping opaque root: {}", opaque);
                    return Ok(Move::Skipped);
                }
                match opaque.sweep(collector)? {
                    Move::Ok { to, bytes_moved } => {
                        unsafe {
                            ptr.write(to);
                        }
                        Ok(Move::Ok {
                            to: (),
                            bytes_moved,
                        })
                    }
                    Move::Skipped => Ok(Move::Skipped),
                }
            }
            Self::Term(ptr) => {
                let term = unsafe { &mut *ptr };
                // Skip all non-boxed or reference-counted types
                if !term.is_box() {
                    trace!(target: "gc", "skipping term root: {}", term);
                    return Ok(Move::Skipped);
                }
                match term.sweep(collector)? {
                    Move::Ok { to, bytes_moved } => {
                        unsafe {
                            ptr.write(to);
                        }
                        Ok(Move::Ok {
                            to: (),
                            bytes_moved,
                        })
                    }
                    Move::Skipped => Ok(Move::Skipped),
                }
            }
        }
    }
}
impl<T> Sweep for Gc<T>
where
    T: Boxable,
{
    default fn sweep<C>(&mut self, collector: &C) -> Result<Move<Self>, AllocError>
    where
        C: CollectionType,
    {
        if self.is_moved() {
            let to = self.forwarded_to();
            return Ok(Move::Ok {
                to: unsafe { Gc::from_raw_parts(to, ()) },
                bytes_moved: 0,
            });
        }

        let ptr = Gc::as_ptr(self);
        if !collector.should_sweep(ptr) {
            return Ok(Move::Skipped);
        }

        let heap = collector.sweep_to(ptr);
        let layout = self.deref().layout_excluding_heap(heap);
        let heap_available = heap.heap_available();
        if heap_available < layout.size() {
            trace!(target: "gc", "insufficient heap space for value {} available vs {} required", heap_available, layout.size());
            return Err(AllocError);
        }
        let bytes_moved = layout.size();
        let cloned = self.deref().unsafe_clone_to_heap(heap);
        unsafe {
            let header_ptr = ptr.cast::<OpaqueTerm>();
            header_ptr.write(cloned.into());
            header_ptr.add(1).write(OpaqueTerm::hole(bytes_moved));
        }
        Ok(Move::Ok {
            to: cloned,
            bytes_moved,
        })
    }
}
impl Sweep for Gc<Cons> {
    fn sweep<C>(&mut self, collector: &C) -> Result<Move<Self>, AllocError>
    where
        C: CollectionType,
    {
        if self.is_move_marker() {
            let to = unsafe { self.forwarded_to() };
            return Ok(Move::Ok { to, bytes_moved: 0 });
        }

        let (ptr, _) = self.to_raw_parts();
        if !collector.should_sweep(ptr) {
            return Ok(Move::Skipped);
        }

        let mut moved = 0;
        let head = {
            let mut head = self.head;
            match head.sweep(collector)? {
                Move::Ok { to, bytes_moved } => {
                    moved += bytes_moved;
                    to
                }
                Move::Skipped => head,
            }
        };
        let tail = {
            let mut tail = self.tail;
            match tail.sweep(collector)? {
                Move::Ok { to, bytes_moved } => {
                    moved += bytes_moved;
                    to
                }
                Move::Skipped => tail,
            }
        };
        let mut to = Cons::new_uninit_in(collector.sweep_to(ptr)).unwrap();
        let to = unsafe {
            to.write(Cons { head, tail });
            to.assume_init()
        };
        moved += mem::size_of::<Cons>();
        self.head = OpaqueTerm::NONE;
        self.tail = to.into();
        Ok(Move::Ok {
            to,
            bytes_moved: moved,
        })
    }
}
impl Sweep for Gc<Tuple> {
    fn sweep<C>(&mut self, collector: &C) -> Result<Move<Self>, AllocError>
    where
        C: CollectionType,
    {
        let (ptr, metadata) = self.to_raw_parts();

        let header_ptr = ptr.cast::<OpaqueTerm>();
        let header = unsafe { *header_ptr };
        if !header.is_header() {
            // If moved, we just need to rewrite the root with the new location
            assert!(header.is_tuple());
            unsafe {
                let ptr = header.as_ptr();
                let to = Gc::<Tuple>::from_raw_parts(ptr, metadata);
                header_ptr.write(to.into());
                header_ptr
                    .add(1)
                    .write(OpaqueTerm::hole(mem::size_of_val(to.deref())));
                return Ok(Move::Ok { to, bytes_moved: 0 });
            }
        }

        if !collector.should_sweep(ptr) {
            return Ok(Move::Skipped);
        }

        // Clone the tuple to the target heap and rewrite the origin tuple header as a move marker
        let mut moved = 0;
        for element in self.as_mut_slice() {
            match element.sweep(collector)? {
                Move::Ok { to, bytes_moved } => {
                    moved += bytes_moved;
                    *element = to;
                }
                Move::Skipped => continue,
            }
        }
        let to = Tuple::from_slice(self.as_slice(), collector.sweep_to(ptr)).unwrap();
        let hole_size = mem::size_of_val(self.deref());
        moved += hole_size;
        unsafe {
            header_ptr.write(to.into());
            header_ptr.add(1).write(OpaqueTerm::hole(hole_size));
        }
        Ok(Move::Ok {
            to,
            bytes_moved: moved,
        })
    }
}
impl Sweep for Gc<Map> {
    fn sweep<C>(&mut self, collector: &C) -> Result<Move<Self>, AllocError>
    where
        C: CollectionType,
    {
        let (ptr, metadata) = self.to_raw_parts();
        if self.is_moved() {
            let dest = self.forwarded_to();
            return Ok(Move::Ok {
                to: unsafe { Gc::from_raw_parts(dest, metadata) },
                bytes_moved: 0,
            });
        }

        if !collector.should_sweep(ptr) {
            return Ok(Move::Skipped);
        }

        let mut to = Map::clone_from(self, collector.sweep_to(ptr)).unwrap();
        let hole_size = mem::size_of_val(self.deref());
        let mut moved = hole_size;

        let kv = to.as_mut_slice();
        for i in 0..kv.len() {
            let term = unsafe { kv.get_unchecked_mut(i) };
            if let Move::Ok { to, bytes_moved } = term.sweep(collector)? {
                moved += bytes_moved;
                *term = to;
            }
        }
        unsafe {
            let header_ptr = ptr.cast::<OpaqueTerm>();
            header_ptr.write(to.into());
            header_ptr.add(1).write(OpaqueTerm::hole(hole_size));
        }
        Ok(Move::Ok {
            to,
            bytes_moved: moved,
        })
    }
}
impl Sweep for Gc<Closure> {
    fn sweep<C>(&mut self, collector: &C) -> Result<Move<Self>, AllocError>
    where
        C: CollectionType,
    {
        let (ptr, metadata) = self.to_raw_parts();
        if self.is_moved() {
            let dest = self.forwarded_to();
            return Ok(Move::Ok {
                to: unsafe { Gc::from_raw_parts(dest, metadata) },
                bytes_moved: 0,
            });
        }

        if !collector.should_sweep(ptr) {
            return Ok(Move::Skipped);
        }

        let mut to = Closure::clone_from(self, collector.sweep_to(ptr)).unwrap();
        let hole_size = mem::size_of_val(self.deref());
        let mut moved = hole_size;
        for element in to.env_mut() {
            if let Move::Ok { to, bytes_moved } = element.sweep(collector)? {
                moved += bytes_moved;
                *element = to;
            }
        }
        unsafe {
            let header_ptr = ptr.cast::<OpaqueTerm>();
            header_ptr.write(to.into());
            header_ptr.add(1).write(OpaqueTerm::hole(hole_size));
        }
        Ok(Move::Ok {
            to,
            bytes_moved: moved,
        })
    }
}
impl Sweep for Gc<BinaryData> {
    fn sweep<C>(&mut self, collector: &C) -> Result<Move<Self>, AllocError>
    where
        C: CollectionType,
    {
        let (ptr, metadata) = self.to_raw_parts();
        if self.is_moved() {
            let dest = self.forwarded_to();
            return Ok(Move::Ok {
                to: unsafe { Gc::from_raw_parts(dest, metadata) },
                bytes_moved: 0,
            });
        }

        if !collector.should_sweep(ptr) {
            return Ok(Move::Skipped);
        }

        let to = self.deref().clone_to_heap(collector.sweep_to(ptr)).unwrap();
        let hole_size = mem::size_of_val(self.deref());
        unsafe {
            let header_ptr = ptr.cast::<OpaqueTerm>();
            header_ptr.write(to.into());
            header_ptr.add(1).write(OpaqueTerm::hole(hole_size));
        }
        Ok(Move::Ok {
            to,
            bytes_moved: hole_size,
        })
    }
}
impl Sweep for Gc<MatchContext> {
    fn sweep<C>(&mut self, collector: &C) -> Result<Move<Self>, AllocError>
    where
        C: CollectionType,
    {
        if self.is_moved() {
            let dest = self.forwarded_to();
            return Ok(Move::Ok {
                to: unsafe { Gc::from_raw_parts(dest, ()) },
                bytes_moved: 0,
            });
        }

        let ptr = Gc::as_ptr(self);
        if !collector.should_sweep(ptr) {
            return Ok(Move::Skipped);
        }

        let to = self.clone_to_heap(collector.sweep_to(ptr)).unwrap();
        unsafe {
            let header_ptr = ptr.cast::<OpaqueTerm>();
            header_ptr.write(to.into());
            header_ptr
                .add(1)
                .write(OpaqueTerm::hole(mem::size_of::<MatchContext>()));
        }
        Ok(Move::Ok {
            to,
            bytes_moved: mem::size_of::<MatchContext>(),
        })
    }
}
impl Sweep<Term> for Gc<BitSlice> {
    fn sweep<C>(&mut self, collector: &C) -> Result<Move<Term>, AllocError>
    where
        C: CollectionType,
    {
        if self.is_moved() {
            let dest = self.forwarded_to();
            let header = unsafe { &*dest.cast::<Header>() };
            match header.tag() {
                Tag::Slice => {
                    return Ok(Move::Ok {
                        to: Term::RefBinary(unsafe { Gc::from_raw_parts(dest, ()) }),
                        bytes_moved: 0,
                    });
                }
                Tag::Binary => {
                    let bin = unsafe { <BinaryData as Boxable>::from_raw_parts(dest, *header) };
                    let is_small = unsafe { (&*bin).flags().is_small() };
                    if is_small {
                        return Ok(Move::Ok {
                            to: Term::HeapBinary(unsafe { Gc::from_raw(bin) }),
                            bytes_moved: 0,
                        });
                    } else {
                        let rc = unsafe { Arc::from_raw(bin.cast_const()) };
                        let cloned = rc.clone();
                        mem::forget(rc);
                        return Ok(Move::Ok {
                            to: Term::RcBinary(cloned),
                            bytes_moved: 0,
                        });
                    }
                }
                _ => panic!("unexpected tag for binary/bit slice"),
            }
        }

        let (ptr, _) = self.to_raw_parts();
        if !collector.should_sweep(ptr) {
            return Ok(Move::Skipped);
        }

        if self.is_owner_literal() {
            let mut to = Gc::<BitSlice>::new_uninit_in(collector.sweep_to(ptr)).unwrap();
            let cloned = unsafe {
                (**self).write_clone_into_raw(to.as_mut_ptr());
                to.assume_init()
            };
            let moved = mem::size_of::<BitSlice>();
            unsafe {
                let header_ptr = ptr.cast::<OpaqueTerm>();
                header_ptr.write(to.into());
                header_ptr.add(1).write(OpaqueTerm::hole(moved));
            }
            return Ok(Move::Ok {
                to: Term::RefBinary(cloned),
                bytes_moved: moved,
            });
        }

        if self.is_owner_refcounted() {
            // If the referenced data fits in a heap binary, clone just the referenced
            // data and release the reference to the owner. Otherwise, clone the bit slice
            // as-is.
            let byte_size = self.byte_size();
            if byte_size > BinaryData::MAX_HEAP_BYTES {
                let mut to = Gc::<BitSlice>::new_uninit_in(collector.sweep_to(ptr)).unwrap();
                let cloned = unsafe {
                    (**self).write_clone_into_raw(to.as_mut_ptr());
                    to.assume_init()
                };
                let moved = mem::size_of::<BitSlice>();
                unsafe {
                    let header_ptr = ptr.cast::<OpaqueTerm>();
                    header_ptr.write(to.into());
                    header_ptr.add(1).write(OpaqueTerm::hole(moved));
                }
                return Ok(Move::Ok {
                    to: Term::RefBinary(cloned),
                    bytes_moved: moved,
                });
            }
            let mut to =
                BinaryData::with_capacity_small(byte_size, collector.sweep_to(ptr)).unwrap();
            to.copy_from_selection(self.as_selection());
            self.owner.maybe_decrement_refcount();
            unsafe {
                let header_ptr = ptr.cast::<OpaqueTerm>();
                header_ptr.write(to.into());
                header_ptr
                    .add(1)
                    .write(OpaqueTerm::hole(mem::size_of::<BitSlice>()));
            }
            return Ok(Move::Ok {
                to: Term::HeapBinary(to),
                bytes_moved: byte_size,
            });
        }

        // The original is a heap binary, so we're going to create a new one from its data
        let byte_size = self.byte_size();
        let mut to = BinaryData::with_capacity_small(byte_size, collector.sweep_to(ptr))?;
        to.copy_from_selection(self.as_selection());
        unsafe {
            let header_ptr = ptr.cast::<OpaqueTerm>();
            header_ptr.write(to.into());
            header_ptr
                .add(1)
                .write(OpaqueTerm::hole(mem::size_of::<BitSlice>()));
        }
        Ok(Move::Ok {
            to: Term::HeapBinary(to),
            bytes_moved: byte_size,
        })
    }
}
impl Sweep for OpaqueTerm {
    fn sweep<C>(&mut self, collector: &C) -> Result<Move<Self>, AllocError>
    where
        C: CollectionType,
    {
        let this = *self;
        // Reference-counted terms must have their ref count incremented
        this.maybe_increment_refcount();
        // Reference-counted values, and any special/immediate values can be returned unchanged
        if this.is_rc() || !this.is_box() || this.is_literal() {
            return Ok(Move::Ok {
                to: this,
                bytes_moved: 0,
            });
        }
        trace!(target: "gc", "sweeping opaque term: {}", self);
        // If the value is already moved, we don't need to do anything but return
        // the forwarded term.
        if let Some(ptr) = this.move_marker() {
            trace!(target: "gc", "term was moved to {:p}", ptr.as_ptr());
            if this.is_nonempty_list() {
                return Ok(Move::Ok {
                    to: ptr.cast::<Cons>().into(),
                    bytes_moved: 0,
                });
            } else if this.is_tuple() {
                let header = ptr.cast::<Header>();
                let arity = unsafe { header.as_ref().arity() };
                let tuple = NonNull::<Tuple>::from_raw_parts(ptr, arity);
                return Ok(Move::Ok {
                    to: tuple.into(),
                    bytes_moved: 0,
                });
            } else {
                assert!(this.is_gcbox());
                let boxed = unsafe { Gc::from_raw(ptr.as_ptr()) };
                return Ok(Move::Ok {
                    to: boxed.into(),
                    bytes_moved: 0,
                });
            }
        }
        // Otherwise we need to sweep the value first
        if this.is_nonempty_list() {
            let mut cons = unsafe { Gc::<Cons>::from_raw_parts(this.as_ptr(), ()) };
            cons.sweep(collector).map(to_opaque_term_move)
        } else if this.is_tuple() {
            let mut tuple = unsafe {
                let ptr = this.as_ptr();
                let header = *ptr.cast::<Header>();
                let ptr = <Tuple as Boxable>::from_raw_parts(ptr, header);
                Gc::from_raw(ptr)
            };
            tuple.sweep(collector).map(to_opaque_term_move)
        } else {
            // Match contexts are not represented in the Term enum, so we have
            // to handle them separately.
            if this.is_match_context() {
                let ptr = unsafe { this.as_ptr().cast::<MatchContext>() };
                let mut mc = unsafe { Gc::from_raw(ptr) };
                mc.sweep(collector).map(to_opaque_term_move)
            } else {
                assert!(this.is_gcbox(), "{:?}", this.r#typeof());
                let mut term: Term = this.into();
                term.sweep(collector).map(to_opaque_term_move)
            }
        }
    }
}
impl Sweep for Term {
    fn sweep<C>(&mut self, collector: &C) -> Result<Move<Self>, AllocError>
    where
        C: CollectionType,
    {
        match self {
            // Boxed terms are handled uniformly
            Term::Cons(boxed) => boxed.sweep(collector).map(to_term_move),
            Term::Tuple(boxed) => boxed.sweep(collector).map(to_term_move),
            Term::BigInt(boxed) => boxed.sweep(collector).map(to_term_move),
            Term::Map(boxed) => boxed.sweep(collector).map(to_term_move),
            Term::Closure(boxed) => boxed.sweep(collector).map(to_term_move),
            Term::Pid(boxed) => boxed.sweep(collector).map(to_term_move),
            Term::Reference(boxed) => boxed.sweep(collector).map(to_term_move),
            Term::HeapBinary(boxed) => boxed.sweep(collector).map(to_term_move),
            Term::RefBinary(boxed) => boxed.sweep(collector),
            // All other term types are trivially cloned
            term => Ok(Move::Ok {
                to: term.clone(),
                bytes_moved: 0,
            }),
        }
    }
}

#[inline]
fn to_opaque_term_move<T: Into<OpaqueTerm>>(m: Move<T>) -> Move<OpaqueTerm> {
    match m {
        Move::Ok { to, bytes_moved } => Move::Ok {
            to: to.into(),
            bytes_moved,
        },
        Move::Skipped => Move::Skipped,
    }
}

#[inline]
fn to_term_move<T: Into<Term>>(m: Move<T>) -> Move<Term> {
    match m {
        Move::Ok { to, bytes_moved } => Move::Ok {
            to: to.into(),
            bytes_moved,
        },
        Move::Skipped => Move::Skipped,
    }
}
