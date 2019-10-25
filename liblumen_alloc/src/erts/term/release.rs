use core::ptr;

use super::prelude::*;

/// This trait provides the ability to attach destructors to types
/// in the term representation, e.g. `ProcBin`, which is a reference-counted
/// object, and needs to decrement the count when a handle to it
/// is no longer used.
///
/// NOTE: Implementations should not follow move markers, as moved
/// terms are always considered live during GC, and should not be
/// released
pub trait Release: Encoded {
    /// Executes the destructor for the underlying term
    fn release(self);
}

impl<T> Release for T where T: Encoded {
    default fn release(self) {
        // Skip entirely for headers
        if self.is_header() {
            return;
        }
        // Skip literals as they are never destroyed
        if self.is_literal() {
            return;
        }

        let term = self.decode().unwrap();
        match self.decode() {
            // Ensure we walk tuples and release all their elements
            Ok(TypedTerm::Tuple(nn)) => {
                for element in nn.as_mut().iter() {
                    element.release()
                }
            }
            // Ensure we walk lists and release all their elements
            Ok(TypedTerm::List(nn)) => {
                let mut cons = nn.as_ref();
                loop {
                    // Do not follow moves
                    if cons.is_move_marker() {
                        break;
                    }

                    // If we reached the end of the list, we're done
                    if cons.head.is_nil() {
                        break;
                    }

                    // Otherwise release the head term
                    cons.head.release();

                    // This is more of a sanity check, as the head will be nil for EOL
                    if cons.tail.is_nil() {
                        break;
                    }

                    // If the tail is proper, move into the cell it represents
                    if cons.tail.is_non_empty_list() {
                        let tail_ptr: *const Cons = cons.tail.dyn_cast();
                        cons = unsafe { &*tail_ptr };
                        continue;
                    }

                    // Otherwise if the tail is improper, release it, and we're done
                    cons.tail.release();
                    break;
                }
            }
            // Ensure ref-counted binaries are released properly
            Ok(TypedTerm::ProcBin(nn)) => {
                unsafe { ptr::drop_in_place(nn.as_ptr()) };
            }
            // Ensure ref-counted resources are released properly
            Ok(TypedTerm::ResourceReference(nn)) => {
                unsafe { ptr::drop_in_place(nn.as_ptr()) };
            }
            // Move markers will be decoded as an `Err`, which is fine
            Err(_) => return,
            // All other types have no destructor
            _ => return,
        }
    }
}
