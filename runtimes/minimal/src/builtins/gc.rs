use core::ptr;

use stackmaps::{FrameInfo, StackMap};

use liblumen_alloc::erts::term::prelude::{Boxed, Encoded, Term};
use lumen_rt_core::process::current_process;

/// On x86_64, calling this function with no arguments will result
/// in effectively calling __lumen_builtin_gc.run with the return address
/// of the caller, as well as the base pointer as arguments.
///
/// When __lumen_builtin_gc.run returns, it will return back to the caller
/// directly, rather than returning through this function.
///
/// The purpose of this hack is to allow us to pass the return address of
/// the caller to the garbage collector so that we can locate the frame
/// information for the caller in the stack map. Using the offsets in the
/// frame info combined with the base pointer allows us to calculate the
/// locations of roots on the stack which the collector will need in order
/// to trace live objects, and update those roots accordingly.
#[naked]
#[inline(never)]
#[export_name = "__lumen_builtin_gc.enter"]
pub unsafe fn builtin_gc_enter() {
    llvm_asm!("
    # Move the return address into %rdi
    popq %rdi
    # Copy the base pointer address into %rsi
    movq %rbp, %rsi
    # Pretend like we called run_gc directly, which
    # will return over us back to the caller
    pushq %rdi
    jmp ___lumen_builtin_gc.run
    "
    :
    :
    :
    : "volatile", "alignstack"
    );
}

//    pub fn garbage_collect(&self, need: usize, roots: &mut [Term]) -> Result<usize, GcError> {

/// When this function is called, it uses the provided return address and base pointer to locate
/// the frame information for the caller, and calculate stack addresses containing roots for the
/// garbage collector to trace and update.
///
/// The offsets in the caller's frame information are relative to the base pointer, and by looking
/// 8 bytes above the base pointer to locate the previous frames return address, we can walk up the
/// stack to locate all roots
#[inline(never)]
#[export_name = "__lumen_builtin_gc.run"]
pub unsafe extern "C" fn builtin_gc_run(
    return_address: *const u8,
    base_pointer: *const u8,
) -> bool {
    let iter = RootsIter::new(StackMap::get(), return_address, base_pointer);
    let roots = iter.collect::<Vec<_>>();
    match current_process().garbage_collect(1, roots) {
        Ok(_) => true,
        Err(err) => panic!("garbage collection failed: {}", err),
    }
}

/// This is an iterator over roots; stack slots containing terms that may refer to
/// objects on the process heap. These roots are found by iterating over the stack map
/// for the frame of the caller to the GC, and walking up frames on the stack until all
/// frames have been visited, or until a frame without a stack map is encountered.
struct RootsIter {
    stack_map: &'static StackMap,
    next: Option<&'static FrameInfo>,
    slot_index: usize,
    base_pointer: *const u8,
    return_address: *const u8,
    done: bool,
}
impl RootsIter {
    #[inline]
    fn new(
        stack_map: &'static StackMap,
        return_address: *const u8,
        base_pointer: *const u8,
    ) -> Self {
        Self {
            stack_map,
            next: None,
            slot_index: 0,
            base_pointer,
            return_address,
            done: false,
        }
    }

    fn done(&mut self) {
        self.done = true;
        self.next = None;
        self.base_pointer = ptr::null();
        self.return_address = ptr::null();
        self.slot_index = 0;
    }
}
impl Iterator for RootsIter {
    type Item = Boxed<Term>;

    fn next(&mut self) -> Option<Self::Item> {
        if std::intrinsics::unlikely(self.done) {
            return None;
        }

        loop {
            if let Some(frame_info) = self.next {
                match frame_info.base_slot(self.slot_index) {
                    Some(slot) => {
                        // Load root for this slot, increment slot index, and update
                        // state for next iteration, returning the loaded root
                        let root_addr =
                            unsafe { self.base_pointer.offset(slot.offset as isize) as *mut Term };
                        let root = unsafe { &*root_addr };
                        self.slot_index += 1;
                        // Skip none values or immediates
                        if root.is_none() || root.is_immediate() {
                            continue;
                        }
                        let boxed = unsafe { Boxed::new_unchecked(root_addr) };
                        break Some(boxed);
                    }
                    None => {
                        // No more slots in this frame, try to move to the next frame
                        // up the stack. We do so by locating the return address 8 bytes
                        // above the current base pointer, loading that frame, and updating
                        // the base pointer, slot index, etc. and trying another iteration
                        let next_return_addr =
                            unsafe { *(self.base_pointer.add(8) as *const *const u8) };
                        if let Some(next_frame_info) = self.stack_map.find_frame(next_return_addr) {
                            self.next = Some(next_frame_info);
                            self.base_pointer =
                                unsafe { self.base_pointer.add(next_frame_info.size_in_bytes) };
                            self.return_address = next_return_addr;
                            self.slot_index = 0;
                            continue;
                        } else {
                            // Can't locate stack map for next frame, so we're done, set everything
                            // to defaults
                            self.done();
                            break None;
                        }
                    }
                }
            } else {
                // No frame loaded, so try to load one
                if let Some(frame_info) = self.stack_map.find_frame(self.return_address) {
                    // We found one, so load the frame and try another iteration
                    self.next = Some(frame_info);
                    self.slot_index = 0;
                    continue;
                } else {
                    // No frame available, so we're done
                    self.done();
                    break None;
                }
            }
        }
    }
}
impl core::iter::FusedIterator for RootsIter {}
